# DROUGA — Computer Vision Pipeline

Full technical reference for the detection and classification pipeline running on the Jetson
Orin Nano during the ERL 2026 competition.

---

## Table of Contents

1. [Hardware and Inputs](#1-hardware-and-inputs)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Stage 1 — Electronic Image Stabilisation (EIS)](#3-stage-1--electronic-image-stabilisation-eis)
4. [Stage 2 — YOLO Detection and ByteTrack Tracking](#4-stage-2--yolo-detection-and-bytetrack-tracking)
5. [Coordinate Remapping (Stabilised → Raw)](#5-coordinate-remapping-stabilised--raw)
6. [Stage 3 — Metric 1: World-Frame Centroid Stability](#6-stage-3--metric-1-world-frame-centroid-stability)
7. [Stage 4 — Metric 2: Intra-Bbox Optical Flow](#7-stage-4--metric-2-intra-bbox-optical-flow)
8. [Stage 5 — Classification Gate](#8-stage-5--classification-gate)
9. [VSLAM Fallback Mode](#9-vslam-fallback-mode)
10. [All Parameters — Reference Table](#10-all-parameters--reference-table)
11. [Tuning Guide](#11-tuning-guide)
12. [Published Topics](#12-published-topics)
13. [Subscribed Topics](#13-subscribed-topics)
14. [Known Limitations and Risks](#14-known-limitations-and-risks)
15. [Planned: Metric 3 — Pose-Based Joint Stability](#15-planned-metric-3--pose-based-joint-stability)
16. [Compute Budget](#16-compute-budget)

---

## 1. Hardware and Inputs

| Component | Model | Role |
|-----------|-------|------|
| Depth camera | Intel RealSense D435**i** | Color + aligned depth + built-in IMU (gyro + accel) |
| Companion computer | Jetson Orin Nano | Runs all CV and ROS2 nodes |
| Visual SLAM | Isaac ROS Visual SLAM | Produces full 6-DOF camera pose at ~30Hz |

The D435i publishes three streams used by the pipeline:

| Topic | Type | Rate | Purpose |
|-------|------|------|---------|
| `/camera/color/image_raw` | sensor_msgs/Image (bgr8) | 30 Hz | Color frame for YOLO |
| `/camera/aligned_depth_to_color/image_raw` | sensor_msgs/Image (uint16, mm) | 30 Hz | Depth aligned to color pixels |
| `/camera/color/camera_info` | sensor_msgs/CameraInfo | 30 Hz | Lens intrinsics (fx, fy, cx, cy) |
| `/camera/imu` | sensor_msgs/Imu | 400 Hz | Gyroscope data (fallback only) |

Isaac ROS Visual SLAM publishes:

| Topic | Type | Rate | Purpose |
|-------|------|------|---------|
| `/visual_slam/tracking/odometry` | nav_msgs/Odometry | ~30 Hz | Camera pose in world frame |
| `/visual_slam/status` | VisualSlamStatus | ~30 Hz | Tracking state (good/bad/lost) |

---

## 2. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  D435i: color frame + depth frame (hardware-synchronised, 30 Hz)    │
│  Isaac ROS VSLAM: T_world_cam odometry                              │
└───────────────────┬─────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Stage 1 — EIS warp   │  Removes vibration BEFORE YOLO sees the frame.
        │  (full frame pre-YOLO)│  Uses VSLAM orientation via adaptive EMA filter.
        └──────────┬────────────┘
                   │ stab_frame (warped BGR)
                   ▼
        ┌───────────────────────┐
        │ Stage 2 — YOLO +      │  Detects mannequins, assigns ByteTrack IDs.
        │ ByteTrack             │  Runs on the stabilised frame.
        └──────────┬────────────┘
                   │ detections in (u_stab, v_stab)
                   ▼
        ┌───────────────────────┐
        │ Coord remap           │  H_stab_inv maps (u_stab,v_stab) → (u_raw,v_raw).
        │ stab → raw frame      │  All depth / 3D uses raw frame coordinates.
        └──────────┬────────────┘
                   │ detections in (u_raw, v_raw)  ┐
                   ├────────────────────────────────┤
                   ▼                                ▼
        ┌───────────────────┐         ┌─────────────────────────────┐
        │ Stage 3 — Metric 1│         │ Stage 4 — Metric 2          │
        │ World centroid    │         │ Intra-bbox optical flow      │
        │ stability         │         │ (ego-compensated)            │
        │                   │         │                              │
        │ depth → 3D camera │         │ H_rot = K R_rel K⁻¹         │
        │ T_world_cam → 3D  │         │ warp prev_raw → curr_raw     │
        │ world             │         │ crop to bbox                 │
        │ rolling σ < 8cm   │         │ Farneback flow               │
        │ → stationary      │         │ p75 < 1.5 px → rigid body   │
        └──────────┬────────┘         └─────────────┬───────────────┘
                   │ centroid_ok                     │ flow_ok
                   └─────────────────┬───────────────┘
                                     ▼
                        ┌───────────────────────┐
                        │ Stage 5 — Gate        │  confirmed (≥25 frames)
                        │                       │  AND centroid_ok
                        │                       │  AND flow_ok
                        └──────────┬────────────┘
                                   │
                                   ▼
                         is_mannequin = True/False

                         Published on /drouga/mannequin_*
```

---

## 3. Stage 1 — Electronic Image Stabilisation (EIS)

### Why it exists

The drone's motors produce high-frequency vibration (50–200 Hz). At 30fps the camera
captures ~one vibration period per frame. This causes YOLO to see a slightly different
pixel layout every frame, which:

- Reduces detection confidence for distant mannequins.
- Makes ByteTrack bounding boxes jitter, corrupting the flow measurements in Metric 2.

EIS warps each raw frame to a "virtual smooth camera" position before YOLO sees it,
so YOLO always sees a stable image.

### How it works — step by step

**1. Get the current camera orientation from VSLAM**

```
R_curr ← quaternion from /visual_slam/tracking/odometry
```

**2. Adaptive alpha**

The smooth reference is maintained by an exponential moving average (EMA). The blend
weight `alpha` controls how fast the smooth reference follows the real camera:

- Low alpha (e.g. 0.08): heavily smoothed — vibration is removed but the reference is
  slow to follow intentional navigation movements. If the drone turns to face a new
  direction, the reference lags and EIS produces a large warp that shows the wrong part
  of the scene.

- High alpha (e.g. 0.6): lightly smoothed — reference follows quickly so no fighting
  navigation, but less vibration is removed.

To get the best of both, we check the inter-frame rotation angle:

```
angle = degrees(prev_R.inv() × R_curr)    # how much did the camera actually rotate?

if angle > eis_aggressive_threshold (default 5°):
    effective_alpha = 0.6   # large intentional movement — follow it
else:
    effective_alpha = eis_alpha (default 0.08)  # tiny vibration — filter it
```

**3. Update the smooth reference**

LERP (linear interpolation) between the current smooth quaternion and the new
measurement, then renormalise:

```
q_smooth ← normalise( (1 - α) × q_smooth  +  α × q_curr )
```

This is an approximation of SLERP, accurate enough when the angular steps are small
(which they always are at 30fps).

> **Double-cover check**: a unit quaternion `q` and `-q` represent the same rotation.
> Before LERP, we check `dot(q_smooth, q_curr)` and flip `q_curr` if it is negative,
> to ensure interpolation takes the short path around the sphere.

**4. Compute the deviation rotation**

The deviation is what R_curr has that the smooth reference does NOT — i.e. the vibration:

```
R_dev = R_smooth⁻¹ × R_curr
```

**5. Build the pixel-space homography**

A pure rotation of the camera maps pixels according to:

```
H = K × R × K⁻¹
```

To remove the deviation (undo it), we apply its inverse:

```
H_stab = K × R_dev⁻¹ × K⁻¹
```

Applying `H_stab` to the raw frame shifts each pixel to where it would be if the camera
had not vibrated, producing `stab_frame`.

We also store `H_stab_inv = inv(H_stab)` — used in Stage 5 to map detections back.

**6. Apply the warp**

```python
stab_frame = cv2.warpPerspective(raw_frame, H_stab, (width, height))
```

YOLO receives `stab_frame`, not `raw_frame`.

### What EIS does NOT correct

- Translational camera shake (only rotation is corrected, because that is what VSLAM
  provides as a quaternion — the full T_world_cam is used only for 3D localisation).
- Exposure / motion blur from very fast movements.
- Wind gust displacements that are not periodic (handled by Metric 1 instead).

---

## 4. Stage 2 — YOLO Detection and ByteTrack Tracking

### YOLO

- Model: YOLOv8m, single class `mannequin` (class 0).
- Input: `stab_frame` (BGR, 640×480 at runtime — same as camera resolution).
- Inference: TensorRT FP16 engine (`best.engine`) on the Jetson GPU for ~28 FPS.
- Fallback: `best.pt` PyTorch model if `best.engine` is not found (~8 FPS).
- Threshold: `conf` parameter (default 0.35). Detections below this are discarded.

### ByteTrack

ByteTrack is invoked through Ultralytics' built-in `model.track()` with `persist=True`.

```python
result = self.model.track(
    stab_frame,
    conf    = self.conf,
    classes = [0],
    tracker = 'bytetrack.yaml',
    persist = True,
    verbose = False
)[0]
```

ByteTrack maintains track IDs across frames using IoU matching + a Kalman filter.
`persist=True` keeps the Kalman filter state alive between `track()` calls.

Every detection gets:
- `box.id`   — integer track ID (consistent across frames for the same physical object)
- `box.conf` — YOLO confidence score
- `box.xyxy` — bounding box corners in stabilised frame coordinates

### Why ByteTrack over DeepSORT

DeepSORT uses a learned re-identification (ReID) network to distinguish individuals.
These networks are trained on human pedestrian datasets, so their embeddings do not
generalise to mannequins — mannequins with identical clothing all look the same to the
ReID network, causing constant ID swaps. ByteTrack uses only IoU + Kalman prediction
and has no ReID component, which works reliably for a single stationary target.

---

## 5. Coordinate Remapping (Stabilised → Raw)

After EIS, YOLO returns detections in `stab_frame` coordinates. But the depth image
and camera intrinsics are in `raw_frame` coordinates (depth pixels map 1:1 to color
pixels from the D435i hardware alignment — in the raw frame).

We must remap every detection centre and bounding box back to raw coordinates before
using depth or computing 3D positions.

```
H_stab_inv = inv(H_stab)

(cx_raw, cy_raw) = perspectiveTransform( (cx_stab, cy_stab),  H_stab_inv )
```

For the bounding box (used for the Metric 2 crop), we transform all four corners:

```
corners_stab = [ (x1,y1), (x2,y1), (x2,y2), (x1,y2) ]
corners_raw  = perspectiveTransform(corners_stab, H_stab_inv)
bbox_raw     = axis-aligned bounding rect of corners_raw
```

After this step every variable ending in `_r` or `_raw` is in the raw camera frame
and can be safely used with the depth image.

---

## 6. Stage 3 — Metric 1: World-Frame Centroid Stability

### What it measures

The 3D world-frame position of the detection centre, tracked over a rolling window.
A mannequin does not move in the world. A walking human does.

### Step by step

**1. Sample depth at the detection centre (raw frame)**

```
depth_m = median( depth_image[ (cy_raw±5) × (cx_raw±5) ] ) / 1000.0
```

A 10×10 pixel median patch (parameter: `depth_window`) is used instead of a single
pixel to smooth over depth holes and sensor noise.

**2. Back-project to camera frame**

Pinhole camera model (inverted):

```
X_cam = (cx_raw - cx_intrinsic) × depth_m / fx
Y_cam = (cy_raw - cy_intrinsic) × depth_m / fy
Z_cam = depth_m
```

ROS/RealSense convention: Z = forward, X = right, Y = down.

**3. Transform to world frame**

```
P_world = T_world_cam × [X_cam, Y_cam, Z_cam, 1]ᵀ
```

`T_world_cam` is the 4×4 homogeneous transform from the VSLAM odometry message,
converting camera-frame points to world-frame (odom-frame) points.

**4. Maintain a rolling window**

For each ByteTrack ID we keep a deque of `world_stability_window` (default 30) world-frame
3D positions.

**5. Compute stability**

```
centroid_std = mean( std( positions_array, axis=0 ) )
```

This averages the standard deviation across the X, Y, Z axes. Units are metres.

```
centroid_ok = centroid_std < world_stability_threshold (default 0.08 m = 8 cm)
```

**Why 8 cm?**

A well-tuned drone in hover typically has ±1–3 cm of position jitter (GPS-denied
VIO drift). 8 cm gives comfortable headroom so a truly stationary mannequin is always
below the threshold, while a walking human (stride ≈ 60 cm) is far above it.

### When is this metric suspended?

- VSLAM status is not TRACKING (slam_state ≠ 1 or 2).
- `latest_odom` has not been received yet.
- Depth at the detection centre is 0 (invalid measurement — occluded or too far/close
  for the D435i).

When suspended, the gate falls through to Metric 2 only.

---

## 7. Stage 4 — Metric 2: Body-Motion Check (flow or pose)

Metric 2 detects body motion that Metric 1 misses — gestures, breathing, weight shifts.
Two implementations are available, selected at launch via `classifier_mode`.

### Option A — Intra-Bbox Optical Flow (`classifier_mode=flow`)

How much the object's body moves independently of the camera between two consecutive
frames, measured inside the bounding box in the raw frame.

This catches humans who are stationary in the world frame but making gestures — arm
waves, head turns, weight shifts — that would fool Metric 1.

### Why the raw frame (not the stabilised frame)?

EIS warps with bilinear interpolation. Across two consecutive frames this introduces
slight blur. The Farneback algorithm computes polynomial expansions of pixel intensities;
blur disrupts these expansions and produces artificial flow signal that looks like body
motion when there is none. Using the original unwarped raw frames avoids this.

### Step by step

**1. Build the ego-motion homography**

We want to cancel the camera's own rotation between the previous frame and the current
frame. The relative transform is:

```
T_rel = T_prev_world_cam⁻¹ × T_curr_world_cam
R_rel = T_rel[:3, :3]
```

From this rotation matrix, we build the pixel-space homography:

```
H_rot = K × R_rel × K⁻¹
```

Applying `H_rot` to the previous raw frame shifts static background pixels to match
where they appear in the current raw frame.

**2. Warp the previous frame**

```python
prev_warped = cv2.warpPerspective(prev_raw_frame, H_rot, (width, height))
```

After this warp, a rigid static point in the scene at the same world position appears
at the same pixel in both `prev_warped` and `raw_frame`. Any residual motion in the crop
is body motion.

**3. Crop to the bounding box**

```
prev_crop = prev_warped[ y1_raw:y2_raw, x1_raw:x2_raw ]
curr_crop = raw_frame  [ y1_raw:y2_raw, x1_raw:x2_raw ]
```

Both converted to greyscale for flow computation.

**4. Farneback dense optical flow**

```python
flow = cv2.calcOpticalFlowFarneback(
    prev_crop, curr_crop, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)
```

`flow` is an array of shape (H, W, 2): for each pixel (u, v), `flow[v, u]` is the
2D displacement vector (dx, dy) in pixels.

**5. Compute mean magnitude**

```
magnitude[u, v] = sqrt(flow[u, v, 0]² + flow[u, v, 1]²)
flow_mag = mean(magnitude)
```

This is stored in a rolling window per track ID. The 75th percentile of that window
is the metric:

```
flow_p75 = percentile(bbox_flow_history[tid], 75)
flow_ok  = flow_p75 < bbox_flow_threshold (default 1.5 px)
```

**Why the 75th percentile?**

Clothing edges and limb joints always show some flow even for a mannequin; using the
mean would set the threshold too high to catch subtle gestures. The 75th percentile
naturally ignores the top 25% of pixels (edges, highlights) and focuses on the bulk
behaviour of the object.

**Minimum bbox size**

Crops smaller than 20×20 pixels are not processed (returns 999.0 — metric is skipped).
For a ~1.7m mannequin at 3m depth the RealSense D435i produces a bounding box of roughly
200×100 pixels, so this limit is only reached for very distant targets (>8m).

### Option B — Pose-Based Joint Stability (`classifier_mode=pose`)

Run YOLOv8-pose on the bounding box crop to estimate 17 COCO keypoints. Track the
**shape** of the skeleton across a rolling window. A mannequin's skeleton shape never
changes. A human's does — even for subtle breathing (shoulders rise 5–10px per cycle).

**Camera motion invariance**: joint positions are expressed relative to the bbox centre
and normalised by the bbox diagonal. Camera movement shifts all joints equally, so it
cancels out in the relative representation. No ego-compensation warp is needed.

**Stability score**: 75th percentile of the per-coordinate standard deviation across
the window. The most-moving joint dominates — one waving hand raises the score even if
the torso is completely still.

```
sigma    = percentile_75( nanstd( pose_history[tid], axis=0 ) )
pose_ok  = sigma < joint_stability_threshold (default 0.008)
```

At a typical 450px bbox diagonal, `0.008` corresponds to ~3.6px of joint movement.

### Option C — Both simultaneously (`classifier_mode=both`)

Runs both flow and pose every frame. Both must pass for `is_mannequin` to be True.
The annotated label shows both values: `flow=0.8px σ=0.0042`.
Use this mode to compare the two metrics side-by-side and decide which to keep.

### Selecting a mode

Use `run_detection.sh` — it asks you at launch:
```
  1) flow  — Farneback optical flow only
  2) pose  — YOLOv8-pose joint stability only
  3) both  — run both, show both scores, both must pass
```

Or pass directly:
```bash
ros2 run drouga_detection detection_node --ros-args -p classifier_mode:=pose
```

---

## 8. Stage 5 — Classification Gate

A track is labelled `mannequin` only when ALL conditions are met simultaneously:

```python
confirmed     = consecutive_hits[tid] >= confirm_frames   # seen long enough
centroid_ok   = centroid_std < world_stability_threshold  # not moving in world (Metric 1)
flow_ok_final = <result of Metric 2, depends on mode>     # not gesturing (Metric 2)

is_mannequin  = confirmed AND centroid_ok AND flow_ok_final
```

`flow_ok_final` resolves based on `classifier_mode`:

| Mode | `flow_ok_final` |
|------|----------------|
| `flow` | `flow_p75 < bbox_flow_threshold` |
| `pose` | `sigma < joint_stability_threshold` |
| `both` | `flow_p75 < bbox_flow_threshold AND sigma < joint_stability_threshold` |

| State | Box colour | Label (flow mode) | Label (pose mode) |
|-------|-----------|-------------------|-------------------|
| Not yet confirmed | Grey | `#ID  N/25` | `#ID  N/25` |
| Confirmed, mannequin | Green | `MANNEQUIN Δ=2.1cm flow=0.8px 3.47m` | `MANNEQUIN Δ=2.1cm σ=0.0042 3.47m` |
| Confirmed, centroid HIGH | Orange | `HUMAN(moving) Δ=12cm flow=0.8px` | `HUMAN(moving) Δ=12cm σ=0.0042` |
| Confirmed, Metric 2 HIGH | Orange | `HUMAN(gesture) Δ=1.2cm flow=3.1px` | `HUMAN(gesture) Δ=1.2cm σ=0.031` |

In `both` mode the label shows: `MANNEQUIN Δ=2.1cm flow=0.8px σ=0.0042 3.47m`

Once a track is labelled mannequin, it stays that way until either metric exceeds its
threshold. This hysteresis prevents flickering if the drone briefly vibrates more.

`consecutive_hits[tid]` is reset to 0 whenever ByteTrack drops the track. This prevents
a human who temporarily leaves the frame and re-enters from being pre-confirmed.

The highest-confidence confirmed mannequin per frame is selected as the single published
target. All others are tracked silently.

---

## 9. VSLAM Fallback Mode

When `visual_slam/status` reports slam_state = LOST (3) or NOT_STARTED (0):

| Component | Normal mode | Fallback mode |
|-----------|-------------|---------------|
| EIS | H_stab from VSLAM orientation | H_stab = identity (raw frame passed to YOLO) |
| Metric 1 | Active — world centroid stability | **Suspended** (no valid world frame) |
| Metric 2 | H_rot from VSLAM T_rel | H_rot from D435i gyro integration |
| Gate | Both metrics required | Only Metric 2 required |

**Gyro integration for Metric 2**

The D435i's built-in gyroscope samples at 400 Hz. During fallback, the pipeline
integrates the angular velocity samples that arrived between the previous frame timestamp
and the current frame timestamp:

```
R = I
for each sample (t, omega) in imu_buffer where t_prev < t <= t_curr:
    dt = t - t_prev
    R  = R × Rotation.from_rotvec(omega × dt)
```

This gives an approximate rotation-only `T_rel` for Metric 2. Translation is NOT
estimated from the IMU (double integration produces too much error). Translational
ego-motion is therefore not removed from the crop during fallback, which means Metric 2
is slightly noisier but still functional.

**When does fallback activate?**

If `isaac_ros_visual_slam_interfaces` is not installed on the machine, the node starts
with `vslam_tracking = True` and never enters fallback automatically. It assumes VSLAM
is always available. This is intentional — on the Mac development machine VSLAM is not
installed, and we do not want the code to behave differently there.

---

## 10. All Parameters — Reference Table

**Detection and tracking**

| Parameter | Type | Default | Unit | What to change it for |
|-----------|------|---------|------|----------------------|
| `model_path` | string | `/home/user/drouga/best.engine` | — | Set to your actual Jetson path before first run |
| `conf` | float | 0.35 | — | Raise if false positives; lower if mannequin is missed |
| `mannequin_class` | int | 0 | — | Do not change (single-class model) |
| `publish_annotated` | bool | true | — | Set false during competition to save ~5ms/frame |
| `confirm_frames` | int | 25 | frames | Raise to reduce false alarms; lower if reaction is too slow |

**EIS (Stage 1)**

| Parameter | Type | Default | Unit | What to change it for |
|-----------|------|---------|------|----------------------|
| `eis_alpha` | float | 0.08 | — | Lower for more smoothing (heavy vibration); raise if EIS fights navigation |
| `eis_aggressive_threshold` | float | 5.0 | degrees | Angle above which alpha jumps to 0.6 to follow intentional turns |

**Metric 1 — world centroid (Stage 3)**

| Parameter | Type | Default | Unit | What to change it for |
|-----------|------|---------|------|----------------------|
| `world_stability_window` | int | 30 | frames | ~1s history; raise to catch slower walkers |
| `world_stability_threshold` | float | 0.08 | metres | Raise in strong wind; lower to be stricter |
| `depth_window` | int | 10 | pixels | Raise on surfaces with many depth holes |

**Metric 2 — body motion (Stage 4)**

| Parameter | Type | Default | Unit | What to change it for |
|-----------|------|---------|------|----------------------|
| `classifier_mode` | string | `flow` | — | `flow`, `pose`, or `both` — select via `run_detection.sh` |
| `bbox_flow_threshold` | float | 1.5 | pixels | (flow/both) Lower to catch subtler gestures; raise if false rejects |
| `pose_model_path` | string | `yolov8n-pose.engine` | — | (pose/both) Path to pose TensorRT engine on Jetson |
| `joint_stability_threshold` | float | 0.008 | normalised | (pose/both) Lower = stricter. At 450px bbox diagonal, 0.008 ≈ 3.6px |
| `joint_stability_window` | int | 40 | frames | (pose/both) ~1.3s history; raise to catch slow movements like breathing |
| `joint_kp_conf` | float | 0.30 | — | (pose/both) Min keypoint confidence to include; raise if pose is noisy |

### Overriding at launch

```bash
ros2 run drouga_detection detection_node --ros-args \
  -p model_path:=/home/mortada/drouga/best.engine \
  -p conf:=0.4 \
  -p confirm_frames:=15 \
  -p world_stability_threshold:=0.12 \
  -p bbox_flow_threshold:=2.0
```

### Changing while running

```bash
ros2 param set /drouga_detection conf 0.45
ros2 param set /drouga_detection bbox_flow_threshold 1.2
```

Note: the node reads parameters once at startup. Runtime changes via `ros2 param set`
are accepted by ROS2 but do NOT take effect until the node is restarted.

---

## 11. Tuning Guide

### Step 1 — Static table test (before mounting on drone)

Place the camera on a tripod pointing at the mannequin. Check the annotated image:

- If the box stays grey after a few seconds: `confirm_frames` too high, or `conf` too high
  (YOLO does not detect). Lower one or both.
- If a person standing next to the mannequin also turns green: metrics too loose.
  Lower `world_stability_threshold` and/or `bbox_flow_threshold`.

### Step 2 — Hand-held walking test

Walk the camera around the room while the mannequin is in view. Check:

- Mannequin box should stay green.
- Your own hand/body (if in frame) should be orange.
- If the mannequin goes orange: `world_stability_threshold` too low (your walking
  introduces position jitter). Raise it slightly.

### Step 3 — Drone mount, motors off

Mount the camera on the drone. Run the motors at idle without takeoff. Check:

- If boxes jitter back to grey: vibration is hurting detection. Lower `eis_alpha`
  (more smoothing). Also check motor balance.
- Confirm the annotated image looks smooth (not shaky).

### Step 4 — Drone hover, no payload

First real EIS test. Hover at 2m altitude facing the mannequin. Tune:

- `eis_alpha`: if the image still shakes, lower it. If EIS fights bank/yaw manoeuvres,
  raise `eis_aggressive_threshold` to 7° or 10°.
- `world_stability_threshold`: hover drift should be well within 8 cm. If not,
  check VSLAM accuracy first.

### Step 5 — Full test with a human bystander

Have a person walk between the drone and the mannequin. Verify:

- Walking person → orange box.
- Person stops next to mannequin and waves → orange box (Metric 2 fires).
- Mannequin → green box throughout.

If walking person turns green: `world_stability_threshold` too loose — lower it.
If person waving turns green: `bbox_flow_threshold` (flow mode) or `joint_stability_threshold`
(pose mode) too loose — lower it.

### Step 6 — Choosing between flow and pose mode

Run in `both` mode during early testing so you can see both scores on every box:

```bash
bash run_detection.sh   # choose 3
```

Look at the label values for the mannequin and for a human standing nearby:

- If `flow=` separates them cleanly → use `flow` mode.
- If `σ=` separates them more clearly → use `pose` mode.
- If neither alone is clean → keep `both`.

Typical observations:
- In calm indoor conditions, `pose` is more stable (no texture/illumination sensitivity).
- Outdoors with wind, `flow` may produce false high values on the mannequin's clothing.
  Switch to `pose` if this happens.

---

## 12. Published Topics

| Topic | Type | Content |
|-------|------|---------|
| `/drouga/mannequin_detected` | std_msgs/Bool | True when a mannequin is confirmed this frame |
| `/drouga/mannequin_pixel` | geometry_msgs/Point | (cx_raw, cy_raw, 0) — pixel centre in raw frame |
| `/drouga/mannequin_confidence` | std_msgs/Float32 | YOLO confidence score (0.0 when none) |
| `/drouga/mannequin_track_id` | std_msgs/Int32 | ByteTrack ID; -1 when no mannequin |
| `/drouga/mannequin_position_3d` | geometry_msgs/PointStamped | 3D world-frame position in metres; only published when VSLAM active and depth valid |
| `/drouga/annotated_image` | sensor_msgs/Image | Debug BGR image with boxes; disable with `publish_annotated:=false` |

QoS for detection results: RELIABLE, depth 10.
QoS for image topics: BEST_EFFORT, depth 1.

---

## 13. Subscribed Topics

| Topic | Type | QoS | Notes |
|-------|------|-----|-------|
| `/camera/color/image_raw` | sensor_msgs/Image | BEST_EFFORT | Synchronised with depth via ApproximateTimeSynchronizer |
| `/camera/aligned_depth_to_color/image_raw` | sensor_msgs/Image | BEST_EFFORT | Synchronised with color; values in mm (uint16) |
| `/camera/color/camera_info` | sensor_msgs/CameraInfo | RELIABLE | Stored once on first message |
| `/visual_slam/tracking/odometry` | nav_msgs/Odometry | RELIABLE | Stored as latest_odom; not synchronised with images |
| `/visual_slam/status` | VisualSlamStatus | RELIABLE | Optional; sets vslam_tracking flag |
| `/camera/imu` | sensor_msgs/Imu | BEST_EFFORT | Gyro buffered at 400 Hz for VSLAM fallback |

### Enabling required streams in the RealSense launch

The D435i IMU and aligned depth are NOT enabled by default. Launch with:

```bash
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true \
  depth_module.profile:=640x480x30 \
  unite_imu_method:=1 \
  enable_gyro:=true \
  enable_accel:=true
```

`unite_imu_method:=1` combines the separate gyro and accel streams into a single
`/camera/imu` topic using linear interpolation. This is required for `_imu_callback`
to receive data.

---

## 14. Known Limitations and Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Beach sand is textureless — depth holes common at flat angles | Medium | Median patch (`depth_window`) tolerates sparse holes; Metric 1 needs only one valid depth per frame |
| Strong wind makes drone drift → world position jitter | Medium | Tune `world_stability_threshold` upward; EIS alpha tuning |
| Mannequin in direct sunlight → overexposed texture → poor Farneback signal | Medium | Ensure `/camera/color/image_raw` is not clipped; consider auto-exposure tuning via RealSense API |
| VSLAM loses tracking during fast turns | Low | Fallback to IMU gyro for Metric 2; Metric 1 suspended but system keeps running |
| D435i depth range: 0.3m – 3m reliable, up to 10m degraded | Medium | At >3m, depth holes increase; `depth_window` should be larger (`-p depth_window:=20`) |
| EIS does not correct translational shake | Low | Translational jitter is smaller than rotational in a typical drone; not worth the complexity of translation correction |
| ByteTrack ID switch if mannequin is occluded for >1s | Low | IoU gate and Kalman predict-only keep IDs for ~30 frames after occlusion; unlikely at >1m altitude |

---

## 15. Pose-Based Joint Stability — Implementation Details

> **Status**: fully integrated into `detection_node.py` as `classifier_mode=pose` (or `both`).
> Prototype script `scripts/live_detect_pose.py` available for laptop-only threshold tuning
> before deploying on the Jetson.

### Concept

Run a COCO-trained YOLOv8-pose model on each detected bounding box to estimate 17 body
keypoints (shoulders, elbows, wrists, hips, knees, ankles, etc.). Track the **shape** of
the skeleton — expressed as normalised joint positions relative to the bounding box centre —
over a rolling window. A mannequin's skeleton shape does not change between frames. A
human's does, even when they appear to be standing still (breathing, micro-movements,
weight shifts).

This metric is **camera-motion-invariant by design**: because we measure relative positions
between joints (not absolute pixel positions), any camera movement that shifts all joints
equally cancels out automatically. No ego-motion compensation is needed for this metric.

### What it catches that the other two metrics miss

| Scenario | Metric 1 | Metric 2 | Metric 3 |
|----------|----------|----------|----------|
| Walking human | ✓ catches | ✓ catches | ✓ catches |
| Gesturing / waving human | ✗ misses (world pos stable) | ✓ catches | ✓ catches |
| Human standing perfectly still, breath only | ✗ misses | ✗ misses (breathing flow below threshold) | ✓ catches (shoulder rise ~5–10px) |
| Mannequin | ✓ passes | ✓ passes | ✓ passes |

### Threshold tuning on the laptop (do before Jetson deployment)

Run the standalone prototype — no ROS2 required:
```bash
python scripts/live_detect_pose.py
```
Stand completely still and note your `σ=` value (breathing alone gives σ ≈ 0.012–0.025).
Wave a hand and note σ (typically > 0.05). Set `joint_stability_threshold` to roughly
half your still-breathing value. The default of `0.008` is conservative — lower it if
a still human is still turning green.

### How the pose metric is selected at runtime

Use `run_detection.sh` and choose option 2 or 3, or pass directly:
```bash
ros2 run drouga_detection detection_node --ros-args \
  -p classifier_mode:=pose \
  -p pose_model_path:=/home/mortada/drouga/yolov8n-pose.engine \
  -p joint_stability_threshold:=0.008 \
  -p joint_stability_window:=40
```

### No custom training needed

The COCO-trained YOLOv8n-pose model is used **only on crops that your custom model
already detected** as mannequin candidates. Your custom model (mAP50=0.958) handles the
detection; the pose model handles the joint measurement. No re-training, no keypoint
annotations on your dataset.

---

## 16. Compute Budget

How heavy is the pipeline? Below are realistic estimates for the Jetson Orin Nano with
the TensorRT FP16 engine.

### Per-frame cost (single detection in frame, ~30 Hz target)

| Stage | Where it runs | Estimated cost | Notes |
|-------|--------------|----------------|-------|
| EIS warp (`warpPerspective`) | GPU | ~1 ms | GPU-accelerated in OpenCV |
| YOLO TensorRT FP16 inference | GPU | ~20–25 ms | Dominant cost. This is the budget. |
| Coordinate remapping (`perspectiveTransform`) | CPU | < 0.1 ms | Negligible |
| Metric 1: depth sample + 3D math | CPU | < 0.5 ms | Just array indexing + matrix multiply |
| Metric 2: Farneback dense flow (per bbox) | CPU | ~3–6 ms | Runs on CPU — main variable cost |
| Annotated image publish (if enabled) | CPU + GPU | ~3–5 ms | Disable with `publish_annotated:=false` |
| **Total (current, 1 detection)** | | **~28–36 ms** | **~28–35 FPS** |

### Cost by classifier_mode (per frame, 1 detection)

| Mode | Additional cost | Total estimate | FPS estimate |
|------|----------------|----------------|-------------|
| `flow` | +3–6 ms (CPU Farneback) | ~28–36 ms | ~28–35 FPS |
| `pose` | +3–5 ms (GPU pose model) | ~28–35 ms | ~28–35 FPS |
| `both` | +6–11 ms (CPU + GPU) | ~31–41 ms | ~24–32 FPS |

`pose` is roughly the same cost as `flow` because it runs on the GPU. `both` runs both
and is the most expensive — only use it for comparison testing, not competition.

### Is it okay?

**Yes, with conditions.**

The `~28 FPS` figure with TensorRT already accounts for the biggest cost — the YOLO
inference. The rest of the pipeline adds at most ~10ms total. On the Jetson Orin Nano:

- The GPU handles EIS and YOLO in parallel with the CPU handling metrics.
- Two detections in frame at once roughly doubles Metric 2 cost (Farneback runs once
  per bbox, on CPU).
- TensorRT engine (`best.engine`) is mandatory for real-time operation. The PyTorch
  fallback (`best.pt`) runs at ~8 FPS — not usable.

**The two things that hurt most:**

1. `publish_annotated:=true` — encoding and publishing a 640×480 BGR image at 30Hz adds
   ~3–5ms. Disable it when not debugging.
2. Metric 2 (Farneback) on large bounding boxes — a full-frame bbox (person very close
   to the camera) can take ~8ms. This is rare in practice since the drone maintains >1m
   distance.

**Metric 3 (pose) is cheaper than Metric 2** because YOLOv8n-pose runs on the GPU,
not the CPU, and operates only on the small crop — not the full frame.
