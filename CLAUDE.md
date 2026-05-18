# Robotics: Autonomous Drone Medical Delivery — ERL Competition

## Competition Context
- Event: European Robotics League (ERL), May 13–15 2026
- Location: Outdoor, under a pier (Scheveningen, Netherlands)
- Environment: GPS-denied, safety net cage, beach obstacles, variable wind/lighting

## Mission
Autonomous drone must:
1. Take off from base, carry a medical payload (1–1.5kg)
2. Navigate obstacles without GPS
3. Detect a mannequin (patient) using onboard vision
4. Land within 1–2 metres of the mannequin (must NOT go closer than 1m)
5. Release the payload while grounded
6. Return and land at base autonomously

## Hardware
- Drone: max 1m diagonal, max 4kg total weight including payload
- Companion computer: Jetson Orin Nano ✓ in hand
- Depth camera: Intel RealSense D435**i** (stereo depth + built-in IMU gyro/accel) ✓ in hand
- Flight controller: Pixhawk 6C running PX4
- Communication: MAVLink / uXRCE-DDS bridge
- Remote monitoring: Ubuntu 24.04 laptop (ROS2 Jazzy), connected via ROS_DOMAIN_ID=42

## Software Stack
- ROS2 Humble (middleware, on Jetson)
- **Isaac ROS Visual SLAM** (6-DOF camera pose at ~30Hz — confirmed running on Jetson)
- Voxblox (3D ESDF mapping from depth data)
- EGO-Planner (local trajectory planning, obstacle avoidance)
- RRT* (global path planner)
- **YOLOv8m** (mannequin detection — single class) ✓ DONE
- **ByteTrack** (target tracking) ✓ DONE
- **EIS** (Electronic Image Stabilisation — pre-YOLO vibration removal via VSLAM) ✓ DONE
- **World-frame centroid stability** (Metric 1 — catches walking humans) ✓ DONE
- **Intra-bbox optical flow / pose joint stability** (Metric 2 — catches gesturing humans, selectable) ✓ DONE
- Visual servoing (precision landing approach) ⬜ TODO
- ArUco marker at base (precision return landing) ⬜ TODO

## Detection Pipeline (current — fully implemented)

### Overview
The node runs on the Jetson at ~30Hz. Per frame:

1. **EIS** — warp raw frame using VSLAM orientation (adaptive EMA filter) to remove drone vibration before YOLO sees it
2. **YOLO + ByteTrack** — detect mannequins on stabilised frame, assign persistent track IDs
3. **Coordinate remap** — map detections back to raw frame coordinates (H_stab_inv) for depth/3D
4. **Metric 1** — back-project detection + depth to world frame via VSLAM pose; track world position variance over 30 frames; low σ → stationary → mannequin candidate
5. **Metric 2** (selectable via `classifier_mode`):
   - `flow`: Farneback dense optical flow inside bbox after ego-compensation warp
   - `pose`: YOLOv8-pose joint shape stability on bbox crop (camera-motion invariant)
   - `both`: run both simultaneously for comparison testing
6. **Gate** — confirmed (≥25 frames) AND Metric 1 OK AND Metric 2 OK → MANNEQUIN

### Key design decisions
- **Single class (`mannequin` only)**: two-class model had cls_loss 3.1 at epoch 40 — mannequin and human look too similar for joint classification
- **No background optical flow for ego-motion**: beach sand is textureless, LK flow fails. Replaced with VSLAM pose
- **Metrics measure world-frame motion, not pixel motion**: pixel residuals are physically meaningless (1px at 0.5m ≠ 1px at 3m). World centroid std in metres is meaningful
- **Pose metric camera-invariant by design**: relative joint positions cancel camera motion without needing any warp
- **ByteTrack over DeepSORT**: DeepSORT ReID embeddings trained on humans confuse mannequins. ByteTrack uses IoU + Kalman only

### VSLAM fallback
When Isaac ROS VSLAM loses tracking: EIS disabled, Metric 1 suspended, Metric 2 continues using D435i IMU gyro integration for ego-compensation

### Published topics
| Topic | Type | Content |
|-------|------|---------|
| `/drouga/mannequin_detected` | Bool | True when mannequin confirmed |
| `/drouga/mannequin_pixel` | Point | Pixel centre in raw frame |
| `/drouga/mannequin_confidence` | Float32 | YOLO score |
| `/drouga/mannequin_track_id` | Int32 | ByteTrack ID (-1 when none) |
| `/drouga/mannequin_position_3d` | PointStamped | World-frame 3D position (metres) |
| `/drouga/annotated_image` | Image | Debug view with boxes |

## Mission State Machine
TAKEOFF → NAVIGATE_TO_ZONE → SEARCH_TARGET →
APPROACH_TARGET → PRECISION_LAND →
RELEASE_PAYLOAD → RETURN → LAND_BASE

## Scoring Rules (critical constraints)
- PENALTY: approaching closer than 1m to mannequin
- PENALTY: manual intervention outside take-off area
- PENALTY: collisions with obstacles
- DISQUALIFICATION: hitting the patient or bystander
- DISQUALIFICATION: flying outside competition area
- Landing accuracy target: 1–2 metres from mannequin

---

## Current Development Status (2026-04-25)

### What is done
- ✅ YOLOv8m trained on Colab (mAP50=0.958, Recall=0.911) → `best.pt`
- ✅ Offline detection + tracking script (`scripts/track_video.py`)
- ✅ Live camera scripts: Mac webcam (`scripts/live_detect.py`), Jetson + D435i (`scripts/live_detect_jetson.py`)
- ✅ Pose-based prototype script (`scripts/live_detect_pose.py`) — laptop only, for threshold tuning
- ✅ ROS2 detection node with full EIS + VSLAM + two-metric pipeline (`ros2_ws/src/drouga_detection/`)
- ✅ Depth fusion → 3D world position published on `/drouga/mannequin_position_3d`
- ✅ EIS (Electronic Image Stabilisation) — adaptive alpha, VSLAM-based, pre-YOLO
- ✅ Metric 1: world-frame centroid stability via VSLAM + D435i depth
- ✅ Metric 2: selectable — Farneback flow (`flow`) or pose joint stability (`pose`) or both (`both`)
- ✅ Interactive launcher `run_detection.sh` — asks flow/pose/both at startup
- ✅ Full Jetson setup guide (`JETSON_SETUP.md`)
- ✅ Full ROS2 setup guide (`ROS2_SETUP.md`)
- ✅ Full CV pipeline technical documentation (`CV_PIPELINE.md`)
- ✅ Remote monitoring from Ubuntu 24.04 laptop via ROS_DOMAIN_ID=42

### What is pending
- ⬜ TensorRT FP16 export on Jetson → `best.engine` (required for 28 FPS; PyTorch fallback = ~8 FPS)
- ⬜ TensorRT export of pose model → `yolov8n-pose.engine`
- ⬜ Tune parameters on real drone footage: `conf`, `world_stability_threshold`, `bbox_flow_threshold` / `joint_stability_threshold`
- ⬜ Decide between `flow` and `pose` mode after real-hardware comparison test
- ⬜ Integrate detection node with mission state machine
- ⬜ Visual servoing for precision landing
- ⬜ ArUco marker for return landing
- ⬜ Full mission state machine
- ⬜ End-to-end flight test

---

## Detection Pipeline — Step-by-Step Progress

### Step 1 — Model Selection ✅ DONE (2026-04-09)
- **Chosen: YOLOv8m** (upgraded from YOLOv8s — accuracy is priority)
- Single class: `mannequin` only (dropped human/bystander class)
- Rationale: single class eliminates classification confusion (loss 3.1 at epoch 40 with 2 classes)
- 28 FPS on Jetson Orin Nano (TensorRT FP16)
- Input resolution: 800×800

### Step 2 & 3 — Dataset ✅ DONE (2026-04-09)
- Source: Roboflow — single class mannequin dataset, 538 empty labels removed

  | Split | Images | Instances |
  |-------|--------|-----------|
  | train | 1,188  | ~3,549    |
  | val   | 321    | ~992      |
  | test  | 170    | ~492      |

### Step 4 — Training ✅ DONE (2026-04-09)
- YOLOv8m, 150 epochs, imgsz=800, AdamW, Google Colab T4
- Output: `best.pt` (22.5MB)

### Step 5 — Validation ✅ DONE (2026-04-09)

  | Metric | Result |
  |--------|--------|
  | mAP50  | **0.958** |
  | mAP50-95 | **0.747** |
  | Precision | **0.939** |
  | Recall | **0.911** |

### Step 6 — Offline Detection + Tracking ✅ DONE (2026-04-15)
- Script: `scripts/track_video.py`, input: `stabilised.mp4`
- ByteTrack + Lucas-Kanade ego-motion (pixel residual method — replaced in ROS2 node by VSLAM)

### Step 7 — Live Detection Scripts ✅ DONE (2026-04-15)
- `scripts/live_detect.py` — Mac webcam
- `scripts/live_detect_jetson.py` — Jetson + D435i
- `scripts/live_detect_pose.py` — pose-based prototype (laptop, for threshold tuning)

### Step 8 — ROS2 Detection Node ✅ DONE (2026-04-15, updated 2026-04-25)
- Package: `ros2_ws/src/drouga_detection/`
- Full EIS + VSLAM + depth + Metric 1 + Metric 2 (flow/pose/both)
- All parameters tunable at launch without rebuilding
- `run_detection.sh` — interactive launcher

### Step 9 — TensorRT FP16 Export ⬜ TODO
- Run on Jetson: `YOLO('best.pt').export(format='engine', half=True, imgsz=800, device=0)`
- Also export pose model: `YOLO('yolov8n-pose.pt').export(format='engine', half=True, imgsz=640, device=0)`

### Step 10 — Real Hardware Testing ⬜ TODO
1. Static test — camera on tripod, tune `conf` and Metric 2 threshold
2. Hand-held walking test — verify ByteTrack IDs, EIS behaviour
3. Run in `both` mode to compare flow vs pose scores side-by-side → pick one
4. Drone mount, motors off — check vibration effect on EIS
5. Drone hover test — tune `eis_alpha`, `world_stability_threshold`
6. Full mission test with human bystander

### Step 11 — Full Stack Integration ⬜ TODO
- Mission state machine subscribes to `/drouga/mannequin_detected` and `/drouga/mannequin_position_3d`
- Visual servoing for precision landing
- ArUco marker for return landing

---

## Key File Locations

| File | Purpose |
|------|---------|
| `best.pt` | Trained YOLOv8m model (22.5MB) |
| `run_detection.sh` | Interactive launcher — asks flow/pose/both at startup |
| `CV_PIPELINE.md` | Full technical documentation of the CV pipeline |
| `JETSON_SETUP.md` | Jetson Orin Nano hardware setup guide |
| `ROS2_SETUP.md` | ROS2 node build and run guide |
| `dataset/data.yaml` | Training config |
| `scripts/train.py` | Training script |
| `scripts/track_video.py` | Offline video detection + tracking |
| `scripts/live_detect.py` | Live detection — Mac webcam |
| `scripts/live_detect_jetson.py` | Live detection — Jetson + D435i |
| `scripts/live_detect_pose.py` | Pose prototype — laptop threshold tuning |
| `ros2_ws/src/drouga_detection/drouga_detection/detection_node.py` | Main ROS2 node |
| `stabiliser/stabilise.py` | Video stabilisation (used for early offline testing only) |

---

## Known Issues / Decisions Made

| Issue | Decision |
|---|---|
| Two-class model had cls_loss 3.1 | Dropped to single class — mannequin only |
| Model metadata says `{0: human, 1: mannequin}` | Model outputs class 0 = mannequin. Metadata is wrong, ignore it |
| DeepSORT worse than ByteTrack | DeepSORT ReID embeddings trained on humans confuse mannequins. Using ByteTrack |
| Pixel residual ego-motion fails on featureless beach sand | Replaced with VSLAM world-frame centroid stability (Metric 1) |
| D435 had no IMU | Upgraded to D435**i** (built-in IMU). IMU used as VSLAM fallback for Metric 2 ego-warp |
| Farneback flow noisy on windy outdoor textures | Pose-based Metric 2 available as alternative — camera-motion invariant, no ego-warp needed |
| Standing-still human indistinguishable from mannequin | Accepted limitation of all CV approaches. Safety guaranteed by 1m hard stop in state machine |
| EIS fights intentional navigation turns | Adaptive alpha: jumps to 0.6 when inter-frame rotation > 5° so EIS follows the turn |
