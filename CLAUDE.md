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
- Companion computer: Jetson Orin Nano
- Depth camera: Intel RealSense D435 (stereo depth, NO IMU — confirmed 2026-04-15)
- IMU: separate external IMU may be available later — not confirmed yet
- Flight controller: Pixhawk 6C running PX4
- Communication: MAVLink/MAVROS over USB
- NOTE: RealSense and Jetson not yet in hand as of 2026-04-09

## Software Stack
- ROS2 Humble (middleware)
- VINS-Fusion or OpenVINS (Visual-Inertial Odometry — localization without GPS)
- Voxblox (3D ESDF mapping from depth data)
- EGO-Planner (local trajectory planning, obstacle avoidance)
- RRT* (global path planner)
- **YOLOv8m** (mannequin detection — single class) ✓ DONE
- **ByteTrack** (target tracking) ✓ DONE
- **Ego-motion compensation** (optical flow homography to separate mannequins from humans) ✓ DONE
- Visual servoing (precision landing approach)
- ArUco marker at base (precision return landing)

## Detection Pipeline
- **Single class: `mannequin` (class 0) only**
- Bystander/human detection dropped — humans handled as obstacles by Voxblox depth data
- Reason for single class: two-class model had classification loss 3.1 at epoch 40 (mannequin and human look too similar)
- Model: YOLOv8m fine-tuned, exported to TensorRT FP16 on Jetson (~28 FPS)
- Confirmation gate: track must appear in 25 consecutive frames (~0.8s) before committing
- Ego-motion compensation: frame-to-frame homography via Lucas-Kanade optical flow
  - Residual = how far a track moves BEYOND camera motion
  - Low 75th-pct residual over 60-frame window → stationary in world → mannequin
  - High residual → moving independently → human/bystander
- Tracking: ByteTrack (IoU + Kalman filter, tightly integrated with YOLO via Ultralytics)
- ROS2 node publishes results to `/drouga/mannequin_detected`, `/drouga/mannequin_pixel`, etc.

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

## Current Development Status (2026-04-15)

### What is done
- ✅ YOLOv8m trained on Colab (mAP50=0.958, Recall=0.911) → `best.pt`
- ✅ Offline detection + tracking script (`scripts/track_video.py`) — ByteTrack + ego-motion
- ✅ Live camera detection script (`scripts/live_detect.py`) — Mac webcam, tested working
- ✅ Jetson live detection script (`scripts/live_detect_jetson.py`) — RealSense D435, no IMU
- ✅ ROS2 detection node (`ros2_ws/src/drouga_detection/`) — built, ready to deploy
- ✅ Full Jetson setup guide (`JETSON_SETUP.md`)
- ✅ Full ROS2 setup guide (`ROS2_SETUP.md`)
- ✅ Video stabiliser (`stabiliser/stabilise.py`) → `stabilised.mp4`

### What is pending (waiting for hardware)
- ⬜ Deploy to Jetson and test on real hardware
- ⬜ TensorRT FP16 export on Jetson → `best.engine`
- ⬜ Tune CONF, CONFIRM_FRAMES, RESIDUAL_THRESHOLD on real drone footage
- ⬜ Add RealSense depth fusion → 3D mannequin world position
- ⬜ Integrate detection node with mission state machine
- ⬜ IMU-fused ego-motion (if separate IMU available)
- ⬜ VIO integration (VINS-Fusion / OpenVINS)
- ⬜ Full mission state machine
- ⬜ TensorRT export + benchmarking

---

## Detection Pipeline — Step-by-Step Progress

### Step 1 — Model Selection ✅ DONE (2026-04-09)
- **Chosen: YOLOv8m** (upgraded from YOLOv8s — accuracy is priority)
- Single class: `mannequin` only (dropped human/bystander class)
- Rationale:
  - Single class eliminates classification confusion (loss was 3.1 at epoch 40 with 2 classes)
  - 28 FPS on Jetson Orin Nano (TensorRT FP16) is sufficient for mission
- Input resolution: 800×800 (larger than default 640 — better for distant mannequins)
- Pretrained COCO weights (transfer learning)

### Step 2 & 3 — Dataset ✅ DONE (2026-04-09)
- Source: Roboflow ("My First Project") — single class mannequin dataset
- Cleaned: 538 empty label files removed
- Final split (RANDOM_SEED=42):

  | Split | Images | Mannequin instances |
  |-------|--------|-------------------|
  | train | 1,188  | ~3,549            |
  | val   | 321    | ~992              |
  | test  | 170    | ~492              |
  | **Total** | **1,679** | **~5,033** |

- Format: YOLOv8 (class_id cx cy w h, normalised)
- Config: `dataset/data.yaml`

### Step 4 — Training ✅ DONE (2026-04-09)
- Script: `scripts/train.py`
- Model: YOLOv8m, epochs=150, imgsz=800, batch=8, device=mps
- Optimiser: AdamW, lr0=0.001, cosine LR schedule, warmup_epochs=5
- Augmentation: hsv_v=0.5, mosaic=1.0, mixup=0.2, copy_paste=0.1, scale=0.6
- Regularisation: dropout=0.1, label_smoothing=0.05
- Trained on Google Colab (T4 GPU) — completed full run, single class (mannequin only)
- Output: `best.pt` (root directory, 22.5MB)
- NOTE: model metadata says 2 classes but it is effectively single-class mannequin (class 0)

### Step 5 — Validate ✅ DONE (2026-04-09)
- Model: `best.pt` — validated on test set (170 images)

  | Metric | Result | Target |
  |--------|--------|--------|
  | mAP50 | **0.958** | >0.80 ✅ |
  | mAP50-95 | **0.747** | >0.55 ✅ |
  | Precision | **0.939** | — ✅ |
  | Recall | **0.911** | >0.85 ✅ |

### Step 6 — Offline Detection + Tracking ✅ DONE (2026-04-15)
- Script: `scripts/track_video.py`
- Input: `stabilised.mp4`
- Tracker: ByteTrack via Ultralytics `model.track()` — tightly integrated with YOLO
- Ego-motion compensation:
  - Lucas-Kanade optical flow estimates frame-to-frame homography
  - RANSAC rejects moving objects, fits to background only
  - Per-track residual = distance between actual and predicted (camera-motion-only) position
  - 75th-pct residual over 60-frame window < 8px → stationary in world → MANNEQUIN (green)
  - High residual → independently moving → HUMAN (orange)
  - < 25 consecutive frames → UNCONFIRMED (grey)
- Key parameters:
  - `CONF = 0.35`
  - `CONFIRM_FRAMES = 25` (~0.8s at 30fps)
  - `RESIDUAL_WINDOW = 60` (2 seconds)
  - `RESIDUAL_THRESHOLD = 8` pixels (75th percentile)
- Why ByteTrack over DeepSORT: faster, no ReID network needed, tightly integrated with YOLO. DeepSORT was tested and performed worse (ReID embeddings trained on humans cause confusion with mannequins)

### Step 7 — Live Detection Scripts ✅ DONE (2026-04-15)
- `scripts/live_detect.py` — Mac webcam, tested working, uses OpenCV VideoCapture
- `scripts/live_detect_jetson.py` — Jetson + RealSense D435 (no IMU), same pipeline
  - IMU lines removed (camera has no IMU)
  - Uses pyrealsense2 color stream
  - Supports `--save` and `--pt` flags

### Step 8 — ROS2 Detection Node ✅ DONE (2026-04-15)
- Package: `ros2_ws/src/drouga_detection/`
- Node: `drouga_detection/detection_node.py`
- Subscribes: `/camera/color/image_raw` (sensor_msgs/Image) from realsense-ros
- Publishes:
  - `/drouga/mannequin_detected` (std_msgs/Bool)
  - `/drouga/mannequin_pixel` (geometry_msgs/Point — pixel centre cx, cy)
  - `/drouga/mannequin_confidence` (std_msgs/Float32)
  - `/drouga/mannequin_track_id` (std_msgs/Int32 — -1 when none)
  - `/drouga/annotated_image` (sensor_msgs/Image — debug view)
- All key parameters are ROS2 parameters (tunable at runtime without rebuilding)
- QoS: image topics use BEST_EFFORT, detection results use RELIABLE
- Setup guide: `ROS2_SETUP.md`

### Step 9 — TensorRT FP16 Export ⬜ TODO (when Jetson available)
- Must run ON the Jetson: `model.export(format='engine', half=True, imgsz=800, device=0)`
- Target: ~28 FPS on Orin Nano
- Output: `best.engine` — used by both `live_detect_jetson.py` and the ROS2 node

### Step 10 — Real Hardware Testing ⬜ TODO (when hardware arrives)
Recommended order:
1. Static test — Jetson on table, camera at mannequin, tune CONF
2. Hand-held walking test — verify ByteTrack holds IDs, ego-motion behaviour
3. Drone mount, motors off — check vibration, USB stability
4. Drone hover test — first real ego-motion test, retune RESIDUAL_THRESHOLD
5. Full mission test

### Step 11 — Depth Fusion ⬜ TODO (when hardware available)
- Use RealSense D435 depth stream to compute 3D world position of mannequin
- Publish as `geometry_msgs/PointStamped` on `/drouga/mannequin_position_3d`
- Required for precision landing approach and 1m safety constraint

### Step 12 — Full Stack Integration ⬜ TODO
- VIO (VINS-Fusion / OpenVINS)
- Mission state machine subscribes to `/drouga/mannequin_detected`
- Visual servoing for precision landing
- ArUco marker for return landing

---

## Stabiliser
- Algorithm: Kalman filter (1D per axis: x, y, rotation) + Lucas-Kanade optical flow
- Cancels: high-frequency motor vibration + low-frequency wind drift
- Script: `stabiliser/stabilise.py`
- Parameters: process_noise=1e-3, measurement_noise=0.1, crop=10%, black border fill
- Input: `ZARA STORE TOUR.mp4` (phone video, 640×360, 30fps, 367s)
- Output: `stabilised.mp4` — used for offline detection testing
- NOTE: no pre-stabilisation in the live pipeline — ego-motion compensation handles camera motion instead

---

## Key File Locations

| File | Purpose |
|------|---------|
| `best.pt` | Trained YOLOv8m model (22.5MB, Colab T4) |
| `dataset/data.yaml` | Training config — paths + class names |
| `dataset/{train,val,test}/` | Images and labels |
| `scripts/train.py` | Training script (YOLOv8m) |
| `scripts/visualise_predictions.py` | Test set visualisation — 4×4 grid |
| `scripts/track_video.py` | Offline video detection + tracking (Mac) |
| `scripts/live_detect.py` | Live detection — Mac webcam |
| `scripts/live_detect_jetson.py` | Live detection — Jetson + RealSense D435 |
| `ros2_ws/src/drouga_detection/` | ROS2 detection package |
| `ros2_ws/src/drouga_detection/drouga_detection/detection_node.py` | ROS2 node source |
| `stabiliser/stabilise.py` | Video stabilisation script |
| `stabilised.mp4` | Stabilised test video for offline detection |
| `runs/drouga_mannequin_v1/weights/best.pt` | Copy of trained model in runs folder |
| `JETSON_SETUP.md` | Full Jetson Orin Nano hardware setup guide |
| `ROS2_SETUP.md` | Full ROS2 node build and run guide |

---

## Known Issues / Decisions Made

| Issue | Decision |
|---|---|
| Two-class model (human + mannequin) had cls_loss 3.1 | Dropped to single class — mannequin only |
| Model metadata says {0: human, 1: mannequin} | Model actually outputs class 0 = mannequin (single-class training). Metadata is wrong |
| DeepSORT worse than ByteTrack | DeepSORT ReID embeddings trained on humans — confuses mannequins. ByteTrack is better |
| Pixel-space stillness filter failed | Drone moves so mannequins move in frame. Replaced with ego-motion compensation |
| RealSense D435 has no IMU | Optical flow used for ego-motion instead. Separate IMU may come later |
| Optical flow slow (~15ms) on featureless surfaces (sand) | Risk noted — will need tuning on real beach footage |
