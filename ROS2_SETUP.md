# ROS2 Detection Node — Setup and Run Guide
## DROUGA Mannequin Detection

---

## Overview

This guide covers building and running the `drouga_detection` ROS2 package on the Jetson Orin Nano. The node takes camera frames from the RealSense, runs YOLO + ByteTrack + ego-motion compensation, and publishes detection results that the rest of the drone stack (mission state machine, PX4) can subscribe to.

```
RealSense D435
      ↓  /camera/color/image_raw
drouga_detection node
      ↓  /drouga/mannequin_detected   (Bool)
      ↓  /drouga/mannequin_pixel      (Point — pixel centre)
      ↓  /drouga/mannequin_confidence (Float32)
      ↓  /drouga/mannequin_track_id   (Int32)
      ↓  /drouga/annotated_image      (Image — debug view)
Mission state machine / PX4
```

---

## Prerequisites

Before following this guide, complete the Jetson hardware setup in `JETSON_SETUP.md`:
- JetPack 6.x flashed
- PyTorch ARM64 installed
- RealSense SDK (librealsense) built from source
- Ultralytics installed
- `best.pt` transferred to the Jetson (and ideally exported to `best.engine`)

---

## Phase 1 — Install ROS2 Humble on Jetson

ROS2 Humble is the LTS release compatible with Ubuntu 22.04 (JetPack 6.x).

### 1.1 Set up locale

```bash
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

### 1.2 Add ROS2 apt repository

```bash
sudo apt install -y software-properties-common curl

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
```

### 1.3 Install ROS2 Humble

```bash
sudo apt install -y ros-humble-desktop
```

> `ros-humble-desktop` includes RViz2 for visualising the annotated image topic. If you want a smaller install without the GUI tools use `ros-humble-ros-base` instead.

### 1.4 Install build tools

```bash
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-pip
```

### 1.5 Source ROS2 in every terminal automatically

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 1.6 Verify ROS2 is working

```bash
ros2 --version
# Expected: ros2 cli version X.X.X
```

---

## Phase 2 — Install ROS2 Dependencies

### 2.1 Install cv_bridge

`cv_bridge` converts between ROS2 `sensor_msgs/Image` messages and OpenCV numpy arrays. The detection node uses it to receive frames from the RealSense topic.

```bash
sudo apt install -y ros-humble-cv-bridge
```

### 2.2 Install realsense2_camera ROS2 package

This package launches the RealSense camera as a ROS2 node and publishes frames as topics.

```bash
sudo apt install -y ros-humble-realsense2-camera
```

Verify it installed:
```bash
ros2 pkg list | grep realsense
# Expected: realsense2_camera
```

### 2.3 Initialise rosdep

`rosdep` automatically installs any missing system dependencies declared in `package.xml`:

```bash
sudo rosdep init
rosdep update
```

---

## Phase 3 — Transfer the Package to Jetson

Copy the `ros2_ws` folder from your Mac to the Jetson via USB drive (same method as in `JETSON_SETUP.md`):

### On your Mac

```bash
# Copy the entire ros2_ws folder to your USB drive
cp -r /Users/mortadamoghazy/Desktop/DROUGA/ros2_ws /Volumes/USBDRIVE/
```

### On the Jetson

```bash
# Copy from USB drive to home directory
cp -r /media/$USER/USBDRIVE/ros2_ws ~/
```

---

## Phase 4 — Fix the Model Path in the Node

The node has a default model path that must match where you put your files on the Jetson.

```bash
nano ~/ros2_ws/src/drouga_detection/drouga_detection/detection_node.py
```

Find this line near the top of the `__init__` method:

```python
self.declare_parameter('model_path', '/home/user/drouga/best.engine')
```

Change `/home/user/` to your actual Jetson username. For example:

```python
self.declare_parameter('model_path', '/home/mortada/drouga/best.engine')
```

Save with `Ctrl+O`, exit with `Ctrl+X`.

---

## Phase 5 — Build the Package

### 5.1 Install package dependencies with rosdep

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 5.2 Build with colcon

```bash
cd ~/ros2_ws
colcon build --packages-select drouga_detection
```

Expected output:
```
Starting >>> drouga_detection
Finished <<< drouga_detection [Xs]
Summary: 1 package finished [Xs]
```

### 5.3 Source the workspace

You must source the workspace after every build so ROS2 knows where to find the package:

```bash
source ~/ros2_ws/install/setup.bash
```

To do this automatically in every terminal:
```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## Phase 6 — Running the Node

You need **two terminals** running simultaneously — one for the RealSense camera, one for the detection node.

### Terminal 1 — Start the RealSense camera node

```bash
ros2 launch realsense2_camera rs_launch.py
```

This starts publishing the color stream to `/camera/color/image_raw`. You should see output like:
```
[realsense2_camera] ... publishing to /camera/color/image_raw
```

Verify the topic is live:
```bash
ros2 topic hz /camera/color/image_raw
# Expected: average rate: 30.000
```

### Terminal 2 — Start the detection node

```bash
ros2 run drouga_detection detection_node
```

Expected output:
```
[drouga_detection]: Model loaded: best.engine
[drouga_detection]: Detection node ready — listening on /camera/color/image_raw
```

---

## Phase 7 — Monitoring the Output

Open additional terminals to inspect what the node is publishing.

### Check if a mannequin is detected

```bash
ros2 topic echo /drouga/mannequin_detected
```
Output when nothing detected:
```yaml
data: false
```
Output when mannequin confirmed:
```yaml
data: true
```

### Check the mannequin pixel position

```bash
ros2 topic echo /drouga/mannequin_pixel
```
Output:
```yaml
x: 312.5   # horizontal pixel (0 = left edge, image_width = right edge)
y: 241.0   # vertical pixel (0 = top, image_height = bottom)
z: 0.0     # always 0 — this is a 2D pixel coordinate, not 3D
```

### Check detection confidence

```bash
ros2 topic echo /drouga/mannequin_confidence
```
Output:
```yaml
data: 0.847
```

### Check the active track ID

```bash
ros2 topic echo /drouga/mannequin_track_id
```
Output:
```yaml
data: 7     # ByteTrack track ID — stays the same for the same mannequin
            # -1 means no mannequin detected
```

### Check publishing rate

```bash
ros2 topic hz /drouga/mannequin_detected
# Should match the camera FPS (~30 Hz)
```

### Visualise the annotated image in RViz2

```bash
# Open RViz2
rviz2
```

Inside RViz2:
1. Click **Add** → **By topic** → select `/drouga/annotated_image` → **Image**
2. You will see the live annotated feed with GREEN/ORANGE/GREY boxes

---

## Phase 8 — Changing Parameters at Runtime

All key parameters can be overridden without editing or rebuilding the code.

### Override at launch

```bash
ros2 run drouga_detection detection_node --ros-args \
  -p conf:=0.4 \
  -p confirm_frames:=15 \
  -p residual_threshold:=12.0 \
  -p publish_annotated:=false
```

### Full list of parameters

| Parameter | Default | What it controls |
|---|---|---|
| `model_path` | `/home/user/drouga/best.engine` | Path to the YOLO model |
| `conf` | `0.35` | Detection confidence threshold |
| `mannequin_class` | `0` | Class index for mannequin in the model |
| `confirm_frames` | `25` | Frames a track must be seen before trusted (~0.8s) |
| `residual_window` | `60` | Frames of motion history to keep (2 seconds) |
| `residual_threshold` | `8.0` | Max pixel residual to classify as stationary |
| `publish_annotated` | `true` | Whether to publish the annotated debug image |

### Change a parameter while the node is already running

```bash
ros2 param set /drouga_detection conf 0.5
ros2 param set /drouga_detection confirm_frames 20
```

---

## Phase 9 — Using a Launch File (optional but recommended)

Instead of starting each node manually in separate terminals, a launch file starts everything together.

Create the file `~/ros2_ws/src/drouga_detection/launch/drouga.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # RealSense camera node
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense',
            parameters=[{
                'color_width':  640,
                'color_height': 480,
                'color_fps':    30.0,
                'enable_depth': False,   # set True when depth fusion is added
            }]
        ),

        # DROUGA detection node
        Node(
            package='drouga_detection',
            executable='detection_node',
            name='drouga_detection',
            parameters=[{
                'model_path':         '/home/mortada/drouga/best.engine',
                'conf':               0.35,
                'confirm_frames':     25,
                'residual_threshold': 8.0,
                'publish_annotated':  True,
            }],
            output='screen'
        ),
    ])
```

Then run the whole stack with one command:

```bash
ros2 launch drouga_detection drouga.launch.py
```

---

## Quick Reference — All Commands

```bash
# ── First time only ──────────────────────────────────────────────────────────

# Build
cd ~/ros2_ws && colcon build --packages-select drouga_detection

# Source (add to ~/.bashrc so you don't have to repeat this)
source ~/ros2_ws/install/setup.bash

# ── Every session ────────────────────────────────────────────────────────────

# Terminal 1 — camera
ros2 launch realsense2_camera rs_launch.py

# Terminal 2 — detection node
ros2 run drouga_detection detection_node

# Terminal 3 — monitor
ros2 topic echo /drouga/mannequin_detected
ros2 topic echo /drouga/mannequin_pixel
ros2 topic hz   /drouga/mannequin_detected

# ── Or launch everything at once ─────────────────────────────────────────────
ros2 launch drouga_detection drouga.launch.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Package drouga_detection not found` | Run `source ~/ros2_ws/install/setup.bash` |
| `No module named ultralytics` | Run `pip3 install ultralytics` |
| `No module named cv_bridge` | Run `sudo apt install ros-humble-cv-bridge` |
| Node starts but no detections at all | Check `/camera/color/image_raw` is publishing: `ros2 topic hz /camera/color/image_raw` |
| `best.engine not found, falling back to best.pt` | TensorRT export not done yet — run Phase 7 of JETSON_SETUP.md |
| Node crashes on first frame | cv_bridge encoding mismatch — check the RealSense is publishing `bgr8` or `rgb8` and that the `desired_encoding` in the node matches |
| Annotated image not showing in RViz | Make sure `publish_annotated` parameter is `true` and the RViz Image display is set to `/drouga/annotated_image` |
| Low publishing rate (< 20 Hz) | Disable annotated image publishing: `-p publish_annotated:=false`. Also ensure TensorRT engine is being used, not best.pt |
