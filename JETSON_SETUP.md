# Jetson Orin Nano + RealSense D435i — Full Setup Guide
## DROUGA Mannequin Detection Pipeline

---

## Overview

This guide sets up the full mannequin detection pipeline on the Jetson Orin Nano using the Intel RealSense D435i camera. By the end you will have:

- YOLO + ByteTrack running in real time at ~28 FPS using TensorRT FP16
- RealSense color stream feeding frames into the model
- Ego-motion compensation to distinguish stationary mannequins from moving humans
- A live annotated display with GREEN (mannequin) / ORANGE (human) / GREY (unconfirmed) boxes

**Prerequisites before starting:**
- Jetson Orin Nano board with power supply
- MicroSD card (64GB+) or NVMe SSD
- Intel RealSense D435i camera + USB 3.0 cable
- A Ubuntu host machine (x86) with NVIDIA SDK Manager installed — needed to flash JetPack
- `best.pt` trained model (already in your DROUGA folder on Mac)

---

## Phase 1 — Flash JetPack on the Jetson

JetPack is NVIDIA's OS image for Jetson. It includes Ubuntu + CUDA + TensorRT + cuDNN pre-installed.

### 1.1 Install SDK Manager on your Ubuntu host machine

Download SDK Manager from:
https://developer.nvidia.com/sdk-manager

Install it:
```bash
sudo dpkg -i sdkmanager_*.deb
sudo apt --fix-broken install
```

### 1.2 Put the Jetson into recovery mode

1. Connect the Jetson to your Ubuntu host via USB-C
2. Hold the **Recovery button** on the Jetson carrier board
3. While holding it, press and release the **Power button**
4. Release the Recovery button after 2 seconds
5. Verify the Jetson is detected:
```bash
lsusb | grep NVIDIA
# Should show something like: NVIDIA Corp. APX
```

### 1.3 Flash with SDK Manager

1. Open SDK Manager on the host
2. Select **Jetson Orin Nano** as the target hardware
3. Select **JetPack 6.x** (latest stable)
4. Select components: **Jetson OS + Jetson SDK Components** (includes CUDA, TensorRT, cuDNN)
5. Click Flash and wait (~30 minutes)

### 1.4 First boot

After flashing completes, the Jetson will boot into Ubuntu. Connect a monitor, keyboard, and mouse for the initial setup (set username, password, timezone).

### 1.5 Verify the installation

```bash
# Check JetPack version
jetson_release

# Check CUDA
nvcc --version
# Expected: Cuda compilation tools, release 12.x

# Check TensorRT
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

---

## Phase 2 — System Dependencies

Run all of these on the Jetson after first boot.

### 2.1 Update the system

```bash
sudo apt update && sudo apt upgrade -y
```

### 2.2 Install build tools and libraries

```bash
sudo apt install -y \
    python3-pip \
    python3-dev \
    cmake \
    git \
    build-essential \
    libssl-dev \
    libusb-1.0-0-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libopencv-dev \
    python3-opencv \
    udev
```

### 2.3 Set up udev rules for RealSense USB access

Without this the camera will fail with permission errors:

```bash
# Download Intel's udev rules
sudo wget -q -O /etc/udev/rules.d/99-realsense-libusb.rules \
  https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules

sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

## Phase 3 — Install PyTorch for Jetson

> **Critical:** Do NOT use `pip install torch`. That installs the x86 CPU-only version and will not use the Jetson GPU.

NVIDIA provides Jetson-specific PyTorch wheels built for ARM64 + CUDA.

### 3.1 Install the Jetson PyTorch wheel

For **JetPack 6.x** (CUDA 12.x):
```bash
pip3 install --no-cache \
  https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+ebedce2.nv24.4-cp310-cp310-linux_aarch64.whl
```

> If you are on JetPack 5.x (CUDA 11.x), go to:
> https://developer.download.nvidia.com/compute/redist/jp/
> and find the v51 or v512 folder for the correct wheel.

### 3.2 Install torchvision

```bash
# Install dependencies first
sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev

# Build torchvision from source (must match your PyTorch version)
git clone --branch v0.18.0 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install --user
cd ..
```

### 3.3 Verify GPU is accessible

```bash
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
"
# Expected output:
# CUDA available: True
# Device: Orin (or similar)
```

---

## Phase 4 — Build and Install RealSense SDK

Intel does not provide pre-built ARM64 packages for Jetson, so we build from source.

### 4.1 Clone librealsense

```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
```

### 4.2 Apply udev rules (if not done in Phase 2)

```bash
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 4.3 Build librealsense with Python bindings

```bash
mkdir build && cd build

cmake .. \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DFORCE_RSUSB_BACKEND=ON

# Use -j4 to avoid out-of-memory errors on Orin Nano (8GB RAM)
make -j4

sudo make install
sudo ldconfig
```

> This takes approximately 20–30 minutes.

### 4.4 Verify RealSense SDK

```bash
python3 -c "import pyrealsense2 as rs; print('RealSense SDK:', rs.__version__)"
```

### 4.5 Test the camera

Plug in the RealSense D435i via USB 3.0 (use a USB 3.0 port — the blue ones), then:

```bash
# List connected devices
python3 -c "
import pyrealsense2 as rs
ctx = rs.context()
for dev in ctx.devices:
    print('Found:', dev.get_info(rs.camera_info.name))
    print('Serial:', dev.get_info(rs.camera_info.serial_number))
"
```

Run the viewer to confirm color + depth streams work:
```bash
realsense-viewer
```

---

## Phase 5 — Install Ultralytics on Jetson

```bash
pip3 install ultralytics lapx
```

Verify:
```bash
python3 -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

---

## Phase 6 — Transfer Files from Mac to Jetson

You are connecting to the Jetson directly (monitor + keyboard + mouse), so use a **USB flash drive** to move files across.

### 6.1 On your Mac — copy files to a USB drive

Plug in a USB drive, then open Terminal on your Mac:

```bash
# Replace /Volumes/USBDRIVE with your actual USB drive name
# You can check the name in Finder under Locations

USB=/Volumes/USBDRIVE

# Trained model
cp /Users/mortadamoghazy/Desktop/DROUGA/best.pt $USB/

# Standalone test script (no ROS2 needed — useful for early hardware testing)
cp /Users/mortadamoghazy/Desktop/DROUGA/scripts/live_detect_jetson.py $USB/

# ROS2 package (for full integration with the drone stack)
cp -r /Users/mortadamoghazy/Desktop/DROUGA/ros2_ws $USB/
```

### 6.2 On the Jetson — copy files from the USB drive

Plug the USB drive into the Jetson, then open a terminal:

```bash
# Create project folder
mkdir -p ~/drouga/scripts

# Find the USB drive mount point
lsblk
# Look for a device like /dev/sda1 — it is usually auto-mounted under /media/USERNAME/DRIVENAME

# Copy files (replace USBDRIVE with your actual drive name)
cp /media/$USER/USBDRIVE/best.pt ~/drouga/
cp /media/$USER/USBDRIVE/live_detect_jetson.py ~/drouga/scripts/
cp -r /media/$USER/USBDRIVE/ros2_ws ~/
```

### 6.3 Fix the model path

The standalone script needs its path updated:

```bash
nano ~/drouga/scripts/live_detect_jetson.py
# Find this line near the top:
#   DROUGA_DIR = Path('/home/user/drouga')
# Change it to your actual username, for example:
#   DROUGA_DIR = Path('/home/mortada/drouga')
# Save with Ctrl+O, exit with Ctrl+X
```

The ROS2 node path is fixed in Phase 4 of `ROS2_SETUP.md`.

> **Two ways to run the pipeline:**
> - **Standalone** (`live_detect_jetson.py`) — no ROS2 needed, good for early hardware testing in isolation
> - **ROS2 node** (`drouga_detection`) — full integration with PX4, VINS-Fusion, and the mission state machine. Follow `ROS2_SETUP.md` for this.

---

## Phase 7 — Export Model to TensorRT FP16

> **This step must run ON the Jetson.** TensorRT compiles the model specifically for the GPU it is running on — you cannot export on Mac and copy the engine file.

```bash
cd ~/drouga

python3 - <<'EOF'
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(
    format='engine',   # TensorRT
    half=True,         # FP16 precision — halves memory, ~2x speed
    imgsz=800,         # must match training resolution
    device=0,          # Jetson GPU
)
print("Done — best.engine is ready")
EOF
```

This takes approximately 10 minutes and only needs to be done once. The result is `best.engine` in `~/drouga/`.

Benchmark to confirm speed:
```bash
python3 -c "
from ultralytics import YOLO
import time

model = YOLO('best.engine')
import numpy as np
dummy = np.zeros((800, 800, 3), dtype='uint8')

# Warm up
for _ in range(5):
    model(dummy, verbose=False)

# Benchmark
t = time.time()
for _ in range(30):
    model(dummy, verbose=False)
fps = 30 / (time.time() - t)
print(f'FPS: {fps:.1f}')
"
```

Expected: **~28 FPS** on Orin Nano with FP16.

---

## Phase 8 — Run the Detection Pipeline

There are two ways to run. Start with the standalone script to verify the hardware works, then move to the ROS2 node for full integration.

### Option A — Standalone (no ROS2, good for initial hardware testing)

First run using `best.pt` (before TensorRT export):
```bash
python3 ~/drouga/scripts/live_detect_jetson.py --pt
```

Full speed run using `best.engine` (after TensorRT export):
```bash
python3 ~/drouga/scripts/live_detect_jetson.py
```

Save output to a video file at the same time:
```bash
python3 ~/drouga/scripts/live_detect_jetson.py --save
# Output saved to ~/drouga/live_output.mp4
```

Press **Q** to quit.

### Option B — ROS2 node (full drone stack integration)

Follow `ROS2_SETUP.md` to build the package, then:

```bash
# Terminal 1 — RealSense camera
ros2 launch realsense2_camera rs_launch.py

# Terminal 2 — detection node
ros2 run drouga_detection detection_node

# Or launch everything at once
ros2 launch drouga_detection drouga.launch.py
```

### What you will see (both options)

| Box colour | Meaning |
|---|---|
| **GREEN** | Confirmed mannequin — stationary in world, seen for ≥0.8s |
| **ORANGE** | Confirmed detection but moving — human / bystander |
| **GREY** | Unconfirmed — less than 25 consecutive frames seen |

---

## Phase 9 — Future Upgrade: IMU-Based Ego-Motion

The current pipeline uses **optical flow** to estimate camera motion (~15ms per frame). Once you are comfortable with the setup, replace it with **IMU gyroscope integration** from the RealSense D435i:

- The D435i has a built-in gyroscope running at 400Hz
- Gyro gives camera rotation directly — no feature matching needed
- Processing time: ~0.5ms vs ~15ms for optical flow
- More accurate under fast drone maneuvers where optical flow can fail

The change is in `live_detect_jetson.py` inside the `EgoMotion` class — replace the Lucas-Kanade optical flow block with:

```python
import pyrealsense2 as rs

# In the RealSense pipeline setup, enable gyro stream:
config.enable_stream(rs.stream.gyro)

# Each frame, get the gyro data:
motion_frame = frames.first_or_default(rs.stream.gyro)
gyro_data    = motion_frame.as_motion_frame().get_motion_data()
# gyro_data.x, gyro_data.y, gyro_data.z are angular velocities (rad/s)

# Integrate over dt to get rotation angle, build rotation matrix H
```

This is the step to do **after** the basic pipeline is working end-to-end.

---

## Phase 10 — ROS2 Integration

The ROS2 `drouga_detection` package is already built and ready. Follow **`ROS2_SETUP.md`** for the full step-by-step guide covering:
- Installing ROS2 Humble on Jetson
- Building the package with `colcon`
- Running the node and monitoring all published topics
- Tuning parameters at runtime
- Using a launch file to start everything at once

The node publishes on these topics:

| Topic | Type | Content |
|---|---|---|
| `/drouga/mannequin_detected` | `std_msgs/Bool` | True when a mannequin is confirmed |
| `/drouga/mannequin_pixel` | `geometry_msgs/Point` | Pixel centre (u, v, 0) |
| `/drouga/mannequin_confidence` | `std_msgs/Float32` | YOLO confidence score |
| `/drouga/mannequin_track_id` | `std_msgs/Int32` | ByteTrack ID (-1 when none) |
| `/drouga/mannequin_position_3d` | `geometry_msgs/PointStamped` | 3D position in camera frame (X, Y, Z metres) |
| `/drouga/annotated_image` | `sensor_msgs/Image` | Debug view with boxes drawn |

---

## Quick Reference — All Commands in Order

```bash
# On Jetson — run these in sequence

# 1. System deps
sudo apt update && sudo apt install -y python3-pip cmake git build-essential \
  libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev libopencv-dev python3-opencv udev

# 2. PyTorch (JetPack 6.x)
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+ebedce2.nv24.4-cp310-cp310-linux_aarch64.whl

# 3. Build RealSense SDK
git clone https://github.com/IntelRealSense/librealsense.git && cd librealsense
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$(which python3) \
         -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DFORCE_RSUSB_BACKEND=ON
make -j4 && sudo make install && sudo ldconfig
cd ~/

# 4. Ultralytics
pip3 install ultralytics lapx

# 5. Transfer files via USB drive
# On Mac: copy best.pt, live_detect_jetson.py, and ros2_ws/ to USB drive
# On Jetson:
#   cp /media/$USER/USBDRIVE/best.pt ~/drouga/
#   cp /media/$USER/USBDRIVE/live_detect_jetson.py ~/drouga/scripts/
#   cp -r /media/$USER/USBDRIVE/ros2_ws ~/

# 6. TensorRT export (run on Jetson)
cd ~/drouga
python3 -c "from ultralytics import YOLO; YOLO('best.pt').export(format='engine', half=True, imgsz=800, device=0)"

# 7a. Run standalone (no ROS2)
python3 ~/drouga/scripts/live_detect_jetson.py

# 7b. Run as ROS2 node (see ROS2_SETUP.md for full setup first)
ros2 launch drouga_detection drouga.launch.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA not available` after installing PyTorch | Wrong wheel — make sure you used the Jetson ARM64 wheel, not the pip default |
| `No module named pyrealsense2` | Build step failed — check cmake output for errors, try `make -j2` instead of `j4` |
| RealSense not detected | Use USB 3.0 port (blue). Try `lsusb` to confirm the device appears |
| TensorRT export fails | Run `sudo nvpmodel -m 0 && sudo jetson_clocks` first to set max power mode |
| Low FPS with best.engine | Run `sudo jetson_clocks` to lock clocks at max frequency |
| Camera permission denied | Run `sudo udevadm control --reload-rules && sudo udevadm trigger` and reconnect the camera |
