#!/bin/bash
# DROUGA detection node launcher
# Usage:  bash run_detection.sh

MODEL_PATH="/home/mortada/drouga/best.engine"
POSE_MODEL_PATH="/home/mortada/drouga/yolov8n-pose.engine"

echo ""
echo "========================================"
echo "  DROUGA — Detection Node Launcher"
echo "========================================"
echo ""
echo "  1) flow  — Farneback optical flow only"
echo "  2) pose  — YOLOv8-pose joint stability only"
echo "  3) both  — run both, show both scores, both must pass"
echo ""
read -p "Enter 1 / 2 / 3: " CHOICE

case "$CHOICE" in
    1) MODE="flow"  ;;
    2) MODE="pose"  ;;
    3) MODE="both"  ;;
    *) echo "Invalid choice. Exiting." ; exit 1 ;;
esac

echo ""
echo "Launching in [$MODE] mode ..."
echo ""

if [[ "$MODE" == "flow" ]]; then
    ros2 run drouga_detection detection_node --ros-args \
        -p model_path:="$MODEL_PATH" \
        -p classifier_mode:=flow
elif [[ "$MODE" == "pose" ]]; then
    ros2 run drouga_detection detection_node --ros-args \
        -p model_path:="$MODEL_PATH" \
        -p classifier_mode:=pose \
        -p pose_model_path:="$POSE_MODEL_PATH"
else
    ros2 run drouga_detection detection_node --ros-args \
        -p model_path:="$MODEL_PATH" \
        -p classifier_mode:=both \
        -p pose_model_path:="$POSE_MODEL_PATH"
fi
