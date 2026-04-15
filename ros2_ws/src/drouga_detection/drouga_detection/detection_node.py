"""
DROUGA — ROS2 Mannequin Detection Node

Subscribes to the RealSense color image topic, runs YOLO + ByteTrack +
ego-motion compensation, and publishes detection results.

Subscribed topics:
  /camera/color/image_raw   (sensor_msgs/Image)  — from realsense-ros

Published topics:
  /drouga/mannequin_detected   (std_msgs/Bool)          — True when a mannequin is confirmed
  /drouga/mannequin_pixel      (geometry_msgs/Point)    — pixel centre (cx, cy, 0)
  /drouga/mannequin_confidence (std_msgs/Float32)       — YOLO confidence of confirmed track
  /drouga/mannequin_track_id   (std_msgs/Int32)         — ByteTrack ID of confirmed mannequin
  /drouga/annotated_image      (sensor_msgs/Image)      — debug view with boxes drawn

Build and run:
  cd ~/drouga/ros2_ws
  colcon build --packages-select drouga_detection
  source install/setup.bash
  ros2 run drouga_detection detection_node
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, deque

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

from ultralytics import YOLO


class EgoMotion:
    """
    Estimates frame-to-frame camera homography using Lucas-Kanade optical flow.
    Projects previous track positions through the homography to find how much
    each track moved beyond camera motion (the residual).
    """
    def __init__(self):
        self.prev_gray = None

    def update(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H = np.eye(3, dtype=np.float32)
        if self.prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(
                self.prev_gray, maxCorners=300, qualityLevel=0.01,
                minDistance=10, blockSize=3
            )
            if pts is not None and len(pts) >= 8:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, pts, None,
                    winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                )
                good_prev = pts[status == 1]
                good_curr = curr_pts[status == 1]
                if len(good_prev) >= 4:
                    H_est, _ = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 3.0)
                    if H_est is not None:
                        H = H_est.astype(np.float32)
        self.prev_gray = gray
        return H

    def project(self, H, cx, cy):
        pt   = np.array([[[cx, cy]]], dtype=np.float32)
        proj = cv2.perspectiveTransform(pt, H)
        return float(proj[0, 0, 0]), float(proj[0, 0, 1])


class DetectionNode(Node):

    def __init__(self):
        super().__init__('drouga_detection')

        # ── Parameters (can be overridden from launch file or command line) ───
        self.declare_parameter('model_path',         '/home/user/drouga/best.engine')
        self.declare_parameter('conf',               0.35)
        self.declare_parameter('mannequin_class',    0)
        self.declare_parameter('confirm_frames',     25)
        self.declare_parameter('residual_window',    60)
        self.declare_parameter('residual_threshold', 8.0)
        self.declare_parameter('publish_annotated',  True)   # set False to save bandwidth

        model_path         = self.get_parameter('model_path').value
        self.conf          = self.get_parameter('conf').value
        self.mann_cls      = self.get_parameter('mannequin_class').value
        self.confirm_frames     = self.get_parameter('confirm_frames').value
        self.residual_window    = self.get_parameter('residual_window').value
        self.residual_threshold = self.get_parameter('residual_threshold').value
        self.publish_annotated  = self.get_parameter('publish_annotated').value

        # Fallback to best.pt if engine not found
        if not Path(model_path).exists():
            fallback = str(Path(model_path).parent / 'best.pt')
            self.get_logger().warn(f'{model_path} not found, falling back to {fallback}')
            model_path = fallback

        # ── Load YOLO model ───────────────────────────────────────────────────
        self.model = YOLO(model_path)
        self.get_logger().info(f'Model loaded: {model_path}')

        # ── Ego-motion + tracking state ───────────────────────────────────────
        self.ego              = EgoMotion()
        self.consecutive_hits = defaultdict(int)
        self.mannequin_tracks = set()
        self.prev_centre:     dict[int, tuple] = {}
        self.residual_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.residual_window)
        )

        # ── cv_bridge — converts ROS Image ↔ OpenCV numpy array ──────────────
        self.bridge = CvBridge()

        # ── QoS profiles ──────────────────────────────────────────────────────
        # Best-effort for image streams (drop frames rather than queue them up)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Reliable for detection results (mission state machine must not miss these)
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── Subscriber ────────────────────────────────────────────────────────
        self.sub_image = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            image_qos
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_detected   = self.create_publisher(Bool,    '/drouga/mannequin_detected',   reliable_qos)
        self.pub_pixel      = self.create_publisher(Point,   '/drouga/mannequin_pixel',      reliable_qos)
        self.pub_conf       = self.create_publisher(Float32, '/drouga/mannequin_confidence', reliable_qos)
        self.pub_track_id   = self.create_publisher(Int32,   '/drouga/mannequin_track_id',   reliable_qos)
        self.pub_annotated  = self.create_publisher(Image,   '/drouga/annotated_image',      image_qos)

        self.get_logger().info('Detection node ready — listening on /camera/color/image_raw')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def residual_p75(self, tid):
        h = self.residual_history[tid]
        return float(np.percentile(h, 75)) if h else 999.0

    # ── Main callback — runs once per incoming image frame ───────────────────

    def image_callback(self, msg: Image):
        # Convert ROS Image message to OpenCV BGR numpy array
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h_f, w_f = frame.shape[:2]

        # Update ego-motion homography for this frame
        H = self.ego.update(frame)

        # YOLO + ByteTrack inference
        result = self.model.track(
            frame,
            conf    = self.conf,
            classes = [self.mann_cls],
            tracker = 'bytetrack.yaml',
            persist = True,
            verbose = False
        )[0]

        active_tids = set()
        best_target = None        # (tid, conf, cx, cy) of the best confirmed mannequin this frame
        best_target_conf = 0.0

        if result.boxes is not None and result.boxes.id is not None:
            for box in result.boxes:
                tid  = int(box.id)
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                active_tids.add(tid)
                self.consecutive_hits[tid] += 1

                # Ego-motion residual
                if tid in self.prev_centre:
                    px, py = self.prev_centre[tid]
                    pred_cx, pred_cy = self.ego.project(H, px, py)
                    self.residual_history[tid].append(
                        np.hypot(cx - pred_cx, cy - pred_cy)
                    )
                self.prev_centre[tid] = (cx, cy)

                # Classify
                p75 = self.residual_p75(tid)
                confirmed = self.consecutive_hits[tid] >= self.confirm_frames
                is_stationary = (
                    len(self.residual_history[tid]) >= self.residual_window // 2
                    and p75 < self.residual_threshold
                )

                if confirmed and is_stationary:
                    self.mannequin_tracks.add(tid)
                elif tid in self.mannequin_tracks and not is_stationary:
                    self.mannequin_tracks.discard(tid)

                is_target = tid in self.mannequin_tracks

                # Track the highest-confidence confirmed mannequin this frame
                if is_target and conf > best_target_conf:
                    best_target      = (tid, conf, cx, cy)
                    best_target_conf = conf

                # Draw on frame (only if annotated image publishing is on)
                if self.publish_annotated:
                    if is_target:
                        color, thickness = (0, 220, 0), 3
                        status = f'MANNEQUIN  res={p75:.1f}px'
                    elif confirmed:
                        color, thickness = (0, 140, 255), 2
                        status = f'MOVING  res={p75:.1f}px'
                    else:
                        color, thickness = (160, 160, 160), 1
                        status = f'{self.consecutive_hits[tid]}/{self.confirm_frames}'

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    label = f'#{tid} {status}  {conf:.2f}'
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
                    if is_target:
                        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 220, 0), -1)

        # Reset consecutive hits for tracks ByteTrack lost this frame
        for tid in list(self.consecutive_hits):
            if tid not in active_tids:
                self.consecutive_hits[tid] = 0

        # ── Publish detection results ─────────────────────────────────────────
        detected = best_target is not None

        self.pub_detected.publish(Bool(data=detected))

        if detected:
            tid, conf, cx, cy = best_target
            self.pub_pixel.publish(Point(x=cx, y=cy, z=0.0))
            self.pub_conf.publish(Float32(data=conf))
            self.pub_track_id.publish(Int32(data=tid))
        else:
            # Publish zeroed values so subscribers don't stall waiting for a message
            self.pub_pixel.publish(Point(x=0.0, y=0.0, z=0.0))
            self.pub_conf.publish(Float32(data=0.0))
            self.pub_track_id.publish(Int32(data=-1))

        # ── Publish annotated image ───────────────────────────────────────────
        if self.publish_annotated:
            n_targets = len([t for t in active_tids if t in self.mannequin_tracks])
            hud = (f'ByteTrack | Active: {len(active_tids)} | '
                   f'Mannequin: {n_targets} | conf={self.conf}')
            cv2.putText(frame, hud, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            annotated_msg.header = msg.header   # keep original timestamp
            self.pub_annotated.publish(annotated_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
