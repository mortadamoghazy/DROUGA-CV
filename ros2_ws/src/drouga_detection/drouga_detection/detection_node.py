"""
DROUGA — ROS2 Mannequin Detection Node
=======================================

Full pipeline (in order of execution per frame):

  1. EIS (Electronic Image Stabilisation)
       The raw frame is warped using the VSLAM orientation to remove drone vibration
       before YOLO sees it. An exponential moving average of the camera orientation
       gives a smooth reference; the deviation from that reference is the vibration.
       Alpha is adaptive: large intentional movements temporarily bypass the filter
       so EIS does not fight the drone's navigation.

  2. YOLO + ByteTrack detection
       YOLO runs on the stabilised frame to detect mannequins. ByteTrack assigns
       consistent IDs across frames. Detected pixel coordinates are then mapped
       back to the raw frame's coordinate system for depth/3D operations.

  3. Metric 1 — World-frame centroid stability
       Each detection is back-projected to a 3D position in the fixed world frame
       using the RealSense depth and the VSLAM camera pose. A rolling window of
       world-frame positions per track is kept. Low standard deviation → the object
       is not moving in the real world → mannequin candidate.
       Catches: walking humans, humans making large movements.

  4. Metric 2 — Intra-bounding-box optical flow after ego-compensation
       The previous raw frame is warped by the camera rotation (from VSLAM delta
       pose) to remove camera motion, then cropped to the bounding box. Dense
       optical flow between the warped crop and the current crop measures how much
       the object's body moved independently of the camera. Low flow → rigid static
       object → mannequin candidate.
       Catches: standing humans waving, gesturing, or shifting weight.

  5. Classification gate
       A track is confirmed as a mannequin only when BOTH metrics are below their
       thresholds AND the track has been seen for at least confirm_frames consecutive
       frames. Either metric exceeding its threshold reclassifies as human.

  Fallback (VSLAM not tracking):
       EIS is disabled. Metric 1 is suspended. Metric 2 continues using raw D435i
       IMU gyroscope integration for the ego-compensation warp.

Subscribed topics:
  /camera/color/image_raw                  (sensor_msgs/Image)      — color frame
  /camera/aligned_depth_to_color/image_raw (sensor_msgs/Image)      — depth aligned to color
  /camera/color/camera_info                (sensor_msgs/CameraInfo) — intrinsics
  /visual_slam/tracking/odometry           (nav_msgs/Odometry)      — VSLAM camera pose
  /visual_slam/status                      (VisualSlamStatus)        — VSLAM state (optional)
  /camera/imu                              (sensor_msgs/Imu)         — D435i IMU fallback

Published topics:
  /drouga/mannequin_detected    (std_msgs/Bool)               — True when mannequin confirmed
  /drouga/mannequin_pixel       (geometry_msgs/Point)         — pixel centre in raw frame
  /drouga/mannequin_confidence  (std_msgs/Float32)            — YOLO confidence score
  /drouga/mannequin_track_id    (std_msgs/Int32)              — ByteTrack ID (-1 when none)
  /drouga/mannequin_position_3d (geometry_msgs/PointStamped)  — world-frame 3D position (metres)
  /drouga/annotated_image       (sensor_msgs/Image)           — debug view on stabilised frame
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, deque

import message_filters
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import Image, CameraInfo, Imu
from std_msgs.msg import Bool, Float32, Int32
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from ultralytics import YOLO

# isaac_ros_visual_slam_interfaces is only available when Isaac ROS is installed.
# We import it optionally so the node still starts on machines without it.
try:
    from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
    _HAVE_SLAM_STATUS = True
except ImportError:
    _HAVE_SLAM_STATUS = False


# ─────────────────────────────────────────────────────────────────────────────
# DetectionNode
# ─────────────────────────────────────────────────────────────────────────────

class DetectionNode(Node):
    """
    Main ROS2 node. One instance runs for the lifetime of the process.
    All state is stored as instance attributes — there is no global state.
    """

    def __init__(self):
        super().__init__('drouga_detection')

        # ── Parameters ────────────────────────────────────────────────────────
        # All parameters can be overridden at launch time with:
        #   ros2 run drouga_detection detection_node --ros-args -p conf:=0.4
        # or changed while running with:
        #   ros2 param set /drouga_detection conf 0.4

        self.declare_parameter('model_path',      '/home/user/drouga/best.engine')
        self.declare_parameter('conf',            0.35)   # YOLO confidence threshold
        self.declare_parameter('mannequin_class', 0)      # class index in the model (0 = mannequin)
        self.declare_parameter('publish_annotated', True) # set False to save bandwidth

        # Confirmation gate — how many consecutive frames a track must be seen
        # before we start trusting its classification (~0.8s at 30fps)
        self.declare_parameter('confirm_frames', 25)

        # EIS parameters
        self.declare_parameter('eis_alpha', 0.08)
        # eis_alpha: exponential moving average weight for the smooth orientation
        # reference.  Smaller → heavier smoothing, more vibration removed but
        # laggier during intentional movements.  0.08 is a good starting point.

        self.declare_parameter('eis_aggressive_threshold', 5.0)
        # eis_aggressive_threshold (degrees): if the camera rotation between two
        # consecutive frames exceeds this, alpha is temporarily raised to 0.6 so
        # the smooth reference catches up and EIS stops fighting navigation.

        # Metric 1 — world-frame centroid stability
        self.declare_parameter('world_stability_window',    30)
        # world_stability_window: number of frames kept in the history for
        # computing position variance (~1 second at 30fps).

        self.declare_parameter('world_stability_threshold', 0.08)
        # world_stability_threshold (metres): if the mean standard deviation of
        # the world-frame 3D position across the window is below this, the object
        # is considered stationary.  8cm is a good starting point — drone hover
        # jitter is typically 2-3cm, so this gives comfortable headroom.

        # Metric 2 — intra-bounding-box optical flow
        self.declare_parameter('bbox_flow_threshold', 1.5)
        # bbox_flow_threshold (pixels): if the 75th-percentile mean optical flow
        # inside the bounding box (after ego-compensation) is below this, the
        # object's body is considered stationary.  1.5px is a starting point;
        # tune downward to catch subtler gestures.

        self.declare_parameter('depth_window', 10)
        # depth_window (pixels): side length of the patch used for median depth
        # sampling at the detection centre.  Larger → more robust to depth holes
        # but less precise for small objects.

        # Read all parameter values into instance variables now.
        # (Subsequent runtime changes via ros2 param set are NOT auto-applied to
        #  these variables — the node would need to be restarted or a param
        #  event handler added.  For a competition pipeline that is acceptable.)
        model_path = self.get_parameter('model_path').value
        self.conf             = self.get_parameter('conf').value
        self.mann_cls         = self.get_parameter('mannequin_class').value
        self.pub_annotated    = self.get_parameter('publish_annotated').value
        self.confirm_frames   = self.get_parameter('confirm_frames').value
        self.eis_alpha        = self.get_parameter('eis_alpha').value
        self.eis_agg_thresh   = self.get_parameter('eis_aggressive_threshold').value
        self.world_win        = self.get_parameter('world_stability_window').value
        self.world_thresh     = self.get_parameter('world_stability_threshold').value
        self.flow_thresh      = self.get_parameter('bbox_flow_threshold').value
        self.depth_window     = self.get_parameter('depth_window').value

        # ── Model loading ─────────────────────────────────────────────────────
        # Fall back to best.pt if the TensorRT engine has not been exported yet.
        if not Path(model_path).exists():
            fallback = str(Path(model_path).parent / 'best.pt')
            self.get_logger().warn(f'{model_path} not found — falling back to {fallback}')
            model_path = fallback
        self.model = YOLO(model_path)
        self.get_logger().info(f'Model loaded: {model_path}')

        # ── ByteTrack / classification state ──────────────────────────────────
        # consecutive_hits[tid]: how many frames in a row this track has been
        # detected.  Reset to 0 when ByteTrack loses the track.
        self.consecutive_hits: dict[int, int]   = defaultdict(int)

        # mannequin_tracks: set of track IDs currently classified as mannequin.
        self.mannequin_tracks: set[int]          = set()

        # world_pos_history[tid]: rolling window of (X, Y, Z) world-frame
        # positions for Metric 1.
        self.world_pos_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.world_win)
        )

        # bbox_flow_history[tid]: rolling window of mean intra-bbox flow
        # magnitudes (pixels) for Metric 2.
        self.bbox_flow_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.world_win)
        )

        # ── EIS state ─────────────────────────────────────────────────────────
        # smooth_R: the current low-pass-filtered camera orientation as a
        # scipy Rotation object.  None until the first VSLAM message arrives.
        self.smooth_R: Rotation | None = None

        # prev_R: camera orientation at the previous frame, used to compute the
        # inter-frame angle change for the adaptive alpha logic.
        self.prev_R:   Rotation | None = None

        # H_stab: the 3×3 homography applied to the current raw frame to produce
        # the stabilised frame fed to YOLO.  Identity when EIS is inactive.
        self.H_stab     = np.eye(3, dtype=np.float32)
        self.H_stab_inv = np.eye(3, dtype=np.float32)

        # ── VSLAM state ───────────────────────────────────────────────────────
        # latest_odom: the most recently received VSLAM odometry message.
        # Stored here so the 2-topic image synchroniser can use it without
        # needing to synchronise 3 topics (which would stall if VSLAM stops).
        self.latest_odom = None

        # vslam_tracking: True while Isaac ROS reports a healthy SLAM state.
        # Set via _slam_status_callback if the message type is importable.
        # Defaults to True so EIS is active from startup.
        self.vslam_tracking: bool = True

        # prev_T_world_cam: camera-to-world transform from the previous frame.
        # Used to compute the relative pose T_rel for Metric 2.
        self.prev_T_world_cam: np.ndarray | None = None

        # prev_raw_frame: the full BGR raw frame from the previous camera tick.
        # Needed by Metric 2 to compute optical flow between consecutive frames.
        self.prev_raw_frame: np.ndarray | None = None

        # ── Camera intrinsics ─────────────────────────────────────────────────
        # Populated once from /camera/color/camera_info.
        # fx, fy: focal lengths in pixels.
        # cx_cam, cy_cam: principal point (optical axis crossing the image plane).
        self.fx = self.fy = self.cx_cam = self.cy_cam = None

        # ── IMU fallback state ────────────────────────────────────────────────
        # When VSLAM is not tracking we use the D435i's built-in gyroscope to
        # estimate the camera rotation between frames for Metric 2.
        # imu_buffer: deque of (timestamp_seconds, angular_velocity_xyz) tuples.
        # last_frame_stamp: ROS time of the previous frame, used to slice the
        # buffer to get only samples from the current inter-frame interval.
        self.imu_buffer:       deque             = deque(maxlen=800)
        self.last_frame_stamp: float | None      = None

        # ── cv_bridge ─────────────────────────────────────────────────────────
        # Converts between ROS Image messages and OpenCV numpy arrays.
        self.bridge = CvBridge()

        # ── QoS profiles ──────────────────────────────────────────────────────
        # BEST_EFFORT for image streams: drop frames rather than queue them up.
        # A queued image is stale and would introduce latency.
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # RELIABLE for detection results: the mission state machine must not
        # miss a mannequin_detected=True message.
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── Subscribers ───────────────────────────────────────────────────────

        # Camera intrinsics — fires continuously but we only store the first
        # message (the calibration does not change at runtime).
        self.create_subscription(
            CameraInfo, '/camera/color/camera_info',
            self._camera_info_callback, reliable_qos
        )

        # VSLAM odometry — stored as self.latest_odom, not synchronised with
        # images.  This way the detection pipeline keeps running even if VSLAM
        # temporarily stops publishing (e.g. tracking lost).
        self.create_subscription(
            Odometry, '/visual_slam/tracking/odometry',
            self._odom_callback, reliable_qos
        )

        # VSLAM status — optional, used to set self.vslam_tracking flag.
        # Only subscribed if the Isaac ROS message package is available.
        if _HAVE_SLAM_STATUS:
            self.create_subscription(
                VisualSlamStatus, '/visual_slam/status',
                self._slam_status_callback, reliable_qos
            )
        else:
            self.get_logger().warn(
                'isaac_ros_visual_slam_interfaces not found — '
                'assuming VSLAM is always tracking. '
                'EIS fallback to IMU will not activate automatically.'
            )

        # D435i IMU — buffered at 400Hz for use as a fallback when VSLAM is
        # not available.  We only use the gyroscope (angular_velocity).
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.create_subscription(Imu, '/camera/imu', self._imu_callback, imu_qos)

        # Color + aligned-depth synchroniser (2 topics).
        # ApproximateTimeSynchronizer fires _synced_callback when it finds a
        # color frame and a depth frame whose timestamps are within slop seconds.
        # The D435i hardware-synchronises these two streams so slop=0.05 is safe.
        sub_color = message_filters.Subscriber(self, Image,
                        '/camera/color/image_raw', qos_profile=image_qos)
        sub_depth = message_filters.Subscriber(self, Image,
                        '/camera/aligned_depth_to_color/image_raw',
                        qos_profile=image_qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_color, sub_depth], queue_size=5, slop=0.05
        )
        self.sync.registerCallback(self._synced_callback)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_detected   = self.create_publisher(Bool,         '/drouga/mannequin_detected',    reliable_qos)
        self.pub_pixel      = self.create_publisher(Point,        '/drouga/mannequin_pixel',       reliable_qos)
        self.pub_conf       = self.create_publisher(Float32,      '/drouga/mannequin_confidence',  reliable_qos)
        self.pub_track_id   = self.create_publisher(Int32,        '/drouga/mannequin_track_id',    reliable_qos)
        self.pub_pos3d      = self.create_publisher(PointStamped, '/drouga/mannequin_position_3d', reliable_qos)
        self.pub_annotated  = self.create_publisher(Image,        '/drouga/annotated_image',       image_qos)

        self.get_logger().info('Detection node ready.')

    # ──────────────────────────────────────────────────────────────────────────
    # Subscriber callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _camera_info_callback(self, msg: CameraInfo):
        """
        Store camera intrinsics from the first CameraInfo message received.

        The intrinsic matrix K is stored row-major as a flat list of 9 values:
            K = [fx,  0, cx,
                  0, fy, cy,
                  0,  0,  1]
        fx, fy: focal lengths in pixels — how many pixels correspond to 1 metre
                at 1 metre depth.
        cx, cy: principal point — where the optical axis hits the image plane,
                usually close to the image centre.
        """
        if self.fx is not None:
            return  # already stored
        self.fx     = msg.k[0]
        self.fy     = msg.k[4]
        self.cx_cam = msg.k[2]
        self.cy_cam = msg.k[5]
        self.get_logger().info(
            f'Intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} '
            f'cx={self.cx_cam:.1f} cy={self.cy_cam:.1f}'
        )

    def _odom_callback(self, msg: Odometry):
        """
        Store the latest VSLAM odometry message.

        This is NOT synchronised with image frames.  The _synced_callback reads
        self.latest_odom whenever it needs a camera pose.  Using a stored value
        like this means the pose may be one frame old but keeps the pipeline
        running even when VSLAM briefly stops publishing.
        """
        self.latest_odom = msg

    def _slam_status_callback(self, msg):
        """
        Update the VSLAM tracking flag from Isaac ROS status messages.

        Isaac ROS Visual SLAM publishes a slam_state integer:
          0 = NOT_STARTED
          1 = TRACKING_GOOD   (full SLAM, loop closure active)
          2 = TRACKING_BAD    (visual odometry only, no loop closure)
          3 = LOST

        We consider states 1 and 2 as "tracking" because even VO-only mode
        gives us a valid pose for ego-motion compensation.

        NOTE: verify the exact integer values on your Isaac ROS version by
        running:  ros2 topic echo /visual_slam/status
        """
        self.vslam_tracking = msg.slam_state in (1, 2)

    def _imu_callback(self, msg: Imu):
        """
        Buffer incoming D435i IMU messages for the VSLAM fallback.

        We store only the timestamp (seconds) and angular velocity (gyroscope)
        because we only need to integrate rotation when VSLAM is unavailable.
        The accelerometer is not used — translational ego-motion compensation
        requires double-integration and accumulates too much error.

        The buffer is capped at 800 samples (~2 seconds at 400Hz).
        """
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        omega = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=np.float64)
        self.imu_buffer.append((stamp, omega))

    # ──────────────────────────────────────────────────────────────────────────
    # Utility helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _K_matrix(self) -> np.ndarray:
        """
        Return the 3×3 camera intrinsic matrix K as a float64 numpy array.
        Returns None if intrinsics have not been received yet.
        """
        if self.fx is None:
            return None
        return np.array([
            [self.fx,       0, self.cx_cam],
            [      0, self.fy, self.cy_cam],
            [      0,       0,           1]
        ], dtype=np.float64)

    def _pose_to_matrix(self, odom_msg: Odometry) -> np.ndarray:
        """
        Convert a nav_msgs/Odometry message to a 4×4 homogeneous transform
        T_world_cam.

        T_world_cam transforms a point expressed in the camera frame into the
        world (odom) frame:
            P_world = T_world_cam @ P_cam

        The odometry message contains:
          pose.pose.position    — translation (x, y, z) in world frame
          pose.pose.orientation — rotation as a unit quaternion (x, y, z, w)

        NOTE: this assumes odom_msg.child_frame_id is the camera optical frame
        (e.g. camera_color_optical_frame).  If it is camera_link or base_link
        you need to apply the additional static TF to the optical frame first.
        Verify with:  ros2 topic echo /visual_slam/tracking/odometry | grep child
        """
        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = [p.x, p.y, p.z]
        return T

    def _sample_depth(self, depth_image: np.ndarray, u: float, v: float) -> float:
        """
        Return the median depth (metres) in a small window centred on (u, v).

        The D435i depth image is uint16 with values in millimetres.
        Zero means 'no measurement' (out of range, occluded, or specular surface).
        We exclude zero values and take the median of the rest.
        A median over a patch is more robust than a single pixel — it tolerates
        holes and edge noise without the outlier sensitivity of a mean.

        Returns 0.0 if no valid depth is found in the window.
        """
        h, w = depth_image.shape
        hw   = self.depth_window // 2
        u0, u1 = max(0, int(u) - hw), min(w, int(u) + hw + 1)
        v0, v1 = max(0, int(v) - hw), min(h, int(v) + hw + 1)
        patch = depth_image[v0:v1, u0:u1].flatten()
        valid = patch[patch > 0]
        if valid.size == 0:
            return 0.0
        return float(np.median(valid)) / 1000.0   # mm → metres

    def _backproject(self, u: float, v: float, depth_m: float):
        """
        Convert pixel (u, v) + depth (metres) → 3D point in camera frame.

        Pinhole camera model inverted:
            X_cam = (u - cx) * Z / fx
            Y_cam = (v - cy) * Z / fy
            Z_cam = depth_m

        ROS/RealSense camera frame convention:
            Z  = forward (depth direction)
            X  = right
            Y  = down
        """
        X = (u - self.cx_cam) * depth_m / self.fx
        Y = (v - self.cy_cam) * depth_m / self.fy
        return X, Y, depth_m

    def _to_world_frame(self, u: float, v: float, depth_m: float,
                        T_world_cam: np.ndarray):
        """
        Back-project a pixel + depth to a 3D position in the world frame.

        Steps:
          1. _backproject  → (X, Y, Z) in camera frame
          2. Homogeneous multiply by T_world_cam  → (X, Y, Z) in world frame

        Returns a (3,) float64 array, or None if depth is invalid or intrinsics
        are not yet known.
        """
        if depth_m <= 0.0 or self.fx is None:
            return None
        X_c, Y_c, Z_c = self._backproject(u, v, depth_m)
        P_cam   = np.array([X_c, Y_c, Z_c, 1.0], dtype=np.float64)
        P_world = T_world_cam @ P_cam
        return P_world[:3]

    def _centroid_stability(self, tid: int) -> float:
        """
        Return the mean standard deviation (metres) of the world-frame 3D
        positions stored for this track ID.

        Lower value → the object barely moved in the world → mannequin candidate.
        Higher value → the object is moving in the world → human.

        Uses np.std across the N×3 position array for each axis, then averages
        the three axis stds to get a single scalar in metres.

        Returns 999.0 if fewer than 5 positions are stored (not enough data).
        """
        h = self.world_pos_history[tid]
        if len(h) < 5:
            return 999.0
        return float(np.mean(np.std(np.array(h), axis=0)))

    def _integrate_imu_rotation(self, t_start: float, t_end: float) -> np.ndarray:
        """
        Integrate buffered D435i gyroscope samples between t_start and t_end
        to get an approximate 3×3 rotation matrix R.

        Used as a fallback for Metric 2 when VSLAM is not tracking.
        Only rotation is estimated — translational ego-motion is not available
        from gyro integration alone.

        Steps for each sample:
          rotvec = omega * dt   (rotation vector: direction = axis, magnitude = angle)
          R_step = Rotation.from_rotvec(rotvec).as_matrix()
          R = R @ R_step        (accumulate)

        Returns np.eye(3) if the buffer contains no samples in [t_start, t_end].
        """
        R = np.eye(3, dtype=np.float64)
        prev_t = t_start
        for stamp, omega in self.imu_buffer:
            if stamp <= t_start:
                continue
            if stamp > t_end:
                break
            dt = stamp - prev_t
            if dt > 0:
                R = R @ Rotation.from_rotvec(omega * dt).as_matrix()
            prev_t = stamp
        return R

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1 — Electronic Image Stabilisation (EIS)
    # ──────────────────────────────────────────────────────────────────────────

    def _update_eis(self, odom_msg: Odometry):
        """
        Compute H_stab: the homography that warps the current raw frame into
        the smooth virtual camera frame, removing vibration.

        Algorithm:
          1. Extract current orientation R_curr from the VSLAM pose.
          2. Compute inter-frame angle change for adaptive alpha.
             If the camera rotated by more than eis_aggressive_threshold degrees
             between frames, temporarily use alpha=0.6 so the smooth reference
             catches up quickly and EIS does not fight intentional navigation.
          3. Update the smooth reference via quaternion LERP:
                q_smooth ← (1-α) * q_smooth + α * q_curr   (normalised)
             This is an approximation of SLERP — accurate for small angular
             steps and computationally cheaper.
          4. Compute the deviation rotation:
                R_dev = R_smooth⁻¹ × R_curr
             This is the part of R_curr that is NOT in the smooth reference —
             i.e. the vibration we want to remove.
          5. Build the pixel-space homography:
                H_stab = K × R_dev⁻¹ × K⁻¹
             Applying R_dev⁻¹ "undoes" the vibration rotation.

        After this call:
          self.H_stab     can be used to warp the raw frame.
          self.H_stab_inv can be used to map stabilised pixels back to raw.
          self.prev_R     is updated to R_curr for the next frame.
        """
        q_msg  = odom_msg.pose.pose.orientation
        R_curr = Rotation.from_quat([q_msg.x, q_msg.y, q_msg.z, q_msg.w])

        # ── Adaptive alpha ────────────────────────────────────────────────────
        if self.prev_R is not None:
            # R_delta is how the camera rotated between this frame and the last.
            # magnitude() returns the rotation angle in radians.
            R_delta    = self.prev_R.inv() * R_curr
            angle_deg  = np.degrees(R_delta.magnitude())
            # If the angle exceeds the threshold the drone is making an
            # intentional manoeuvre.  We raise alpha to 0.6 so the smooth
            # reference follows quickly and the warp stays small.
            alpha = 0.6 if angle_deg > self.eis_agg_thresh else self.eis_alpha
        else:
            alpha = self.eis_alpha

        # ── Update smooth reference via quaternion LERP ───────────────────────
        if self.smooth_R is None:
            # First frame — initialise with current orientation, no warp.
            self.smooth_R = R_curr
            self.prev_R   = R_curr
            self.H_stab     = np.eye(3, dtype=np.float32)
            self.H_stab_inv = np.eye(3, dtype=np.float32)
            return

        q_smooth = self.smooth_R.as_quat()   # [x, y, z, w]
        q_curr   = R_curr.as_quat()

        # Ensure both quaternions are in the same hemisphere to avoid the
        # "double-cover" problem where q and -q represent the same rotation
        # but lerp would take the long way round.
        if np.dot(q_smooth, q_curr) < 0.0:
            q_curr = -q_curr

        q_new = (1.0 - alpha) * q_smooth + alpha * q_curr
        q_new /= np.linalg.norm(q_new)   # renormalise after lerp
        self.smooth_R = Rotation.from_quat(q_new)

        # ── Compute deviation and build homography ────────────────────────────
        # R_dev maps smooth camera orientation to actual camera orientation.
        # Applying R_dev_inv to the frame brings it back to the smooth viewpoint.
        R_dev    = self.smooth_R.inv() * R_curr   # small rotation ≈ vibration
        R_dev_mat = R_dev.as_matrix()

        K = self._K_matrix()
        if K is None:
            # Intrinsics not yet received — identity warp until they arrive.
            self.H_stab     = np.eye(3, dtype=np.float32)
            self.H_stab_inv = np.eye(3, dtype=np.float32)
        else:
            K_inv = np.linalg.inv(K)
            # H_stab removes the deviation: warp actual frame → smooth frame
            H = (K @ np.linalg.inv(R_dev_mat) @ K_inv).astype(np.float32)
            self.H_stab     = H
            self.H_stab_inv = np.linalg.inv(H).astype(np.float32)

        self.prev_R = R_curr

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 3 — Metric 2 helper: intra-bounding-box optical flow
    # ──────────────────────────────────────────────────────────────────────────

    def _intra_bbox_flow(self, raw_frame: np.ndarray, T_rel: np.ndarray,
                         x1: int, y1: int, x2: int, y2: int) -> float:
        """
        Measure independent body motion inside a bounding box by computing
        dense optical flow after warping out the camera's own rotation.

        Why we use the raw frame (not the stabilised one):
            EIS uses bilinear interpolation during warping which slightly blurs
            the image.  Accumulated over two frames, this blur can produce
            artificial flow signal.  Using raw frames for the crop avoids this.

        Steps:
          1. Extract the rotation component R_rel from T_rel (the relative
             camera transform between the previous and current frame).
          2. Build a homography H_rot = K × R_rel × K⁻¹.
             This is the pixel-level transformation due to camera rotation alone.
          3. Warp the previous raw frame with H_rot so that a static rigid point
             appears at the same pixel in both the warped previous frame and the
             current frame.
          4. Crop both frames to the bounding box.
          5. Run Farneback dense optical flow on the greyscale crops.
          6. Return the mean flow vector magnitude across all pixels in the crop.
             Low → nothing moved inside the box → mannequin.
             High → something moved inside the box → human.

        Returns 999.0 if:
          - No previous frame exists yet (first frame).
          - Intrinsics are not yet known.
          - The bounding box is smaller than 20×20 px (object too far to measure).
        """
        if self.prev_raw_frame is None or self.fx is None:
            return 999.0

        # Reject tiny crops — Farneback is unreliable on very few pixels.
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return 999.0

        h, w = raw_frame.shape[:2]

        # ── Build rotation-only ego-motion homography ─────────────────────────
        R_rel = T_rel[:3, :3]   # 3×3 rotation from T_rel (4×4)
        K     = self._K_matrix()
        H_rot = (K @ R_rel @ np.linalg.inv(K)).astype(np.float32)

        # ── Warp previous frame to cancel camera rotation ─────────────────────
        # After this warp, a truly static pixel in the scene should be at the
        # same (u, v) in prev_warped and raw_frame.
        prev_warped = cv2.warpPerspective(self.prev_raw_frame, H_rot, (w, h))

        # ── Crop to bounding box and convert to greyscale ─────────────────────
        prev_crop = cv2.cvtColor(prev_warped[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        curr_crop = cv2.cvtColor(raw_frame[y1:y2,   x1:x2], cv2.COLOR_BGR2GRAY)

        # ── Dense optical flow (Farneback algorithm) ──────────────────────────
        # Farneback computes a dense flow field: for every pixel (u, v) in the
        # previous crop it estimates a 2D displacement (dx, dy) to the current
        # crop.  Parameters:
        #   pyr_scale  = 0.5  — image pyramid scale (0.5 halves each level)
        #   levels     = 3    — number of pyramid levels
        #   winsize    = 15   — averaging window size (larger = smoother)
        #   iterations = 3    — iterations per pyramid level
        #   poly_n     = 5    — pixel neighbourhood for polynomial expansion
        #   poly_sigma = 1.2  — Gaussian smoothing for polynomial expansion
        flow = cv2.calcOpticalFlowFarneback(
            prev_crop, curr_crop, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Magnitude of each pixel's flow vector
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(np.mean(magnitude))

    # ──────────────────────────────────────────────────────────────────────────
    # Main callback — runs once per synchronised color + depth pair (~30 Hz)
    # ──────────────────────────────────────────────────────────────────────────

    def _synced_callback(self, color_msg: Image, depth_msg: Image):
        """
        Entry point for each camera frame.  Executes all five pipeline stages
        in sequence and publishes results.
        """

        # ── Convert ROS messages to numpy arrays ──────────────────────────────
        raw_frame   = self.bridge.imgmsg_to_cv2(color_msg,  desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg,  desired_encoding='passthrough')
        # depth_image dtype is uint16, values are millimetres.
        h_f, w_f = raw_frame.shape[:2]

        # ── Retrieve current VSLAM pose ───────────────────────────────────────
        # We use the most recently buffered odometry message.
        # T_world_cam transforms camera-frame points to world-frame points.
        T_world_cam = None
        if self.latest_odom is not None and self.vslam_tracking:
            T_world_cam = self._pose_to_matrix(self.latest_odom)

        # ── STAGE 1: Electronic Image Stabilisation (EIS) ─────────────────────
        if T_world_cam is not None:
            # VSLAM is available — use full EIS pipeline.
            self._update_eis(self.latest_odom)
            stab_frame = cv2.warpPerspective(raw_frame, self.H_stab, (w_f, h_f))
            # H_stab_inv is used later to map stabilised detection coordinates
            # back to raw frame coordinates for depth sampling.
        else:
            # VSLAM not available — pass the raw frame unchanged to YOLO.
            # H_stab and H_stab_inv remain as identity matrices.
            stab_frame = raw_frame

        # ── STAGE 2: YOLO + ByteTrack detection on stabilised frame ───────────
        # persist=True tells Ultralytics to keep ByteTrack state between calls
        # so that track IDs are consistent across frames.
        result = self.model.track(
            stab_frame,
            conf    = self.conf,
            classes = [self.mann_cls],
            tracker = 'bytetrack.yaml',
            persist = True,
            verbose = False
        )[0]

        # Compute T_rel: how the camera moved relative to the previous frame.
        # Used by _intra_bbox_flow to remove camera rotation from the crop.
        if T_world_cam is not None and self.prev_T_world_cam is not None:
            # Full 6-DOF relative transform from VSLAM (most accurate).
            T_rel = np.linalg.inv(self.prev_T_world_cam) @ T_world_cam
        elif self.last_frame_stamp is not None:
            # VSLAM not tracking — use D435i gyro integration (rotation only).
            curr_stamp = color_msg.header.stamp.sec + \
                         color_msg.header.stamp.nanosec * 1e-9
            R_imu = self._integrate_imu_rotation(self.last_frame_stamp, curr_stamp)
            T_rel = np.eye(4, dtype=np.float64)
            T_rel[:3, :3] = R_imu
        else:
            T_rel = np.eye(4, dtype=np.float64)

        # ── Per-detection processing ──────────────────────────────────────────
        active_tids      = set()
        best_target      = None        # (tid, conf, cx_raw, cy_raw, P_world)
        best_target_conf = 0.0

        # annotation_frame: the stabilised frame with boxes drawn on it.
        # We annotate on the stabilised frame so the display looks smooth.
        annotation_frame = stab_frame.copy() if self.pub_annotated else None

        if result.boxes is not None and result.boxes.id is not None:
            for box in result.boxes:
                tid  = int(box.id)
                conf = float(box.conf)

                # Bounding box in stabilised frame coordinates.
                x1_s, y1_s, x2_s, y2_s = map(int, box.xyxy[0])
                cx_s = (x1_s + x2_s) / 2.0
                cy_s = (y1_s + y2_s) / 2.0

                # ── Map stabilised coordinates back to raw frame ──────────────
                # Depth image and intrinsics are in raw frame coordinates.
                # We must use raw coordinates for all 3D computations.
                # H_stab_inv is the inverse of the EIS warp, mapping each
                # stabilised pixel back to its original raw pixel location.
                pt_s = np.array([[[cx_s, cy_s]]], dtype=np.float32)
                pt_r = cv2.perspectiveTransform(pt_s, self.H_stab_inv)
                cx_r, cy_r = float(pt_r[0, 0, 0]), float(pt_r[0, 0, 1])

                # Map bbox corners back to raw frame for intra-bbox flow crop.
                corners_s = np.array(
                    [[[x1_s, y1_s], [x2_s, y1_s],
                      [x2_s, y2_s], [x1_s, y2_s]]], dtype=np.float32
                )
                corners_r = cv2.perspectiveTransform(corners_s, self.H_stab_inv)
                x1_r = int(np.clip(corners_r[0, :, 0].min(), 0, w_f - 1))
                y1_r = int(np.clip(corners_r[0, :, 1].min(), 0, h_f - 1))
                x2_r = int(np.clip(corners_r[0, :, 0].max(), 0, w_f - 1))
                y2_r = int(np.clip(corners_r[0, :, 1].max(), 0, h_f - 1))

                active_tids.add(tid)
                self.consecutive_hits[tid] += 1

                # ── STAGE 3 — Metric 1: world-frame centroid stability ─────────
                centroid_std = 999.0
                centroid_ok  = False

                if T_world_cam is not None:
                    depth_m  = self._sample_depth(depth_image, cx_r, cy_r)
                    P_world  = self._to_world_frame(cx_r, cy_r, depth_m, T_world_cam)
                    if P_world is not None:
                        self.world_pos_history[tid].append(P_world)
                    centroid_std = self._centroid_stability(tid)
                    centroid_ok  = centroid_std < self.world_thresh
                else:
                    depth_m = self._sample_depth(depth_image, cx_r, cy_r)
                    P_world = None

                # ── STAGE 3 — Metric 2: intra-bbox optical flow ───────────────
                flow_mag = self._intra_bbox_flow(
                    raw_frame, T_rel, x1_r, y1_r, x2_r, y2_r
                )
                self.bbox_flow_history[tid].append(flow_mag)
                flow_p75 = float(
                    np.percentile(list(self.bbox_flow_history[tid]), 75)
                ) if self.bbox_flow_history[tid] else 999.0
                flow_ok = flow_p75 < self.flow_thresh

                # ── STAGE 5 — Classification gate ─────────────────────────────
                confirmed = self.consecutive_hits[tid] >= self.confirm_frames

                # Require at least half the window filled before trusting metrics.
                # This prevents false mannequin labels in the first few frames.
                has_history = len(self.bbox_flow_history[tid]) >= 5

                if T_world_cam is not None:
                    # Normal mode: both metrics must agree.
                    world_ready  = len(self.world_pos_history[tid]) >= self.world_win // 2
                    is_stationary = (has_history and world_ready
                                     and centroid_ok and flow_ok)
                else:
                    # VSLAM fallback: only flow metric available.
                    # Less reliable but keeps the pipeline running.
                    is_stationary = has_history and flow_ok

                if confirmed and is_stationary:
                    self.mannequin_tracks.add(tid)
                elif tid in self.mannequin_tracks and not is_stationary:
                    self.mannequin_tracks.discard(tid)

                is_target = tid in self.mannequin_tracks

                # Track the highest-confidence confirmed mannequin this frame.
                if is_target and conf > best_target_conf:
                    best_target      = (tid, conf, cx_r, cy_r, P_world)
                    best_target_conf = conf

                # ── Annotate the stabilised frame ─────────────────────────────
                if self.pub_annotated and annotation_frame is not None:
                    if is_target:
                        colour    = (0, 220, 0)   # green
                        thickness = 3
                        status = (f'MANNEQUIN '
                                  f'Δ={centroid_std*100:.1f}cm '
                                  f'flow={flow_p75:.1f}px '
                                  f'{depth_m:.2f}m')
                    elif confirmed:
                        colour    = (0, 140, 255)  # orange
                        thickness = 2
                        reason    = 'moving' if not centroid_ok else 'gesture'
                        status = (f'HUMAN({reason}) '
                                  f'Δ={centroid_std*100:.1f}cm '
                                  f'flow={flow_p75:.1f}px')
                    else:
                        colour    = (160, 160, 160)  # grey
                        thickness = 1
                        status = f'{self.consecutive_hits[tid]}/{self.confirm_frames}'

                    cv2.rectangle(annotation_frame,
                                  (x1_s, y1_s), (x2_s, y2_s), colour, thickness)
                    label = f'#{tid} {status}  {conf:.2f}'
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                    cv2.rectangle(annotation_frame,
                                  (x1_s, y1_s - th - 6), (x1_s + tw + 4, y1_s),
                                  colour, -1)
                    cv2.putText(annotation_frame, label, (x1_s + 2, y1_s - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
                    if is_target:
                        cv2.circle(annotation_frame,
                                   (int(cx_s), int(cy_s)), 5, (0, 220, 0), -1)

        # ── Reset hit counter for tracks ByteTrack dropped this frame ─────────
        for tid in list(self.consecutive_hits):
            if tid not in active_tids:
                self.consecutive_hits[tid] = 0

        # ── Publish detection results ─────────────────────────────────────────
        detected = best_target is not None
        self.pub_detected.publish(Bool(data=detected))

        if detected:
            tid, conf, cx_r, cy_r, P_world = best_target
            # Pixel position is in raw frame coordinates.
            self.pub_pixel.publish(Point(x=cx_r, y=cy_r, z=0.0))
            self.pub_conf.publish(Float32(data=conf))
            self.pub_track_id.publish(Int32(data=tid))

            # 3D position in world frame — only published when VSLAM is active
            # and depth is valid.  Published in world frame (odom) so the
            # mission state machine can navigate to it directly.
            if P_world is not None:
                msg3d = PointStamped()
                msg3d.header = color_msg.header
                msg3d.header.frame_id = self.latest_odom.header.frame_id
                msg3d.point.x = float(P_world[0])
                msg3d.point.y = float(P_world[1])
                msg3d.point.z = float(P_world[2])
                self.pub_pos3d.publish(msg3d)
        else:
            # Always publish zeroed values so subscribers don't stall waiting.
            self.pub_pixel.publish(Point(x=0.0, y=0.0, z=0.0))
            self.pub_conf.publish(Float32(data=0.0))
            self.pub_track_id.publish(Int32(data=-1))

        # ── Publish annotated image ───────────────────────────────────────────
        if self.pub_annotated and annotation_frame is not None:
            n_mann = len([t for t in active_tids if t in self.mannequin_tracks])
            mode   = 'VSLAM' if T_world_cam is not None else 'IMU-fallback'
            hud = (f'DROUGA | {mode} | '
                   f'Active: {len(active_tids)} | Mannequin: {n_mann} | '
                   f'conf={self.conf}')
            cv2.putText(annotation_frame, hud, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
            cv2.putText(annotation_frame,
                        'GREEN=mannequin  ORANGE=human  GREY=unconfirmed',
                        (8, h_f - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (180, 180, 180), 1)
            ann_msg = self.bridge.cv2_to_imgmsg(annotation_frame, encoding='bgr8')
            ann_msg.header = color_msg.header
            self.pub_annotated.publish(ann_msg)

        # ── Save state for next frame ─────────────────────────────────────────
        # These are read at the start of the next _synced_callback call.
        self.prev_raw_frame    = raw_frame.copy()   # copy before annotation modifies it
        self.prev_T_world_cam  = T_world_cam
        self.last_frame_stamp  = (color_msg.header.stamp.sec
                                  + color_msg.header.stamp.nanosec * 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

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
