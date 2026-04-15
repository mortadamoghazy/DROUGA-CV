"""
Live YOLO + ByteTrack mannequin detection on Jetson Orin Nano + RealSense D435i.

Uses:
  - RealSense color stream for frames
  - Optical flow (Lucas-Kanade) for ego-motion compensation
  - TensorRT FP16 model (best.engine) for ~28 FPS on Jetson

Run:
    python3 scripts/live_detect_jetson.py
    python3 scripts/live_detect_jetson.py --save    # also saves to live_output.mp4
    python3 scripts/live_detect_jetson.py --pt      # use best.pt instead of best.engine (slower)

Press Q to quit.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO
import pyrealsense2 as rs

# ── Config ────────────────────────────────────────────────────────────────────
DROUGA_DIR      = Path('/home/user/drouga')          # path on Jetson
MODEL_ENGINE    = DROUGA_DIR / 'best.engine'         # TensorRT FP16 — fast
MODEL_PT        = DROUGA_DIR / 'best.pt'             # fallback if engine not exported yet
CONF            = 0.35
MANNEQUIN_CLASS = 0

CONFIRM_FRAMES     = 25
RESIDUAL_WINDOW    = 60
RESIDUAL_THRESHOLD = 8 # pixels: 75th-pct residual above this → human (Lower = more strict mannequin classification)

# RealSense stream settings
RS_WIDTH  = 640
RS_HEIGHT = 480
RS_FPS    = 30

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true', help='Save output to live_output.mp4')
parser.add_argument('--pt',   action='store_true', help='Use best.pt instead of best.engine')
args = parser.parse_args()

MODEL_PATH = MODEL_PT if args.pt else MODEL_ENGINE
if not MODEL_PATH.exists():
    print(f"WARNING: {MODEL_PATH} not found, falling back to best.pt")
    MODEL_PATH = MODEL_PT

# ── Load model ────────────────────────────────────────────────────────────────
model = YOLO(str(MODEL_PATH))
print(f"Model loaded: {MODEL_PATH.name}")
print(f"Press Q to quit\n")

# ── RealSense pipeline ────────────────────────────────────────────────────────
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)
# IMU streams removed — camera does not have IMU
# Ego-motion is handled by optical flow instead (EgoMotion class)

pipeline.start(config)
print(f"RealSense started: {RS_WIDTH}x{RS_HEIGHT} @ {RS_FPS}fps")

# ── Ego-motion estimator (optical flow fallback — upgrade to IMU later) ───────
class EgoMotion:
    """
    Estimates frame-to-frame camera homography using Lucas-Kanade optical flow.
    NOTE: When ready, replace this with IMU gyro integration from RealSense
    for much faster (~0.5ms vs ~15ms) and more accurate ego-motion estimation.
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

ego = EgoMotion()

# ── Tracking state ────────────────────────────────────────────────────────────
consecutive_hits = defaultdict(int)
mannequin_tracks = set()
prev_centre:      dict[int, tuple] = {}
residual_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=RESIDUAL_WINDOW))

def residual_p75(tid):
    h = residual_history[tid]
    return float(np.percentile(h, 75)) if h else 999.0

# ByteTrack tracker — initialise once, feed frames manually
tracker_gen = model.track(
    source  = 0,            # placeholder — we override with RealSense frames below
    conf    = CONF,
    classes = [MANNEQUIN_CLASS],
    tracker = 'bytetrack.yaml',
    stream  = True,
    verbose = False,
)

# ── Video writer (optional) ───────────────────────────────────────────────────
vid_out = None

# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    frame_idx = 0
    while True:
        # Get RealSense frames
        frames      = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())   # BGR numpy array
        h_f, w_f = frame.shape[:2]

        # Init video writer on first frame
        if args.save and vid_out is None:
            fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = DROUGA_DIR / 'live_output.mp4'
            vid_out  = cv2.VideoWriter(str(out_path), fourcc, RS_FPS, (w_f, h_f))
            print(f"Saving to {out_path}")

        H = ego.update(frame)

        # YOLO + ByteTrack inference
        result     = model.track(frame, conf=CONF, classes=[MANNEQUIN_CLASS],
                                 tracker='bytetrack.yaml', persist=True, verbose=False)[0]
        active_tids = set()

        if result.boxes is not None and result.boxes.id is not None:
            for box in result.boxes:
                tid  = int(box.id)
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                active_tids.add(tid)
                consecutive_hits[tid] += 1

                if tid in prev_centre:
                    px, py = prev_centre[tid]
                    pred_cx, pred_cy = ego.project(H, px, py)
                    residual_history[tid].append(np.hypot(cx - pred_cx, cy - pred_cy))
                prev_centre[tid] = (cx, cy)

                p75 = residual_p75(tid)
                confirmed = consecutive_hits[tid] >= CONFIRM_FRAMES
                is_stationary = (
                    len(residual_history[tid]) >= RESIDUAL_WINDOW // 2
                    and p75 < RESIDUAL_THRESHOLD
                )
                if confirmed and is_stationary:
                    mannequin_tracks.add(tid)
                elif tid in mannequin_tracks and not is_stationary:
                    mannequin_tracks.discard(tid)

                is_target = tid in mannequin_tracks

                if is_target:
                    color, thickness = (0, 220, 0), 3
                    status = f'MANNEQUIN  res={p75:.1f}px'
                elif confirmed:
                    color, thickness = (0, 140, 255), 2
                    status = f'MOVING  res={p75:.1f}px'
                else:
                    color, thickness = (160, 160, 160), 1
                    status = f'{consecutive_hits[tid]}/{CONFIRM_FRAMES}'

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label = f'#{tid} {status}  {conf:.2f}'
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                if is_target:
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 220, 0), -1)

        for tid in list(consecutive_hits):
            if tid not in active_tids:
                consecutive_hits[tid] = 0

        n_targets = len([t for t in active_tids if t in mannequin_tracks])
        hud = (f"ByteTrack + {MODEL_PATH.stem}  |  "
               f"Active: {len(active_tids)}  |  Mannequin: {n_targets}  |  "
               f"conf={CONF}")
        cv2.putText(frame, hud, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(frame,
                    'GREEN=mannequin  ORANGE=moving  GREY=unconfirmed  |  Q to quit',
                    (8, h_f - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        cv2.imshow('DROUGA — Jetson Live', frame)
        if vid_out is not None:
            vid_out.write(frame)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    if vid_out is not None:
        vid_out.release()
        print("Saved live_output.mp4")
    print(f"Stopped after {frame_idx} frames.")
