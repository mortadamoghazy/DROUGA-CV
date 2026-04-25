"""
DROUGA — Live Pose-Based Mannequin/Human Classification
========================================================

Uses YOLOv8-pose (COCO) to estimate 17 body keypoints on every detected
person. Classifies each track as MANNEQUIN or HUMAN by measuring how much
the joints move relative to each other over a rolling window of frames.

Key insight: a mannequin's joints do not move relative to each other — the
body is rigid. A human's joints shift even when the person appears still
(breathing, micro-movements, weight shifts, gestures). By measuring the
SHAPE of the skeleton (not its position) we get a camera-motion-invariant
signal with no background feature extraction required.

Run:
    python scripts/live_detect_pose.py

Optional flags:
    --model  yolov8s-pose.pt   use a larger model (more accurate, slower)
    --conf   0.45              YOLO confidence threshold
    --thresh 0.025             joint stability threshold (lower = stricter)
    --window 25                rolling window size in frames
    --save                     save output to pose_output.mp4

Press Q to quit.
"""

import argparse
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO


# ── COCO keypoint index names (for annotation labels) ─────────────────────────
KP_NAMES = [
    'nose',
    'left_eye', 'right_eye',
    'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]

# Skeleton connections to draw — pairs of keypoint indices.
SKELETON = [
    (5, 6),   # shoulder to shoulder
    (5, 7),   (7, 9),    # left arm
    (6, 8),   (8, 10),   # right arm
    (5, 11),  (6, 12),   # torso sides
    (11, 12),            # hip to hip
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',  default='yolov8n-pose.pt',
                   help='YOLOv8-pose model. Downloaded automatically on first run.')
    p.add_argument('--conf',   type=float, default=0.40,
                   help='YOLO confidence threshold (default 0.40)')
    p.add_argument('--thresh', type=float, default=0.008,
                   help='Joint stability threshold — p75 normalised std across all '
                        'joint coordinates below this → mannequin (default 0.008). '
                        'At bbox diagonal ~450px this equals ~3.6px of joint movement. '
                        'Lower to catch subtler motion (try 0.005); raise if a truly '
                        'static mannequin keeps getting labelled HUMAN.')
    p.add_argument('--window', type=int,   default=40,
                   help='Rolling window of frames used to measure stability (default 40 '
                        '= ~1.3s at 30fps). Longer window catches slow drifts like '
                        'breathing or slow weight shifts.')
    p.add_argument('--confirm', type=int,  default=30,
                   help='Consecutive frames seen before classifying (default 30 = 1s). '
                        'Higher = less chance of a transient still-human being labelled '
                        'mannequin before they move again.')
    p.add_argument('--kp_conf', type=float, default=0.30,
                   help='Min keypoint confidence to include in measurement (default 0.30). '
                        'Lower than before so more joints are included — more joints '
                        'means any subtle movement anywhere in the body is captured.')
    p.add_argument('--save',   action='store_true',
                   help='Save annotated output to pose_output.mp4')
    return p.parse_args()


def normalised_pose_vector(keypoints_xy: np.ndarray,
                            keypoints_conf: np.ndarray,
                            bbox_xyxy: np.ndarray,
                            kp_conf_min: float):
    """
    Convert raw keypoints into a normalised pose vector that is invariant to
    the person's position and size in the frame.

    Normalisation:
        1. Subtract the bounding box centre — removes translation.
        2. Divide by the bounding box diagonal — removes scale.

    Only keypoints with confidence >= kp_conf_min are included; the rest are
    filled with NaN so they do not contribute to the variance computation.

    Returns a 1D float32 array of length 34 (17 keypoints × 2 coordinates).
    NaN entries are keypoints that were not visible or not confident enough.
    """
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    diag = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + 1e-6   # avoid /0

    vec = np.full(34, np.nan, dtype=np.float32)
    for i, (xy, conf) in enumerate(zip(keypoints_xy, keypoints_conf)):
        if conf >= kp_conf_min:
            vec[i * 2]     = (xy[0] - cx) / diag
            vec[i * 2 + 1] = (xy[1] - cy) / diag
    return vec


def joint_stability(history: deque) -> float:
    """
    Compute how much the normalised pose vector has changed across the
    rolling window.

    Steps:
      1. Stack the history into a (T, 34) matrix.
      2. For each of the 34 coordinates, compute the std across T frames —
         ignoring NaN entries (invisible keypoints).
      3. Return the 75th PERCENTILE of the per-coordinate stds.

    Why p75 instead of mean:
        Using the mean would average out joints that barely moved (e.g. the
        feet of a standing person) against joints that moved a lot (e.g. a
        waving hand). That dilution lets subtle gestures slip through.
        The 75th percentile gives weight to the joints that move the MOST —
        so ANY body part moving more than the threshold is enough to classify
        as human. One waving hand → HUMAN, even if the rest of the body is
        completely still.

    A low value means the skeleton shape has not changed → rigid body →
    mannequin candidate. A high value means joints have been moving relative
    to each other → human.

    Returns 999.0 if fewer than 5 frames are in the window (not enough data).
    """
    if len(history) < 5:
        return 999.0
    mat = np.array(history, dtype=np.float32)   # shape (T, 34)
    # nanstd ignores NaN (occluded keypoints) — requires at least 2 valid samples
    col_stds = np.nanstd(mat, axis=0)
    # Drop coordinates that were never visible (all NaN → nanstd returns NaN)
    valid = col_stds[~np.isnan(col_stds)]
    if valid.size == 0:
        return 999.0
    # 75th percentile: the joints that move the most dominate the score.
    # One waving hand raises the score even if the torso is perfectly still.
    return float(np.percentile(valid, 75))


def draw_skeleton(frame, keypoints_xy, keypoints_conf, kp_conf_min, colour):
    """Draw skeleton lines and joint dots on the frame."""
    for i1, i2 in SKELETON:
        if keypoints_conf[i1] >= kp_conf_min and keypoints_conf[i2] >= kp_conf_min:
            p1 = (int(keypoints_xy[i1][0]), int(keypoints_xy[i1][1]))
            p2 = (int(keypoints_xy[i2][0]), int(keypoints_xy[i2][1]))
            cv2.line(frame, p1, p2, colour, 2)
    for i, (xy, conf) in enumerate(zip(keypoints_xy, keypoints_conf)):
        if conf >= kp_conf_min:
            cv2.circle(frame, (int(xy[0]), int(xy[1])), 4, colour, -1)


def main():
    args = parse_args()

    # YOLOv8-pose model — downloads yolov8n-pose.pt automatically on first run
    # (about 6MB). Switch to yolov8s-pose.pt or yolov8m-pose.pt for better
    # keypoint accuracy on small/distant people, at the cost of speed.
    model = YOLO(args.model)
    print(f'[pose] Model loaded: {args.model}')
    print(f'[pose] Joint stability threshold : {args.thresh}  (lower = stricter)')
    print(f'[pose] Rolling window             : {args.window} frames')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam (device 0)')

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('pose_output.mp4', fourcc, fps_cam, (w, h))

    # Per-track rolling history of normalised pose vectors.
    pose_history: dict[int, deque] = defaultdict(
        lambda: deque(maxlen=args.window)
    )
    # How many consecutive frames each track has been detected.
    consecutive_hits: dict[int, int] = defaultdict(int)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ── YOLO-pose inference with ByteTrack ───────────────────────────────
        # persist=True keeps ByteTrack state between calls so IDs are stable.
        results = model.track(
            frame,
            conf    = args.conf,
            persist = True,
            verbose = False,
        )[0]

        active_tids = set()

        if (results.boxes is not None
                and results.boxes.id is not None
                and results.keypoints is not None):

            for box, kps in zip(results.boxes, results.keypoints):
                tid  = int(box.id)
                conf = float(box.conf)

                # kps.xy  : shape (1, 17, 2) — pixel coordinates
                # kps.conf: shape (1, 17)    — per-keypoint confidence
                xy_all   = kps.xy[0].cpu().numpy()       # (17, 2)
                conf_all = kps.conf[0].cpu().numpy()     # (17,)
                bbox     = box.xyxy[0].cpu().numpy()     # (4,) x1y1x2y2

                # ── Build normalised pose vector ──────────────────────────────
                pose_vec = normalised_pose_vector(
                    xy_all, conf_all, bbox, args.kp_conf
                )
                pose_history[tid].append(pose_vec)

                active_tids.add(tid)
                consecutive_hits[tid] += 1

                # ── Measure joint stability ───────────────────────────────────
                stability = joint_stability(pose_history[tid])
                confirmed = consecutive_hits[tid] >= args.confirm
                is_mannequin = confirmed and stability < args.thresh

                # ── Choose display style ──────────────────────────────────────
                x1, y1, x2, y2 = map(int, bbox)
                if is_mannequin:
                    colour    = (0, 220, 0)       # green
                    thickness = 3
                    label = (f'#{tid} MANNEQUIN  '
                             f'σ={stability:.4f}  {conf:.2f}')
                elif confirmed:
                    colour    = (0, 140, 255)     # orange
                    thickness = 2
                    label = (f'#{tid} HUMAN  '
                             f'σ={stability:.4f}  {conf:.2f}')
                else:
                    colour    = (160, 160, 160)   # grey
                    thickness = 1
                    label = (f'#{tid} {consecutive_hits[tid]}/{args.confirm}  '
                             f'σ={stability:.4f}')

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

                # Label background + text
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.rectangle(frame,
                              (x1, y1 - th - 6), (x1 + tw + 4, y1),
                              colour, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

                # Skeleton
                draw_skeleton(frame, xy_all, conf_all, args.kp_conf, colour)

        # Reset hit counter for tracks ByteTrack dropped this frame.
        for tid in list(consecutive_hits):
            if tid not in active_tids:
                consecutive_hits[tid] = 0

        # ── HUD ──────────────────────────────────────────────────────────────
        n_mann = sum(
            1 for tid in active_tids
            if (consecutive_hits[tid] >= args.confirm
                and joint_stability(pose_history[tid]) < args.thresh)
        )
        hud = (f'DROUGA-pose | active:{len(active_tids)} mann:{n_mann} '
               f'| thresh={args.thresh:.3f}  window={args.window}')
        cv2.putText(frame, hud, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        cv2.putText(frame,
                    'GREEN=mannequin  ORANGE=human  GREY=unconfirmed  '
                    'σ=joint p75-std (lower=more still)   Q=quit',
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (180, 180, 180), 1)

        cv2.imshow('DROUGA pose', frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
