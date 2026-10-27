"""PART A of the rules-hardening task: analysis only, no code changes.
Investigates whether src/rules.py's elbow_straight rule firing on
shoulderAbduction1.mp4 (measured elbow angle 106.4 deg, below the
provisional 150 deg threshold -- see config/exercises/shoulder_abduction.
json) reflects a real elbow bend or landmark/Z noise, and whether 150 deg
is even consistent with how training-correct reps look under the SAME
compute_elbow_angle() function.

Runs on RAW (unsmoothed) landmarks deliberately -- this is the "before"
picture that motivates PART B's smoothing work, not a post-smoothing
check.

Outputs (scratch/, gitignored):
  - elbow_angle_vs_time.png: elbow_angle_deg across the one detected rep.
  - elbow_min_frame.png: the frame at the rep's minimum elbow angle (its
    worst bend), with the selected side's shoulder/elbow/wrist drawn.
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import _fk_frame_to_landmarks, discover_rep_files, forward_kinematics_frame, select_side_for_rep, sibling_path
import pandas as pd
from src.feature_extraction import check_confidence, compute_elbow_angle, extract_features, select_side
from src.pipeline import RepDetector
from src.pose_estimation import get_pose_landmarks

CLIP = "data/recordings/shoulderAbduction1.mp4"
EXERCISE = "shoulder_abduction"
SCRATCH_DIR = Path("scratch")


def extract_live_rep():
    """Runs the one real clip through RepDetector (raw, unsmoothed
    landmarks) and returns the single detected rep's angle_deg,
    elbow_angle_deg, visibility (per required landmark), and the raw
    frame + landmarks at each contributing frame, so the exact frame at
    the rep's elbow-angle minimum can be recovered afterward."""
    RepDetectorClass = RepDetector
    cap = cv2.VideoCapture(CLIP)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = RepDetectorClass()

    angle_buf, elbow_buf, vis_buf, frame_buf, landmarks_buf = [], [], [], [], []
    reps = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        landmarks = get_pose_landmarks(frame)
        visible, failed, features, elbow_val = False, [], None, None
        side_vis = None
        if landmarks is not None:
            side = select_side(landmarks)
            visible, failed = check_confidence(landmarks, side, EXERCISE)
            features = extract_features(landmarks, EXERCISE) if visible else None
            elbow_val = compute_elbow_angle(landmarks)
            side_vis = {
                "shoulder": landmarks[side["shoulder"]][3],
                "elbow": landmarks[side["elbow"]][3],
                "wrist": landmarks[side["wrist"]][3],
            }

        angle_deg = features["angle_deg"] if features else None
        side_label = features["side"] if features else None

        prev_state = detector.state
        event, payload = detector.update(visible, angle_deg, side_label, failed)
        new_state = detector.state

        contributed = visible and (prev_state == "MOVING" or (prev_state == "RESTING" and new_state == "MOVING"))
        if contributed:
            angle_buf.append(angle_deg)
            elbow_buf.append(elbow_val)
            vis_buf.append(side_vis)
            frame_buf.append(frame.copy())
            landmarks_buf.append(landmarks)

        if event == "rep_complete":
            reps.append({
                "angle_deg": list(angle_buf), "elbow_angle_deg": list(elbow_buf),
                "visibility": list(vis_buf), "frames": list(frame_buf), "landmarks": list(landmarks_buf),
            })

        if prev_state == "MOVING" and new_state == "RESTING":
            angle_buf, elbow_buf, vis_buf, frame_buf, landmarks_buf = [], [], [], [], []

    cap.release()
    return reps, fps


def training_correct_elbow_stats():
    """Per-rep min elbow angle (its worst-bend moment) across every
    training-correct m07 rep, via FK positions + the SAME
    compute_elbow_angle() the live app uses -- never a second
    implementation."""
    per_rep_mins = []
    for parsed in discover_rep_files("data/ui_prmd", 7, modality="kinect", source="positions"):
        if not parsed.is_correct:
            continue
        pos_df = pd.read_csv(parsed.path, sep=r"\s+|,", engine="python", header=None)
        ang_path = sibling_path(parsed.path, "positions")
        ang_df = pd.read_csv(ang_path, sep=r"\s+|,", engine="python", header=None)
        ang_df_named = ang_df.copy()
        ang_df_named.columns = [f"col_{i}" for i in range(ang_df_named.shape[1])]
        side = select_side_for_rep(ang_df_named, EXERCISE, "kinect")

        elbow_vals = []
        for pos_row, ang_row in zip(pos_df.to_numpy(), ang_df.to_numpy()):
            fk_pos = forward_kinematics_frame(pos_row, ang_row)
            landmarks = _fk_frame_to_landmarks(fk_pos, side)
            elbow_vals.append(compute_elbow_angle(landmarks))
        per_rep_mins.append(min(elbow_vals))
    return per_rep_mins


def draw_landmarks_frame(frame, landmarks, side: dict) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    pts = {}
    for role, idx in side.items():
        x, y = landmarks[idx][0] * w, landmarks[idx][1] * h
        pts[role] = (int(x), int(y))
        cv2.circle(annotated, pts[role], 6, (0, 255, 255), -1)
    cv2.line(annotated, pts["shoulder"], pts["elbow"], (0, 255, 0), 2)
    cv2.line(annotated, pts["elbow"], pts["wrist"], (0, 255, 0), 2)
    return annotated


def main() -> None:
    SCRATCH_DIR.mkdir(exist_ok=True)

    print("[1/3] Extracting the live rep from shoulderAbduction1.mp4 (raw, unsmoothed landmarks)...")
    reps, fps = extract_live_rep()
    if not reps:
        print("No rep detected -- cannot proceed with PART A.")
        return
    if len(reps) > 1:
        print(f"NOTE: {len(reps)} reps detected (expected 1 from prior runs); using rep 1 only.")
    rep = reps[0]
    elbow = rep["elbow_angle_deg"]
    angle = rep["angle_deg"]
    n = len(elbow)
    t = np.arange(n) / fps
    min_idx = int(np.argmin(elbow))
    print(f"    rep: n={n} frames, elbow min={elbow[min_idx]:.1f} deg at frame {min_idx} (t={t[min_idx]:.2f}s), "
          f"abduction angle at that frame={angle[min_idx]:.1f} deg")

    # local window around the minimum, to see if it's a single-frame spike
    # or a sustained dip (a real bend should look like the latter).
    lo, hi = max(0, min_idx - 5), min(n, min_idx + 6)
    print(f"    elbow angle, frames {lo}-{hi - 1} (around the minimum): "
          f"{[round(v, 1) for v in elbow[lo:hi]]}")
    vis_at_min = rep["visibility"][min_idx]
    print(f"    landmark visibility at that frame: shoulder={vis_at_min['shoulder']:.2f}  "
          f"elbow={vis_at_min['elbow']:.2f}  wrist={vis_at_min['wrist']:.2f}")

    frame_to_frame_std = float(np.std(np.diff(elbow)))
    print(f"    frame-to-frame elbow-angle change, std over the whole rep: {frame_to_frame_std:.2f} deg")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, elbow, marker=".", label="elbow_angle_deg")
    ax.axhline(150, color="red", linestyle="--", label="min_elbow_straight_deg (150, provisional)")
    ax.axvline(t[min_idx], color="gray", linestyle=":", label=f"minimum (t={t[min_idx]:.2f}s)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("elbow angle (deg)")
    ax.set_title("shoulderAbduction1.mp4 -- elbow angle vs time (raw, unsmoothed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SCRATCH_DIR / "elbow_angle_vs_time.png", dpi=150)
    print(f"    saved {SCRATCH_DIR.resolve() / 'elbow_angle_vs_time.png'}")

    peak_frame = rep["frames"][min_idx]
    peak_landmarks = rep["landmarks"][min_idx]
    side = select_side(peak_landmarks)
    annotated = draw_landmarks_frame(peak_frame, peak_landmarks, side)
    cv2.putText(annotated, f"elbow={elbow[min_idx]:.1f} deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(SCRATCH_DIR / "elbow_min_frame.png"), annotated)
    print(f"    saved {SCRATCH_DIR.resolve() / 'elbow_min_frame.png'}")

    print("\n[2/3] Training-correct m07 per-rep minimum elbow angle (FK positions, same compute_elbow_angle)...")
    mins = training_correct_elbow_stats()
    arr = np.asarray(mins)
    print(f"    n={len(arr)} reps")
    print(f"    min={arr.min():.1f}  p5={np.percentile(arr, 5):.1f}  median={np.median(arr):.1f}  "
          f"p95={np.percentile(arr, 95):.1f}  max={arr.max():.1f}")

    print("\n[3/3] Is the 150 deg threshold consistent with training-correct reps?")
    below_150 = int((arr < 150).sum())
    print(f"    training-correct reps with per-rep-min elbow angle < 150 deg: {below_150}/{len(arr)}")
    if below_150 == 0:
        print("    -> 150 deg sits ABOVE every training-correct rep's own elbow bend; consistent with the rule as a loose ceiling.")
    else:
        print("    -> 150 deg is BELOW some training-correct reps' own elbow bend -- the threshold would flag reps UI-PRMD itself labeled correct.")

    print("\nAssessment (evidence above, not a code change):")
    if frame_to_frame_std > 15 and 0 < min_idx < n - 1:
        spike = abs(elbow[min_idx] - elbow[max(0, min_idx - 1)]) > 20 or abs(elbow[min_idx] - elbow[min(n - 1, min_idx + 1)]) > 20
        print(f"    frame-to-frame std is high ({frame_to_frame_std:.1f} deg); "
              f"{'the minimum looks like an isolated single-frame spike' if spike else 'but the dip around the minimum is NOT an isolated spike (neighbors are close)'}.")
    else:
        print(f"    frame-to-frame std is modest ({frame_to_frame_std:.1f} deg); the dip is not obviously single-frame noise.")


if __name__ == "__main__":
    main()
