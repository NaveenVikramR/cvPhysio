"""PART C measurement: src/smoothing.py's before/after effect on
shoulderAbduction1.mp4 -- the only clip this hardening task touches
(rotation is explicitly out of scope). Runs AFTER this task's PART A
confidence-gate fix and segmenter fix are in place (both already affect
rep count/shape on their own), so this isolates smoothing's own effect
against that corrected baseline, not the pre-fix behavior. Reports, per
run (no smoothing vs smoothing with config/smoothing.json's params):
  - rep count
  - std of frame-to-frame abduction angle_deg change, pooled over every
    frame RepDetector accepted into a rep buffer across the whole clip
  - number of reps rejected by the segmenter, broken down by reason
    (too_short, too_long, lost_tracking)
"""

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rules as rules_module
from src import smoothing as smoothing_module
from src.pose_estimation import get_pose_landmarks

CLIP = "data/recordings/shoulderAbduction1.mp4"
EXERCISE = "shoulder_abduction"


def load_rep_detector_and_process_frame():
    """exec()-slice app.py's RepDetector class + process_frame() out of
    its source text -- same technique as the other scripts/*.py scripts
    in this project, needed because app.py now parses CLI args and
    launches Streamlit/headless logic at import time."""
    import statistics
    from collections import Counter as _Counter
    from collections import deque

    from src.feature_extraction import check_confidence, compute_elbow_angle, compute_trunk_lean_deg, extract_features, select_side

    source = Path("app.py").read_text()
    start = source.index("CALIBRATION_FRAMES = 15")
    end = source.index("\ndef build_smoother")
    namespace = {
        "statistics": statistics, "Counter": _Counter, "deque": deque,
        "check_confidence": check_confidence, "extract_features": extract_features,
        "select_side": select_side, "compute_elbow_angle": compute_elbow_angle,
        "compute_trunk_lean_deg": compute_trunk_lean_deg,
    }
    exec(source[start:end], namespace)
    return namespace["RepDetector"], namespace["process_frame"]


def run_clip(smoother, fps: float) -> dict:
    RepDetectorClass, process_frame = load_rep_detector_and_process_frame()
    required_roles = rules_module.required_joint_roles()
    cap = cv2.VideoCapture(CLIP)
    detector = RepDetectorClass()
    parallel = {"elbow": [], "trunk": [], "total_frames": 0}

    rep_count = 0
    discard_reasons = Counter()
    all_rep_angles = []  # pooled frame-to-frame deltas, across every rep buffer frame

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        frame_idx += 1
        landmarks = get_pose_landmarks(frame)
        event, payload, _visible, _failed = process_frame(
            detector, parallel, landmarks, EXERCISE, smoother, t, required_roles,
        )

        if event == "rep_complete":
            rep_count += 1
            all_rep_angles.append(payload["angle_deg"])
        elif event == "rep_discarded":
            discard_reasons[payload["reason_code"]] += 1

    cap.release()

    deltas = []
    for angles in all_rep_angles:
        deltas.extend(np.diff(angles).tolist())
    std_delta = float(np.std(deltas)) if deltas else float("nan")

    return {
        "rep_count": rep_count,
        "std_frame_to_frame_deg": std_delta,
        "discard_reasons": dict(discard_reasons),
        "reps_rejected_by_segmenter": sum(discard_reasons.values()),
        "rep_lengths": [len(a) for a in all_rep_angles],
    }


def main() -> None:
    config = smoothing_module.load_config()
    cap = cv2.VideoCapture(CLIP)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    print("Running WITHOUT smoothing...")
    before = run_clip(smoother=None, fps=fps)

    print("Running WITH smoothing (config/smoothing.json's min_cutoff/beta/d_cutoff/max_hold_frames)...")
    smoother = smoothing_module.LandmarkSmoother(
        min_cutoff=config["min_cutoff"], beta=config["beta"],
        d_cutoff=config["d_cutoff"], max_hold_frames=config["max_hold_frames"],
    )
    after = run_clip(smoother=smoother, fps=fps)

    print(f"\n{'metric':40s} {'before':>12s} {'after':>12s}")
    print(f"{'rep count':40s} {before['rep_count']:>12d} {after['rep_count']:>12d}")
    print(f"{'std frame-to-frame angle change (deg)':40s} {before['std_frame_to_frame_deg']:>12.3f} {after['std_frame_to_frame_deg']:>12.3f}")
    print(f"{'reps rejected by segmenter':40s} {before['reps_rejected_by_segmenter']:>12d} {after['reps_rejected_by_segmenter']:>12d}")
    print(f"{'rep lengths (frames)':40s} {str(before['rep_lengths']):>12s} {str(after['rep_lengths']):>12s}")
    print(f"\ndiscard reasons before: {before['discard_reasons']}")
    print(f"discard reasons after:  {after['discard_reasons']}")

    if before["rep_count"] != after["rep_count"]:
        print(f"\nREP COUNT CHANGED: {before['rep_count']} -> {after['rep_count']} with smoothing applied.")
    else:
        print(f"\nRep count unchanged by smoothing ({before['rep_count']} both runs).")


if __name__ == "__main__":
    main()
