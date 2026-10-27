"""PART C tuning script: grid search over src/smoothing.py's min_cutoff
and beta, evaluated on every calibration-set clip in data/clips_manifest.
csv. A cell (min_cutoff, beta) passes only if ALL of these hold on EVERY
calibration clip, against that clip's own no-smoothing baseline:
  - rep count unchanged
  - peak-angle shift <= 2 deg
  - rep-end lag <= 3 frames
  - frame-to-frame angle std reduced by >= 30%

If fewer than 3 calibration clips exist, this prints "insufficient
clips to tune" and does NOT run the grid -- picking smoothing params
from 1-2 clips risks overfitting noise in those specific clips, and the
task explicitly calls for leaving smoothing off by default in that case
(already true: config/smoothing.json's "enabled" is false). This script
never writes config/smoothing.json itself even when a cell does pass --
it only reports which cells pass, so a human decides whether to apply one.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rules as rules_module
from src import smoothing as smoothing_module
from src.pipeline import RepDetector, process_frame
from src.pose_estimation import get_pose_landmarks
from scripts.evaluate_rules import read_manifest

MANIFEST_PATH = Path("data/clips_manifest.csv")
MIN_CUTOFF_GRID = [0.5, 1.0, 2.0]
BETA_GRID = [0.0, 0.1, 0.3, 0.7]
MIN_CALIBRATION_CLIPS = 3

PASS_REP_COUNT_UNCHANGED = True
PASS_MAX_PEAK_SHIFT_DEG = 2.0
PASS_MAX_REP_END_LAG_FRAMES = 3
PASS_MIN_STD_REDUCTION_FRAC = 0.30


def run_clip(clip_path: str, exercise: str, smoother, required_roles, RepDetectorClass, process_frame) -> dict:
    """Returns per-rep angle_deg sequences and end-frame indices, for one
    clip, with or without smoothing."""
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = RepDetectorClass()
    parallel = {"elbow": [], "trunk": [], "total_frames": 0}

    reps = []  # list of {"angle_deg": [...], "end_frame": int}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        landmarks = get_pose_landmarks(frame)
        event, payload, _visible, _failed = process_frame(
            detector, parallel, landmarks, exercise, smoother, t, required_roles,
        )
        if event == "rep_complete":
            reps.append({"angle_deg": payload["angle_deg"], "end_frame": frame_idx})
        frame_idx += 1
    cap.release()
    return {"reps": reps, "fps": fps}


def evaluate_cell(clip_path: str, exercise: str, min_cutoff: float, beta: float,
                   baseline: dict, required_roles, RepDetectorClass, process_frame) -> dict:
    """Compares a (min_cutoff, beta) smoothed run against this clip's
    no-smoothing baseline. Returns the 4 pass criteria plus the
    underlying measured numbers."""
    smoother = smoothing_module.LandmarkSmoother(
        min_cutoff=min_cutoff, beta=beta, d_cutoff=1.0, max_hold_frames=5,
    )
    smoothed = run_clip(clip_path, exercise, smoother, required_roles, RepDetectorClass, process_frame)

    rep_count_before, rep_count_after = len(baseline["reps"]), len(smoothed["reps"])
    rep_count_ok = rep_count_before == rep_count_after

    peak_shift_ok, rep_end_lag_ok, std_reduction_ok = True, True, True
    peak_shift, rep_end_lag, std_reduction = None, None, None

    if rep_count_ok and rep_count_before > 0:
        peak_shifts, lags, reductions = [], [], []
        for rep_b, rep_a in zip(baseline["reps"], smoothed["reps"]):
            peak_shifts.append(abs(max(rep_b["angle_deg"]) - max(rep_a["angle_deg"])))
            lags.append(abs(rep_b["end_frame"] - rep_a["end_frame"]))
            std_b = np.std(np.diff(rep_b["angle_deg"])) if len(rep_b["angle_deg"]) > 1 else 0.0
            std_a = np.std(np.diff(rep_a["angle_deg"])) if len(rep_a["angle_deg"]) > 1 else 0.0
            reductions.append((std_b - std_a) / std_b if std_b > 0 else 0.0)

        peak_shift = max(peak_shifts)
        rep_end_lag = max(lags)
        std_reduction = min(reductions)

        peak_shift_ok = peak_shift <= PASS_MAX_PEAK_SHIFT_DEG
        rep_end_lag_ok = rep_end_lag <= PASS_MAX_REP_END_LAG_FRAMES
        std_reduction_ok = std_reduction >= PASS_MIN_STD_REDUCTION_FRAC
    elif rep_count_before == 0:
        # nothing to compare against on this clip -- can't claim any criterion holds
        peak_shift_ok = rep_end_lag_ok = std_reduction_ok = False

    passed = rep_count_ok and peak_shift_ok and rep_end_lag_ok and std_reduction_ok
    return {
        "passed": passed, "rep_count_ok": rep_count_ok,
        "peak_shift_ok": peak_shift_ok, "peak_shift": peak_shift,
        "rep_end_lag_ok": rep_end_lag_ok, "rep_end_lag": rep_end_lag,
        "std_reduction_ok": std_reduction_ok, "std_reduction": std_reduction,
    }


def main() -> None:
    rows = read_manifest(MANIFEST_PATH)
    calibration_rows = [r for r in rows if r["set"] == "calibration" and Path(r["clip"]).exists()]

    print(f"Calibration-set clips found on disk: {len(calibration_rows)} "
          f"(need >= {MIN_CALIBRATION_CLIPS} to tune safely)")
    for row in calibration_rows:
        print(f"  - {row['clip']} ({row['exercise']})")

    if len(calibration_rows) < MIN_CALIBRATION_CLIPS:
        print("\ninsufficient clips to tune")
        print("Leaving smoothing off by default (config/smoothing.json's \"enabled\" stays false). "
              "Add at least 3 calibration-set clips to data/clips_manifest.csv and re-run this script.")
        return

    RepDetectorClass = RepDetector
    required_roles = rules_module.required_joint_roles()

    print("\nMeasuring no-smoothing baselines...")
    baselines = {}
    for row in calibration_rows:
        baselines[row["clip"]] = run_clip(row["clip"], row["exercise"], None, required_roles, RepDetectorClass, process_frame)
        print(f"  {row['clip']}: {len(baselines[row['clip']]['reps'])} rep(s) at baseline")

    print(f"\nGrid: min_cutoff in {MIN_CUTOFF_GRID}, beta in {BETA_GRID}  "
          f"(criteria: rep count unchanged, peak shift <= {PASS_MAX_PEAK_SHIFT_DEG} deg, "
          f"rep-end lag <= {PASS_MAX_REP_END_LAG_FRAMES} frames, std reduction >= {PASS_MIN_STD_REDUCTION_FRAC:.0%})")

    header = "min_cutoff\\beta  " + "  ".join(f"{b:>6.2f}" for b in BETA_GRID)
    print(header)

    passing_cells = []
    for min_cutoff in MIN_CUTOFF_GRID:
        row_cells = []
        for beta in BETA_GRID:
            cell_results = [
                evaluate_cell(row["clip"], row["exercise"], min_cutoff, beta, baselines[row["clip"]],
                               required_roles, RepDetectorClass, process_frame)
                for row in calibration_rows
            ]
            cell_passed = all(r["passed"] for r in cell_results)
            row_cells.append("PASS" if cell_passed else "fail")
            if cell_passed:
                passing_cells.append((min_cutoff, beta))
        print(f"{min_cutoff:>14.2f}  " + "  ".join(f"{c:>6s}" for c in row_cells))

    if passing_cells:
        print(f"\nCells passing all criteria on all {len(calibration_rows)} calibration clips: {passing_cells}")
        print("Not auto-applying -- a human should review and update config/smoothing.json.")
    else:
        print("\nNo (min_cutoff, beta) cell passed all 4 criteria on all calibration clips. "
              "Leaving smoothing off (config/smoothing.json's \"enabled\" stays false).")


if __name__ == "__main__":
    main()
