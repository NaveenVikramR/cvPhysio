"""PART C measurement-only experiment: does a cheap, Z-free proxy for
shoulder rotation separate training-correct, training-incorrect, and live
reps any better than compute_shoulder_rotation_angle() (which
scripts/diagnose_misclassification.py already showed is dominated by
MediaPipe's unreliable monocular Z)? This script only measures and
reports -- it does NOT touch src/rules.py, src/feature_extraction.py, or
app.py, and nothing it computes is wired into the live app.

rotation_proxy = (wrist_x - elbow_x) / forearm_length

- wrist_x, elbow_x: in-plane X only (screen-horizontal), never Z.
- forearm_length: the (X, Y) 2D distance between elbow and wrist --
  deliberately NOT the 3D distance. diagnose_misclassification.py already
  measured compute_shoulder_rotation_angle() swinging from 180deg to
  12.9deg on the same real frame as Z is scaled 0x->5x -- i.e. Z on this
  live pipeline cannot be trusted. Folding Z into this proxy's
  denominator would reintroduce exactly the instability this proxy is
  trying to avoid, so forearm_length is computed from X/Y only, on both
  the FK (training) and live sides, so the two stay comparable.

Reuses the project's established techniques: FK joint positions via
src/dataset_loader.py (same functions scripts/train_classifier.py's
dataset loading goes through), and RepDetector-based rep segmentation via
src.pipeline.RepDetector (the shared module app.py and every diagnostic
script import rep-detection from).
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import (
    SIDE_ROLE_JOINTS,
    discover_rep_files,
    forward_kinematics_frame,
    select_side_for_rep,
    sibling_path,
)
from src.feature_extraction import LEFT, check_confidence, select_side
from src.pipeline import RepDetector
from src.pose_estimation import get_pose_landmarks

RAW_DIR = "data/ui_prmd"
MOVEMENT = 9
LIVE_CLIPS = [
    "data/recordings/ShoulderInternalExternal1.mp4",
    "data/recordings/ShoulderInteralExternal2.mp4",
    "data/recordings/physitrack_standing_rotation.mp4",
]
SCRATCH_DIR = Path("scratch")


def proxy_from_points(elbow_xy: np.ndarray, wrist_xy: np.ndarray) -> float:
    forearm_length = float(np.linalg.norm(wrist_xy[:2] - elbow_xy[:2]))  # X/Y only -- see module docstring
    if forearm_length == 0:
        return float("nan")
    return float((wrist_xy[0] - elbow_xy[0]) / forearm_length)


def training_rep_proxy_range(parsed) -> float:
    pos_df = pd.read_csv(parsed.path, sep=r"\s+|,", engine="python", header=None)
    ang_path = sibling_path(parsed.path, "positions")
    ang_df = pd.read_csv(ang_path, sep=r"\s+|,", engine="python", header=None)

    ang_df_named = ang_df.copy()
    ang_df_named.columns = [f"col_{i}" for i in range(ang_df_named.shape[1])]
    side = select_side_for_rep(ang_df_named, "shoulder_rotation", "kinect")
    roles = SIDE_ROLE_JOINTS[side]

    values = []
    for pos_row, ang_row in zip(pos_df.to_numpy(), ang_df.to_numpy()):
        fk_pos = forward_kinematics_frame(pos_row, ang_row)
        elbow = np.asarray(fk_pos[roles["elbow"]], dtype=float)
        wrist = np.asarray(fk_pos[roles["wrist"]], dtype=float)
        values.append(proxy_from_points(elbow, wrist))
    values = np.asarray(values)
    return float(np.nanmax(values) - np.nanmin(values))


def training_group_ranges(is_correct: bool) -> list[float]:
    ranges = []
    for parsed in discover_rep_files(RAW_DIR, MOVEMENT, modality="kinect", source="positions"):
        if parsed.is_correct != is_correct:
            continue
        ranges.append(training_rep_proxy_range(parsed))
    return ranges


def live_clip_rep_ranges(video_path: str, RepDetectorClass) -> list[float]:
    cap = cv2.VideoCapture(video_path)
    detector = RepDetectorClass()
    proxy_buf = []
    rep_ranges = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        landmarks = get_pose_landmarks(frame)
        visible, failed, angle_deg, side_label, proxy_val = False, [], None, None, None
        if landmarks is not None:
            side_dict = select_side(landmarks)
            visible, failed = check_confidence(landmarks, side_dict, "shoulder_rotation")
            elbow = np.asarray(landmarks[side_dict["elbow"]][:3], dtype=float)
            wrist = np.asarray(landmarks[side_dict["wrist"]][:3], dtype=float)
            proxy_val = proxy_from_points(elbow, wrist)
            if visible:
                from src.feature_extraction import compute_shoulder_rotation_angle
                angle_deg = compute_shoulder_rotation_angle(landmarks)
                side_label = "LEFT" if side_dict is LEFT else "RIGHT"

        prev_state = detector.state
        event, _payload = detector.update(visible, angle_deg, side_label, failed)
        new_state = detector.state

        contributed = visible and (prev_state == "MOVING" or (prev_state == "RESTING" and new_state == "MOVING"))
        if contributed:
            proxy_buf.append(proxy_val)

        if event == "rep_complete":
            arr = np.asarray(proxy_buf)
            rep_ranges.append(float(np.nanmax(arr) - np.nanmin(arr)))

        if prev_state == "MOVING" and new_state == "RESTING":
            proxy_buf = []

    cap.release()
    return rep_ranges


def percentile_row(values: list[float]) -> dict:
    arr = np.asarray(values)
    return {
        "n": len(arr),
        "min": float(np.min(arr)),
        "p5": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def print_row(label: str, row: dict) -> None:
    print(f"  {label:28s} n={row['n']:3d}  min={row['min']:7.3f}  p5={row['p5']:7.3f}  "
          f"median={row['median']:7.3f}  p95={row['p95']:7.3f}  max={row['max']:7.3f}")


def main() -> None:
    print("=" * 78)
    print("PART C -- rotation_proxy = (wrist_x - elbow_x) / forearm_length(X,Y 2D)")
    print("Measurement only. Not wired into src/rules.py or app.py.")
    print("=" * 78)

    print("\n[1/3] Training-correct m09 reps (FK positions)...")
    correct_ranges = training_group_ranges(is_correct=True)
    print("[2/3] Training-incorrect m09 reps (FK positions)...")
    incorrect_ranges = training_group_ranges(is_correct=False)

    print("[3/3] Live rotation clips (RepDetector-segmented)...")
    RepDetectorClass = RepDetector
    live_ranges_by_clip = {}
    for clip in LIVE_CLIPS:
        ranges = live_clip_rep_ranges(clip, RepDetectorClass)
        live_ranges_by_clip[clip] = ranges
        print(f"    {clip}: {len(ranges)} rep(s) -> ranges={[round(r, 3) for r in ranges]}")
    live_ranges_all = [r for ranges in live_ranges_by_clip.values() for r in ranges]

    print("\nPer-rep range of the proxy, by group:")
    print_row("training-correct (m09)", percentile_row(correct_ranges))
    print_row("training-incorrect (m09)", percentile_row(incorrect_ranges))
    if live_ranges_all:
        print_row("live rotation clips (all)", percentile_row(live_ranges_all))
    else:
        print("  live rotation clips (all)   -- NO reps detected, cannot compute percentiles")

    if live_ranges_all:
        lo, hi = min(correct_ranges), max(correct_ranges)
        inside = [r for r in live_ranges_all if lo <= r <= hi]
        print(f"\nLive reps whose proxy range falls inside the training-correct range [{lo:.3f}, {hi:.3f}]: "
              f"{len(inside)}/{len(live_ranges_all)}")
        go_no_go = "GO" if len(inside) == len(live_ranges_all) else (
            "PARTIAL" if inside else "NO-GO")
        print(f"\nVERDICT: {go_no_go} -- "
              + ("all live reps fall inside the training-correct proxy range." if go_no_go == "GO"
                 else "some but not all live reps fall inside the training-correct proxy range." if go_no_go == "PARTIAL"
                 else "no live reps fall inside the training-correct proxy range; "
                      "this proxy does not obviously fix the mismatch either."))
    else:
        print("\nVERDICT: NO-GO -- no live reps were detected by RepDetector in any rotation clip, "
              "so this proxy cannot be evaluated against live data at all.")

    SCRATCH_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        0,
        max([*correct_ranges, *incorrect_ranges, *(live_ranges_all or [0])], default=1.0),
        30,
    )
    ax.hist(correct_ranges, bins=bins, alpha=0.5, label=f"training-correct (n={len(correct_ranges)})", color="tab:green")
    ax.hist(incorrect_ranges, bins=bins, alpha=0.5, label=f"training-incorrect (n={len(incorrect_ranges)})", color="tab:red")
    if live_ranges_all:
        ax.hist(live_ranges_all, bins=bins, alpha=0.5, label=f"live clips (n={len(live_ranges_all)})", color="tab:blue")
    ax.set_xlabel("per-rep range of rotation_proxy (dimensionless)")
    ax.set_ylabel("rep count")
    ax.set_title("rotation_proxy per-rep range by group")
    ax.legend()
    fig.tight_layout()
    out_path = SCRATCH_DIR / "rotation_proxy_ranges.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved histogram: {out_path.resolve()}")


if __name__ == "__main__":
    main()
