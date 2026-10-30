"""Recording QC for one own-clip, per Amendment 2 of docs/preregistered_own_clip_eval.md.
Read-only. Uses the pose model and rep detector only: it never scores a rule, and
prints nothing about peak angle, elbow, trunk or tempo.

    python scripts/qc_clip.py data/recordings/abd_reference_correct_01.mp4

Three checks, each PASS or FAIL:
  1. arms_down: every measured shoulder angle in the first and last 1.5 s is below
     45 deg, and each window has at least one measured frame.
  2. rep_count: detected reps equal the planned count (from data/own_manifest.csv,
     or --expected-reps).
  3. tracking: no lost_tracking discard, and no run of frames failing the visibility
     gate longer than the detector's own tolerance.

Exit code 0 if all pass, 1 if any fail, 2 if the clip can't be checked.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.feature_extraction import extract_features
from src.pipeline import CONFIDENCE_TOLERANCE_FRAMES

MANIFEST_PATH = Path("data/own_manifest.csv")
EXERCISE = "shoulder_abduction"
REST_WINDOW_S = 1.5
ARMS_DOWN_BELOW_DEG = 45.0


def collect_detector_trace(video_path: Path) -> dict:
    """Runs the clip through the same pose call, gate and Session the evaluator
    uses. {"frames": [{"t", "angle", "visible"}], "rep_count", "lost_tracking", "duration_s"}."""
    from scripts.evaluate_own_clips import resolve_clip_fps
    from src import pose_estimation
    from src import rules as rules_module
    from src.pipeline import Session, build_smoother

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps, _source = resolve_clip_fps(video_path, cap)
    required_roles = rules_module.required_joint_roles()  # the gate's joints; no rule is evaluated
    session = Session(EXERCISE)
    session.start()
    smoother = build_smoother()
    frames, rep_count, lost_tracking, index = [], 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = index / fps
        index += 1
        landmarks = pose_estimation.get_pose_landmarks(frame)
        event, payload, visible, _failed = session.process_frame(landmarks, smoother, t, required_roles)
        features = extract_features(landmarks, EXERCISE) if visible else None
        frames.append({"t": t, "angle": features["angle_deg"] if features else None, "visible": visible})
        rep_count += event == "rep_complete"
        lost_tracking += event == "rep_discarded" and payload["reason_code"] == "lost_tracking"
    cap.release()
    return {"frames": frames, "rep_count": rep_count, "lost_tracking": lost_tracking, "duration_s": index / fps}


def check_arms_down(frames: list[dict], duration_s: float) -> dict:
    windows = {
        "first": [f["angle"] for f in frames if f["t"] < REST_WINDOW_S and f["angle"] is not None],
        "last": [f["angle"] for f in frames if f["t"] >= duration_s - REST_WINDOW_S and f["angle"] is not None],
    }
    details = []
    passed = True
    for name, angles in windows.items():
        if not angles:
            passed = False
            details.append(f"{name} {REST_WINDOW_S:g} s: no measured frames")
        else:
            passed = passed and max(angles) < ARMS_DOWN_BELOW_DEG
            details.append(f"{name} {REST_WINDOW_S:g} s: highest {max(angles):.1f} deg")
    return {"passed": passed, "detail": "; ".join(details)}


def check_rep_count(detected: int, planned: int) -> dict:
    return {"passed": detected == planned, "detail": f"detected {detected}, planned {planned}"}


def longest_gated_run(frames: list[dict]) -> int:
    longest = current = 0
    for frame in frames:
        current = 0 if frame["visible"] else current + 1
        longest = max(longest, current)
    return longest


def check_tracking(frames: list[dict], lost_tracking: int) -> dict:
    run = longest_gated_run(frames)
    return {
        "passed": lost_tracking == 0 and run <= CONFIDENCE_TOLERANCE_FRAMES,
        "detail": f"lost_tracking discards {lost_tracking}; longest gated-out run {run} frames "
                  f"(limit {CONFIDENCE_TOLERANCE_FRAMES})",
    }


def run_checks(trace: dict, planned_reps: int) -> dict:
    return {
        "arms_down": check_arms_down(trace["frames"], trace["duration_s"]),
        "rep_count": check_rep_count(trace["rep_count"], planned_reps),
        "tracking": check_tracking(trace["frames"], trace["lost_tracking"]),
    }


def planned_reps_from_manifest(video_path: Path, manifest_path: Path) -> int | None:
    if not manifest_path.exists():
        return None
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if Path(row["clip"]).name == video_path.name:
                return int(row["expected_reps"])
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector-only recording QC for one clip (read-only).")
    parser.add_argument("clip", type=Path)
    parser.add_argument("--expected-reps", type=int, default=None, help="default: the clip's manifest row")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.clip.exists():
        print(f"No clip at {args.clip}.")
        return 2
    planned = args.expected_reps if args.expected_reps is not None else planned_reps_from_manifest(args.clip, args.manifest)
    if planned is None:
        print(f"No planned rep count for {args.clip.name}: it isn't in {args.manifest}. Pass --expected-reps.")
        return 2
    trace = collect_detector_trace(args.clip)
    if not trace["frames"]:
        print(f"{args.clip} has no readable frames.")
        return 2

    checks = run_checks(trace, planned)
    print(f"QC {args.clip.name} ({trace['duration_s']:.1f} s)")
    for name, check in checks.items():
        print(f"  {'PASS' if check['passed'] else 'FAIL'}  {name}: {check['detail']}")
    all_passed = all(check["passed"] for check in checks.values())
    print(f"Overall: {'PASS, keep this take' if all_passed else 'FAIL, re-record this clip'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
