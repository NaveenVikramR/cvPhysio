"""Measures elbow-angle noise on a held pose: how far the elbow reading strays
when the arm isn't moving and is kept straight. Run it yourself, standing in
frame at the recording distance, with both arms straight:

    python scripts/measure_elbow_noise.py

Preview -> SPACE -> 5 s countdown -> arms out at shoulder height, held for
--seconds -> 5 s countdown -> arms down, held for --seconds. ESC quits without
saving. Elbow angle uses the same landmark space and formula as the rules scorer.

Prints mean, std, p5, p95 and the fraction of frames below 135 deg per pose,
for each arm and for the arm the scorer would select. Statistics use frames
where that arm's elbow and wrist both pass the visibility gate; the gate pass
rate is reported beside them. Per-frame readings and the summary go to
results/elbow_noise_<tag>.json with --tag, otherwise results/elbow_noise.json
(gitignored). A repeated tag overwrites that tag's file. No video is stored and
nothing uses the network.

This describes one person in one setup. It isn't a threshold derivation.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.record_clip import draw_preview
from src.feature_extraction import LEFT, MIN_VISIBILITY, RIGHT, compute_angle, select_side

RESULTS_PATH = REPO_ROOT / "results" / "elbow_noise.json"
ELBOW_THRESHOLD_DEG = 135.0
COUNTDOWN_S = 5
POSES = (("arm_out", "ARMS OUT at shoulder height, elbows straight"), ("arm_down", "ARMS DOWN, elbows straight"))
ARMS = {"left": LEFT, "right": RIGHT}
KEY_ESC = 27


def frame_reading(landmarks: list) -> dict:
    """One frame's elbow angle and visibilities for both arms, plus the side
    the scorer would select."""
    reading = {"selected": "left" if select_side(landmarks) is LEFT else "right"}
    for arm, joints in ARMS.items():
        shoulder, elbow, wrist = (landmarks[joints[role]] for role in ("shoulder", "elbow", "wrist"))
        reading[arm] = {"elbow_deg": compute_angle(shoulder, elbow, wrist), "elbow_vis": elbow[3], "wrist_vis": wrist[3]}
    return reading


def arm_values(frames: list[dict], arm: str) -> tuple[list[float], int]:
    """(gated, finite elbow angles, frames considered) for "left", "right" or
    "selected"; frames with no pose count as considered but not gated."""
    values, considered = [], 0
    for frame in frames:
        considered += 1
        if frame.get("reading") is None:
            continue
        side = frame["reading"]["selected"] if arm == "selected" else arm
        data = frame["reading"][side]
        if data["elbow_vis"] >= MIN_VISIBILITY and data["wrist_vis"] >= MIN_VISIBILITY and np.isfinite(data["elbow_deg"]):
            values.append(data["elbow_deg"])
    return values, considered


def summarize(values: list[float], considered: int) -> dict:
    summary = {"n_frames": considered, "n_gated": len(values),
               "gate_pass_fraction": len(values) / considered if considered else None}
    if not values:
        return {**summary, "mean": None, "std": None, "p5": None, "p95": None, "frac_below_threshold": None}
    data = np.asarray(values, dtype=float)
    return {
        **summary, "mean": float(data.mean()), "std": float(data.std(ddof=0)),
        "p5": float(np.percentile(data, 5)), "p95": float(np.percentile(data, 95)),
        "frac_below_threshold": float((data < ELBOW_THRESHOLD_DEG).mean()),
    }


def summarize_session(frames: list[dict]) -> dict:
    return {
        pose: {arm: summarize(*arm_values([f for f in frames if f["pose"] == pose], arm))
               for arm in ("left", "right", "selected")}
        for pose, _instruction in POSES
    }


def format_summary(summary: dict) -> list[str]:
    lines = [f"{'pose':9s} {'arm':9s} {'gated/frames':>13s} {'mean':>7s} {'std':>6s} {'p5':>7s} {'p95':>7s} {'<135':>7s}"]
    for pose, arms in summary.items():
        for arm, s in arms.items():
            counts = f"{s['n_gated']}/{s['n_frames']}"
            if s["mean"] is None:
                lines.append(f"{pose:9s} {arm:9s} {counts:>13s}   no gated frames")
            else:
                lines.append(f"{pose:9s} {arm:9s} {counts:>13s} {s['mean']:7.1f} {s['std']:6.1f} {s['p5']:7.1f} "
                             f"{s['p95']:7.1f} {s['frac_below_threshold']:7.1%}")
    return lines


def git_provenance() -> tuple[str, bool]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        return commit, bool(status.strip())
    except Exception:
        return "unknown", True


def results_path(tag: str | None) -> Path:
    return RESULTS_PATH.with_name(f"elbow_noise_{tag}.json") if tag else RESULTS_PATH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure elbow-angle noise on held straight-arm poses.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=10.0, help="how long to hold EACH pose")
    parser.add_argument("--tag", default=None, help="save to results/elbow_noise_<tag>.json (letters, digits, _ and - only)")
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("--tag may only contain letters, digits, _ and -")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from src import pose_estimation  # loads the pose model at import time

    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    capture = pose_estimation.open_webcam(args.camera)
    frames: list[dict] = []
    resolution = None

    def step(status: str, colour: tuple) -> tuple[int, object]:
        nonlocal resolution
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Camera stopped delivering frames")
        resolution = [frame.shape[1], frame.shape[0]]
        normalized, world = pose_estimation.get_pose_landmarks_pair(frame)
        cv2.imshow("measure_elbow_noise", draw_preview(frame, normalized, status, colour))
        key = cv2.waitKey(1) & 0xFF
        return key, world if landmark_space == "world" else normalized

    try:
        while True:
            key, _landmarks = step("Frame head to hips, both arms visible. SPACE = start, ESC = quit", (255, 255, 255))
            if key == KEY_ESC:
                print("Quit. Nothing saved.")
                return 1
            if key == ord(" "):
                break
        pose_times = {}
        for pose, instruction in POSES:
            deadline = time.monotonic() + COUNTDOWN_S
            while (remaining := deadline - time.monotonic()) > 0:
                key, _landmarks = step(f"{instruction} in {int(remaining) + 1}...", (0, 255, 255))
                if key == KEY_ESC:
                    print("Quit. Nothing saved.")
                    return 1
            start = time.monotonic()
            n_before = len(frames)
            while (elapsed := time.monotonic() - start) < args.seconds:
                key, landmarks = step(f"HOLD: {instruction} {args.seconds - elapsed:4.1f}s", (0, 0, 255))
                if key == KEY_ESC:
                    print("Quit. Nothing saved.")
                    return 1
                frames.append({"pose": pose, "t": elapsed, "reading": frame_reading(landmarks) if landmarks else None})
            pose_times[pose] = (len(frames) - n_before) / (time.monotonic() - start)
    finally:
        capture.release()
        cv2.destroyAllWindows()

    summary = summarize_session(frames)
    commit, dirty = git_provenance()
    output_path = results_path(args.tag)
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps({
        "commit": commit, "dirty": dirty, "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tag": args.tag,
        "camera_index": args.camera, "resolution": resolution, "landmark_space": landmark_space,
        "pose_model": pose_estimation.configured_pose_model(), "seconds_per_pose": args.seconds,
        "measured_fps_per_pose": pose_times, "elbow_threshold_deg": ELBOW_THRESHOLD_DEG,
        "summary": summary, "frames": frames,
    }, indent=1) + "\n")
    print("\n".join(format_summary(summary)))
    print(f"\nMeasured fps per pose: { {k: round(v, 1) for k, v in pose_times.items()} }")
    print(f"Saved {output_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
