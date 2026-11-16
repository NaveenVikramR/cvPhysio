"""Phase 1 rotation diagnosis -- measurement only. Runs the real live
detection path (src.pipeline.Session, rules confidence roles, world
landmarks from config/pose.json) over the demo shoulder-rotation clips and
reports, per clip, evidence for three questions:

  Q1 depth: with the arm at the side, rotation about the upper arm is
     mostly depth motion for a front camera, so the 3D angle necessarily
     relies on MediaPipe's monocular z (an image-plane version of the same
     angle is degenerate: always 0 or 180). Compares the 3D angle with a
     depth-free image-space proxy (image_proxy) over gated frames, and
     reports how much of the forearm's length lies along z at rest.
  Q2 segmentation: does RepDetector split one out-and-back rotation into
     several segments? Records, for every completed or discarded segment,
     its direction relative to the baseline (above = outward of rest,
     below = inward), its range and its frame count.
  Q3 camera angle: is the resting forearm pointing mostly at the camera
     (the z share in Q1), which a 45-degree view would reduce?

The demo clips are unlabeled, so nothing here is an accuracy figure. Does
not modify any src/ module or config.

Usage: python scripts/diagnose_rotation.py [clip ...]
Writes results/diag_rotation.md, results/diag/rotation_<clip>.png and
results/diag/rotation_<clip>_frames.csv (results/ is gitignored).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.landmark_cache import get_or_build, landmarks_at
from src import pose_estimation
from src import rules as rules_module
from src.feature_extraction import RIGHT, compute_shoulder_rotation_angle, select_side
from src.pipeline import REST_BAND_DEG, Session

EXERCISE = "shoulder_rotation"
DEFAULT_CLIPS = [
    "data/recordings/ShoulderInternalExternal1.mp4",
    "data/recordings/ShoulderInteralExternal2.mp4",
    "data/recordings/physitrack_standing_rotation.mp4",
]
REPORT_PATH = Path("results/diag_rotation.md")
DIAG_DIR = Path("results/diag")


def image_proxy(norm_landmarks, width: int, height: int) -> float:
    """Depth-free rotation proxy from image-space landmarks (the idea of
    scripts/rotation_proxy_experiment.py's proxy_from_points): horizontal
    forearm offset over 2D forearm length, in pixels so the non-square
    frame doesn't distort it. -1..1. Sign flipped for the LEFT arm, so for
    both arms larger means the hand is further toward the subject's own
    side (outward) in the image."""
    side = select_side(norm_landmarks)
    e, w = norm_landmarks[side["elbow"]], norm_landmarks[side["wrist"]]
    dx, dy = (w[0] - e[0]) * width, (w[1] - e[1]) * height
    n = float(np.hypot(dx, dy))
    if n == 0:
        return float("nan")
    # Subject faces the camera: their RIGHT side is image-left (smaller x).
    return float((-dx if side is RIGHT else dx) / n)


def forearm_z_share(landmarks) -> float:
    """|z| of the selected side's elbow->wrist vector over its 3D length:
    0 = forearm in the image plane, 1 = pointing straight at the camera."""
    side = select_side(landmarks)
    e = np.asarray(landmarks[side["elbow"]][:3], dtype=float)
    w = np.asarray(landmarks[side["wrist"]][:3], dtype=float)
    v = w - e
    n = float(np.linalg.norm(v))
    return float(abs(v[2]) / n) if n > 0 else float("nan")


def segment_summary(buffer: list[float], baseline: float) -> dict:
    """Direction of a detected segment relative to the resting baseline."""
    arr = np.asarray(buffer, dtype=float)
    up, down = float(arr.max() - baseline), float(baseline - arr.min())
    return {"n_frames": len(arr), "min": float(arr.min()), "max": float(arr.max()),
            "range": float(arr.max() - arr.min()), "baseline": float(baseline),
            "direction": "outward(above)" if up >= down else "inward(below)"}


def run_clip(path: Path, landmark_space: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    cache = get_or_build(path)
    fps = float(cache["fps"])
    session = Session(EXERCISE)
    session.start()
    roles = rules_module.required_joint_roles()
    frames, segments = [], []
    for i in range(cache["frame_count"]):
        lm = landmarks_at(cache, i, landmark_space)
        det = session.detector
        buffer_before = list(det.rep_buffer) if det.state == "MOVING" else []
        baseline_before = det.baseline if det.state != "CALIBRATING" else float("nan")
        event, payload, visible, _failed = session.process_frame(lm, None, i / fps, roles)
        row = {"frame": i, "t": i / fps, "state": det.state, "visible": visible,
               "baseline": det.baseline if det.state != "CALIBRATING" else np.nan,
               "angle_3d": np.nan, "proxy_2d": np.nan, "z_share": np.nan, "vis_elbow": np.nan, "vis_wrist": np.nan}
        if lm is not None:
            side = select_side(lm)
            row["vis_elbow"], row["vis_wrist"] = lm[side["elbow"]][3], lm[side["wrist"]][3]
            row["angle_3d"] = compute_shoulder_rotation_angle(lm)
            norm = landmarks_at(cache, i, "normalized")
            if norm is not None:
                row["proxy_2d"] = image_proxy(norm, cache["width"], cache["height"])
            row["z_share"] = forearm_z_share(lm)
        frames.append(row)
        if event == "rep_complete":
            segments.append({"end_frame": i, "outcome": "rep_complete",
                             **segment_summary(payload["angle_deg"], baseline_before)})
        elif event == "rep_discarded" and buffer_before:
            segments.append({"end_frame": i, "outcome": payload["reason_code"],
                             **segment_summary(buffer_before, baseline_before)})
    return pd.DataFrame(frames), pd.DataFrame(segments), fps


def plot_clip(name: str, frames: pd.DataFrame, segments: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax = axes[0]
    ax.plot(frames.t, frames.angle_3d, lw=0.8, label="rotation angle (3D, world)")
    ax.plot(frames.t, frames.baseline, "k--", lw=0.8, label="detector baseline")
    ax.fill_between(frames.t, frames.baseline - REST_BAND_DEG, frames.baseline + REST_BAND_DEG,
                    color="grey", alpha=0.2, label=f"rest band +/-{REST_BAND_DEG:g}")
    for _, s in segments.iterrows():
        t_end = frames.t.iloc[int(s.end_frame)]
        color = "green" if s.outcome == "rep_complete" else "red"
        ax.axvline(t_end, color=color, lw=0.8, alpha=0.6)
    ax.set_ylabel("deg")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"{name}: green = rep completed, red = discarded")
    axes[1].plot(frames.t, frames.z_share, lw=0.8, label="forearm z share (world)")
    axes[1].plot(frames.t, frames.proxy_2d, lw=0.8, label="image proxy (depth-free)")
    axes[1].set_ylim(-1, 1)
    axes[1].legend(fontsize=7)
    axes[2].plot(frames.t, frames.vis_elbow, lw=0.8, label="elbow")
    axes[2].plot(frames.t, frames.vis_wrist, lw=0.8, label="wrist")
    axes[2].axhline(0.5, color="k", ls=":", lw=0.8)
    axes[2].set_ylabel("visibility")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def clip_report(name: str, frames: pd.DataFrame, segments: pd.DataFrame, fps: float) -> list[str]:
    detected = frames.dropna(subset=["angle_3d"])
    rest = frames[(frames.state == "RESTING")].dropna(subset=["z_share"])
    jitter_3d = float(np.nanmedian(np.abs(np.diff(detected.angle_3d))))
    gated = frames[frames.visible].dropna(subset=["angle_3d", "proxy_2d"])
    r = float(gated.angle_3d.corr(gated.proxy_2d, method="spearman")) if len(gated) > 2 else float("nan")
    lines = [f"## {name}", "",
             f"- fps {fps:.2f}, {len(frames)} frames, pose detected on {len(detected)}, "
             f"confidence gate passed on {int(frames.visible.sum())}",
             f"- Q1: Spearman r(3D angle, depth-free image proxy) over {len(gated)} gated frames = {r:.2f}; "
             f"median frame-to-frame change of the 3D angle {jitter_3d:.2f} deg",
             f"- Q3: forearm z share while RESTING: median {rest.z_share.median():.2f} "
             f"(n={len(rest)} frames)" if len(rest) else "- Q3: no RESTING frames",
             ""]
    if segments.empty:
        lines += ["- Q2: no segments detected", ""]
        return lines
    counts = segments.groupby(["outcome", "direction"]).size()
    lines.append("- Q2: segments by outcome and direction: " +
                 ", ".join(f"{o}/{d} = {n}" for (o, d), n in counts.items()))
    lines += ["", "| end t (s) | outcome | direction | frames | min | max | range | baseline |",
              "|---|---|---|---|---|---|---|---|"]
    for _, s in segments.iterrows():
        lines.append(f"| {frames.t.iloc[int(s.end_frame)]:.1f} | {s.outcome} | {s.direction} | {int(s.n_frames)} | "
                     f"{s['min']:.1f} | {s['max']:.1f} | {s['range']:.1f} | {s.baseline:.1f} |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="*", default=DEFAULT_CLIPS)
    args = parser.parse_args()
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Rotation diagnosis (measurement only, unlabeled demo clips)", "",
             f"landmark_space `{landmark_space}`, detector REST_BAND_DEG {REST_BAND_DEG:g}. "
             "Nothing here is an accuracy figure.", ""]
    for clip in args.clips:
        path = Path(clip)
        frames, segments, fps = run_clip(path, landmark_space)
        frames.to_csv(DIAG_DIR / f"rotation_{path.stem}_frames.csv", index=False)
        plot_clip(path.name, frames, segments, DIAG_DIR / f"rotation_{path.stem}.png")
        lines += clip_report(path.name, frames, segments, fps)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {REPORT_PATH} and plots in {DIAG_DIR}")


if __name__ == "__main__":
    main()
