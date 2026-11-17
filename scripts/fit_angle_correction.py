"""F2 step 2: compare ways of correcting MediaPipe's shoulder (abduction)
angle offset against motion capture. Measurement only -- nothing here
changes src/ or config/.

Ground truth: REHAB24-6 Ex1 mocap, hip-shoulder-elbow with the ANATOMICAL
right hip (docs/landmark_fidelity.md, finding 1). Frames come from
scripts/landmark_fidelity_report.collect_frames() (right arm, live
confidence gate, cached landmarks).

Every correction is fitted on CALIBRATION subjects only and judged on
EVALUATION subjects:
  A none        the raw angle.
  B offset      subtract the calibration subjects' mean error.
  C linear      invert a straight-line fit (mediapipe = a * mocap + b) from
                the calibration subjects.
  D rest-anchor per recording: offset = median MediaPipe angle over the first
                REST_FRAMES of its first annotated rep minus the calibration
                subjects' mean TRUE angle over the same kind of frames. Uses
                nothing from the evaluation subject's own mocap, so it is what
                a live session could do during calibration.

Fitted on the front view (the recording setup). Half-profile is reported
too, with the front-fitted corrections, for reference.

Usage: python scripts/fit_angle_correction.py
Writes results/angle_correction.md (gitignored).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.benchmark_rehab24 as bm
from scripts.landmark_fidelity_report import collect_frames
from src import pose_estimation

REPORT_PATH = Path("results/angle_correction.md")
REST_FRAMES = 15  # the live detector's CALIBRATION_FRAMES
FIT_VIEW = "front"
METHODS = ("A none", "B offset", "C linear", "D rest-anchor")


def add_recording_ids(df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["recording"] = df["rep"].map(manifest["video_id"]) + "/" + df["camera"]
    return df


def rest_frames(df: pd.DataFrame) -> pd.DataFrame:
    """The first REST_FRAMES valid frames of each recording's first annotated rep."""
    df = df.dropna(subset=["mp_shoulder", "mocap_shoulder"])
    first_rep = df.groupby("recording")["rep"].transform("min")
    first = df[df["rep"] == first_rep].sort_values(["recording", "frame"])
    return first.groupby("recording").head(REST_FRAMES)


def fit(calib: pd.DataFrame) -> dict:
    c = calib.dropna(subset=["mp_shoulder", "mocap_shoulder"])
    slope, intercept = np.polyfit(c["mocap_shoulder"], c["mp_shoulder"], 1)
    return {"offset": float((c["mp_shoulder"] - c["mocap_shoulder"]).mean()),
            "slope": float(slope), "intercept": float(intercept),
            "true_rest": float(rest_frames(c)["mocap_shoulder"].mean())}


def corrected(df: pd.DataFrame, params: dict, method: str) -> pd.Series:
    mp = df["mp_shoulder"]
    if method == "A none":
        return mp
    if method == "B offset":
        return mp - params["offset"]
    if method == "C linear":
        return (mp - params["intercept"]) / params["slope"]
    rest = rest_frames(df).groupby("recording")["mp_shoulder"].median()
    per_recording = rest - params["true_rest"]
    return mp - df["recording"].map(per_recording)


def error_stats(est: pd.Series, truth: pd.Series) -> dict:
    e = (est - truth).dropna()
    return {"n": len(e), "bias": float(e.mean()), "mae": float(e.abs().mean()),
            "p90": float(e.abs().quantile(0.9))}


def main() -> None:
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    manifest, split = bm.load_manifest_and_split()
    commit, dirty = bm.git_provenance()
    df = add_recording_ids(collect_frames(manifest, split, landmark_space), manifest)
    params = fit(df[(df.view == FIT_VIEW) & (df.group == "calibration")])
    fit_subjects = sorted(int(s) for s in df[(df.view == FIT_VIEW) & (df.group == "calibration")].subject.unique())

    lines = ["# Shoulder angle offset correction (REHAB24-6 Ex1, right arm)", "",
             f"_commit `{commit}`{' (dirty)' if dirty else ''}, landmark_space `{landmark_space}`_", "",
             f"Fitted on {FIT_VIEW}-view calibration subjects {fit_subjects} only "
             f"(subject 6 has no front view, so this is two subjects). Truth: mocap with the anatomical hip.", "",
             f"Fitted: offset {params['offset']:+.2f} deg; linear mediapipe = {params['slope']:.3f} x mocap "
             f"{params['intercept']:+.2f}; calibration subjects' mean true rest angle {params['true_rest']:.2f} deg "
             f"(first {REST_FRAMES} frames of each recording's first rep).", "",
             "| view | group | method | frames | bias | MAE | 90th pct abs error |", "|---|---|---|---|---|---|---|"]
    per_subject = []
    for view in ("front", "half-profile"):
        for group in ("calibration", "evaluation"):
            g = df[(df.view == view) & (df.group == group)].dropna(subset=["mp_shoulder", "mocap_shoulder"])
            if g.empty:
                continue
            for method in METHODS:
                s = error_stats(corrected(g, params, method), g["mocap_shoulder"])
                lines.append(f"| {view} | {group} | {method} | {s['n']} | {s['bias']:+.2f} | {s['mae']:.2f} | "
                             f"{s['p90']:.2f} |")
                if view == "front" and group == "evaluation":
                    for subj, gs in g.groupby("subject"):
                        ss = error_stats(corrected(gs, params, method), gs["mocap_shoulder"])
                        per_subject.append((int(subj), method, ss["bias"], ss["mae"]))
    lines += ["", "## Front view, evaluation subjects: MAE per subject", "",
              "| subject | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    table = pd.DataFrame(per_subject, columns=["subject", "method", "bias", "mae"])
    for subj, g in table.groupby("subject"):
        m = dict(zip(g.method, g.mae))
        lines.append(f"| {subj} | " + " | ".join(f"{m[x]:.2f}" for x in METHODS) + " |")
    text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(text)
    print(text)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
