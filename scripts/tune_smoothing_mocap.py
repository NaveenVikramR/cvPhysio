"""F2 step 3: tune One Euro landmark smoothing (src/smoothing.py) against
REHAB24-6 Ex1 motion capture, using cached landmarks (no new recordings).

Each recording span (front and half-profile; profile is unsupported) runs
through src.pipeline.Session(side="auto") with a LandmarkSmoother, exactly
as the live app does: the smoother is fed every frame from Start, it never
restarts per rep. The smoothed landmarks Session sees are captured, and the
right-arm angles on annotated rep frames are compared with mocap
(scripts/landmark_fidelity_report.py's mocap_angles/mediapipe_angles).

Per (min_cutoff, beta) cell, against the same recordings with smoothing off:
  shoulder MAE debiased  mean |error - group bias| (a constant offset is
                         removed separately by config/angle_correction.json)
  shoulder jitter        median |frame-to-frame change of the error| in a rep
  added lag              median over reps of (lag with smoothing - lag off),
                         in frames; lag = the shift k in [-MAX_LAG, MAX_LAG]
                         minimising mean |(mp(t) - mean) - (mocap(t-k) - mean)|
  peak shift             mean over reps of |peak with smoothing - peak off|
  rep counts             Session rep count vs the annotated count

SELECTION RULE (fixed before any result was seen; calibration subjects
1, 2, 6 only, front and half-profile):
  1. A cell is eligible only if, on EVERY calibration recording, its
     |detected - annotated| rep count is no worse than with smoothing off,
     AND over all calibration reps its median added lag <= MAX_ADDED_LAG
     frames AND its mean peak shift <= MAX_PEAK_SHIFT_DEG.
  2. Among eligible cells, pick the lowest front-view calibration shoulder
     MAE debiased. It must beat smoothing off by >= MIN_GAIN_DEG, else no
     cell is chosen and smoothing stays off.
Only then are the evaluation subjects (3, 4, 5, 7, 8, 9) run, for the chosen
cell and for smoothing off only.

d_cutoff 1.0 and max_hold_frames 5 stay at config/smoothing.json's values.
Never writes config/smoothing.json.

Usage: python scripts/tune_smoothing_mocap.py
Writes results/smoothing_mocap.md and results/smoothing_mocap_calibration.csv
(gitignored).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.benchmark_rehab24 as bm
from scripts import landmark_cache
from scripts.landmark_fidelity_report import fidelity, mediapipe_angles, mocap_angles
from src import pose_estimation
from src import rules as rules_module
from src import smoothing as smoothing_module
from src.pipeline import Session

REPORT_PATH = Path("results/smoothing_mocap.md")
CSV_PATH = Path("results/smoothing_mocap_calibration.csv")
VIEWS = ("front", "half-profile")
MIN_CUTOFF_GRID = (0.3, 0.5, 1.0, 2.0, 3.0, 5.0)
BETA_GRID = (0.0, 0.3, 1.0, 3.0, 10.0, 30.0)
D_CUTOFF = 1.0
MAX_HOLD_FRAMES = 5
MAX_LAG = 15
MAX_ADDED_LAG = 3.0
MAX_PEAK_SHIFT_DEG = 2.0
MIN_GAIN_DEG = 0.1
OFF = None  # the "smoothing off" cell


class RecordingSmoother:
    """Wraps a smoother (or none) and remembers what it returned for each
    frame, so the angles measured are exactly the landmarks Session used."""

    def __init__(self, inner):
        self.inner = inner
        self.last = None

    def update(self, landmarks, t):
        self.last = self.inner.update(landmarks, t) if self.inner is not None else landmarks
        return self.last


def best_lag(mp: np.ndarray, truth: np.ndarray, max_lag: int = MAX_LAG) -> float:
    """Shift k (frames) for which mp(t) best matches truth(t - k) after
    removing each series' mean; positive k means mp trails the truth.
    NaN if fewer than 10 paired frames survive at every shift."""
    mp, truth = np.asarray(mp, dtype=float), np.asarray(truth, dtype=float)
    best, best_err = np.nan, np.inf
    for k in range(-max_lag, max_lag + 1):
        a = mp[k:] if k >= 0 else mp[:k]
        b = truth[:len(truth) - k] if k >= 0 else truth[-k:]
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 10:
            continue
        err = np.mean(np.abs((a[ok] - a[ok].mean()) - (b[ok] - b[ok].mean())))
        if err < best_err:
            best, best_err = float(k), err
    return best


def run_recording(rec: dict, cell, landmark_space: str) -> tuple[pd.DataFrame, int]:
    """Frames of every annotated rep (with smoothed MediaPipe and mocap
    angles) and the Session rep count, for one recording and one cell."""
    inner = (smoothing_module.LandmarkSmoother(cell[0], cell[1], D_CUTOFF, MAX_HOLD_FRAMES)
             if cell is not OFF else None)
    smoother = RecordingSmoother(inner)
    cache, joints, reps = rec["cache"], rec["joints"], rec["reps"]
    start, end = rec["span"]
    end = min(end, cache["frame_count"] - 1)
    rep_of = {f: i for i, r in reps.iterrows() for f in range(int(r.start_frame), int(r.end_frame) + 1)}
    session = Session(bm.EXERCISE, side="auto")
    roles = rules_module.required_joint_roles()
    fps, detected, rows = cache["fps"], 0, []
    for f in range(start, end + 1):
        if f == start:
            session.start()
        lm = landmark_cache.landmarks_at(cache, f, landmark_space)
        event, *_ = session.process_frame(lm, smoother, f / fps, roles)
        detected += event == "rep_complete"
        if f in rep_of and f < len(joints):
            rows.append({"rep": rep_of[f], "frame": f, **mocap_angles(joints[f]), **mediapipe_angles(smoother.last)})
    session.stop()
    return pd.DataFrame(rows), detected


def rep_stats(frames: pd.DataFrame) -> pd.DataFrame:
    out = []
    for rep, g in frames.groupby("rep"):
        mp, truth = g["mp_shoulder"].to_numpy(), g["mocap_shoulder"].to_numpy()
        out.append({"rep": rep, "lag": best_lag(mp, truth),
                    "peak": float(np.nanmax(mp)) if np.isfinite(mp).any() else np.nan})
    return pd.DataFrame(out)


def evaluate_cell(recordings: list[dict], cell, landmark_space: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(frames with recording metadata, per-recording rep counts and per-rep stats)."""
    frames, per_rec = [], []
    for rec in recordings:
        df, detected = run_recording(rec, cell, landmark_space)
        df = df.assign(recording=rec["id"], view=rec["view"], subject=rec["subject"])
        frames.append(df)
        stats = rep_stats(df).assign(recording=rec["id"], view=rec["view"], subject=rec["subject"],
                                     detected=detected, annotated=rec["annotated"])
        per_rec.append(stats)
    return pd.concat(frames, ignore_index=True), pd.concat(per_rec, ignore_index=True)


def summarize(cell, frames: pd.DataFrame, reps: pd.DataFrame, off_reps: pd.DataFrame) -> dict:
    merged = reps.merge(off_reps[["recording", "rep", "lag", "peak", "detected"]], on=["recording", "rep"],
                        suffixes=("", "_off"))
    counts = merged.drop_duplicates("recording")
    worse = int(((counts.detected - counts.annotated).abs() > (counts.detected_off - counts.annotated).abs()).sum())
    row = {"min_cutoff": None if cell is OFF else cell[0], "beta": None if cell is OFF else cell[1],
           "recordings_count_worse": worse,
           "exact_counts": f"{int((counts.detected == counts.annotated).sum())}/{len(counts)}",
           "added_lag": float(np.nanmedian(merged.lag - merged.lag_off)),
           "peak_shift": float(np.nanmean(np.abs(merged.peak - merged.peak_off)))}
    for view in VIEWS:
        g = frames[frames.view == view]
        if g.empty:
            continue
        s = fidelity(g["mp_shoulder"], g["mocap_shoulder"], g["recording"] + ":" + g["rep"].astype(str))
        e = fidelity(g["mp_elbow"], g["mocap_elbow"])
        row[f"{view}_shoulder_mae_debiased"] = s["mae_debiased"]
        row[f"{view}_shoulder_jitter"] = s["jitter"]
        row[f"{view}_shoulder_r"] = s["r"]
        row[f"{view}_elbow_mae_debiased"] = e["mae_debiased"]
    return row


def select_cell(table: pd.DataFrame) -> tuple | None:
    """The selection rule in the module docstring. `table` has one row per
    cell, including the OFF row (min_cutoff None)."""
    off = table[table.min_cutoff.isna()].iloc[0]
    cells = table[table.min_cutoff.notna()]
    eligible = cells[(cells.recordings_count_worse == 0) & (cells.added_lag <= MAX_ADDED_LAG)
                     & (cells.peak_shift <= MAX_PEAK_SHIFT_DEG)]
    if eligible.empty:
        return None
    best = eligible.sort_values("front_shoulder_mae_debiased").iloc[0]
    if off.front_shoulder_mae_debiased - best.front_shoulder_mae_debiased < MIN_GAIN_DEG:
        return None
    return float(best.min_cutoff), float(best.beta)


def load_recordings(manifest: pd.DataFrame, subjects: set[int]) -> list[dict]:
    out = []
    for _, rec in manifest.drop_duplicates(["video_id", "camera"]).iterrows():
        if rec.view not in VIEWS or int(rec.subject) not in subjects:
            continue
        video = bm.DATA_DIR / "videos" / rec.video
        joints = bm.load_mocap_joints(rec.video_id)
        if joints is None or not video.exists():
            continue
        all_reps = manifest[(manifest.video_id == rec.video_id) & (manifest.camera == rec.camera)]
        out.append({"id": f"{rec.video_id}/{rec.camera}", "view": rec.view, "subject": int(rec.subject),
                    "cache": landmark_cache.get_or_build(video), "joints": joints,
                    "reps": all_reps[~all_reps.mocap_erroneous.astype(bool)], "annotated": len(all_reps),
                    "span": bm.recording_span(manifest, rec.video_id, rec.camera)})
    return out


def fmt_row(r: dict) -> str:
    cell = "off" if r["min_cutoff"] is None or pd.isna(r["min_cutoff"]) else f"{r['min_cutoff']:g} / {r['beta']:g}"
    v = lambda k, d=2: "-" if k not in r or pd.isna(r[k]) else f"{r[k]:.{d}f}"
    return (f"| {cell} | {r['exact_counts']} | {r['recordings_count_worse']} | {v('added_lag', 1)} | "
            f"{v('peak_shift')} | {v('front_shoulder_mae_debiased')} | {v('front_shoulder_jitter')} | "
            f"{v('half-profile_shoulder_mae_debiased')} | {v('half-profile_shoulder_jitter')} | "
            f"{v('front_elbow_mae_debiased')} |")


HEADER = ["| min_cutoff / beta | exact rep counts | recordings with worse count | added lag (frames) | "
          "peak shift | front shoulder MAE debiased | front jitter | half-profile MAE debiased | "
          "half-profile jitter | front elbow MAE debiased |", "|---|---|---|---|---|---|---|---|---|---|"]


def main() -> None:
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    manifest, split = bm.load_manifest_and_split()
    commit, dirty = bm.git_provenance()

    calib = load_recordings(manifest, set(split["calibration_subjects"]))
    off_frames, off_reps = evaluate_cell(calib, OFF, landmark_space)
    rows = [summarize(OFF, off_frames, off_reps, off_reps)]
    for mc in MIN_CUTOFF_GRID:
        for beta in BETA_GRID:
            frames, reps = evaluate_cell(calib, (mc, beta), landmark_space)
            rows.append(summarize((mc, beta), frames, reps, off_reps))
            print(fmt_row(rows[-1]), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(CSV_PATH, index=False)
    chosen = select_cell(table)

    lines = ["# One Euro smoothing tuned against mocap (REHAB24-6 Ex1, right arm)", "",
             f"_commit `{commit}`{' (dirty)' if dirty else ''}, landmark_space `{landmark_space}`, "
             f"d_cutoff {D_CUTOFF}, max_hold_frames {MAX_HOLD_FRAMES}, Session side=auto_", "",
             f"Selection rule (scripts/tune_smoothing_mocap.py docstring): rep count never worse on any calibration "
             f"recording, median added lag <= {MAX_ADDED_LAG:g} frames, mean peak shift <= {MAX_PEAK_SHIFT_DEG:g} deg; "
             f"then lowest front-view shoulder MAE debiased, beating off by >= {MIN_GAIN_DEG:g} deg.", "",
             f"## Calibration subjects {split['calibration_subjects']} ({len(calib)} recordings)", "", *HEADER,
             *[fmt_row(r) for r in rows], "",
             f"**Chosen:** {'none -- smoothing stays off' if chosen is None else f'min_cutoff {chosen[0]:g}, beta {chosen[1]:g}'}"]

    if chosen is not None:
        evals = load_recordings(manifest, set(split["evaluation_subjects"]))
        e_off_frames, e_off_reps = evaluate_cell(evals, OFF, landmark_space)
        e_frames, e_reps = evaluate_cell(evals, chosen, landmark_space)
        lines += ["", f"## Evaluation subjects {split['evaluation_subjects']} ({len(evals)} recordings), "
                      f"chosen cell and off only", "", *HEADER,
                  fmt_row(summarize(OFF, e_off_frames, e_off_reps, e_off_reps)),
                  fmt_row(summarize(chosen, e_frames, e_reps, e_off_reps))]
    text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(text)
    print(text)
    print(f"Wrote {REPORT_PATH} and {CSV_PATH}")


if __name__ == "__main__":
    main()
