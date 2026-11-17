"""F2 step 4: tune RepDetector's opt-in settings (src/pipeline.py:
start_frames, end_frames, baseline) for rep counting on REHAB24-6 Ex1.

Every front and half-profile recording span runs through
src.pipeline.Session(side="auto", detector=settings) as the live app does
(no smoothing, raw angles; a constant angle offset can't change the
detector, which works relative to its baseline). Detected rep counts are
compared with the annotated counts.

Grid: baseline {running, stable, fixed} x start_frames {1, 2, 3, 5} x
end_frames {1, 2, 3, 5}. running/1/1 is the current detector.

SELECTION RULE (fixed before any result was seen; calibration subjects
1, 2, 6 only, 9 recordings):
  1. Eligible: on EVERY calibration recording, |detected - annotated| is no
     larger than the current detector's.
  2. Rank eligible cells by: most recordings counted exactly, then lowest
     total |detected - annotated|, then fewest departures from the current
     detector (baseline "running" before "stable" before "fixed", then the
     smallest start_frames + end_frames).
  3. The top cell is chosen only if it beats the current detector on
     exact recordings or on total |error|; otherwise nothing changes.
Only then are the evaluation subjects (3, 4, 5, 7, 8, 9) run, for the
chosen cell and the current detector only.

Usage: python scripts/tune_rep_detector.py
Writes results/rep_detector.md and results/rep_detector_calibration.csv
(gitignored). Changes no config; the live app keeps the default detector.
"""

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.benchmark_rehab24 as bm
from scripts import landmark_cache
from src import pose_estimation
from src import rules as rules_module
from src.pipeline import BASELINE_MODES, DEFAULT_DETECTOR_SETTINGS, Session

REPORT_PATH = Path("results/rep_detector.md")
CSV_PATH = Path("results/rep_detector_calibration.csv")
VIEWS = ("front", "half-profile")
START_GRID = (1, 2, 3, 5)
END_GRID = (1, 2, 3, 5)
CURRENT = dict(DEFAULT_DETECTOR_SETTINGS)


def grid() -> list[dict]:
    return [{"baseline": b, "start_frames": s, "end_frames": e}
            for b, s, e in itertools.product(BASELINE_MODES, START_GRID, END_GRID)]


def count_reps(rec: dict, settings: dict, landmark_space: str) -> tuple[int, int]:
    """(completed reps, discarded movements) for one recording span."""
    cache = rec["cache"]
    start, end = rec["span"]
    end = min(end, cache["frame_count"] - 1)
    session = Session(bm.EXERCISE, side="auto", detector=settings)
    roles = rules_module.required_joint_roles()
    done = discarded = 0
    for f in range(start, end + 1):
        if f == start:
            session.start()
        event, *_ = session.process_frame(landmark_cache.landmarks_at(cache, f, landmark_space), None,
                                          f / cache["fps"], roles)
        done += event == "rep_complete"
        discarded += event == "rep_discarded"
    return done, discarded


def run_cell(recordings: list[dict], settings: dict, landmark_space: str) -> pd.DataFrame:
    rows = []
    for rec in recordings:
        done, discarded = count_reps(rec, settings, landmark_space)
        rows.append({**settings, "recording": rec["id"], "view": rec["view"], "subject": rec["subject"],
                     "annotated": rec["annotated"], "detected": done, "discarded": discarded,
                     "abs_error": abs(done - rec["annotated"]), "exact": int(done == rec["annotated"])})
    return pd.DataFrame(rows)


def departure(settings: dict) -> tuple[int, int]:
    return BASELINE_MODES.index(settings["baseline"]), settings["start_frames"] + settings["end_frames"]


def select_cell(per_recording: pd.DataFrame) -> dict | None:
    """The selection rule in the module docstring. `per_recording` holds one
    row per (cell, recording) and includes the CURRENT cell."""
    keys = list(CURRENT)
    is_current = (per_recording[keys] == pd.Series(CURRENT)).all(axis=1)
    current = per_recording[is_current].set_index("recording")
    cur_exact, cur_err = int(current.exact.sum()), int(current.abs_error.sum())
    candidates = []
    for values, g in per_recording.groupby(keys):
        settings = dict(zip(keys, values))
        g = g.set_index("recording")
        if (g.abs_error > current.abs_error.reindex(g.index)).any():
            continue
        candidates.append((-int(g.exact.sum()), int(g.abs_error.sum()), departure(settings), settings))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[:3])
    neg_exact, err, _dep, best = candidates[0]
    if best == CURRENT or (-neg_exact <= cur_exact and err >= cur_err):
        return None
    return best


def load_recordings(manifest: pd.DataFrame, subjects: set[int]) -> list[dict]:
    out = []
    for _, rec in manifest.drop_duplicates(["video_id", "camera"]).iterrows():
        video = bm.DATA_DIR / "videos" / rec.video
        view = bm.recording_view(manifest, rec.video_id, rec.camera)
        if not set(view.split("+")) <= set(VIEWS) or int(rec.subject) not in subjects or not video.exists():
            continue
        out.append({"id": f"{rec.video_id}/{rec.camera}", "view": view, "subject": int(rec.subject),
                    "cache": landmark_cache.get_or_build(video),
                    "annotated": len(manifest[(manifest.video_id == rec.video_id) & (manifest.camera == rec.camera)]),
                    "span": bm.recording_span(manifest, rec.video_id, rec.camera)})
    return out


def label(settings: dict) -> str:
    return f"{settings['baseline']} / {settings['start_frames']} / {settings['end_frames']}"


def summary_line(settings: dict, g: pd.DataFrame) -> str:
    parts = []
    for view in RECORDING_VIEWS:
        v = g[g.view == view]
        parts.append(f"{bm.rate_with_ci(int(v.exact.sum()), len(v))} | {(v.detected - v.annotated).mean():+.2f}"
                     if len(v) else "- | -")
    return (f"| {label(settings)} | {bm.rate_with_ci(int(g.exact.sum()), len(g))} | {int(g.abs_error.sum())} | "
            f"{int(g.discarded.sum())} | " + " | ".join(parts) + " |")


RECORDING_VIEWS = ("front", "half-profile", "front+half-profile")  # bm.recording_view labels
HEADER = ["| baseline / start_frames / end_frames | exact (all) | total abs error | discarded | "
          + " | ".join(f"{v} exact | {v} mean signed error" for v in RECORDING_VIEWS) + " |",
          "|---|---|---|---|" + "---|---|" * len(RECORDING_VIEWS)]


def main() -> None:
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    manifest, split = bm.load_manifest_and_split()
    commit, dirty = bm.git_provenance()

    calib = load_recordings(manifest, set(split["calibration_subjects"]))
    per_recording = pd.concat([run_cell(calib, s, landmark_space) for s in grid()], ignore_index=True)
    per_recording.to_csv(CSV_PATH, index=False)
    chosen = select_cell(per_recording)
    keys = list(CURRENT)

    lines = ["# RepDetector settings vs rep counts (REHAB24-6 Ex1)", "",
             f"_commit `{commit}`{' (dirty)' if dirty else ''}, landmark_space `{landmark_space}`, "
             f"Session side=auto, no smoothing_", "",
             "Selection rule: scripts/tune_rep_detector.py docstring. Current detector = running / 1 / 1.", "",
             f"## Calibration subjects {split['calibration_subjects']} ({len(calib)} recordings)", "", *HEADER]
    for values, g in per_recording.groupby(keys, sort=False):
        lines.append(summary_line(dict(zip(keys, values)), g))
    lines += ["", f"**Chosen:** {'none -- the current detector stays' if chosen is None else label(chosen)}"]

    if chosen is not None:
        evals = load_recordings(manifest, set(split["evaluation_subjects"]))
        lines += ["", f"## Evaluation subjects {split['evaluation_subjects']} ({len(evals)} recordings), "
                      "chosen cell and current only", "", *HEADER]
        detail = []
        for settings in (CURRENT, chosen):
            g = run_cell(evals, settings, landmark_space)
            lines.append(summary_line(settings, g))
            detail.append(g.assign(cell=label(settings)))
        d = pd.concat(detail)
        lines += ["", "| recording | view | annotated | " + " | ".join(label(s) for s in (CURRENT, chosen)) + " |",
                  "|---|---|---|---|---|"]
        for rec, g in d.groupby("recording", sort=False):
            counts = dict(zip(g.cell, g.detected))
            lines.append(f"| {rec} | {g.view.iloc[0]} | {g.annotated.iloc[0]} | "
                         + " | ".join(str(counts[label(s)]) for s in (CURRENT, chosen)) + " |")
    text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(text)
    print(text)
    print(f"Wrote {REPORT_PATH} and {CSV_PATH}")


if __name__ == "__main__":
    main()
