"""F2 step 1: does locking the measured arm help rep detection? Runs
src.pipeline.Session over every REHAB24-6 Ex1 recording span (the same
Start/Stop span convention as scripts/benchmark_rehab24.py part (a)) once
per arm mode and compares detected rep counts with the dataset's annotated
counts.

Modes: per_frame (the original behaviour, select_side() every frame),
auto (lock to the arm that completes the first rep), RIGHT (Ex1 always
uses the right arm, so this is the correct fixed choice) and LEFT (the
wrong fixed choice, as a floor).

Also reports, for every completed rep, whether its frames were measured on
a mix of arms (only possible in per_frame mode).

No threshold is tuned here, so all subjects are reported, with calibration
(1, 2, 6) and evaluation (3, 4, 5, 7, 8, 9) subjects shown separately.
Landmarks come from the per-video cache; MediaPipe never runs if every
video is cached.

Usage: python scripts/evaluate_side_lock.py
Writes results/side_lock.md and results/side_lock.csv (gitignored).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.benchmark_rehab24 as bm
from scripts import landmark_cache
from src import pose_estimation
from src import rules as rules_module
from src.pipeline import Session

MODES = ("per_frame", "auto", "RIGHT", "LEFT")
REPORT_PATH = Path("results/side_lock.md")
CSV_PATH = Path("results/side_lock.csv")
VIEW_ORDER = ["front", "half-profile", "profile"]


def run_span(cache: dict, start: int, end: int, mode: str, landmark_space: str) -> dict:
    """One recording span through Session(side=mode)."""
    end = min(end, cache["frame_count"] - 1)
    roles = rules_module.required_joint_roles()
    session = Session(bm.EXERCISE, side=mode)
    fps = cache["fps"]
    reps, mixed = 0, 0
    for f in range(cache["frame_count"]):
        if f == start:
            session.start()
        lm = landmark_cache.landmarks_at(cache, f, landmark_space)
        event, payload, _v, _fl = session.process_frame(lm, None, f / fps, roles)
        if f == end:
            session.stop()
        if event == "rep_complete":
            reps += 1
            mixed += len(set(payload["side"])) > 1
    return {"detected": reps, "mixed_side_reps": mixed, "locked_side": session.locked_side}


def collect(manifest: pd.DataFrame, split: dict, landmark_space: str) -> pd.DataFrame:
    calib = set(split["calibration_subjects"])
    recordings = manifest.drop_duplicates(subset=["video_id", "camera"])
    rows = []
    for _, rec in recordings.iterrows():
        video_path = bm.DATA_DIR / "videos" / rec["video"]
        if not video_path.exists():
            continue
        annotated = len(manifest[(manifest.video_id == rec.video_id) & (manifest.camera == rec.camera)])
        start, end = bm.recording_span(manifest, rec.video_id, rec.camera)
        cache = landmark_cache.get_or_build(video_path)
        for mode in MODES:
            r = run_span(cache, start, end, mode, landmark_space)
            rows.append({"subject": int(rec.subject), "group": "calibration" if int(rec.subject) in calib else "evaluation",
                         "video_id": rec.video_id, "camera": rec.camera, "view": rec.view, "mode": mode,
                         "annotated": annotated, **r, "signed_error": r["detected"] - annotated,
                         "exact": int(r["detected"] == annotated)})
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, commit: str, dirty: bool, landmark_space: str) -> str:
    lines = ["# Arm locking vs rep detection (REHAB24-6 Ex1, right arm)", "",
             f"_commit `{commit}`{' (dirty)' if dirty else ''}, landmark_space `{landmark_space}`_", "",
             "Exact = detected reps equal the annotated count for a recording. Mixed-arm reps = completed reps "
             "whose frames were measured on both arms.", "",
             "| view | group | mode | recordings | exact match (95% CI) | mean signed error | mixed-arm reps | auto locked RIGHT |",
             "|---|---|---|---|---|---|---|---|"]
    for view in VIEW_ORDER:
        for group in ("calibration", "evaluation"):
            for mode in MODES:
                g = df[(df.view == view) & (df.group == group) & (df["mode"] == mode)]
                if g.empty:
                    continue
                k, n = int(g.exact.sum()), len(g)
                locked = (f"{int((g.locked_side == 'RIGHT').sum())}/{int(g.locked_side.notna().sum())}"
                          if mode == "auto" else "-")
                lines.append(f"| {view} | {group} | {mode} | {n} | {bm.rate_with_ci(k, n)} | "
                             f"{g.signed_error.mean():+.2f} | {int(g.mixed_side_reps.sum())}/{int(g.detected.sum())} | "
                             f"{locked} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    manifest, split = bm.load_manifest_and_split()
    commit, dirty = bm.git_provenance()
    df = collect(manifest, split, landmark_space)
    df.to_csv(CSV_PATH, index=False)
    text = report(df, commit, dirty, landmark_space)
    REPORT_PATH.write_text(text)
    print(text)
    print(f"Wrote {REPORT_PATH} and {CSV_PATH}")


if __name__ == "__main__":
    main()
