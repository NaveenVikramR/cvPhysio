"""Derives config/exercises/benchmarks/rehab24_abduction.json ONLY from
calibration-subject CORRECT reps, FRONT VIEW ONLY (view == "front" in
data/rehab24_manifest.csv -- this excludes cam18 entirely and excludes
half-profile/profile recordings on cam17, since mixing camera angles
into one percentile pool was found to make the derived peak_angle rule
nearly a no-op: see the previous, all-cameras version of this file's
commit message). Annotated rep boundaries, under landmark_space=world
(read from config/pose.json, NOT forced/mutated by this script; aborts
if it isn't "world"). Calibration subjects only; evaluation subjects
are never read. Does not run the full benchmark (scripts/
benchmark_rehab24.py main()) -- reuses only its rep_feature_sequences()
helper for calibration-subject correct front-view reps. Does not touch
config/exercises/shoulder_abduction.json.

Allowed range per rule = [p5, p95] of the measured feature across those
front-view reps' annotated spans:
  - peak_angle: target_peak_deg=(p5+p95)/2, tolerance_deg=(p95-p5)/2,
    so [target-tolerance, target+tolerance] == [p5, p95] exactly.
  - elbow_straight: LOWER bound only = p5 of elbow_min.
  - trunk_lean: UPPER bound only = p95 of trunk_max.
  - tempo: tempo_min_s=p5, tempo_max_s=p95 of tempo_s.
cue_cooldown_s/min_valid_fraction/cues are NOT data-derived -- carried
over unchanged from config/exercises/shoulder_abduction.json's own
operational defaults, since this benchmark config still needs them to
be loadable by src/rules.py's score_rep().

Prints n reps per subject, n total, the percentile table, and a
WARNING if total n < 30 (small-sample caveat -- front view alone is a
much smaller slice of the calibration data than all cameras/views
combined). Output: config/exercises/benchmarks/rehab24_abduction.json,
with n, subjects, views, date, commit hash and "derived_from" recorded
inside the file.
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_rehab24 import DATA_DIR, git_provenance, load_manifest_and_split, rep_feature_sequences
from src import pose_estimation

OUTPUT_PATH = Path("config/exercises/benchmarks/rehab24_abduction.json")
MIN_N_WARNING = 30


def collect_calibration_correct_front_rep_features(manifest: pd.DataFrame, split: dict) -> pd.DataFrame:
    calib_correct_front = manifest[
        manifest["subject"].isin(split["calibration_subjects"])
        & (manifest["label"] == "correct")
        & (manifest["view"] == "front")
    ]
    rows = []
    for _, rep in calib_correct_front.iterrows():
        video_path = DATA_DIR / "videos" / rep["video"]
        if not video_path.exists():
            continue
        seq, fps = rep_feature_sequences(video_path, int(rep["start_frame"]), int(rep["end_frame"]))
        if len(seq["angle_deg"]) == 0:
            continue
        rows.append({
            "subject": int(rep["subject"]), "video_id": rep["video_id"], "camera": rep["camera"], "view": rep["view"],
            "peak_angle": max(seq["angle_deg"]), "elbow_min": min(seq["elbow_angle_deg"]),
            "trunk_max": max(seq["trunk_lean_deg"]), "tempo_s": len(seq["angle_deg"]) / fps,
        })
    return pd.DataFrame(rows)


def main() -> None:
    commit, dirty = git_provenance()
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    print(f"Provenance: commit={commit}  dirty={dirty}  landmark_space={landmark_space}")
    if landmark_space != "world":
        print(f"ABORTING: config/pose.json landmark_space={landmark_space!r}, not 'world'. This config must be "
              "derived under landmark_space=world -- set config/pose.json's landmark_space to 'world' and re-run.")
        return

    manifest, split = load_manifest_and_split()
    print(f"Calibration subjects: {split['calibration_subjects']} (evaluation subjects never read)")

    print("Computing per-rep features for every calibration-subject CORRECT front-view rep "
          "(camera==cam17, view=='front' only -- cam18 and half-profile/profile excluded)...")
    df = collect_calibration_correct_front_rep_features(manifest, split)
    if df.empty:
        print("No calibration-subject correct front-view reps produced usable features -- nothing written.")
        return

    subjects_used = sorted(int(s) for s in df["subject"].unique())
    views_used = sorted(df["view"].unique().tolist())
    cameras_used = sorted(df["camera"].unique().tolist())
    print(f"\n{len(df)} calibration-subject CORRECT front-view reps used, subjects: {subjects_used}, "
          f"views: {views_used}, cameras: {cameras_used}")
    print("n reps per subject:")
    per_subject_counts = df.groupby("subject").size()
    print(per_subject_counts.to_string())
    if len(df) < MIN_N_WARNING:
        print(f"\nWARNING: total n={len(df)} < {MIN_N_WARNING} -- front-view-only calibration data is small. "
              "Percentiles below (especially p5/p95) are based on very few reps and should be treated with "
              "proportional caution.")

    percentiles = {}
    print("\n| feature | n | p5 | median | p95 |")
    print("|---|---|---|---|---|")
    for col in ("peak_angle", "elbow_min", "trunk_max", "tempo_s"):
        p5, median, p95 = df[col].quantile(0.05), df[col].median(), df[col].quantile(0.95)
        percentiles[col] = {"p5": float(p5), "median": float(median), "p95": float(p95)}
        print(f"| {col} | {len(df)} | {p5:.1f} | {median:.1f} | {p95:.1f} |")

    target_peak = (percentiles["peak_angle"]["p5"] + percentiles["peak_angle"]["p95"]) / 2
    tolerance = (percentiles["peak_angle"]["p95"] - percentiles["peak_angle"]["p5"]) / 2

    config = {
        "exercise": "shoulder_abduction",
        "notice": "Derived automatically from REHAB24-6 calibration-subject CORRECT reps, FRONT VIEW ONLY "
                  "(see derived_from/views below) -- a benchmark/reference config for scripts/"
                  "benchmark_rehab24.py --config, NOT the live shipped config "
                  "(config/exercises/shoulder_abduction.json is untouched by this file). A licensed "
                  "physiotherapist must still review before any of this is used for real feedback. "
                  f"n={len(df)} reps -- {'BELOW' if len(df) < MIN_N_WARNING else 'at/above'} the "
                  f"{MIN_N_WARNING}-rep caution threshold used when this file was generated.",
        "derived_from": "REHAB24-6 calibration subjects",
        "landmark_space": landmark_space,
        "n_reps": len(df),
        "n_reps_per_subject": {str(s): int(c) for s, c in per_subject_counts.items()},
        "subjects": subjects_used,
        "views": views_used,
        "cameras": cameras_used,
        "date": date.today().isoformat(),
        "commit": commit,
        "percentiles": percentiles,
        "target_peak_deg": {"value": round(target_peak, 1), "status": "derived",
                             "comment": "(p5+p95)/2 of peak_angle across calibration-subject correct front-view reps"},
        "tolerance_deg": {"value": round(tolerance, 1), "status": "derived",
                           "comment": "(p95-p5)/2 of peak_angle, so [target-tolerance,target+tolerance]==[p5,p95]"},
        "min_elbow_straight_deg": {"value": round(percentiles["elbow_min"]["p5"], 1), "status": "derived",
                                    "comment": "p5 of elbow_min (lower bound only)"},
        "max_trunk_lean_deg": {"value": round(percentiles["trunk_max"]["p95"], 1), "status": "derived",
                                "comment": "p95 of trunk_max (upper bound only)"},
        "tempo_min_s": {"value": round(percentiles["tempo_s"]["p5"], 2), "status": "derived", "comment": "p5 of tempo_s"},
        "tempo_max_s": {"value": round(percentiles["tempo_s"]["p95"], 2), "status": "derived", "comment": "p95 of tempo_s"},
        "cue_cooldown_s": {"value": 8, "status": "carried_over",
                           "comment": "not data-derived -- same operational default as config/exercises/shoulder_abduction.json"},
        "min_valid_fraction": {"value": 0.7, "status": "carried_over",
                                "comment": "not data-derived -- same operational default as config/exercises/shoulder_abduction.json"},
        "cues": {
            "peak_angle": "Raise your arm closer to the target height.",
            "elbow_straight": "Keep your elbow straighter as you raise your arm.",
            "trunk_lean": "Keep your torso upright -- try not to lean.",
            "tempo": "Move at a steadier, more controlled pace.",
            "insufficient_data": "Keep your whole arm in view.",
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(config, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
