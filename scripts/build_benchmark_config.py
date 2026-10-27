"""Derives config/exercises/benchmarks/rehab24_abduction.json ONLY from
calibration-subject CORRECT reps -- ALL cameras/views, matching the
established calibration_percentiles() precedent in scripts/
benchmark_rehab24.py (no camera/view restriction there either) -- under
landmark_space=world (read from config/pose.json, NOT forced/mutated by
this script; aborts if it isn't "world"). Calibration subjects only;
evaluation subjects are never read. Does not run the full benchmark
(scripts/benchmark_rehab24.py main()) -- reuses only its
rep_feature_sequences() helper for calibration-subject correct reps.
Does not touch config/exercises/shoulder_abduction.json.

Allowed range per rule = [p5, p95] of the measured feature across those
58 reps' annotated spans:
  - peak_angle: target_peak_deg=(p5+p95)/2, tolerance_deg=(p95-p5)/2,
    so [target-tolerance, target+tolerance] == [p5, p95] exactly.
  - elbow_straight: LOWER bound only = p5 of elbow_min.
  - trunk_lean: UPPER bound only = p95 of trunk_max.
  - tempo: tempo_min_s=p5, tempo_max_s=p95 of tempo_s.
cue_cooldown_s/min_valid_fraction/cues are NOT data-derived -- carried
over unchanged from config/exercises/shoulder_abduction.json's own
operational defaults, since this benchmark config still needs them to
be loadable by src/rules.py's score_rep().

Output: config/exercises/benchmarks/rehab24_abduction.json, with n
reps, subjects, date, commit hash and "derived_from" recorded inside
the file.
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


def collect_calibration_correct_rep_features(manifest: pd.DataFrame, split: dict) -> pd.DataFrame:
    calib_correct = manifest[manifest["subject"].isin(split["calibration_subjects"]) & (manifest["label"] == "correct")]
    rows = []
    for _, rep in calib_correct.iterrows():
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

    print("Computing per-rep features for every calibration-subject CORRECT rep (all cameras/views)...")
    df = collect_calibration_correct_rep_features(manifest, split)
    if df.empty:
        print("No calibration-subject correct reps produced usable features -- nothing written.")
        return

    subjects_used = sorted(int(s) for s in df["subject"].unique())
    print(f"\n{len(df)} calibration-subject CORRECT reps used, subjects: {subjects_used}")
    print(df.groupby(["subject", "camera", "view"]).size().to_string())

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
        "notice": "Derived automatically from REHAB24-6 calibration-subject CORRECT reps (see derived_from "
                  "below) -- a benchmark/reference config for scripts/benchmark_rehab24.py --config, NOT the "
                  "live shipped config (config/exercises/shoulder_abduction.json is untouched by this file). "
                  "A licensed physiotherapist must still review before any of this is used for real feedback.",
        "derived_from": "REHAB24-6 calibration subjects",
        "landmark_space": landmark_space,
        "n_reps": len(df),
        "subjects": subjects_used,
        "date": date.today().isoformat(),
        "commit": commit,
        "percentiles": percentiles,
        "target_peak_deg": {"value": round(target_peak, 1), "status": "derived",
                             "comment": "(p5+p95)/2 of peak_angle across calibration-subject correct reps"},
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
