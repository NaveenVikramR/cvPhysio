"""STEP 3 of the reference-scoring task: derive reference_relative
tolerances from CALIBRATION subjects, front view only (evaluation
subjects never read). For each calibration subject with front-view
CORRECT reps: reference = that subject's first 3 correct reps in
annotated order (video_id then repetition_number), via
rules.build_reference(); then, for each of that subject's REMAINING
correct reps (never the 3 reference reps themselves), the one-sided
deviation of each feature from that subject's own reference median:
  - peak_angle / elbow_min: max(0, ref_median - measured) -- only
    penalize falling BELOW the reference (more range/straighter is
    never "bad", matching score_rep()'s lower-bound-only rule).
  - trunk_max: max(0, measured - ref_median) -- only penalize going
    ABOVE the reference (less lean is never "bad").
  - tempo_s: |measured - ref_median| / ref_median -- a fractional
    magnitude, since tempo's rule is two-sided (too fast OR too slow).
Pooled over BOTH calibration subjects' remaining reps, tolerance per
feature = p95 of that one-sided deviation. Landmarks sourced entirely
from the landmark cache (--rescore-only semantics; every video already
has a cache entry from the earlier full benchmark run).

Output: config/exercises/benchmarks/rehab24_abduction_reference.json
(mode="reference_relative"), with n (reference + deviation reps),
subjects, date, commit hash and derived_from recorded inside. The
absolute-mode fallback keys are copied from the already-derived
front-view absolute config (config/exercises/benchmarks/
rehab24_abduction.json), since score_rep() needs them whenever this
mode has no reference supplied. Warns if n (deviation reps) < 30.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.benchmark_rehab24 as bm
from src import pose_estimation
from src import rules as rules_module

OUTPUT_PATH = Path("config/exercises/benchmarks/rehab24_abduction_reference.json")
ABSOLUTE_FALLBACK_PATH = Path("config/exercises/benchmarks/rehab24_abduction.json")
MIN_N_WARNING = 30
N_REFERENCE_REPS = 3


def calibration_front_correct_reps(manifest, split):
    return manifest[
        manifest["subject"].isin(split["calibration_subjects"])
        & (manifest["view"] == "front")
        & (manifest["label"] == "correct")
    ].sort_values(["subject", "video_id", "repetition_number"])


def main() -> None:
    bm.RESCORE_ONLY = True  # never touch MediaPipe -- every video must already be cached
    commit, dirty = bm.git_provenance()
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    print(f"Provenance: commit={commit}  dirty={dirty}  landmark_space={landmark_space}  rescore_only=True")
    if landmark_space != "world":
        print("ABORTING: set config/pose.json's landmark_space to 'world' and re-run.")
        return

    manifest, split = bm.load_manifest_and_split()
    calib_front_correct = calibration_front_correct_reps(manifest, split)
    print(f"Calibration subjects: {split['calibration_subjects']} (evaluation subjects never read)")
    print(f"Calibration-subject front-view correct reps available: {len(calib_front_correct)}")
    print(calib_front_correct.groupby("subject").size().to_string())

    deviations = {"peak_angle": [], "elbow_min": [], "trunk_max": [], "tempo_s": []}
    subjects_used = []
    n_reference_reps = 0
    per_subject_reference = {}
    fps_ref = 30.0

    for subject in sorted(calib_front_correct["subject"].unique()):
        sub_reps = calib_front_correct[calib_front_correct["subject"] == subject]
        if len(sub_reps) <= N_REFERENCE_REPS:
            print(f"  subject {subject}: only {len(sub_reps)} correct front-view reps -- need > "
                  f"{N_REFERENCE_REPS} (reference + at least 1 deviation rep) -- skipping.")
            continue

        reference_rows = sub_reps.iloc[:N_REFERENCE_REPS]
        remaining_rows = sub_reps.iloc[N_REFERENCE_REPS:]

        reference_reps = []
        for _, rep in reference_rows.iterrows():
            video_path = bm.DATA_DIR / "videos" / rep["video"]
            seq, fps_ref = bm.rep_feature_sequences(video_path, int(rep["start_frame"]), int(rep["end_frame"]))
            if len(seq["angle_deg"]) > 0:
                reference_reps.append(seq)
        if len(reference_reps) < N_REFERENCE_REPS:
            print(f"  subject {subject}: fewer than {N_REFERENCE_REPS} usable reference reps -- skipping.")
            continue

        reference = rules_module.build_reference(f"calib_subject_{subject}", "shoulder_abduction", reference_reps, fps_ref)
        medians = reference["medians"]
        per_subject_reference[str(int(subject))] = medians
        n_reference_reps += len(reference_reps)
        print(f"  subject {subject}: reference medians = {medians}  "
              f"({len(remaining_rows)} remaining correct reps for deviations)")

        for _, rep in remaining_rows.iterrows():
            video_path = bm.DATA_DIR / "videos" / rep["video"]
            seq, fps = bm.rep_feature_sequences(video_path, int(rep["start_frame"]), int(rep["end_frame"]))
            if len(seq["angle_deg"]) == 0:
                continue
            feat = rules_module.rep_summary_features(seq, fps)
            deviations["peak_angle"].append(max(0.0, medians["peak_angle"] - feat["peak_angle"]))
            deviations["elbow_min"].append(max(0.0, medians["elbow_min"] - feat["elbow_min"]))
            deviations["trunk_max"].append(max(0.0, feat["trunk_max"] - medians["trunk_max"]))
            deviations["tempo_s"].append(abs(feat["tempo_s"] - medians["tempo_s"]) / medians["tempo_s"])
        subjects_used.append(int(subject))

    n = len(deviations["peak_angle"])
    print(f"\nn reference reps ({N_REFERENCE_REPS} per subject): {n_reference_reps}  "
          f"n deviation reps (pooled): {n}  subjects used: {subjects_used}")
    if n < MIN_N_WARNING:
        print(f"WARNING: n={n} < {MIN_N_WARNING} -- these tolerances are derived from very few reps "
              "and should be treated with proportional caution.")
    if n == 0:
        print("No deviation reps available -- nothing written.")
        return

    print("\n| feature | n | p95 one-sided deviation |")
    print("|---|---|---|")
    p95 = {}
    for feat, vals in deviations.items():
        p95[feat] = float(np.percentile(vals, 95))
        print(f"| {feat} | {len(vals)} | {p95[feat]:.3f} |")

    absolute_config = json.loads(ABSOLUTE_FALLBACK_PATH.read_text())

    config = {
        "exercise": "shoulder_abduction",
        "mode": "reference_relative",
        "notice": "Derived automatically from REHAB24-6 CALIBRATION subjects, front view only (see "
                  "derived_from/subjects below) -- NOT the live shipped config. A licensed "
                  f"physiotherapist must still review before use. n={n} deviation reps -- "
                  f"{'BELOW' if n < MIN_N_WARNING else 'at/above'} the {MIN_N_WARNING}-rep caution threshold.",
        "derived_from": "REHAB24-6 calibration subjects (front view only)",
        "landmark_space": landmark_space,
        "n_reference_reps": n_reference_reps,
        "n_deviation_reps": n,
        "subjects": subjects_used,
        "date": date.today().isoformat(),
        "commit": commit,
        "per_subject_reference_medians": per_subject_reference,
        "tol_peak_deg": {"value": round(p95["peak_angle"], 1), "status": "derived",
                          "comment": "p95 of max(0, ref_median - measured), pooled over calibration subjects' remaining correct reps"},
        "tol_elbow_deg": {"value": round(p95["elbow_min"], 1), "status": "derived",
                           "comment": "p95 of max(0, ref_median - measured)"},
        "tol_trunk_deg": {"value": round(p95["trunk_max"], 1), "status": "derived",
                           "comment": "p95 of max(0, measured - ref_median)"},
        "tol_tempo_frac": {"value": round(p95["tempo_s"], 3), "status": "derived",
                            "comment": "p95 of |measured - ref_median| / ref_median"},
        # absolute-mode fallback, copied from the already-derived front-view absolute config --
        # score_rep() needs these whenever reference_relative mode has no reference supplied.
        "target_peak_deg": absolute_config["target_peak_deg"],
        "tolerance_deg": absolute_config["tolerance_deg"],
        "min_elbow_straight_deg": absolute_config["min_elbow_straight_deg"],
        "max_trunk_lean_deg": absolute_config["max_trunk_lean_deg"],
        "tempo_min_s": absolute_config["tempo_min_s"],
        "tempo_max_s": absolute_config["tempo_max_s"],
        "cue_cooldown_s": absolute_config["cue_cooldown_s"],
        "min_valid_fraction": absolute_config["min_valid_fraction"],
        "cues": absolute_config["cues"],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(config, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
