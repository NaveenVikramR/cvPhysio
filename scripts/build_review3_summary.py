"""PART B of the oracle-ceiling task: writes results/review3_summary.md,
consolidating every table produced across this project's "review 3"
evaluation work (tags review3-eval, review3-eval2, review3-eval3) into
one file. Cache-only, no MediaPipe -- every number here is either read
from an existing results/*.csv (results/rehab24_rep_detection.csv, from
run 1) or recomputed from the landmark cache + mocap .npy files via the
same functions those earlier scripts already used (scripts.
benchmark_rehab24, scripts.evaluate_reference_scoring, scripts.
evaluate_oracle_ceiling), so every table is reproducible from the same
cache without re-running MediaPipe.

Plain tables, one neutral caption per table -- no interpretation beyond
that (see the chat response for anything-suspicious/assumptions
commentary, which deliberately does NOT belong in this file).
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.benchmark_rehab24 as bm
import scripts.evaluate_oracle_ceiling as oracle
import scripts.evaluate_reference_scoring as ers
import scripts.per_subject_fidelity as psf
from src import pose_estimation
from src import rules as rules_module

RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "review3_summary.md"
REP_DETECTION_CSV = RESULTS_DIR / "rehab24_rep_detection.csv"
REFERENCE_CONFIG_PATH = Path("config/exercises/benchmarks/rehab24_abduction_reference.json")

RUN1_SOURCE = "run 1 (results/rehab24_benchmark.md), commit `10774e0`, tag `review3-eval`"
RUN2_SOURCE = "run 2 (results/diag_reference_evaluation.md), commit `ba90f19`, tag `review3-eval2`"
ORACLE_SOURCE = "oracle (results/diag_oracle_ceiling.md), this commit, tag `review3-eval3`"


def collect_view_frame_pairs(manifest: pd.DataFrame, view: str) -> pd.DataFrame:
    sub = manifest[manifest["view"] == view]
    rows = []
    for _, rep in sub.iterrows():
        video_path = bm.DATA_DIR / "videos" / rep["video"]
        joints = bm.load_mocap_joints(rep["video_id"])
        if not video_path.exists() or joints is None:
            continue
        try:
            mp_angles = bm.mediapipe_angle_series_for_rep(video_path, int(rep["start_frame"]), int(rep["end_frame"]))
        except RuntimeError as exc:
            print(f"  SKIPPED {rep['video_id']} ({view}, subject {rep['subject']}): {exc}")
            continue
        mocap_angles = [
            bm.mocap_shoulder_trunk_angle(joints[f]) for f in range(int(rep["start_frame"]), int(rep["end_frame"]) + 1)
            if f < len(joints)
        ]
        n = min(len(mp_angles), len(mocap_angles))
        for i in range(n):
            if mp_angles[i] is not None:
                rows.append({"subject": int(rep["subject"]), "mediapipe_deg": mp_angles[i], "mocap_deg": mocap_angles[i]})
    return pd.DataFrame(rows)


def table1_measurement(manifest: pd.DataFrame) -> list[str]:
    front_df = collect_view_frame_pairs(manifest, "front")
    profile_df = collect_view_frame_pairs(manifest, "profile")

    lines = ["## 1. Measurement: per-subject front-view fidelity (world landmarks vs mocap)\n",
             f"_Source: scripts/per_subject_fidelity.py, {ORACLE_SOURCE}._\n",
             "| subject | n_frames | slope | intercept | mean_error | r |", "|---|---|---|---|---|---|"]
    fit_table = psf.per_subject_fit(front_df)
    for _, row in fit_table.iterrows():
        lines.append(f"| {int(row['subject'])} | {int(row['n_frames'])} | {row['slope']} | {row['intercept']} | "
                      f"{row['mean_error']} | {row['r']} |")

    front_r = front_df["mediapipe_deg"].corr(front_df["mocap_deg"]) if not front_df.empty else float("nan")
    profile_r = profile_df["mediapipe_deg"].corr(profile_df["mocap_deg"]) if not profile_df.empty else float("nan")
    front_error = front_df["mediapipe_deg"] - front_df["mocap_deg"]
    mae_after_mean_removal = (front_error - front_error.mean()).abs().mean()

    lines.append("\n_Caption: pooled correlation by view, and front-view MAE after removing the pooled mean error._\n")
    lines.append("| metric | value | n |")
    lines.append("|---|---|---|")
    lines.append(f"| pooled r, front view | {front_r:.3f} | {len(front_df)} |")
    lines.append(f"| pooled r, profile view | {profile_r:.3f} | {len(profile_df)} |")
    lines.append(f"| front-view MAE after removing mean error | {mae_after_mean_removal:.2f} deg | {len(front_df)} |")
    return lines


def table2_rep_detection() -> list[str]:
    df = pd.read_csv(REP_DETECTION_CSV, comment="#")
    lines = ["\n## 2. Rep detection: exact-match and within +/-1 rep, by view\n",
             f"_Source: {REP_DETECTION_CSV} ({RUN1_SOURCE})._\n",
             "| view | n recordings | exact-match (95% CI) | within +/-1 rep (95% CI) | mean signed_error | std signed_error |",
             "|---|---|---|---|---|---|"]
    for view in sorted(df["view"].unique()):
        sub = df[df["view"] == view]
        k_exact, n = int(sub["exact_match"].sum()), len(sub)
        k_within1 = int((sub["abs_error"] <= 1).sum())
        lines.append(f"| {view} | {n} | {bm.rate_with_ci(k_exact, n)} | {bm.rate_with_ci(k_within1, n)} | "
                      f"{sub['signed_error'].mean():+.2f} | {sub['signed_error'].std():.2f} |")
    return lines


def table3_scorer_vs_labels(manifest: pd.DataFrame, split: dict, commit: str, landmark_space: str) -> list[str]:
    bm.RESCORE_ONLY = True
    config_benchmark = json.loads(bm.BENCHMARK_CONFIG_PATH.read_text())
    config_demo = json.loads(bm.DEMO_CONFIG_PATH.read_text())
    config_reference = json.loads(REFERENCE_CONFIG_PATH.read_text())

    run1_benchmark = bm.part_c_rules_vs_labels_front(manifest, split, config_benchmark)
    run1_demo = bm.part_c_rules_vs_labels_front(manifest, split, config_demo)
    run2 = ers.build_run2_rows(manifest, split, config_reference)

    config_oracle_abs = oracle.derive_absolute_config(manifest, split, commit, landmark_space)
    config_oracle_ref = oracle.derive_reference_relative_config(manifest, split, commit, landmark_space)
    for key in ("target_peak_deg", "tolerance_deg", "min_elbow_straight_deg", "max_trunk_lean_deg",
                "tempo_min_s", "tempo_max_s"):
        config_oracle_ref[key] = config_oracle_abs[key]
    oracle_abs_df = oracle.evaluate_oracle_absolute(manifest, split, config_oracle_abs)
    oracle_ref_df = oracle.evaluate_oracle_reference(manifest, split, config_oracle_ref)

    lines = ["\n## 3. Scorer vs labels: one row per run (evaluation subjects, front view)\n",
             f"_Source: run 1/run 2 reproduced + oracle computed fresh from cache in this script "
             f"({RUN1_SOURCE}; {RUN2_SOURCE}; {ORACLE_SOURCE})._\n",
             "| run | false-fail rate | detection rate | balanced accuracy | insufficient_data rate |",
             "|---|---|---|---|---|"]
    for label, df in (
        ("run 1: benchmark config (absolute, MediaPipe)", run1_benchmark),
        ("run 1: demo config (absolute, MediaPipe)", run1_demo),
        ("run 2: reference_relative (MediaPipe)", run2),
        ("oracle: absolute (mocap)", oracle_abs_df),
        ("oracle: reference_relative (mocap)", oracle_ref_df),
    ):
        if df.empty:
            lines.append(f"| {label} | N/A | N/A | N/A | N/A |")
            continue
        r = bm.scorer_rates(df)
        ba = oracle.balanced_accuracy(r)
        lines.append(f"| {label} | {bm.rate_with_ci(*r['false_fail'])} | {bm.rate_with_ci(*r['detection'])} | "
                      f"{f'{ba:.1%}' if ba is not None else 'N/A'} | {bm.rate_with_ci(*r['insufficient'])} |")
    return lines


def collect_rep_features_all_labels(manifest: pd.DataFrame, view: str = "front") -> pd.DataFrame:
    sub = manifest[manifest["view"] == view]
    rows = []
    for _, rep in sub.iterrows():
        video_path = bm.DATA_DIR / "videos" / rep["video"]
        try:
            seq_mp, fps = bm.rep_feature_sequences(video_path, int(rep["start_frame"]), int(rep["end_frame"]))
        except RuntimeError as exc:
            print(f"  SKIPPED (mediapipe) {rep['video_id']}: {exc}")
            continue
        if len(seq_mp["angle_deg"]) == 0:
            continue
        feat_mp = rules_module.rep_summary_features(seq_mp, fps)

        seq_mocap = oracle.mocap_rep_sequence(rep)
        if seq_mocap is None or len(seq_mocap["angle_deg"]) == 0:
            continue
        feat_mocap = rules_module.rep_summary_features(seq_mocap, oracle.fps_for_rep(rep))

        rows.append({
            "subject": int(rep["subject"]), "label": rep["label"],
            "label_correct": 1 if rep["label"] == "correct" else 0,
            "mp_peak_angle": feat_mp["peak_angle"], "mp_elbow_min": feat_mp["elbow_min"], "mp_trunk_max": feat_mp["trunk_max"],
            "mocap_peak_angle": feat_mocap["peak_angle"], "mocap_elbow_min": feat_mocap["elbow_min"], "mocap_trunk_max": feat_mocap["trunk_max"],
        })
    return pd.DataFrame(rows)


def table4_per_subject_auc(manifest: pd.DataFrame) -> list[str]:
    df = collect_rep_features_all_labels(manifest, view="front")
    lines = ["\n## 4. Per-subject AUC (correct vs incorrect), mocap vs world landmarks, front view\n",
             "_Source: this script, computed fresh from the landmark cache + mocap .npy files._\n",
             "| subject | n correct | n incorrect | feature | AUC (mocap) | AUC (world landmarks) |",
             "|---|---|---|---|---|---|"]
    feature_cols = (("peak_angle", "mocap_peak_angle", "mp_peak_angle"),
                     ("elbow_min", "mocap_elbow_min", "mp_elbow_min"),
                     ("trunk_max", "mocap_trunk_max", "mp_trunk_max"))
    for subject in sorted(df["subject"].unique()):
        sub = df[df["subject"] == subject]
        n_correct = int((sub["label"] == "correct").sum())
        n_incorrect = int((sub["label"] == "incorrect").sum())
        inconclusive = n_correct < 10 or n_incorrect < 10
        tag = " (inconclusive, n<10 per label)" if inconclusive else ""
        for feat, mocap_col, mp_col in feature_cols:
            if sub["label_correct"].nunique() < 2:
                auc_mocap_s = auc_world_s = "N/A"
            else:
                auc_mocap_s = f"{roc_auc_score(sub['label_correct'], sub[mocap_col]):.3f}{tag}"
                auc_world_s = f"{roc_auc_score(sub['label_correct'], sub[mp_col]):.3f}{tag}"
            lines.append(f"| {subject} | {n_correct} | {n_incorrect} | {feat} | {auc_mocap_s} | {auc_world_s} |")
    return lines


def table5_limitations() -> list[str]:
    return [
        "\n## 5. Limitations\n",
        "_Plain list, no scores -- see each cited source for the number behind each item._\n",
        "- Absolute benchmark config (config/exercises/benchmarks/rehab24_abduction.json) derived from n=15 "
        "calibration-subject correct front-view reps (subjects 1, 2 only -- subject 6 has no front-view "
        "recording).",
        "- Reference-relative tolerances (config/exercises/benchmarks/rehab24_abduction_reference.json) "
        "derived from n=9 deviation reps (subject 1: 7, subject 2: 2), below the 30-rep caution threshold "
        "used when deriving them.",
        "- Oracle (mocap) configs derived from the same n=15 (absolute) / n=9 (reference) calibration reps "
        "as above -- the oracle isolates MediaPipe's measurement error, not the small-sample derivation risk, "
        "which is shared with the non-oracle configs.",
        "- The derived reference-relative tol_elbow_deg (0.4 deg, MediaPipe) is far tighter than the oracle's "
        "mocap-derived tol_elbow_deg (7.7 deg) on the same 9 reps -- an artefact of deriving a tight "
        "tolerance from very few, likely unusually self-consistent MediaPipe measurements, not a property "
        "of real elbow-straightness variation.",
        "- cam18 (profile view) is excluded from every threshold derivation and from the front-view-only "
        "evaluations (runs 1/2/oracle) -- rep detection and fidelity are reported for it in tables 1-2 "
        "above, but it never informs a threshold.",
        "- No itemized codebook of which specific mistake each subject performed exists in the released "
        "REHAB24-6 files (Segmentation.csv/.txt, joints_names.txt) -- \"incorrect\" is a holistic "
        "physiotherapist judgment, confirmed absent in an earlier diagnostic (results/diag_rehab24.md Part 4).",
        "- tempo_s is not independently validated as a feature: both mocap_tempo_s and mediapipe-derived "
        "tempo_s are computed from the SAME annotated (start_frame, end_frame) span / fps, so any tempo "
        "agreement reported anywhere in this project reflects shared annotation boundaries, not an "
        "independent measurement of rep duration.",
    ]


def main() -> None:
    commit, dirty = bm.git_provenance()
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    print(f"Provenance: commit={commit}  dirty={dirty}  landmark_space={landmark_space}")

    manifest, split = bm.load_manifest_and_split()

    print("[1/5] Measurement table...")
    t1 = table1_measurement(manifest)
    print("[2/5] Rep detection table...")
    t2 = table2_rep_detection()
    print("[3/5] Scorer vs labels (one row per run)...")
    t3 = table3_scorer_vs_labels(manifest, split, commit, landmark_space)
    print("[4/5] Per-subject AUC table...")
    t4 = table4_per_subject_auc(manifest)
    print("[5/5] Limitations list...")
    t5 = table5_limitations()

    lines = [
        "# Review 3 evaluation -- consolidated summary\n",
        f"_commit: `{commit}` (working tree {'dirty' if dirty else 'clean'})  tag: `review3-eval3`_\n",
        f"_landmark_space: `{landmark_space}`_\n",
        "Consolidates results from tags `review3-eval` (run 1), `review3-eval2` (run 2), and `review3-eval3` "
        "(oracle ceiling, this tag). Every table below cites its own source commit/tag. Plain tables, one "
        "neutral caption per table.\n",
    ]
    lines.extend(t1)
    lines.extend(t2)
    lines.extend(t3)
    lines.extend(t4)
    lines.extend(t5)

    RESULTS_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
