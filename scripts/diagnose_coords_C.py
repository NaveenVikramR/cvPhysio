"""Reconciles results/diag_coords_C.md: the first version of this script
computed "peak_bias" from a GLOBAL per-recording excursion threshold
(mocap_angle within 5 deg of that recording's own max), which is a
DIFFERENT definition than scripts/diagnose_coord_space.py's original
per-rep peak_bias (one point per annotated rep, at that rep's own true
mocap-angle maximum, pooled over reps) -- that mismatch, not a labelling
error in the earlier offset number, is why -14.86 (global mean error)
didn't sit between -9.51 (rest) and the first version's +3.35 (global-
threshold "peak"): they were never measuring the same frame population.
This version uses ONE definition throughout, matching the original
diagnose_coord_space.py: rest = frames within 5 deg of the pooled
recording-wide mocap-angle minimum; peak = one frame per annotated rep,
at that rep's own mocap-angle maximum within its [start_frame,end_frame]
span (ALL annotated reps, correct AND incorrect -- camera/view-filtered
only, same as the original). See PART 1's own explanation paragraph.

ANALYSIS ONLY -- no src/, config/ or threshold changes. Calibration
subjects [1, 2, 6], front camera (cam17) only; subject 6 has no cam17/
front recording in this dataset (confirmed below), so every number here
is drawn only from subjects 1 and 2. Evaluation subjects are never read.
Does not start any benchmark.

PART 1: ONE table (fit slope/intercept, mean error, error std, MAE, MAE
after subtracting the mean error, rest bias, peak bias, with n stated
for each), plus the reconciliation explanation above.
PART 2: for peak angle, elbow min, trunk max, under mocap AND under
variant C: reps per subject per label, per-subject AUC (correct vs
incorrect, computed WITHIN each subject separately, not pooled across
subjects) and the mean of those per-subject AUCs -- flags any subject
with only one label (AUC undefined for that subject/feature).

Output: results/diag_coords_C.md (commit hash + dirty flag at the top).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_rehab24 import DATA_DIR, git_provenance, load_manifest_and_split
from scripts.diagnose_coord_space import full_recording_variants
from scripts.diagnose_rehab24 import CALIBRATION_SUBJECTS

RESULTS_DIR = Path("results")
DIAG_DIR = RESULTS_DIR / "diag"
VARIANT = "C_world"
FEATURES = ("peak_angle", "elbow_min", "trunk_max")


def load_or_build_recording_df(video_id: str, video_filename: str) -> pd.DataFrame:
    cache_path = DIAG_DIR / f"coords_{video_id}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)
    video_path = DATA_DIR / "videos" / video_filename
    df = full_recording_variants(video_path, video_id)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def recording_fps(video_filename: str) -> float:
    cap = cv2.VideoCapture(str(DATA_DIR / "videos" / video_filename))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


# ===========================================================================
# PART 1 -- ONE reconciled frame-level table
# ===========================================================================

def frame_level_table(all_df: pd.DataFrame, annotated_reps: pd.DataFrame) -> dict:
    mp_arr = all_df[f"{VARIANT}_angle"].to_numpy()
    mocap_arr = all_df["mocap_angle"].to_numpy()
    valid = ~np.isnan(mp_arr) & ~np.isnan(mocap_arr)
    mp_v, mocap_v = mp_arr[valid], mocap_arr[valid]

    fit = scipy_stats.linregress(mocap_v, mp_v)
    diff = mp_v - mocap_v
    mean_error = float(np.mean(diff))
    error_std = float(np.std(diff))
    mae = float(np.mean(np.abs(diff)))
    mae_after_mean_removed = float(np.mean(np.abs(diff - mean_error)))

    # rest: pooled-recording definition, as in diagnose_coord_space.py
    rest_threshold = np.nanmin(mocap_arr) + 5.0
    rest_mask = valid & (mocap_arr <= rest_threshold)
    rest_bias = float(np.mean(mp_arr[rest_mask] - mocap_arr[rest_mask])) if rest_mask.sum() else None
    rest_n = int(rest_mask.sum())

    # peak: ONE frame per annotated rep (its own mocap-angle maximum within
    # [start_frame, end_frame]) -- matches diagnose_coord_space.py exactly,
    # NOT a global per-recording excursion threshold (that was the bug in
    # the first version of this script -- see module docstring).
    peak_diffs = []
    for _, rep in annotated_reps.iterrows():
        s, e = int(rep["start_frame"]), min(int(rep["end_frame"]), len(mocap_arr) - 1)
        if e <= s:
            continue
        seg = mocap_arr[s:e + 1]
        if np.all(np.isnan(seg)):
            continue
        peak_local = s + int(np.nanargmax(seg))
        if peak_local < len(mp_arr) and not np.isnan(mp_arr[peak_local]):
            peak_diffs.append(mp_arr[peak_local] - mocap_arr[peak_local])
    peak_bias = float(np.mean(peak_diffs)) if peak_diffs else None
    peak_n = len(peak_diffs)

    return {
        "n": int(valid.sum()), "slope": round(fit.slope, 3), "intercept": round(fit.intercept, 1),
        "mean_error": round(mean_error, 2), "error_std": round(error_std, 2),
        "mae": round(mae, 2), "mae_after_mean_removed": round(mae_after_mean_removed, 2),
        "rest_bias": round(rest_bias, 2) if rest_bias is not None else None, "rest_n": rest_n,
        "peak_bias": round(peak_bias, 2) if peak_bias is not None else None, "peak_n": peak_n,
    }


# ===========================================================================
# PART 2 -- per-rep features, per-subject AUC (not pooled)
# ===========================================================================

def per_rep_features_for_recording(df: pd.DataFrame, reps: pd.DataFrame) -> list[dict]:
    mocap_angle, mocap_elbow, mocap_trunk = df["mocap_angle"].to_numpy(), df["mocap_elbow"].to_numpy(), df["mocap_trunk"].to_numpy()
    mp_angle, mp_elbow, mp_trunk = df[f"{VARIANT}_angle"].to_numpy(), df[f"{VARIANT}_elbow"].to_numpy(), df[f"{VARIANT}_trunk"].to_numpy()
    rows = []
    for _, rep in reps.iterrows():
        s, e = int(rep["start_frame"]), min(int(rep["end_frame"]), len(df) - 1)
        if e <= s:
            continue
        mocap_peak, mp_peak = np.nanmax(mocap_angle[s:e + 1]), np.nanmax(mp_angle[s:e + 1])
        mocap_elbow_min, mp_elbow_min = np.nanmin(mocap_elbow[s:e + 1]), np.nanmin(mp_elbow[s:e + 1])
        mocap_trunk_max, mp_trunk_max = np.nanmax(mocap_trunk[s:e + 1]), np.nanmax(mp_trunk[s:e + 1])
        if np.isnan(mp_peak):
            continue
        rows.append({
            "subject": int(rep["subject"]), "video_id": rep["video_id"], "label": rep["label"],
            "label_correct": 1 if rep["label"] == "correct" else 0,
            "mocap_peak_angle": mocap_peak, "mp_peak_angle": mp_peak,
            "mocap_elbow_min": mocap_elbow_min, "mp_elbow_min": mp_elbow_min,
            "mocap_trunk_max": mocap_trunk_max, "mp_trunk_max": mp_trunk_max,
        })
    return rows


def reps_per_subject_per_label(rep_df: pd.DataFrame) -> list[str]:
    counts = rep_df.groupby(["subject", "label"]).size().unstack(fill_value=0)
    lines = ["| subject | correct | incorrect |", "|---|---|---|"]
    for subject in sorted(rep_df["subject"].unique()):
        row = counts.loc[subject]
        lines.append(f"| {subject} | {row.get('correct', 0)} | {row.get('incorrect', 0)} |")
    return lines


def per_subject_auc(rep_df: pd.DataFrame) -> tuple[list[str], dict]:
    """Per-subject AUC (correct vs incorrect) computed WITHIN each
    subject's own reps only, for each feature under mocap and under
    variant C, plus the mean of those per-subject AUCs -- NOT the
    pooled-across-subjects AUC, which would let between-subject
    differences (e.g. a systematically larger range of motion for one
    subject) masquerade as the feature's within-subject discriminative
    power.
    """
    col_map = {
        "peak_angle": ("mocap_peak_angle", "mp_peak_angle"),
        "elbow_min": ("mocap_elbow_min", "mp_elbow_min"),
        "trunk_max": ("mocap_trunk_max", "mp_trunk_max"),
    }
    subjects = [int(s) for s in sorted(rep_df["subject"].unique())]
    flagged = [s for s in subjects if rep_df[rep_df["subject"] == s]["label_correct"].nunique() < 2]

    lines = ["| feature | source | " + " | ".join(f"subj {s} AUC" for s in subjects) + " | mean (within-subject) |",
              "|---|---|" + "---|" * len(subjects) + "---|"]
    means = {}
    for feature, (mocap_col, mp_col) in col_map.items():
        for source, col in (("mocap", mocap_col), (VARIANT, mp_col)):
            per_subj = []
            for subject in subjects:
                sub_df = rep_df[rep_df["subject"] == subject]
                if sub_df["label_correct"].nunique() < 2 or sub_df[col].nunique() < 2:
                    per_subj.append(None)
                else:
                    per_subj.append(roc_auc_score(sub_df["label_correct"], sub_df[col]))
            valid_vals = [v for v in per_subj if v is not None]
            mean_auc = float(np.mean(valid_vals)) if valid_vals else None
            means[(feature, source)] = mean_auc
            cells = " | ".join(f"{v:.3f}" if v is not None else "N/A (one label)" for v in per_subj)
            lines.append(f"| {feature} | {source} | {cells} | "
                          f"{round(mean_auc, 3) if mean_auc is not None else 'N/A'} |")
    if flagged:
        lines.append(f"\n**Subject(s) {flagged} have only one label present** -- AUC is undefined for them "
                      "(marked N/A above), and they're excluded from the mean-within-subject AUC.\n")
    else:
        lines.append(f"\nAll subjects ({subjects}) have both labels present -- no subject flagged.\n")
    return lines, means


def main() -> None:
    commit, dirty = git_provenance()
    print(f"Provenance: commit={commit}  dirty={dirty}")

    manifest, _split = load_manifest_and_split()
    calib_cam17_front = manifest[
        manifest["subject"].isin(CALIBRATION_SUBJECTS) & (manifest["camera"] == "cam17") & (manifest["view"] == "front")
    ]
    recordings = calib_cam17_front.drop_duplicates(subset=["video_id"])[["subject", "video_id", "video"]]

    lines = [
        "# Variant C (world landmarks) vs mocap -- reconciled characterisation\n",
        f"_commit: `{commit}` (working tree {'dirty' if dirty else 'clean'})_\n",
        f"Calibration subjects: {CALIBRATION_SUBJECTS}, front camera (cam17) only. Evaluation subjects never read. "
        "No benchmark was run for this.\n",
    ]

    subjects_present = sorted(recordings["subject"].unique().tolist())
    subjects_missing = [s for s in CALIBRATION_SUBJECTS if s not in subjects_present]
    lines.append(f"Recordings used: {recordings['video_id'].tolist()} (subjects {subjects_present}).")
    if subjects_missing:
        lines.append(f"**Subject(s) {subjects_missing} have no cam17/front recording in this dataset** -- confirmed "
                      f"by filtering data/rehab24_manifest.csv for subject in {CALIBRATION_SUBJECTS}, camera=='cam17', "
                      f"view=='front': only {recordings['video_id'].tolist()} match. Every number below is drawn "
                      f"only from subjects {subjects_present}.\n")

    print("Loading/building per-frame variant-C + mocap series for each recording...")
    per_recording_dfs = {}
    for _, rec in recordings.iterrows():
        print(f"  {rec['video_id']}...")
        per_recording_dfs[rec["video_id"]] = load_or_build_recording_df(rec["video_id"], rec["video"])
    all_df = pd.concat(per_recording_dfs.values(), ignore_index=True)

    # ---- PART 1 ----
    fit = frame_level_table(all_df, calib_cam17_front)
    print(f"Frame-level table (variant {VARIANT}): {fit}")
    lines.append("\n## PART 1 -- reconciled frame-level table\n")
    lines.append(f"n={fit['n']} pooled frames across {recordings['video_id'].tolist()}.\n")
    lines.append("| slope | intercept | mean_error (mp-mocap) | error_std | MAE | MAE after subtracting mean_error "
                  "| rest_bias (n) | peak_bias (n) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(f"| {fit['slope']} | {fit['intercept']} | {fit['mean_error']} | {fit['error_std']} | {fit['mae']} | "
                  f"{fit['mae_after_mean_removed']} | {fit['rest_bias']} (n={fit['rest_n']}) | "
                  f"{fit['peak_bias']} (n={fit['peak_n']}) |\n")
    lines.append(f"**rest frames**: pooled across all {recordings['video_id'].tolist()} frames, any frame whose "
                 "mocap_angle is within 5 deg of the pooled minimum mocap_angle across those recordings "
                 f"(n={fit['rest_n']} frames). **peak frames**: ONE frame per annotated rep (correct AND "
                 "incorrect, camera/view-filtered only -- all 25 reps), at that specific rep's own mocap_angle "
                 f"maximum within its annotated [start_frame, end_frame] span (n={fit['peak_n']} reps/frames, "
                 "one point per rep, not per-frame).\n")
    lines.append("**Reconciling the earlier inconsistency (offset -14.86 vs rest -9.51 / peak +3.35):** that was "
                 "a methodology bug in the first version of this script, not a labelling error in the earlier "
                 "mean_error number. The first version's \"peak_bias\" used a GLOBAL per-recording excursion "
                 "threshold (any frame within 5 deg of that recording's own single highest mocap_angle), which "
                 "is a different, less principled population of frames than diagnose_coord_space.py's original "
                 "per-rep peak definition (used in results/diag_coords.md) -- and could draw its handful of "
                 "frames disproportionately from whichever single rep happened to reach the highest excursion "
                 f"in a recording. This version uses that same original per-rep definition throughout: mean_error="
                 f"{fit['mean_error']} (pooled over ALL {fit['n']} frames, including the ascending/descending "
                 f"phase between rest and peak, which is most of the recording), rest_bias={fit['rest_bias']} and "
                 f"peak_bias={fit['peak_bias']} are each a SMALL SUBSET of those frames -- there's no mathematical "
                 "requirement that the pooled mean sit between them, since the remaining (non-rest, non-peak) "
                 "frames in the ascending/descending phase carry most of the weight and are not shown separately "
                 f"here. (For reference: with slope={fit['slope']} and intercept={fit['intercept']}, the affine "
                 f"fit alone predicts error = {fit['slope'] - 1:.3f}*mocap_angle + {fit['intercept']}, which is "
                 "a smooth, continuously-varying function of mocap_angle, not a single constant -- consistent "
                 "with rest_bias and peak_bias differing from each other and from the pooled mean_error.)\n")

    # ---- PART 2 ----
    print("Computing per-rep features and per-subject AUC...")
    manifest_subj = manifest.set_index("video_id")["subject"].to_dict()
    rep_rows = []
    for video_id, df in per_recording_dfs.items():
        reps = calib_cam17_front[calib_cam17_front["video_id"] == video_id].copy()
        reps["subject"] = manifest_subj[video_id]
        rep_rows.extend(per_rep_features_for_recording(df, reps))
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(DIAG_DIR / "coords_C_per_rep_reconciled.csv", index=False)
    print(f"  {len(rep_df)} reps total (labels: {rep_df['label'].value_counts().to_dict()})")

    lines.append("\n## PART 2 -- reps per subject per label, and per-subject AUC (not pooled)\n")
    lines.extend(reps_per_subject_per_label(rep_df))
    lines.append("")
    auc_lines, means = per_subject_auc(rep_df)
    lines.extend(auc_lines)
    print(f"Mean within-subject AUC: {means}")

    (RESULTS_DIR / "diag_coords_C.md").write_text("\n".join(lines))
    print(f"\nWrote {RESULTS_DIR / 'diag_coords_C.md'}")


if __name__ == "__main__":
    main()
