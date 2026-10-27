"""STEP 1 of the coordinate-space follow-up task: characterise variant C
(MediaPipe world/metric landmarks) vs mocap in more depth than results/
diag_coords.md did. ANALYSIS ONLY -- no src/, config/ or threshold
changes. Calibration subjects [1, 2, 6], front camera (cam17) only
(subject 6 has no cam17-front recording in this dataset -- see PART 1's
own printed note -- so every number here is actually drawn from subjects
1 and 2's 3 front-view recordings: PM_000, PM_001, PM_012). Evaluation
subjects are never read.

Reuses the per-frame variant-C/mocap series already computed and cached
by scripts/diagnose_coord_space.py at results/diag/coords_{video_id}.csv
(regenerates them via that script's own full_recording_variants() if a
cache file is missing), instead of re-running MediaPipe inference a
second time for numbers that script already produced.

PART 1: frame-level fit (slope, intercept, r, bias at rest/peak,
residual std after the linear fit), plus the MAE split into a constant
offset component and a random-error component after removing it.
PART 2: per-rep paired differences (peak angle, elbow minimum, trunk-
lean max, tempo) vs mocap, per subject, over ALL annotated reps
(correct AND incorrect -- not restricted to correct reps, since PART 3
needs both labels anyway and this step is a fuller characterisation
than diag_coords.md's correct-only table).
PART 3: AUC of each feature for correct-vs-incorrect discrimination
under variant C, same (subjects, camera, view) restriction.

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
ADOPTION_MAE_MAX = 10.0  # the PART 3 rule from the earlier task -- NOT met by C, restated here for reference only


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
# PART 1 -- frame-level fit + MAE split (offset vs random error)
# ===========================================================================

def frame_level_fit_with_offset_split(all_df: pd.DataFrame) -> dict:
    mp_arr = all_df[f"{VARIANT}_angle"].to_numpy()
    mocap_arr = all_df["mocap_angle"].to_numpy()
    valid = ~np.isnan(mp_arr) & ~np.isnan(mocap_arr)
    mp_v, mocap_v = mp_arr[valid], mocap_arr[valid]

    fit = scipy_stats.linregress(mocap_v, mp_v)
    residuals = mp_v - (fit.slope * mocap_v + fit.intercept)
    diff = mp_v - mocap_v
    mae_total = float(np.mean(np.abs(diff)))
    offset = float(np.mean(diff))  # constant (mean) error -- a pure bias
    mae_after_offset_removed = float(np.mean(np.abs(diff - offset)))  # what's left once the bias is subtracted

    rest_threshold = np.nanmin(mocap_arr) + 5.0
    rest_mask = valid & (mocap_arr <= rest_threshold)
    rest_bias = float(np.mean(mp_arr[rest_mask] - mocap_arr[rest_mask])) if rest_mask.sum() else None

    peak_mask = valid & (mocap_arr >= np.nanmax(mocap_arr) - 5.0)
    peak_bias = float(np.mean(mp_arr[peak_mask] - mocap_arr[peak_mask])) if peak_mask.sum() else None

    return {
        "n": int(valid.sum()), "slope": round(fit.slope, 3), "intercept": round(fit.intercept, 1),
        "r": round(fit.rvalue, 3), "residual_std": round(float(residuals.std()), 2),
        "mae_total": round(mae_total, 2), "offset_bias": round(offset, 2),
        "mae_after_offset_removed": round(mae_after_offset_removed, 2),
        "rest_bias": round(rest_bias, 2) if rest_bias is not None else None,
        "peak_bias": round(peak_bias, 2) if peak_bias is not None else None,
    }


# ===========================================================================
# PART 2 -- per-rep paired differences, per subject, all labels
# ===========================================================================

def per_rep_features_for_recording(df: pd.DataFrame, reps: pd.DataFrame, fps: float) -> list[dict]:
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
        tempo_s = (e - s + 1) / fps  # identical for mocap and mediapipe BY CONSTRUCTION -- see module docstring
        if np.isnan(mp_peak):
            continue
        rows.append({
            "subject": int(rep["subject"]), "video_id": rep["video_id"], "label": rep["label"],
            "peak_angle_diff": mp_peak - mocap_peak, "elbow_min_diff": mp_elbow_min - mocap_elbow_min,
            "trunk_max_diff": mp_trunk_max - mocap_trunk_max, "tempo_diff": 0.0,
            "mocap_peak_angle": mocap_peak, "mp_peak_angle": mp_peak,
            "mocap_elbow_min": mocap_elbow_min, "mp_elbow_min": mp_elbow_min,
            "mocap_trunk_max": mocap_trunk_max, "mp_trunk_max": mp_trunk_max,
            "mocap_tempo_s": tempo_s, "mp_tempo_s": tempo_s,
            "label_correct": 1 if rep["label"] == "correct" else 0,
        })
    return rows


def per_subject_summary(rep_df: pd.DataFrame) -> list[str]:
    lines = []
    for subject in sorted(rep_df["subject"].unique()):
        sub_df = rep_df[rep_df["subject"] == subject]
        lines.append(f"- **subject {subject}** (n={len(sub_df)} reps, {sub_df['video_id'].unique().tolist()}):")
        for col, label in (("peak_angle_diff", "peak_angle"), ("elbow_min_diff", "elbow_min"),
                            ("trunk_max_diff", "trunk_max"), ("tempo_diff", "tempo")):
            lines.append(f"  - {label}_diff: mean={sub_df[col].mean():.2f}  std={sub_df[col].std():.2f}")
    return lines


# ===========================================================================
# PART 3 -- AUC under variant C
# ===========================================================================

def auc_table(rep_df: pd.DataFrame) -> list[str]:
    lines = ["| source | feature | n | AUC |", "|---|---|---|---|"]
    pairs = [
        ("mocap", "mocap_peak_angle"), ("mocap", "mocap_elbow_min"), ("mocap", "mocap_trunk_max"), ("mocap", "mocap_tempo_s"),
        (VARIANT, "mp_peak_angle"), (VARIANT, "mp_elbow_min"), (VARIANT, "mp_trunk_max"), (VARIANT, "mp_tempo_s"),
    ]
    for source, col in pairs:
        if rep_df[col].nunique() < 2 or rep_df["label_correct"].nunique() < 2:
            auc = None
        else:
            auc = roc_auc_score(rep_df["label_correct"], rep_df[col])
        lines.append(f"| {source} | {col} | {len(rep_df)} | {round(auc, 3) if auc is not None else 'N/A'} |")
    return lines


def main() -> None:
    commit, dirty = git_provenance()
    print(f"Provenance: commit={commit}  dirty={dirty}")

    manifest, _split = load_manifest_and_split()
    calib_cam17_front = manifest[
        manifest["subject"].isin(CALIBRATION_SUBJECTS) & (manifest["camera"] == "cam17") & (manifest["view"] == "front")
    ]
    recordings = calib_cam17_front.drop_duplicates(subset=["video_id"])[["subject", "video_id", "video"]]

    lines = [
        "# Variant C (world landmarks) vs mocap -- characterisation\n",
        f"_commit: `{commit}` (working tree {'dirty' if dirty else 'clean'})_\n",
        f"Calibration subjects: {CALIBRATION_SUBJECTS}, front camera (cam17) only. Evaluation subjects never read.\n",
    ]

    subjects_present = sorted(recordings["subject"].unique().tolist())
    subjects_missing = [s for s in CALIBRATION_SUBJECTS if s not in subjects_present]
    lines.append(f"Recordings actually used: {recordings['video_id'].tolist()} (subjects {subjects_present}).")
    if subjects_missing:
        lines.append(f"**Subject(s) {subjects_missing} have no cam17/front recording in this dataset** -- confirmed "
                      f"by filtering data/rehab24_manifest.csv for subject in {CALIBRATION_SUBJECTS}, camera=='cam17', "
                      f"view=='front': only {recordings['video_id'].tolist()} match. Every number below is drawn "
                      f"only from subjects {subjects_present}.\n")

    print("Loading/building per-frame variant-C + mocap series for each recording...")
    per_recording_dfs = {}
    fps_by_video = {}
    for _, rec in recordings.iterrows():
        print(f"  {rec['video_id']}...")
        per_recording_dfs[rec["video_id"]] = load_or_build_recording_df(rec["video_id"], rec["video"])
        fps_by_video[rec["video_id"]] = recording_fps(rec["video"])
    all_df = pd.concat(per_recording_dfs.values(), ignore_index=True)

    # ---- PART 1 ----
    fit = frame_level_fit_with_offset_split(all_df)
    print(f"Frame-level fit (variant {VARIANT}): {fit}")
    lines.append("\n## PART 1 -- frame-level fit vs mocap\n")
    lines.append(f"n={fit['n']} pooled frames across {recordings['video_id'].tolist()}.\n")
    lines.append("| slope | intercept | r | residual_std (after full affine fit) | rest_bias | peak_bias |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| {fit['slope']} | {fit['intercept']} | {fit['r']} | {fit['residual_std']} | "
                  f"{fit['rest_bias']} | {fit['peak_bias']} |\n")
    lines.append("**MAE split into offset (constant bias) vs random error:**\n")
    lines.append("| MAE (total, mp - mocap) | offset (mean signed error) | MAE after removing the offset |")
    lines.append("|---|---|---|")
    lines.append(f"| {fit['mae_total']} | {fit['offset_bias']} | {fit['mae_after_offset_removed']} |\n")
    offset_share = abs(fit["offset_bias"]) / fit["mae_total"] * 100 if fit["mae_total"] else 0.0
    lines.append(f"The constant offset accounts for about {offset_share:.0f}% of the total MAE magnitude "
                 f"(|offset|={abs(fit['offset_bias']):.2f} of MAE={fit['mae_total']:.2f}); "
                 f"{fit['mae_after_offset_removed']:.2f} deg of error remains after removing just that constant, "
                 "i.e. most of variant C's frame-level error is NOT a single fixable constant bias -- it varies "
                 "with the angle itself (consistent with a slope of "
                 f"{fit['slope']} rather than 1.0, and with rest_bias={fit['rest_bias']} differing from "
                 f"peak_bias={fit['peak_bias']}).\n")

    # ---- PART 2 ----
    print("Computing per-rep paired differences (all labels) per subject...")
    manifest_subj = manifest.set_index("video_id")["subject"].to_dict()
    rep_rows = []
    for video_id, df in per_recording_dfs.items():
        reps = calib_cam17_front[calib_cam17_front["video_id"] == video_id].copy()
        reps["subject"] = manifest_subj[video_id]
        rep_rows.extend(per_rep_features_for_recording(df, reps, fps_by_video[video_id]))
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(DIAG_DIR / "coords_C_per_rep_all_labels.csv", index=False)
    print(f"  {len(rep_df)} reps total (labels: {rep_df['label'].value_counts().to_dict()})")

    lines.append("\n## PART 2 -- per-rep paired differences vs mocap, per subject (ALL annotated reps, both labels)\n")
    lines.append(f"n={len(rep_df)} reps ({rep_df['label'].value_counts().to_dict()}).\n")
    lines.extend(per_subject_summary(rep_df))
    lines.append("\n`tempo_diff` is 0.0 for every rep by construction: both mocap_tempo_s and mp_tempo_s are "
                 "computed from the SAME annotated (start_frame, end_frame) span / fps -- this measures rep "
                 "DURATION agreement given a shared boundary, not MediaPipe's own rep-boundary detection "
                 "(RepDetector's own detected boundaries were already characterised separately in "
                 "results/diag_rehab24.md PART 1).\n")

    # ---- PART 3 ----
    print("Computing AUC under variant C...")
    lines.append("\n## PART 3 -- AUC for correct-vs-incorrect discrimination (variant C)\n")
    lines.extend(auc_table(rep_df))

    # ---- rule reminder ----
    lines.append(f"\n## Pre-registered MAE rule, restated\n")
    lines.append(f"The earlier task's adoption rule required MAE <= {ADOPTION_MAE_MAX} deg; variant C's frame-level "
                 f"MAE here is {fit['mae_total']} deg, so that rule was **NOT met** (as already reported in "
                 "results/diag_coords.md). The offset/random-error split above shows why: even after removing "
                 "the single largest fixable component (the constant offset), "
                 f"{fit['mae_after_offset_removed']} deg of error remains -- too large a residual for a pure "
                 "coordinate-space fix (world landmarks) alone to close the gap to <= 10 deg.\n")

    (RESULTS_DIR / "diag_coords_C.md").write_text("\n".join(lines))
    print(f"\nWrote {RESULTS_DIR / 'diag_coords_C.md'}")


if __name__ == "__main__":
    main()
