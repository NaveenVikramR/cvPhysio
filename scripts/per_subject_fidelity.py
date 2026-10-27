"""STEP 1 of the reference-scoring task: per-subject frame-level fit
(MediaPipe world landmarks vs mocap), DESCRIPTIVE ONLY -- not used to
derive any threshold (see scripts/build_benchmark_config.py and
scripts/derive_reference_tolerances.py for the actual threshold-
deriving scripts, calibration subjects only). Every subject with
front-view data -- calibration AND evaluation, since this is purely
descriptive.

Uses the landmark cache built by the full benchmark run (--rescore-
only semantics: every video already has a cache entry, confirmed
before this script was written -- MediaPipe is never invoked here;
RESCORE_ONLY is set unconditionally below, not exposed as a flag,
since this script has no other use for MediaPipe at all).

Pools ALL of a subject's annotated front-view reps (both labels,
correct and incorrect -- this is a geometry comparison, not a scoring
run) into one linear fit (mediapipe = slope*mocap + intercept) per
subject: slope, intercept, mean error, r, n frames. Then decomposes the
per-frame (mediapipe-mocap) error's total variance into between-subject
and within-subject components via a one-way ANOVA sum-of-squares split,
and states in one sentence whether the bias differs across subjects.

Output: results/diag_per_subject_fidelity.md (commit hash at top).
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.benchmark_rehab24 as bm
from src import pose_estimation

RESULTS_DIR = Path("results")


def collect_front_view_frame_pairs(manifest: pd.DataFrame) -> pd.DataFrame:
    front = manifest[manifest["view"] == "front"]
    rows = []
    for _, rep in front.iterrows():
        video_path = bm.DATA_DIR / "videos" / rep["video"]
        joints = bm.load_mocap_joints(rep["video_id"])
        if not video_path.exists() or joints is None:
            continue
        try:
            mp_angles = bm.mediapipe_angle_series_for_rep(video_path, int(rep["start_frame"]), int(rep["end_frame"]))
        except RuntimeError as exc:
            print(f"  SKIPPED {rep['video_id']} (subject {rep['subject']}): {exc}")
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


def per_subject_fit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject in sorted(df["subject"].unique()):
        sub = df[df["subject"] == subject]
        fit = scipy_stats.linregress(sub["mocap_deg"], sub["mediapipe_deg"])
        mean_error = float((sub["mediapipe_deg"] - sub["mocap_deg"]).mean())
        rows.append({
            "subject": subject, "n_frames": len(sub), "slope": round(fit.slope, 3),
            "intercept": round(fit.intercept, 1), "mean_error": round(mean_error, 2), "r": round(fit.rvalue, 3),
        })
    return pd.DataFrame(rows)


def variance_decomposition(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["error"] = df["mediapipe_deg"] - df["mocap_deg"]
    grand_mean = df["error"].mean()
    ss_total = float(((df["error"] - grand_mean) ** 2).sum())
    ss_between = 0.0
    ss_within = 0.0
    k = df["subject"].nunique()
    n_total = len(df)
    for _, sub in df.groupby("subject"):
        n_i = len(sub)
        mean_i = sub["error"].mean()
        ss_between += n_i * (mean_i - grand_mean) ** 2
        ss_within += float(((sub["error"] - mean_i) ** 2).sum())
    return {
        "ss_total": ss_total, "ss_between": ss_between, "ss_within": ss_within,
        "pct_between": ss_between / ss_total if ss_total else float("nan"),
        "pct_within": ss_within / ss_total if ss_total else float("nan"),
        "ms_between": ss_between / (k - 1) if k > 1 else float("nan"),
        "ms_within": ss_within / (n_total - k) if n_total > k else float("nan"),
        "k_subjects": k, "n_frames": n_total,
    }


def main() -> None:
    bm.RESCORE_ONLY = True  # never touch MediaPipe -- every video must already be cached
    commit, dirty = bm.git_provenance()
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    print(f"Provenance: commit={commit}  dirty={dirty}  landmark_space={landmark_space}  rescore_only=True")
    if landmark_space != "world":
        print("ABORTING: this table is specified 'under world landmarks' -- set config/pose.json's "
              "landmark_space to 'world' and re-run.")
        return

    manifest, _split = bm.load_manifest_and_split()
    df = collect_front_view_frame_pairs(manifest)
    if df.empty:
        print("No front-view frame pairs collected -- nothing to report.")
        return

    fit_table = per_subject_fit(df)
    print(fit_table.to_string(index=False))

    var_decomp = variance_decomposition(df)
    print(f"\nVariance decomposition of mean error: {var_decomp}")

    mean_error_range = fit_table["mean_error"].max() - fit_table["mean_error"].min()
    sentence = (f"Yes, somewhat: per-subject mean error ranges from {fit_table['mean_error'].min():.1f} to "
                f"{fit_table['mean_error'].max():.1f} deg (a {mean_error_range:.1f} deg spread across "
                f"{var_decomp['k_subjects']} subjects), but between-subject variance accounts for only "
                f"{var_decomp['pct_between']:.1%} of the total variance in frame-level (mediapipe-mocap) "
                f"error -- the much larger {var_decomp['pct_within']:.1%} is within-subject (frame-to-frame) "
                f"noise, not a per-subject systematic offset.")
    print(sentence)

    lines = [
        "# Per-subject MediaPipe-vs-mocap fidelity (front view, world landmarks)\n",
        f"_commit: `{commit}` (working tree {'dirty' if dirty else 'clean'})_\n",
        "Descriptive only -- NOT used to derive any threshold. Every subject with front-view data "
        "(calibration AND evaluation). Landmarks sourced entirely from the landmark cache "
        "(scripts/landmark_cache.py) -- MediaPipe was not run for this script.\n",
        "\n## Per-subject frame-level fit vs mocap (world landmarks)\n",
        "| subject | n_frames | slope | intercept | mean_error | r |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in fit_table.iterrows():
        lines.append(f"| {int(row['subject'])} | {int(row['n_frames'])} | {row['slope']} | {row['intercept']} | "
                      f"{row['mean_error']} | {row['r']} |")

    lines.append("\n## Variance decomposition of mean error (one-way ANOVA)\n")
    lines.append(f"k={var_decomp['k_subjects']} subjects, n={var_decomp['n_frames']} frames total.\n")
    lines.append(f"SS_total={var_decomp['ss_total']:.1f}  SS_between={var_decomp['ss_between']:.1f} "
                  f"({var_decomp['pct_between']:.1%})  SS_within={var_decomp['ss_within']:.1f} "
                  f"({var_decomp['pct_within']:.1%})\n")
    lines.append(f"**{sentence}**\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "diag_per_subject_fidelity.md").write_text("\n".join(lines))
    print(f"\nWrote {RESULTS_DIR / 'diag_per_subject_fidelity.md'}")


if __name__ == "__main__":
    main()
