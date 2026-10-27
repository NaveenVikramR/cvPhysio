"""PART A of the oracle-ceiling task: scores the SAME reps with the SAME
rule structure (src.rules.score_rep(), absolute AND reference_relative
modes) but with features computed from the mocap 3D joints instead of
MediaPipe -- an UPPER-BOUND analysis using ground-truth motion-capture
features, run AFTER runs 1 (results/rehab24_benchmark.md) and 2
(results/diag_reference_evaluation.md) on the SAME evaluation subjects.
This is NOT a deployable scorer (mocap is never available at inference
time) -- it answers "how well could rules-based scoring possibly do if
MediaPipe's measurement error were zero?", isolating the rule
design/threshold question from MediaPipe's own geometric error.

Cache-only, no MediaPipe at all (mocap features need only the .npy
joint files + data/rehab24_manifest.csv; fps is read from the existing
landmark_cache sidecar JSON, never from re-opening a video). Reps
flagged mocap_erroneous=True are excluded from every mocap computation
(none were, on calibration or evaluation front-view data -- checked
and reported below, not assumed).

Derivation (calibration subjects, front view, correct reps only):
  - absolute mode: [p5, p95] per feature (same formulas as
    scripts/build_benchmark_config.py, mocap features instead of
    MediaPipe).
  - reference_relative mode: per calibration subject, reference =
    first 3 correct reps (annotated order); tolerance per feature =
    p95 of the one-sided deviation of remaining correct reps, pooled
    over calibration subjects (same method as scripts/
    derive_reference_tolerances.py, mocap features instead of
    MediaPipe, K=3).

Evaluation (evaluation subjects, front view) -- exactly as run 1/run 2:
  - oracle absolute: every front-view rep (correct + incorrect) scored
    against the mocap-derived absolute config.
  - oracle reference_relative: per evaluation subject, reference =
    that subject's own first 3 correct front-view reps (mocap
    features); remaining reps (3 reference reps excluded from every
    metric) scored against the mocap-derived reference_relative config
    with that subject's reference.

Output: results/diag_oracle_ceiling.md (commit hash, dirty flag,
landmark_space, date, and the "upper-bound, not deployable" header at
the top).
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import landmark_cache
import scripts.benchmark_rehab24 as bm
import scripts.diagnose_rehab24 as dr
from src import pose_estimation
from src import rules as rules_module

RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "diag_oracle_ceiling.md"
N_REFERENCE_REPS = 3
MIN_N_WARNING = 30


def fps_for_rep(rep) -> float:
    video_path = bm.DATA_DIR / "videos" / rep["video"]
    cached = landmark_cache.load_cache(video_path)
    if cached is None:
        raise RuntimeError(f"No landmark_cache entry for {video_path} -- cache-only, no MediaPipe fallback here.")
    return cached["fps"]


def mocap_rep_sequence(rep) -> dict | None:
    """Same {"angle_deg","elbow_angle_deg","trunk_lean_deg","valid_fraction"}
    shape rep_feature_sequences() returns, but from mocap 3D joints over
    the rep's ANNOTATED [start_frame, end_frame] span -- no confidence
    gating exists for mocap, so valid_fraction is always 1.0 (every
    frame has a joint position) once a rep isn't mocap_erroneous."""
    if bool(rep.get("mocap_erroneous", False)):
        return None
    joints = bm.load_mocap_joints(rep["video_id"])
    if joints is None:
        return None
    start_frame, end_frame = int(rep["start_frame"]), int(rep["end_frame"])
    end_frame = min(end_frame, len(joints) - 1)
    if end_frame < start_frame:
        return None
    angle = [bm.mocap_shoulder_trunk_angle(joints[f]) for f in range(start_frame, end_frame + 1)]
    elbow = [dr.mocap_elbow_angle(joints[f]) for f in range(start_frame, end_frame + 1)]
    trunk = [dr.mocap_trunk_lean_deg(joints[f]) for f in range(start_frame, end_frame + 1)]
    return {"angle_deg": angle, "elbow_angle_deg": elbow, "trunk_lean_deg": trunk, "valid_fraction": 1.0}


def calibration_front_correct(manifest: pd.DataFrame, split: dict) -> pd.DataFrame:
    return manifest[
        manifest["subject"].isin(split["calibration_subjects"])
        & (manifest["view"] == "front") & (manifest["label"] == "correct")
    ].sort_values(["subject", "video_id", "repetition_number"])


def evaluation_front(manifest: pd.DataFrame, split: dict) -> pd.DataFrame:
    return manifest[
        manifest["subject"].isin(split["evaluation_subjects"]) & (manifest["view"] == "front")
    ].sort_values(["subject", "video_id", "repetition_number"])


# ---------------------------------------------------------------------------
# Derivation: absolute ([p5,p95]) and reference_relative (p95 one-sided dev)
# ---------------------------------------------------------------------------

def derive_absolute_config(manifest: pd.DataFrame, split: dict, commit: str, landmark_space: str) -> dict:
    calib_correct = calibration_front_correct(manifest, split)
    rows = []
    excluded_erroneous = 0
    for _, rep in calib_correct.iterrows():
        if bool(rep.get("mocap_erroneous", False)):
            excluded_erroneous += 1
            continue
        seq = mocap_rep_sequence(rep)
        if seq is None:
            continue
        fps = fps_for_rep(rep)
        feat = rules_module.rep_summary_features(seq, fps)
        rows.append({"subject": int(rep["subject"]), **feat})
    df = pd.DataFrame(rows)
    print(f"  [absolute derivation] n={len(df)} calibration-subject correct front-view reps "
          f"(excluded_mocap_erroneous={excluded_erroneous}), subjects={sorted(df['subject'].unique().tolist()) if not df.empty else []}")

    percentiles = {col: {"p5": float(df[col].quantile(0.05)), "p95": float(df[col].quantile(0.95))}
                   for col in ("peak_angle", "elbow_min", "trunk_max", "tempo_s")}
    target_peak = (percentiles["peak_angle"]["p5"] + percentiles["peak_angle"]["p95"]) / 2
    tolerance = (percentiles["peak_angle"]["p95"] - percentiles["peak_angle"]["p5"]) / 2

    return {
        "exercise": "shoulder_abduction", "mode": "absolute",
        "notice": "ORACLE -- derived from mocap (ground-truth) features, calibration subjects, front view. "
                  "Upper-bound analysis only, NOT a deployable scorer (mocap is unavailable at inference time).",
        "derived_from": "REHAB24-6 calibration subjects (front view, mocap features)",
        "landmark_space": landmark_space, "n_reps": len(df),
        "subjects": sorted(int(s) for s in df["subject"].unique()) if not df.empty else [],
        "date": date.today().isoformat(), "commit": commit, "percentiles": percentiles,
        "target_peak_deg": {"value": round(target_peak, 1), "status": "derived"},
        "tolerance_deg": {"value": round(tolerance, 1), "status": "derived"},
        "min_elbow_straight_deg": {"value": round(percentiles["elbow_min"]["p5"], 1), "status": "derived"},
        "max_trunk_lean_deg": {"value": round(percentiles["trunk_max"]["p95"], 1), "status": "derived"},
        "tempo_min_s": {"value": round(percentiles["tempo_s"]["p5"], 2), "status": "derived"},
        "tempo_max_s": {"value": round(percentiles["tempo_s"]["p95"], 2), "status": "derived"},
        "cue_cooldown_s": {"value": 8, "status": "carried_over"},
        "min_valid_fraction": {"value": 0.7, "status": "carried_over"},
        "cues": {"peak_angle": "", "elbow_straight": "", "trunk_lean": "", "tempo": "", "insufficient_data": ""},
    }


def derive_reference_relative_config(manifest: pd.DataFrame, split: dict, commit: str, landmark_space: str) -> dict:
    calib_correct = calibration_front_correct(manifest, split)
    deviations = {"peak_angle": [], "elbow_min": [], "trunk_max": [], "tempo_s": []}
    subjects_used = []
    n_reference_reps = 0

    for subject in sorted(calib_correct["subject"].unique()):
        sub_reps = calib_correct[calib_correct["subject"] == subject]
        sub_reps = sub_reps[~sub_reps.apply(lambda r: bool(r.get("mocap_erroneous", False)), axis=1)]
        if len(sub_reps) <= N_REFERENCE_REPS:
            print(f"  [reference derivation] subject {subject}: only {len(sub_reps)} usable correct front-view "
                  f"reps -- need > {N_REFERENCE_REPS} -- skipping.")
            continue
        reference_rows = sub_reps.iloc[:N_REFERENCE_REPS]
        remaining_rows = sub_reps.iloc[N_REFERENCE_REPS:]

        reference_reps_seq = [mocap_rep_sequence(rep) for _, rep in reference_rows.iterrows()]
        reference_reps_seq = [s for s in reference_reps_seq if s is not None]
        if len(reference_reps_seq) < N_REFERENCE_REPS:
            print(f"  [reference derivation] subject {subject}: fewer than {N_REFERENCE_REPS} usable "
                  f"reference reps -- skipping.")
            continue
        fps_ref = fps_for_rep(reference_rows.iloc[0])
        reference = rules_module.build_reference(f"oracle_calib_{subject}", "shoulder_abduction", reference_reps_seq, fps_ref)
        medians = reference["medians"]
        n_reference_reps += len(reference_reps_seq)

        for _, rep in remaining_rows.iterrows():
            seq = mocap_rep_sequence(rep)
            if seq is None:
                continue
            fps = fps_for_rep(rep)
            feat = rules_module.rep_summary_features(seq, fps)
            deviations["peak_angle"].append(max(0.0, medians["peak_angle"] - feat["peak_angle"]))
            deviations["elbow_min"].append(max(0.0, medians["elbow_min"] - feat["elbow_min"]))
            deviations["trunk_max"].append(max(0.0, feat["trunk_max"] - medians["trunk_max"]))
            deviations["tempo_s"].append(abs(feat["tempo_s"] - medians["tempo_s"]) / medians["tempo_s"])
        subjects_used.append(int(subject))

    n = len(deviations["peak_angle"])
    print(f"  [reference derivation] n_reference_reps={n_reference_reps}  n_deviation_reps={n}  "
          f"subjects={subjects_used}")
    if n < MIN_N_WARNING:
        print(f"  WARNING: n={n} < {MIN_N_WARNING} -- oracle reference tolerances derived from very few reps.")

    p95 = {feat: (float(np.percentile(vals, 95)) if vals else None) for feat, vals in deviations.items()}

    return {
        "exercise": "shoulder_abduction", "mode": "reference_relative",
        "notice": "ORACLE -- derived from mocap (ground-truth) features, calibration subjects, front view. "
                  "Upper-bound analysis only, NOT a deployable scorer (mocap is unavailable at inference time).",
        "derived_from": "REHAB24-6 calibration subjects (front view, mocap features)",
        "landmark_space": landmark_space, "n_reference_reps": n_reference_reps, "n_deviation_reps": n,
        "subjects": subjects_used, "date": date.today().isoformat(), "commit": commit,
        "tol_peak_deg": {"value": round(p95["peak_angle"], 1) if p95["peak_angle"] is not None else None, "status": "derived"},
        "tol_elbow_deg": {"value": round(p95["elbow_min"], 1) if p95["elbow_min"] is not None else None, "status": "derived"},
        "tol_trunk_deg": {"value": round(p95["trunk_max"], 1) if p95["trunk_max"] is not None else None, "status": "derived"},
        "tol_tempo_frac": {"value": round(p95["tempo_s"], 3) if p95["tempo_s"] is not None else None, "status": "derived"},
        # absolute fallback (required by score_rep() whenever no reference is supplied) --
        # reuse the oracle absolute config's own thresholds, computed separately above.
        "cue_cooldown_s": {"value": 8, "status": "carried_over"},
        "min_valid_fraction": {"value": 0.7, "status": "carried_over"},
        "cues": {"peak_angle": "", "elbow_straight": "", "trunk_lean": "", "tempo": "", "insufficient_data": ""},
    }


# ---------------------------------------------------------------------------
# Evaluation -- exactly as run 1 / run 2
# ---------------------------------------------------------------------------

def evaluate_oracle_absolute(manifest: pd.DataFrame, split: dict, config: dict) -> pd.DataFrame:
    eval_front = evaluation_front(manifest, split)
    rows = []
    for _, rep in eval_front.iterrows():
        seq = mocap_rep_sequence(rep)
        if seq is None:
            status, failed_rules = "insufficient_data", []
        else:
            fps = fps_for_rep(rep)
            result = rules_module.score_rep(seq, config, fps)
            status, failed_rules = result["status"], result["failed_rules"]
        rows.append({"subject": int(rep["subject"]), "view": rep["view"], "label": rep["label"],
                      "status": status, "failed_rules": ";".join(failed_rules)})
    return pd.DataFrame(rows)


def evaluate_oracle_reference(manifest: pd.DataFrame, split: dict, config: dict) -> pd.DataFrame:
    eval_front = evaluation_front(manifest, split)
    rows = []
    for subject in sorted(eval_front["subject"].unique()):
        sub_reps = eval_front[eval_front["subject"] == subject]
        correct_reps = sub_reps[sub_reps["label"] == "correct"].sort_values(["video_id", "repetition_number"])
        if len(correct_reps) <= N_REFERENCE_REPS:
            print(f"  [oracle reference eval] subject {subject}: only {len(correct_reps)} correct front-view "
                  f"reps -- skipping entirely.")
            continue
        reference_rows = correct_reps.iloc[:N_REFERENCE_REPS]
        reference_reps_seq = [mocap_rep_sequence(rep) for _, rep in reference_rows.iterrows()]
        reference_reps_seq = [s for s in reference_reps_seq if s is not None]
        if len(reference_reps_seq) < N_REFERENCE_REPS:
            print(f"  [oracle reference eval] subject {subject}: fewer than {N_REFERENCE_REPS} usable "
                  f"reference reps -- skipping entirely.")
            continue
        fps_ref = fps_for_rep(reference_rows.iloc[0])
        reference = rules_module.build_reference(f"oracle_eval_{subject}", "shoulder_abduction", reference_reps_seq, fps_ref)

        remaining = sub_reps.drop(reference_rows.index)
        for _, rep in remaining.iterrows():
            seq = mocap_rep_sequence(rep)
            if seq is None:
                status, failed_rules = "insufficient_data", []
            else:
                fps = fps_for_rep(rep)
                result = rules_module.score_rep(seq, config, fps, reference=reference)
                status, failed_rules = result["status"], result["failed_rules"]
            rows.append({"subject": int(subject), "view": rep["view"], "label": rep["label"],
                         "status": status, "failed_rules": ";".join(failed_rules)})
    return pd.DataFrame(rows)


def balanced_accuracy(rates: dict) -> float | None:
    k_ff, n_ff = rates["false_fail"]
    k_det, n_det = rates["detection"]
    if n_ff == 0 or n_det == 0:
        return None
    return ((k_det / n_det) + (1 - k_ff / n_ff)) / 2


def summary_row(label: str, df: pd.DataFrame) -> list[str]:
    if df.empty:
        return [f"| {label} | N/A | N/A | N/A | N/A | N/A |"]
    r = bm.scorer_rates(df)
    ba = balanced_accuracy(r)
    return [f"| {label} | {bm.rate_with_ci(*r['false_fail'])} | {bm.rate_with_ci(*r['detection'])} | "
            f"{f'{ba:.1%}' if ba is not None else 'N/A'} | {bm.rate_with_ci(*r['insufficient'])} | "
            f"{r['rule_fire_counts']} |"]


def per_subject_rows(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["N/A\n"]
    lines = ["| subject | n reps | false-fail (95% CI) | detection (95% CI) | balanced accuracy | "
              "insufficient_data (95% CI) | which rule fired |", "|---|---|---|---|---|---|---|"]
    for subject in sorted(df["subject"].unique()):
        sub = df[df["subject"] == subject]
        r = bm.scorer_rates(sub)
        ba = balanced_accuracy(r)
        lines.append(f"| {subject} | {len(sub)} | {bm.rate_with_ci(*r['false_fail'])} | "
                      f"{bm.rate_with_ci(*r['detection'])} | {f'{ba:.1%}' if ba is not None else 'N/A'} | "
                      f"{bm.rate_with_ci(*r['insufficient'])} | {r['rule_fire_counts']} |")
    return lines


def main() -> None:
    commit, dirty = bm.git_provenance()
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    today = date.today().isoformat()
    print(f"Provenance: commit={commit}  dirty={dirty}  landmark_space={landmark_space}  date={today}")

    manifest, split = bm.load_manifest_and_split()

    print("\n[derive] oracle absolute config (calibration subjects, front, mocap)...")
    config_absolute = derive_absolute_config(manifest, split, commit, landmark_space)
    print("[derive] oracle reference_relative config (calibration subjects, front, mocap)...")
    config_reference = derive_reference_relative_config(manifest, split, commit, landmark_space)
    # the reference_relative config needs absolute fallback keys too (score_rep() requires them)
    for key in ("target_peak_deg", "tolerance_deg", "min_elbow_straight_deg", "max_trunk_lean_deg",
                "tempo_min_s", "tempo_max_s"):
        config_reference[key] = config_absolute[key]

    print("\n[evaluate] oracle absolute on evaluation subjects, front view...")
    oracle_absolute_df = evaluate_oracle_absolute(manifest, split, config_absolute)
    print("[evaluate] oracle reference_relative on evaluation subjects, front view...")
    oracle_reference_df = evaluate_oracle_reference(manifest, split, config_reference)

    lines = [
        "# Oracle ceiling: mocap-feature scoring (upper bound, NOT deployable)\n",
        f"_commit: `{commit}` (working tree {'dirty' if dirty else 'clean'})_\n",
        f"_landmark_space: `{landmark_space}`  date: {today}_\n",
        "**UPPER-BOUND ANALYSIS using ground-truth mocap features (ignoring MediaPipe's own measurement "
        "error entirely) -- NOT a deployable scorer, since mocap is never available at real inference time. "
        "Run AFTER run 1 (results/rehab24_benchmark.md) and run 2 (results/diag_reference_evaluation.md) on "
        "the SAME evaluation subjects, front view, excluding each oracle-reference run's own 3 reference "
        "reps from every metric, same as run 2.**\n",
        "\n## Oracle absolute config (derived)\n",
        f"```json\n{json.dumps({k: v for k, v in config_absolute.items() if k not in ('cues',)}, indent=2, default=str)}\n```\n",
        "\n## Oracle reference_relative config (derived)\n",
        f"```json\n{json.dumps({k: v for k, v in config_reference.items() if k not in ('cues',)}, indent=2, default=str)}\n```\n",
        "\n## Pooled (evaluation subjects, front view)\n",
        "| run | false-fail rate | detection rate | balanced accuracy | insufficient_data rate | which rule fired |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(summary_row("oracle absolute (mocap)", oracle_absolute_df))
    lines.extend(summary_row("oracle reference_relative (mocap)", oracle_reference_df))

    lines.append("\n## Per subject -- oracle absolute (mocap)\n")
    lines.extend(per_subject_rows(oracle_absolute_df))
    lines.append("\n## Per subject -- oracle reference_relative (mocap)\n")
    lines.extend(per_subject_rows(oracle_reference_df))

    RESULTS_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
