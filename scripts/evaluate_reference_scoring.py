"""STEP 4 of the reference-scoring task: evaluate reference_relative
scoring on EVALUATION subjects, front view, entirely from the landmark
cache (--rescore-only semantics; every video already has a cache entry
from the earlier full benchmark run -- MediaPipe is never invoked here).

This is a SECOND evaluation of the SAME evaluation subjects already
scored in run 1 (results/rehab24_benchmark.md's Part (c): absolute
benchmark config vs absolute demo config). Run 1's numbers are
REPRODUCED here -- recomputed from the exact same code
(scripts.benchmark_rehab24.part_c_rules_vs_labels_front/scorer_rates)
and the exact same cache, not hand-copied -- so both runs sit in one
table, side by side, and are guaranteed consistent.

Run 2 (reference_relative): for each evaluation subject, reference =
that subject's own first 3 correct front-view reps in annotated order
(video_id then repetition_number) -- simulating a therapist recording
a per-patient reference at the start of that patient's own session.
ALL remaining reps (correct reps beyond the first 3, and every
incorrect rep) are scored against config/exercises/benchmarks/
rehab24_abduction_reference.json using that per-patient reference; the
3 reference reps themselves are excluded from every metric below. A
subject with <= 3 correct front-view reps has no remaining reps to
score and is skipped entirely (printed, not silently dropped).

Metrics (pooled and per subject, Wilson 95% CIs): false-fail rate
(correct reps), detection rate (incorrect reps), balanced accuracy
((sensitivity + specificity) / 2), insufficient_data rate, which rule
fired.

Output: results/diag_reference_evaluation.md (commit hash, dirty flag,
landmark_space, config paths, date at the top).
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.benchmark_rehab24 as bm
from src import pose_estimation
from src import rules as rules_module

RESULTS_DIR = Path("results")
REFERENCE_CONFIG_PATH = Path("config/exercises/benchmarks/rehab24_abduction_reference.json")
N_REFERENCE_REPS = 3


def evaluation_front_reps(manifest: pd.DataFrame, split: dict) -> pd.DataFrame:
    return manifest[
        manifest["subject"].isin(split["evaluation_subjects"]) & (manifest["view"] == "front")
    ].sort_values(["subject", "video_id", "repetition_number"])


def build_run2_rows(manifest: pd.DataFrame, split: dict, reference_config: dict) -> pd.DataFrame:
    eval_front = evaluation_front_reps(manifest, split)
    rows = []
    for subject in sorted(eval_front["subject"].unique()):
        sub_reps = eval_front[eval_front["subject"] == subject]
        correct_reps = sub_reps[sub_reps["label"] == "correct"].sort_values(["video_id", "repetition_number"])
        if len(correct_reps) <= N_REFERENCE_REPS:
            print(f"  subject {subject}: only {len(correct_reps)} correct front-view reps -- need > "
                  f"{N_REFERENCE_REPS} (reference + at least 1 remaining rep) -- SKIPPED entirely.")
            continue

        reference_rows = correct_reps.iloc[:N_REFERENCE_REPS]
        reference_reps_seq = []
        fps_ref = 30.0
        for _, rep in reference_rows.iterrows():
            video_path = bm.DATA_DIR / "videos" / rep["video"]
            seq, fps_ref = bm.rep_feature_sequences(video_path, int(rep["start_frame"]), int(rep["end_frame"]))
            if len(seq["angle_deg"]) > 0:
                reference_reps_seq.append(seq)
        if len(reference_reps_seq) < N_REFERENCE_REPS:
            print(f"  subject {subject}: fewer than {N_REFERENCE_REPS} usable reference reps -- SKIPPED entirely.")
            continue
        reference = rules_module.build_reference(f"eval_subject_{subject}", "shoulder_abduction", reference_reps_seq, fps_ref)
        print(f"  subject {subject}: reference (first {N_REFERENCE_REPS} correct reps) medians = {reference['medians']}")

        remaining = sub_reps.drop(reference_rows.index)  # excludes the 3 reference reps from every metric
        for _, rep in remaining.iterrows():
            video_path = bm.DATA_DIR / "videos" / rep["video"]
            seq, fps = bm.rep_feature_sequences(video_path, int(rep["start_frame"]), int(rep["end_frame"]))
            if len(seq["angle_deg"]) == 0:
                status, failed_rules = "insufficient_data", []
            else:
                result = rules_module.score_rep(seq, reference_config, fps, reference=reference)
                status, failed_rules = result["status"], result["failed_rules"]
            rows.append({
                "subject": int(subject), "view": rep["view"], "label": rep["label"],
                "status": status, "failed_rules": ";".join(failed_rules),
            })
    return pd.DataFrame(rows)


def balanced_accuracy(rates: dict) -> float | None:
    k_ff, n_ff = rates["false_fail"]
    k_det, n_det = rates["detection"]
    if n_ff == 0 or n_det == 0:
        return None
    sensitivity = k_det / n_det  # incorrect reps correctly scored fail
    specificity = 1 - (k_ff / n_ff)  # correct reps correctly scored pass
    return (sensitivity + specificity) / 2


def run_summary_lines(label: str, df: pd.DataFrame) -> list[str]:
    if df.empty:
        return [f"| {label} | N/A | N/A | N/A | N/A | N/A | N/A |"]
    r = bm.scorer_rates(df)
    ba = balanced_accuracy(r)
    return [f"| {label} | {bm.rate_with_ci(*r['false_fail'])} | {bm.rate_with_ci(*r['detection'])} | "
            f"{f'{ba:.1%}' if ba is not None else 'N/A'} | {bm.rate_with_ci(*r['insufficient'])} | "
            f"{r['rule_fire_counts']} |"]


def per_subject_lines(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["N/A\n"]
    lines = ["| subject | n reps | false-fail (95% CI) | detection (95% CI) | balanced accuracy | "
              "insufficient_data (95% CI) | which rule fired |",
              "|---|---|---|---|---|---|---|"]
    for subject in sorted(df["subject"].unique()):
        sub = df[df["subject"] == subject]
        r = bm.scorer_rates(sub)
        ba = balanced_accuracy(r)
        lines.append(f"| {subject} | {len(sub)} | {bm.rate_with_ci(*r['false_fail'])} | "
                      f"{bm.rate_with_ci(*r['detection'])} | {f'{ba:.1%}' if ba is not None else 'N/A'} | "
                      f"{bm.rate_with_ci(*r['insufficient'])} | {r['rule_fire_counts']} |")
    return lines


def main() -> None:
    bm.RESCORE_ONLY = True  # never touch MediaPipe -- every video must already be cached
    commit, dirty = bm.git_provenance()
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    today = date.today().isoformat()
    print(f"Provenance: commit={commit}  dirty={dirty}  landmark_space={landmark_space}  date={today}  "
          f"rescore_only=True")
    if landmark_space != "world":
        print("ABORTING: set config/pose.json's landmark_space to 'world' and re-run.")
        return

    manifest, split = bm.load_manifest_and_split()
    print(f"Evaluation subjects: {split['evaluation_subjects']} (front view only)")

    config_benchmark = json.loads(bm.BENCHMARK_CONFIG_PATH.read_text())
    config_demo = json.loads(bm.DEMO_CONFIG_PATH.read_text())
    config_reference = json.loads(REFERENCE_CONFIG_PATH.read_text())

    print("\n[run 1, reproduced] benchmark config (absolute)...")
    run1_benchmark = bm.part_c_rules_vs_labels_front(manifest, split, config_benchmark)
    print("[run 1, reproduced] demo config (absolute)...")
    run1_demo = bm.part_c_rules_vs_labels_front(manifest, split, config_demo)

    print("\n[run 2] reference_relative (per-subject reference = first 3 correct reps)...")
    run2 = build_run2_rows(manifest, split, config_reference)

    lines = [
        "# Reference-relative scoring evaluation (run 2) -- evaluation subjects, front view\n",
        f"_commit: `{commit}` (working tree {'dirty' if dirty else 'clean'})_\n",
        f"_landmark_space: `{landmark_space}`_\n",
        f"_configs: benchmark=`{bm.BENCHMARK_CONFIG_PATH}`  demo=`{bm.DEMO_CONFIG_PATH}`  "
        f"reference_relative=`{REFERENCE_CONFIG_PATH}`_\n",
        f"_date: {today}_\n",
        "**This file reports a SECOND evaluation of the SAME evaluation subjects already scored in run 1 "
        "(results/rehab24_benchmark.md's Part (c)). Run 1's numbers are reproduced here from the exact same "
        "code and cache (not hand-copied) and quoted in the table below, next to run 2 (reference_relative, "
        "per-subject reference = each subject's own first 3 correct front-view reps, excluded from every "
        "metric).**\n",
        "\n## Pooled (all evaluation subjects, front view)\n",
        "| run | false-fail rate | detection rate | balanced accuracy | insufficient_data rate | which rule fired |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(run_summary_lines("run 1: benchmark config (absolute)", run1_benchmark))
    lines.extend(run_summary_lines("run 1: demo config (absolute)", run1_demo))
    lines.extend(run_summary_lines("run 2: reference_relative", run2))

    lines.append("\n## Per subject -- run 1: benchmark config (absolute)\n")
    lines.extend(per_subject_lines(run1_benchmark))
    lines.append("\n## Per subject -- run 1: demo config (absolute)\n")
    lines.extend(per_subject_lines(run1_demo))
    lines.append("\n## Per subject -- run 2: reference_relative\n")
    lines.extend(per_subject_lines(run2))

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "diag_reference_evaluation.md").write_text("\n".join(lines))
    print(f"\nWrote {RESULTS_DIR / 'diag_reference_evaluation.md'}")


if __name__ == "__main__":
    main()
