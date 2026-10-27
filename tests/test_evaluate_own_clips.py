"""Tests for scripts/evaluate_own_clips.py's manifest parser and pure
metrics functions, on SYNTHETIC/placeholder data only -- no real video,
no MediaPipe. process_clip_reps() and main() (the only functions that
touch real video files) are NOT exercised here or anywhere else in this
session: this script's harness was built and tested on synthetic rows
only, per the task that introduced it.
"""

from pathlib import Path

import pytest

from scripts.evaluate_own_clips import (
    derive_tolerances,
    fault_detection_rates,
    false_fail_rate,
    insufficient_data_rate,
    manifest_hash,
    one_sided_deviation,
    rate_with_ci,
    read_manifest,
    rep_count_metrics,
    wilson_interval,
    wrong_rules_on_correct,
)


# --- manifest parser -----------------------------------------------------

def write_manifest(tmp_path, rows):
    path = tmp_path / "own_manifest.csv"
    header = "clip,role,intended,expected_reps,subject,notes"
    lines = [header] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_read_manifest_parses_synthetic_rows(tmp_path):
    path = write_manifest(tmp_path, [
        ("clips/ref.mp4", "reference", "correct", "3", "demo", "ref clip"),
        ("clips/eval_lowarm.mp4", "eval", "fault_lowarm", "5", "demo", "lowarm fault"),
    ])
    rows = read_manifest(path)
    assert len(rows) == 2
    assert rows[0]["role"] == "reference"
    assert rows[0]["clip"] == "clips/ref.mp4"
    assert rows[1]["intended"] == "fault_lowarm"
    assert rows[1]["expected_reps"] == "5"


def test_read_manifest_returns_empty_list_when_missing(tmp_path):
    assert read_manifest(tmp_path / "does_not_exist.csv") == []


def test_manifest_hash_is_deterministic_and_changes_with_content(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("a,b\n1,2\n")
    h1 = manifest_hash(path)
    h2 = manifest_hash(path)
    assert h1 == h2
    path.write_text("a,b\n1,3\n")
    h3 = manifest_hash(path)
    assert h1 != h3


def test_manifest_hash_is_a_fixed_string_when_file_missing(tmp_path):
    assert manifest_hash(tmp_path / "missing.csv") == "no-manifest"


# --- wilson_interval / rate_with_ci --------------------------------------

def test_wilson_interval_zero_n_is_nan():
    lo, hi = wilson_interval(0, 0)
    assert lo != lo and hi != hi  # nan != nan


def test_wilson_interval_bounds_are_sane():
    lo, hi = wilson_interval(8, 10)
    assert 0.0 <= lo < 0.8 < hi <= 1.0


def test_rate_with_ci_formats_n_zero():
    assert rate_with_ci(0, 0) == "N/A (n=0)"


def test_rate_with_ci_contains_fraction_and_percent():
    s = rate_with_ci(5, 5)
    assert "5/5" in s and "100.0%" in s


# --- one_sided_deviation --------------------------------------------------

def test_one_sided_deviation_peak_below_reference_is_positive():
    measured = {"peak_angle": 80.0}
    ref = {"peak_angle": 100.0}
    assert one_sided_deviation("peak_angle", measured, ref) == 20.0


def test_one_sided_deviation_peak_above_reference_is_zero():
    measured = {"peak_angle": 120.0}
    ref = {"peak_angle": 100.0}
    assert one_sided_deviation("peak_angle", measured, ref) == 0.0


def test_one_sided_deviation_elbow_below_reference_is_positive():
    measured = {"elbow_min": 140.0}
    ref = {"elbow_min": 150.0}
    assert one_sided_deviation("elbow_min", measured, ref) == 10.0


def test_one_sided_deviation_trunk_above_reference_is_positive():
    measured = {"trunk_max": 25.0}
    ref = {"trunk_max": 10.0}
    assert one_sided_deviation("trunk_max", measured, ref) == 15.0


def test_one_sided_deviation_trunk_below_reference_is_zero():
    measured = {"trunk_max": 5.0}
    ref = {"trunk_max": 10.0}
    assert one_sided_deviation("trunk_max", measured, ref) == 0.0


def test_one_sided_deviation_tempo_is_two_sided_fraction():
    measured = {"tempo_s": 4.0}
    ref = {"tempo_s": 5.0}
    assert one_sided_deviation("tempo_s", measured, ref) == pytest.approx(0.2)
    measured_fast = {"tempo_s": 6.0}
    assert one_sided_deviation("tempo_s", measured_fast, ref) == pytest.approx(0.2)


def test_one_sided_deviation_raises_on_unknown_feature():
    with pytest.raises(ValueError):
        one_sided_deviation("not_a_feature", {}, {})


# --- derive_tolerances ----------------------------------------------------

def make_synthetic_rep(peak=100.0, elbow=150.0, trunk=10.0, n_frames=90):
    angle = [0.0] + [peak] * n_frames + [0.0]
    elbow_seq = [elbow] * len(angle)
    trunk_seq = [trunk] * len(angle)
    return {"angle_deg": angle, "elbow_angle_deg": elbow_seq, "trunk_lean_deg": trunk_seq}


def test_derive_tolerances_on_synthetic_reps():
    reference_medians = {"peak_angle": 100.0, "elbow_min": 150.0, "trunk_max": 10.0, "tempo_s": 3.0}
    reps = [
        make_synthetic_rep(peak=90.0, elbow=145.0, trunk=15.0, n_frames=88),   # tempo 3.0s
        make_synthetic_rep(peak=85.0, elbow=140.0, trunk=20.0, n_frames=88),
    ]
    result = derive_tolerances(reps, fps=30.0, reference_medians=reference_medians)
    assert result["n"] == 2
    assert result["tol_peak_deg"] >= 10.0  # p95 of [10, 15] deviations is close to 15
    assert result["tol_trunk_deg"] >= 5.0


def test_derive_tolerances_empty_reps_returns_n_zero_and_none_tolerances():
    result = derive_tolerances([], fps=30.0, reference_medians={"peak_angle": 1, "elbow_min": 1, "trunk_max": 1, "tempo_s": 1})
    assert result["n"] == 0
    assert result["tol_peak_deg"] is None


# --- rep_count_metrics -----------------------------------------------------

def test_rep_count_metrics_exact_match():
    r = rep_count_metrics(detected=10, expected=10)
    assert r["exact_match"] is True
    assert r["within_1"] is True
    assert r["signed_error"] == 0


def test_rep_count_metrics_within_one_but_not_exact():
    r = rep_count_metrics(detected=9, expected=10)
    assert r["exact_match"] is False
    assert r["within_1"] is True
    assert r["signed_error"] == -1


def test_rep_count_metrics_outside_one():
    r = rep_count_metrics(detected=7, expected=10)
    assert r["exact_match"] is False
    assert r["within_1"] is False
    assert r["signed_error"] == -3


# --- false_fail_rate / fault_detection_rates / wrong_rules_on_correct /
#     insufficient_data_rate -- all on synthetic score_rep()-shaped dicts

def result(status, failed_rules=None):
    return {"status": status, "failed_rules": failed_rules or []}


def test_false_fail_rate_counts_fail_among_scorable_only():
    correct = [result("pass"), result("fail", ["peak_angle"]), result("insufficient_data")]
    k, n = false_fail_rate(correct)
    assert (k, n) == (1, 2)  # insufficient_data excluded from the denominator


def test_fault_detection_rates_any_vs_expected_rule():
    by_intended = {
        "fault_lowarm": [
            result("fail", ["peak_angle"]),
            result("fail", ["trunk_lean"]),  # flagged, but NOT by the expected rule
            result("pass"),
        ],
    }
    rates = fault_detection_rates(by_intended)
    assert rates["fault_lowarm"]["rule_id"] == "peak_angle"
    assert rates["fault_lowarm"]["any"] == (2, 3)
    assert rates["fault_lowarm"]["expected_rule"] == (1, 3)


def test_fault_detection_rates_excludes_insufficient_data_from_denominator():
    by_intended = {"fault_trunk": [result("fail", ["trunk_lean"]), result("insufficient_data")]}
    rates = fault_detection_rates(by_intended)
    assert rates["fault_trunk"]["any"] == (1, 1)


def test_fault_detection_rates_covers_all_four_fault_types():
    rates = fault_detection_rates({})
    assert set(rates.keys()) == {"fault_lowarm", "fault_trunk", "fault_elbow", "fault_fast"}
    for r in rates.values():
        assert r["any"] == (0, 0)


def test_wrong_rules_on_correct_only_counts_failed_reps():
    correct = [result("pass"), result("fail", ["peak_angle", "tempo"]), result("fail", ["peak_angle"])]
    counts = wrong_rules_on_correct(correct)
    assert counts["peak_angle"] == 2
    assert counts["tempo"] == 1


def test_insufficient_data_rate():
    all_results = [result("pass"), result("fail"), result("insufficient_data"), result("insufficient_data")]
    k, n = insufficient_data_rate(all_results)
    assert (k, n) == (2, 4)
