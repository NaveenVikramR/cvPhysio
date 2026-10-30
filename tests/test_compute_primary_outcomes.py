"""scripts/compute_primary_outcomes.py on synthetic evaluator CSVs. Every path is
under tmp_path; no clip, config or result is touched."""

import csv
import subprocess

import pytest

from scripts import compute_primary_outcomes as cpo

HEADER = ["clip", "intended", "rep_index", "demo_status", "demo_failed_rules", "reference_status", "reference_failed_rules"]
PROVENANCE = {"commit": "abc", "dirty": "False", "landmark_space": "world", "date": "2026-10-30"}


def rep(intended, failed="", status=None, index=1):
    status = status or ("fail" if failed else "pass")
    return {"clip": f"{intended}.mp4", "intended": intended, "rep_index": index, "demo_status": status,
            "demo_failed_rules": failed, "reference_status": status, "reference_failed_rules": failed}


def write_csv(tmp_path, rows, provenance=PROVENANCE):
    path = tmp_path / "own_clips_eval.csv"
    with path.open("w", newline="") as f:
        for key, value in provenance.items():
            f.write(f"# {key}: {value}\n")
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return path


def planned_rows(correct=None, lowarm=None, trunk=None, fast=None, elbow=None):
    """The planned 6 correct + 3 per fault, all behaving as intended by default."""
    return [
        *(correct if correct is not None else [rep("correct") for _ in range(6)]),
        *(lowarm if lowarm is not None else [rep("fault_lowarm", "peak_angle") for _ in range(3)]),
        *(trunk if trunk is not None else [rep("fault_trunk", "trunk_lean") for _ in range(3)]),
        *(fast if fast is not None else [rep("fault_fast", "tempo") for _ in range(3)]),
        *(elbow if elbow is not None else [rep("fault_elbow", "elbow_straight") for _ in range(3)]),
    ]


def outcome_verdict(rows, mode="demo"):
    return cpo.verdict(cpo.mode_outcomes(rows, mode))


# --- the primary outcomes -----------------------------------------------------

def test_clean_planned_set_passes():
    assert outcome_verdict(planned_rows()) == ("PASS", [])


def test_elbow_failures_on_correct_reps_do_not_count_as_false_fails():
    correct = [rep("correct", "elbow_straight") for _ in range(6)]
    outcomes = cpo.mode_outcomes(planned_rows(correct=correct), "demo")
    assert outcomes["false_fail"] == (0, 6)
    assert outcomes["all_rule_false_fail"] == (6, 6)
    assert cpo.verdict(outcomes)[0] == "PASS"


@pytest.mark.parametrize("rule", ["peak_angle", "trunk_lean", "tempo"])
def test_each_primary_rule_counts_as_a_false_fail(rule):
    correct = [rep("correct", f"elbow_straight;{rule}")] + [rep("correct") for _ in range(5)]
    assert cpo.mode_outcomes(planned_rows(correct=correct), "demo")["false_fail"] == (1, 6)


def test_false_fail_boundary_one_of_six_passes_two_of_six_fails():
    one = [rep("correct", "tempo")] + [rep("correct") for _ in range(5)]
    two = [rep("correct", "tempo") for _ in range(2)] + [rep("correct") for _ in range(4)]
    assert outcome_verdict(planned_rows(correct=one))[0] == "PASS"
    result, reasons = outcome_verdict(planned_rows(correct=two))
    assert result == "FAIL"
    assert "false-fail 2/6" in reasons[0]


def test_detection_needs_the_expected_rule_not_any_rule():
    lowarm = [rep("fault_lowarm", "tempo") for _ in range(3)]
    outcomes = cpo.mode_outcomes(planned_rows(lowarm=lowarm), "demo")
    assert outcomes["faults"]["fault_lowarm"] == (0, 3)
    assert cpo.verdict(outcomes)[0] == "FAIL"


@pytest.mark.parametrize("group, rule", [("fault_lowarm", "peak_angle"), ("fault_trunk", "trunk_lean"), ("fault_fast", "tempo")])
def test_each_fault_needs_three_of_three_with_the_planned_reps(group, rule):
    reps = [rep(group, rule), rep(group, rule), rep(group, "elbow_straight")]
    key = {"fault_lowarm": "lowarm", "fault_trunk": "trunk", "fault_fast": "fast"}[group]
    result, reasons = outcome_verdict(planned_rows(**{key: reps}))
    assert result == "FAIL"
    assert f"{group} detected 2/3" in reasons[0]


def test_the_thresholds_are_rates_so_more_reps_can_tolerate_a_miss():
    lowarm = [rep("fault_lowarm", "peak_angle") for _ in range(3)] + [rep("fault_lowarm", "")]
    assert outcome_verdict(planned_rows(lowarm=lowarm))[0] == "PASS"  # 3/4 = 75%


def test_a_missed_bent_elbow_never_affects_the_verdict():
    elbow = [rep("fault_elbow", "") for _ in range(3)]
    outcomes = cpo.mode_outcomes(planned_rows(elbow=elbow), "demo")
    assert outcomes["exploratory"] == (0, 3)
    assert cpo.verdict(outcomes) == ("PASS", [])


def test_modes_are_scored_independently():
    rows = planned_rows()
    rows[0] = {**rows[0], "reference_status": "fail", "reference_failed_rules": "tempo"}
    rows[1] = {**rows[1], "reference_status": "fail", "reference_failed_rules": "tempo"}
    assert outcome_verdict(rows, "demo")[0] == "PASS"
    assert outcome_verdict(rows, "reference")[0] == "FAIL"


# --- inconclusive -------------------------------------------------------------

def test_insufficient_reps_leave_the_denominator_and_make_a_group_short():
    correct = [rep("correct", "tempo", status="insufficient_data")] + [rep("correct") for _ in range(5)]
    outcomes = cpo.mode_outcomes(planned_rows(correct=correct), "demo")
    assert outcomes["false_fail"] == (0, 5)
    assert outcomes["insufficient"] == (1, 18)
    result, reasons = cpo.verdict(outcomes)
    assert result == "INCONCLUSIVE"
    assert "correct: 5 scorable reps, 6 planned" in reasons[0]


def test_too_many_insufficient_reps_is_inconclusive_even_with_enough_scorable():
    extra = [rep("correct", status="insufficient_data") for _ in range(6)]
    result, reasons = outcome_verdict(planned_rows() + extra)
    assert result == "INCONCLUSIVE"
    assert "insufficient_data on 6/24" in reasons[0]


def test_a_missing_exploratory_group_is_still_inconclusive():
    assert outcome_verdict(planned_rows(elbow=[]))[0] == "INCONCLUSIVE"


def test_inconclusive_outranks_a_failure():
    two = [rep("correct", "tempo") for _ in range(2)] + [rep("correct") for _ in range(3)]
    assert outcome_verdict(planned_rows(correct=two))[0] == "INCONCLUSIVE"


# --- CSV reading --------------------------------------------------------------

def test_reads_the_evaluators_provenance_header_and_rows(tmp_path):
    provenance, rows = cpo.read_eval_csv(write_csv(tmp_path, planned_rows()))
    assert provenance["commit"] == "abc"
    assert provenance["dirty"] == "False"
    assert len(rows) == 18
    assert rows[6]["demo_failed_rules"] == "peak_angle"


def test_missing_columns_are_refused(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("clip,intended\nx.mp4,correct\n")
    with pytest.raises(ValueError, match="missing columns"):
        cpo.read_eval_csv(path)


# --- pins ---------------------------------------------------------------------

def test_pins_match_the_preregistration_document():
    text = open("docs/preregistered_own_clip_eval.md", encoding="utf-8").read()
    for path, blob in cpo.PINNED_BLOBS.items():
        assert path in text
        assert blob in text


def test_pin_check_passes_on_this_repos_head():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert cpo.pin_check(head) == (True, [])


def test_pin_check_fails_on_a_commit_before_the_pins():
    # d67563e predates the pinned src/rules.py.
    ok, problems = cpo.pin_check("d67563e")
    assert ok is False
    assert any("src/rules.py" in problem for problem in problems)


@pytest.mark.parametrize("commit", [None, "unknown"])
def test_pin_check_without_a_usable_commit_fails(commit):
    assert cpo.pin_check(commit)[0] is False


# --- the script ---------------------------------------------------------------

def snapshot(directory):
    return {p: p.read_bytes() for p in directory.iterdir()}


def test_main_prints_outcomes_with_intervals_and_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cpo, "pin_check", lambda commit: (True, []))
    path = write_csv(tmp_path, planned_rows())
    before = snapshot(tmp_path)
    assert cpo.main([str(path)]) == 0
    assert snapshot(tmp_path) == before
    out = capsys.readouterr().out
    assert "=== demo" in out and "=== reference" in out
    assert "0/6 (0.0%, 95% CI [" in out
    assert "3/3 (100.0%, 95% CI [" in out
    assert "Exploratory, no pass criterion" in out
    assert "Verdict: PASS" in out
    assert "isn't evidence of accuracy" in out


def test_main_flags_files_that_do_not_match_the_pins(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cpo, "pin_check", lambda commit: (False, ["src/rules.py: differs"]))
    assert cpo.main([str(write_csv(tmp_path, planned_rows()))]) == 0
    assert "NOT VALID as the pre-registered evaluation" in capsys.readouterr().out


def test_main_warns_about_a_dirty_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cpo, "pin_check", lambda commit: (True, []))
    path = write_csv(tmp_path, planned_rows(), {**PROVENANCE, "dirty": "True"})
    assert cpo.main([str(path)]) == 0
    assert "wasn't recorded as clean" in capsys.readouterr().out


def test_main_explains_a_missing_csv(tmp_path, capsys):
    assert cpo.main([str(tmp_path / "absent.csv")]) == 1
    assert "No evaluator output" in capsys.readouterr().out
