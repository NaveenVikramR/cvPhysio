"""The advisory_rules config option, on synthetic reps. The default must keep
today's verdict exactly, and no shipped config may turn it on."""

import csv
import json
from pathlib import Path

import pytest

from src import rules as rules_module
from src import ui_theme
from src.feedback import message_for_rules_result
from src.pipeline import apply_advisory_rules
from src.tracking import RepRecord, SessionTracker

FPS = 10.0
BASE_CONFIG = rules_module.load_config("shoulder_abduction")
ADVISORY_CONFIG = {**BASE_CONFIG, "advisory_rules": ["elbow_straight"]}
GOOD_ANGLE = [10.0, 50.0, 90.0, 50.0, 10.0] * 6  # peak 90, 3.0 s at 10 fps


def make_rep(elbow=170.0, angle=None, trunk=5.0, valid_fraction=1.0):
    angle = angle or GOOD_ANGLE
    return {"angle_deg": angle, "elbow_angle_deg": [elbow] * len(angle), "trunk_lean_deg": [trunk] * len(angle),
            "valid_fraction": valid_fraction}


def score(rep, config):
    return apply_advisory_rules(rules_module.score_rep(rep, config, FPS), config)


BENT = make_rep(elbow=120.0)


# --- default behaviour --------------------------------------------------------

@pytest.mark.parametrize("rep", [make_rep(), BENT, make_rep(trunk=30.0), make_rep(elbow=120.0, trunk=30.0),
                                 make_rep(elbow=120.0, valid_fraction=0.3)])
def test_without_the_key_the_verdict_is_exactly_score_reps(rep):
    raw = rules_module.score_rep(rep, BASE_CONFIG, FPS)
    result = apply_advisory_rules(raw, BASE_CONFIG)
    assert result["status"] == raw["status"]
    assert result["failed_rules"] == raw["failed_rules"]
    assert result["advisory_failed"] == []


def test_no_shipped_config_turns_it_on():
    for path in Path("config/exercises").rglob("*.json"):
        assert "advisory_rules" not in json.loads(path.read_text()), path


# --- the verdict --------------------------------------------------------------

def test_an_advisory_failure_alone_passes_the_rep_and_is_still_reported():
    assert rules_module.score_rep(BENT, ADVISORY_CONFIG, FPS)["status"] == "fail"
    result = score(BENT, ADVISORY_CONFIG)
    assert result["status"] == "pass"
    assert result["failed_rules"] == []
    assert result["advisory_failed"] == ["elbow_straight"]
    assert result["rules"]["elbow_straight"]["passed"] is False
    assert result["rules"]["elbow_straight"]["measured"] == 120.0


def test_a_counted_rule_still_fails_the_rep():
    result = score(make_rep(elbow=120.0, trunk=30.0), ADVISORY_CONFIG)
    assert result["status"] == "fail"
    assert result["failed_rules"] == ["trunk_lean"]
    assert result["advisory_failed"] == ["elbow_straight"]


def test_a_straight_arm_has_no_advisory_failure():
    result = score(make_rep(), ADVISORY_CONFIG)
    assert (result["status"], result["advisory_failed"]) == ("pass", [])


def test_insufficient_data_is_never_turned_into_a_pass():
    result = score(make_rep(elbow=120.0, valid_fraction=0.3), ADVISORY_CONFIG)
    assert result["status"] == "insufficient_data"
    assert result["advisory_failed"] == ["elbow_straight"]


def test_an_unknown_advisory_rule_is_refused():
    with pytest.raises(ValueError, match="advisory_rules"):
        score(make_rep(), {**BASE_CONFIG, "advisory_rules": ["elbow"]})


def test_the_input_result_is_not_mutated():
    raw = rules_module.score_rep(BENT, ADVISORY_CONFIG, FPS)
    apply_advisory_rules(raw, ADVISORY_CONFIG)
    assert raw["status"] == "fail"
    assert raw["failed_rules"] == ["elbow_straight"]


# --- feedback -----------------------------------------------------------------

def test_feedback_mentions_the_advisory_rule_on_a_pass():
    message = message_for_rules_result(score(BENT, ADVISORY_CONFIG))
    assert message.startswith("Good form!")
    assert "Advisory: elbow_straight" in message


def test_feedback_keeps_counted_and_advisory_rules_apart_on_a_fail():
    message = message_for_rules_result(score(make_rep(elbow=120.0, trunk=30.0), ADVISORY_CONFIG))
    assert message == "Needs adjustment: trunk_lean (Advisory: elbow_straight)"


def test_default_feedback_text_is_unchanged():
    assert message_for_rules_result(score(make_rep(), BASE_CONFIG)) == "Good form!"
    assert message_for_rules_result(score(BENT, BASE_CONFIG)) == "Needs adjustment: elbow_straight"


# --- rule chip ----------------------------------------------------------------

def chips_for(result, config):
    return {
        rule_id: ui_theme.render_rule_chip(rule_id, r["measured"], tuple(r["allowed"]), r["passed"],
                                           advisory=rule_id in config.get("advisory_rules", []))
        for rule_id, r in result["rules"].items()
    }


def test_a_passing_rep_never_shows_a_failure_chip():
    result = score(BENT, ADVISORY_CONFIG)
    assert result["status"] == "pass"
    chips = chips_for(result, ADVISORY_CONFIG)
    assert not any(ui_theme.CANTALOUPE in chip or "cvp-chip-fail" in chip for chip in chips.values())


def test_advisory_chip_is_amber_and_keeps_its_text_label():
    chip = chips_for(score(BENT, ADVISORY_CONFIG), ADVISORY_CONFIG)["elbow_straight"]
    assert ui_theme.AMBER in chip
    assert "cvp-chip-advisory" in chip
    assert "elbow_straight: 120.0" in chip
    assert "advisory" in chip.split("cvp-chip-text")[1]


def test_advisory_rule_that_passes_shows_the_normal_pass_chip():
    chip = chips_for(score(make_rep(), ADVISORY_CONFIG), ADVISORY_CONFIG)["elbow_straight"]
    assert "cvp-chip-pass" in chip
    assert ui_theme.AMBER not in chip


def test_a_counted_rule_failure_keeps_the_failure_chip():
    chips = chips_for(score(make_rep(elbow=120.0, trunk=30.0), ADVISORY_CONFIG), ADVISORY_CONFIG)
    assert "cvp-chip-fail" in chips["trunk_lean"]
    assert "cvp-chip-advisory" in chips["elbow_straight"]


def test_default_chip_is_unchanged():
    chip = ui_theme.render_rule_chip("elbow_straight", 120.0, (135.0, 180.0), False)
    assert "cvp-chip-fail" in chip
    assert ui_theme.CANTALOUPE in chip
    assert "advisory" not in chip


# --- log and FORM -------------------------------------------------------------

def session_with(reps, config):
    tracker = SessionTracker(user_id="demo", exercise="shoulder_abduction")
    for rep in reps:
        result = score(rep, config)
        tracker.add_rep(RepRecord(
            exercise="shoulder_abduction", side="LEFT", angle_deg=max(rep["angle_deg"]), status=result["status"],
            confidence=None, failed_rules=";".join(result["failed_rules"]),
            advisory_failed_rules=";".join(result["advisory_failed"]),
        ))
    return tracker


def test_advisory_failures_are_logged_and_do_not_lower_form(tmp_path):
    reps = [make_rep(), BENT, BENT, make_rep(trunk=30.0)]
    tracker = session_with(reps, ADVISORY_CONFIG)
    summary = tracker.summary()
    assert (summary["passed_reps"], summary["failed_reps"], summary["accuracy_pct"]) == (3, 1, 75.0)
    with tracker.save(logs_dir=tmp_path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["advisory_failed_rules"] for r in rows] == ["", "elbow_straight", "elbow_straight", ""]
    assert [r["failed_rules"] for r in rows] == ["", "", "", "trunk_lean"]


def test_form_with_the_default_config_still_counts_the_elbow():
    summary = session_with([make_rep(), BENT, BENT, make_rep(trunk=30.0)], BASE_CONFIG).summary()
    assert (summary["passed_reps"], summary["failed_reps"], summary["accuracy_pct"]) == (1, 3, 25.0)
