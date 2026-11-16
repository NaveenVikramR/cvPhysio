"""config/exercises/shoulder_rotation.json (open_questions.md #11): loads via
src.rules.load_config, works with the unchanged score_rep(), and keeps the
properties scripts/build_rotation_config.py documents."""

import json
from pathlib import Path

import pytest

from src import rules as rules_module
from src.pipeline import apply_advisory_rules

ROTATION = Path("config/exercises/shoulder_rotation.json")


@pytest.fixture(scope="module")
def config():
    return rules_module.load_config("shoulder_rotation")


def _rep(peak, elbow_min=90.0, n=90, fps=30.0):
    angle = [90.0] * (n - 1) + [peak]
    elbow = [95.0] * (n - 1) + [elbow_min]
    return {"angle_deg": angle, "elbow_angle_deg": elbow, "trunk_lean_deg": [2.0] * n, "valid_fraction": 1.0}


def test_has_every_key_score_rep_needs(config):
    abduction = rules_module.load_config("shoulder_abduction")
    needed = {"target_peak_deg", "tolerance_deg", "min_elbow_straight_deg", "max_trunk_lean_deg",
              "tempo_min_s", "tempo_max_s", "cue_cooldown_s", "min_valid_fraction", "cues"}
    assert needed <= set(config)
    assert set(config["cues"]) == set(abduction["cues"])


def test_peak_band_includes_180(config):
    lo, hi = rules_module._absolute_bounds(config)["peak_angle"]
    assert hi >= 180.0 and lo < 180.0


def test_full_range_rep_passes(config):
    result = apply_advisory_rules(rules_module.score_rep(_rep(180.0), config, 30.0), config)
    assert result["status"] == "pass"


def test_short_range_rep_fails_peak(config):
    result = apply_advisory_rules(rules_module.score_rep(_rep(150.0), config, 30.0), config)
    assert result["status"] == "fail" and result["failed_rules"] == ["peak_angle"]


def test_elbow_is_advisory_only(config):
    assert config["advisory_rules"] == ["elbow_straight"]
    result = apply_advisory_rules(rules_module.score_rep(_rep(180.0, elbow_min=40.0), config, 30.0), config)
    assert result["status"] == "pass" and result["advisory_failed"] == ["elbow_straight"]


def test_non_derived_values_are_flagged(config):
    for key in ("max_trunk_lean_deg", "tempo_min_s", "tempo_max_s"):
        assert config[key]["status"] == "provisional"
        assert "NOT data-derived" in config[key]["comment"]


def test_derivation_and_held_out_subjects_disjoint(config):
    assert not set(config["derivation_subjects"]) & set(config["held_out_subjects"])
    assert len(config["derivation_subjects"]) + len(config["held_out_subjects"]) == 10


def test_abduction_config_untouched_by_this_feature():
    # The pre-registration pins shoulder_abduction.json's blob id; this file
    # must not have been regenerated or edited alongside the rotation config.
    import subprocess
    blob = subprocess.run(["git", "hash-object", "config/exercises/shoulder_abduction.json"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert blob == "79c406f505c73c2ea87671b3773ea81e99e16848"
