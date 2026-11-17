"""F2 step 2: the opt-in constant shoulder-angle offset
(config/angle_correction.json, docs/landmark_fidelity.md)."""

import subprocess

import pytest

import app
from src.feature_extraction import correct_angle, extract_features, load_angle_correction
from src.pipeline import Session
from tests.test_side_lock import body, rep_frames, run

CORRECTION = {"shoulder_abduction": -4.82}


def test_repo_config_corrects_abduction_only():
    correction = load_angle_correction()
    assert correction == {"shoulder_abduction": pytest.approx(-4.82)}
    assert "shoulder_rotation" not in correction  # no mocap truth for rotation yet


def test_correct_angle_is_opt_in_and_clipped():
    assert correct_angle(50.0, "shoulder_abduction", None) == 50.0
    assert correct_angle(50.0, "shoulder_rotation", CORRECTION) == 50.0
    assert correct_angle(50.0, "shoulder_abduction", CORRECTION) == pytest.approx(54.82)
    assert correct_angle(178.0, "shoulder_abduction", CORRECTION) == 180.0
    assert correct_angle(3.0, "shoulder_abduction", {"shoulder_abduction": 5.0}) == 0.0


def test_extract_features_default_is_raw():
    lm = body(right_deg=60.0, left_deg=10.0)
    raw = extract_features(lm, "shoulder_abduction", "RIGHT")
    fixed = extract_features(lm, "shoulder_abduction", "RIGHT", CORRECTION)
    assert raw["angle_deg"] == pytest.approx(60.0, abs=1e-6)
    assert fixed["angle_deg"] == pytest.approx(64.82, abs=1e-6)


@pytest.mark.parametrize("side", ["RIGHT", "auto", "per_frame"])
def test_session_same_reps_angles_shifted(side):
    def completes(correction):
        session = Session("shoulder_abduction", side=side, correction=correction)
        session.start()
        return [p for e, p in run(session, rep_frames("right", n_reps=2)) if e == "rep_complete"]

    raw, fixed = completes(None), completes(CORRECTION)
    assert len(raw) == len(fixed)
    for r, f in zip(raw, fixed):
        assert len(r["angle_deg"]) == len(f["angle_deg"])
        assert f["angle_deg"] == pytest.approx([a + 4.82 for a in r["angle_deg"]])
        assert f["elbow_angle_deg"] == r["elbow_angle_deg"]  # only the primary angle is corrected


def test_app_applies_correction_to_rules_scorer_only():
    assert app.active_angle_correction("rules") == load_angle_correction()
    assert app.active_angle_correction("rules", enabled=False) is None
    assert app.active_angle_correction("classifier") is None


def test_reference_mismatch_warning():
    legacy = {"name": "old", "exercise": "shoulder_abduction"}  # saved before the correction existed
    assert app.reference_correction_warning(legacy, None) is None
    warning = app.reference_correction_warning(legacy, CORRECTION)
    assert warning and "raw angles" in warning and "4.8 deg lower" in warning
    tagged = dict(legacy, angle_offset_deg=-4.82)
    assert app.reference_correction_warning(tagged, CORRECTION) is None
    assert "4.8 deg higher" in app.reference_correction_warning(tagged, None)


@pytest.mark.parametrize("path,blob", [
    ("src/rules.py", "4e85c2b7"),
    ("config/pose.json", "c15ab25f"),
    ("config/exercises/shoulder_abduction.json", "79c406f5"),
    ("scripts/evaluate_own_clips.py", "4382f70f"),
])
def test_pinned_files_untouched(path, blob):
    out = subprocess.run(["git", "hash-object", path], capture_output=True, text=True, check=True).stdout
    assert out.startswith(blob)
