"""Tests for src/rules.py's rule-based live scorer, on synthetic rep
arrays (no video, no UI-PRMD) -- each rule tested in isolation, pass and
fail, plus the cue cooldown.
"""

import pytest

from src.rules import CueCooldown, load_config, score_rep

FPS = 30.0


def make_config(
    target_peak_deg=90, tolerance_deg=15, min_elbow_straight_deg=150,
    max_trunk_lean_deg=15, tempo_min_s=1.0, tempo_max_s=6.0,
):
    return {
        "target_peak_deg": {"value": target_peak_deg, "status": "provisional"},
        "tolerance_deg": {"value": tolerance_deg, "status": "provisional"},
        "min_elbow_straight_deg": {"value": min_elbow_straight_deg, "status": "provisional"},
        "max_trunk_lean_deg": {"value": max_trunk_lean_deg, "status": "provisional"},
        "tempo_min_s": {"value": tempo_min_s, "status": "provisional"},
        "tempo_max_s": {"value": tempo_max_s, "status": "provisional"},
        "cue_cooldown_s": {"value": 8, "status": "provisional"},
        "cues": {
            "peak_angle": "peak cue", "elbow_straight": "elbow cue",
            "trunk_lean": "trunk cue", "tempo": "tempo cue",
        },
    }


def make_rep(n_frames, peak=90.0, elbow=170.0, trunk=5.0):
    """n_frames at the given plateau values, framed by a short ramp up
    from 0 and back down so angle_deg genuinely has `peak` as its max."""
    angle = [0.0] + [peak] * n_frames + [0.0]
    elbow_seq = [elbow] * len(angle)
    trunk_seq = [trunk] * len(angle)
    return {"angle_deg": angle, "elbow_angle_deg": elbow_seq, "trunk_lean_deg": trunk_seq}


def test_all_rules_pass_within_thresholds():
    config = make_config()
    rep = make_rep(n_frames=60, peak=90.0, elbow=170.0, trunk=5.0)  # 62 frames / 30fps ~= 2.07s
    result = score_rep(rep, config, fps=FPS)
    assert result["passed"] is True
    assert result["failed_rules"] == []
    for rule_id, r in result["rules"].items():
        assert r["passed"] is True, rule_id


def test_peak_angle_rule_fails_when_too_low():
    config = make_config(target_peak_deg=90, tolerance_deg=15)
    rep = make_rep(n_frames=60, peak=50.0, elbow=170.0, trunk=5.0)
    result = score_rep(rep, config, fps=FPS)
    assert "peak_angle" in result["failed_rules"]
    assert result["rules"]["peak_angle"]["measured"] == 50.0
    assert result["rules"]["peak_angle"]["allowed"] == [75, 105]
    assert result["passed"] is False


def test_peak_angle_rule_fails_when_too_high():
    config = make_config(target_peak_deg=90, tolerance_deg=15)
    rep = make_rep(n_frames=60, peak=140.0, elbow=170.0, trunk=5.0)
    result = score_rep(rep, config, fps=FPS)
    assert "peak_angle" in result["failed_rules"]


def test_elbow_straight_rule_fails_when_bent():
    config = make_config(min_elbow_straight_deg=150)
    rep = make_rep(n_frames=60, peak=90.0, elbow=100.0, trunk=5.0)
    result = score_rep(rep, config, fps=FPS)
    assert "elbow_straight" in result["failed_rules"]
    assert result["rules"]["elbow_straight"]["measured"] == 100.0
    assert result["passed"] is False


def test_elbow_straight_rule_passes_when_near_straight():
    config = make_config(min_elbow_straight_deg=150)
    rep = make_rep(n_frames=60, peak=90.0, elbow=175.0, trunk=5.0)
    result = score_rep(rep, config, fps=FPS)
    assert "elbow_straight" not in result["failed_rules"]


def test_trunk_lean_rule_fails_when_over_limit():
    config = make_config(max_trunk_lean_deg=15)
    rep = make_rep(n_frames=60, peak=90.0, elbow=170.0, trunk=30.0)
    result = score_rep(rep, config, fps=FPS)
    assert "trunk_lean" in result["failed_rules"]
    assert result["rules"]["trunk_lean"]["measured"] == 30.0
    assert result["passed"] is False


def test_trunk_lean_rule_passes_when_within_limit():
    config = make_config(max_trunk_lean_deg=15)
    rep = make_rep(n_frames=60, peak=90.0, elbow=170.0, trunk=10.0)
    result = score_rep(rep, config, fps=FPS)
    assert "trunk_lean" not in result["failed_rules"]


def test_tempo_rule_fails_when_too_fast():
    config = make_config(tempo_min_s=1.0, tempo_max_s=6.0)
    rep = make_rep(n_frames=5, peak=90.0, elbow=170.0, trunk=5.0)  # 7 frames / 30fps ~= 0.23s
    result = score_rep(rep, config, fps=FPS)
    assert "tempo" in result["failed_rules"]
    assert result["passed"] is False


def test_tempo_rule_fails_when_too_slow():
    config = make_config(tempo_min_s=1.0, tempo_max_s=6.0)
    rep = make_rep(n_frames=300, peak=90.0, elbow=170.0, trunk=5.0)  # 302 frames / 30fps ~= 10s
    result = score_rep(rep, config, fps=FPS)
    assert "tempo" in result["failed_rules"]


def test_tempo_rule_passes_within_range():
    config = make_config(tempo_min_s=1.0, tempo_max_s=6.0)
    rep = make_rep(n_frames=60, peak=90.0, elbow=170.0, trunk=5.0)
    result = score_rep(rep, config, fps=FPS)
    assert "tempo" not in result["failed_rules"]


def test_multiple_rules_can_fail_at_once():
    config = make_config()
    rep = make_rep(n_frames=60, peak=40.0, elbow=90.0, trunk=40.0)
    result = score_rep(rep, config, fps=FPS)
    assert set(result["failed_rules"]) == {"peak_angle", "elbow_straight", "trunk_lean"}
    assert result["passed"] is False


def test_score_rep_rejects_mismatched_lengths():
    config = make_config()
    rep = {"angle_deg": [1.0, 2.0], "elbow_angle_deg": [1.0], "trunk_lean_deg": [1.0, 2.0]}
    with pytest.raises(ValueError):
        score_rep(rep, config, fps=FPS)


def test_load_config_reads_real_shoulder_abduction_file():
    config = load_config("shoulder_abduction")
    for key in (
        "target_peak_deg", "tolerance_deg", "min_elbow_straight_deg",
        "max_trunk_lean_deg", "tempo_min_s", "tempo_max_s", "cue_cooldown_s",
    ):
        assert config[key]["status"] == "provisional"
    assert "cues" in config


def test_load_config_raises_clear_error_for_unknown_exercise():
    with pytest.raises(FileNotFoundError):
        load_config("not_a_real_exercise")


# --- CueCooldown -------------------------------------------------------

def test_cue_fires_first_time():
    cooldown = CueCooldown(cooldown_s=8)
    assert cooldown.should_fire("peak_angle", now=0.0) is True


def test_cue_suppressed_within_cooldown_window():
    cooldown = CueCooldown(cooldown_s=8)
    cooldown.record("peak_angle", now=0.0)
    assert cooldown.should_fire("peak_angle", now=3.0) is False


def test_cue_fires_again_after_cooldown_elapses():
    cooldown = CueCooldown(cooldown_s=8)
    cooldown.record("peak_angle", now=0.0)
    assert cooldown.should_fire("peak_angle", now=8.0) is True


def test_cue_fires_exactly_at_boundary():
    cooldown = CueCooldown(cooldown_s=8)
    cooldown.record("peak_angle", now=10.0)
    assert cooldown.should_fire("peak_angle", now=17.999) is False
    assert cooldown.should_fire("peak_angle", now=18.0) is True


def test_different_cue_ids_have_independent_cooldowns():
    cooldown = CueCooldown(cooldown_s=8)
    cooldown.record("peak_angle", now=0.0)
    assert cooldown.should_fire("trunk_lean", now=1.0) is True


def test_cues_to_speak_returns_only_failed_rules_not_in_cooldown():
    cooldown = CueCooldown(cooldown_s=8)
    cue_text = {"peak_angle": "peak cue", "trunk_lean": "trunk cue", "tempo": "tempo cue"}

    messages = cooldown.cues_to_speak(["peak_angle", "trunk_lean"], cue_text, now=0.0)
    assert set(messages) == {"peak cue", "trunk cue"}

    # peak_angle fails again immediately -- still cooling down, suppressed.
    # tempo fails for the first time -- fires.
    messages = cooldown.cues_to_speak(["peak_angle", "tempo"], cue_text, now=1.0)
    assert messages == ["tempo cue"]


def test_cues_to_speak_records_firing_so_repeats_are_suppressed():
    cooldown = CueCooldown(cooldown_s=8)
    cue_text = {"peak_angle": "peak cue"}

    first = cooldown.cues_to_speak(["peak_angle"], cue_text, now=0.0)
    second = cooldown.cues_to_speak(["peak_angle"], cue_text, now=2.0)
    third = cooldown.cues_to_speak(["peak_angle"], cue_text, now=9.0)

    assert first == ["peak cue"]
    assert second == []
    assert third == ["peak cue"]
