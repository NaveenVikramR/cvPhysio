"""The alternative per-rep elbow statistic (config key "elbow_statistic"), on
synthetic reps. The default must stay the old minimum."""

import json
from pathlib import Path

import pytest

from src import rules as rules_module
from src.feature_extraction import LEFT
from src.pipeline import RepDetector, process_frame

FPS = 10.0
BASE_CONFIG = rules_module.load_config("shoulder_abduction")


def make_rep(angle, elbow, **extra):
    return {"angle_deg": angle, "elbow_angle_deg": elbow, "trunk_lean_deg": [5.0] * len(angle),
            "valid_fraction": 1.0, **extra}


def hold_config():
    return {**BASE_CONFIG, "elbow_statistic": "hold_median"}


# Peak at index 5 (t=0.5 s), so at 10 fps the window is indices 2..8.
ANGLE = [10, 30, 50, 70, 85, 90, 85, 70, 50, 30, 10]


def test_hold_median_uses_only_frames_within_the_window():
    elbow = [100, 100, 170, 171, 172, 173, 174, 175, 176, 100, 100]
    assert rules_module.hold_window_elbow(make_rep(ANGLE, elbow), FPS) == 173


def test_hold_window_edge_frames_are_included():
    elbow = [100, 100, 150, 150, 150, 150, 150, 150, 150, 100, 100]
    assert rules_module.hold_window_elbow(make_rep(ANGLE, elbow), FPS) == 150


def test_hold_window_follows_the_fps():
    elbow = [100, 100, 100, 100, 170, 170, 170, 100, 100, 100, 100]
    assert rules_module.hold_window_elbow(make_rep(ANGLE, elbow), 5.0) == 170  # +/-0.3 s is only +/-1 frame here
    assert rules_module.hold_window_elbow(make_rep(ANGLE, elbow), FPS) == 100  # +/-3 frames


def test_hold_window_uses_real_timestamps_across_gated_gaps():
    """Frames were gated out around the peak: the outer frames are two indices
    away but really 0.4 s, outside the window."""
    angle = [50, 60, 90, 60, 50]
    elbow = [100, 100, 170, 170, 100]
    rep = make_rep(angle, elbow, frame_t=[0.0, 0.1, 0.4, 0.5, 0.8])
    assert rules_module.hold_window_elbow(rep, FPS) == 170  # median of 100, 170, 170
    assert rules_module.hold_window_elbow(make_rep(angle, elbow), FPS) == 100  # index timing pulls in the outer frames


def test_hold_median_ignores_a_one_frame_glitch_that_min_would_fail():
    elbow = [170.0] * 11
    elbow[4] = 90.0
    rep = make_rep(ANGLE, elbow)
    assert rules_module.score_rep(rep, BASE_CONFIG, FPS)["rules"]["elbow_straight"]["passed"] is False
    assert rules_module.score_rep(rep, hold_config(), FPS)["rules"]["elbow_straight"]["passed"] is True


def test_hold_median_still_fails_a_held_bend():
    elbow = [170, 170, 120, 120, 120, 120, 120, 120, 120, 170, 170]
    result = rules_module.score_rep(make_rep(ANGLE, elbow), hold_config(), FPS)
    assert result["rules"]["elbow_straight"]["measured"] == 120
    assert "elbow_straight" in result["failed_rules"]


def test_hold_median_misses_a_bend_outside_the_window_by_design():
    elbow = [170, 100, 170, 170, 170, 170, 170, 170, 170, 170, 170]
    rep = make_rep(ANGLE, elbow)
    assert "elbow_straight" in rules_module.score_rep(rep, BASE_CONFIG, FPS)["failed_rules"]
    assert "elbow_straight" not in rules_module.score_rep(rep, hold_config(), FPS)["failed_rules"]


# --- selection and defaults ---------------------------------------------------

def test_default_is_the_old_minimum_when_the_key_is_absent():
    assert rules_module.elbow_statistic({}) == "min"
    elbow = [170.0] * 11
    elbow[1] = 140.0
    assert rules_module.score_rep(make_rep(ANGLE, elbow), {**BASE_CONFIG}, FPS)["rules"]["elbow_straight"]["measured"] == 140.0


def test_shipped_configs_keep_the_old_statistic():
    for path in Path("config/exercises").rglob("*.json"):
        assert "elbow_statistic" not in json.loads(path.read_text()), path


def test_min_can_be_selected_explicitly():
    elbow = [170.0] * 11
    elbow[1] = 140.0
    config = {**BASE_CONFIG, "elbow_statistic": "min"}
    assert rules_module.score_rep(make_rep(ANGLE, elbow), config, FPS)["rules"]["elbow_straight"]["measured"] == 140.0


def test_unknown_statistic_is_refused():
    with pytest.raises(ValueError, match="elbow_statistic"):
        rules_module.score_rep(make_rep(ANGLE, [170.0] * 11), {**BASE_CONFIG, "elbow_statistic": "mean"}, FPS)


# --- reference ----------------------------------------------------------------

def test_reference_records_and_uses_its_statistic():
    elbow = [170.0] * 11
    elbow[1] = 140.0
    rep = make_rep(ANGLE, elbow)
    old = rules_module.build_reference("r", "shoulder_abduction", [rep], FPS)
    new = rules_module.build_reference("r", "shoulder_abduction", [rep], FPS, "hold_median")
    assert (old["elbow_statistic"], old["medians"]["elbow_min"]) == ("min", 140.0)
    assert (new["elbow_statistic"], new["medians"]["elbow_min"]) == ("hold_median", 170.0)


def reference_config(statistic):
    return {**BASE_CONFIG, "mode": "reference_relative", "elbow_statistic": statistic,
            "tol_peak_deg": {"value": 10}, "tol_elbow_deg": {"value": 10},
            "tol_trunk_deg": {"value": 10}, "tol_tempo_frac": {"value": 0.5}}


def test_mismatched_reference_statistic_warns():
    rep = make_rep(ANGLE, [170.0] * 11)
    reference = rules_module.build_reference("r", "shoulder_abduction", [rep], FPS)  # built with min
    assert "re-record" in rules_module.score_rep(rep, reference_config("hold_median"), FPS, reference=reference)["warning"]
    assert rules_module.score_rep(rep, reference_config("min"), FPS, reference=reference)["warning"] is None


def test_older_reference_without_the_key_counts_as_min():
    rep = make_rep(ANGLE, [170.0] * 11)
    reference = rules_module.build_reference("r", "shoulder_abduction", [rep], FPS)
    del reference["elbow_statistic"]
    assert rules_module.score_rep(rep, reference_config("min"), FPS, reference=reference)["warning"] is None


# --- pipeline timestamps ------------------------------------------------------

def landmarks(arm_up, visibility=0.9):
    points = [(0.1 * i, 0.5, 0.0, 0.2) for i in range(33)]
    elbow = (0.3, 0.0, 0.0, visibility) if arm_up else (0.0, 0.3, 0.0, visibility)
    points[LEFT["shoulder"]] = (0.0, 0.0, 0.0, 0.9)
    points[LEFT["hip"]] = (0.0, 0.5, 0.0, 0.9)
    points[LEFT["elbow"]] = elbow
    points[LEFT["wrist"]] = (elbow[0] * 2, elbow[1] * 2, 0.0, 0.9)
    return points


def test_frame_timestamps_stay_aligned_with_gated_frames_and_reset_between_reps():
    detector = RepDetector()
    parallel = {"elbow": [], "trunk": [], "total_frames": 0}
    payloads = []
    t = 0.0

    def step(points):
        nonlocal t
        event, payload, _visible, _failed = process_frame(detector, parallel, points, "shoulder_abduction", None, t)
        t += 0.1
        if event == "rep_complete":
            payloads.append(payload)

    for _ in range(2):
        for _ in range(20):
            step(landmarks(False))
        for i in range(20):
            step(landmarks(True, 0.2 if 5 <= i < 8 else 0.9))
        step(landmarks(False))

    assert len(payloads) == 2
    for payload in payloads:
        assert len(payload["frame_t"]) == len(payload["angle_deg"]) == len(payload["elbow_angle_deg"])
        assert payload["frame_t"] == sorted(payload["frame_t"])
    assert max(b - a for a, b in zip(payloads[0]["frame_t"], payloads[0]["frame_t"][1:])) == pytest.approx(0.4)
    assert payloads[1]["frame_t"][0] > payloads[0]["frame_t"][-1]
