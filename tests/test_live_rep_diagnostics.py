"""Per-rep diagnostics and timing for the live path, on synthetic landmarks."""

import csv
from datetime import datetime, timezone

import pytest

from src import rules as rules_module
from src.feature_extraction import LEFT
from src.pipeline import RepDetector, live_rep_fps, process_frame
from src.tracking import RepRecord, SessionTracker, rep_start_timestamp

REST_FRAMES = 20
REP_FRAMES = 30


def make_landmarks(arm_up: bool, wrist_visibility: float = 0.9):
    """Left arm selected: the right side is kept below the left's visibility."""
    landmarks = [(0.1 * i, 0.5, 0.0, 0.2) for i in range(33)]
    elbow = (0.3, 0.0, 0.0, 0.9) if arm_up else (0.0, 0.3, 0.0, 0.9)
    landmarks[LEFT["shoulder"]] = (0.0, 0.0, 0.0, 0.9)
    landmarks[LEFT["hip"]] = (0.0, 0.5, 0.0, 0.9)
    landmarks[LEFT["elbow"]] = elbow
    landmarks[LEFT["wrist"]] = (elbow[0] * 2, elbow[1] * 2, 0.0, wrist_visibility)
    return landmarks


def run_one_rep(frame_dt: float, low_wrist_frames: int = 0, required_roles=None):
    """Rest, then one rep of REP_FRAMES raised frames plus the frame that
    returns to rest. Returns the rep_complete payload."""
    detector = RepDetector()
    parallel = {"elbow": [], "trunk": [], "total_frames": 0}
    t = 100.0
    for _ in range(REST_FRAMES):
        process_frame(detector, parallel, make_landmarks(False), "shoulder_abduction", None, t, required_roles)
        t += frame_dt
    for i in range(REP_FRAMES):
        wrist = 0.4 if 5 <= i < 5 + low_wrist_frames else 0.9  # mid-rep, once the rep has started
        process_frame(detector, parallel, make_landmarks(True, wrist), "shoulder_abduction", None, t, required_roles)
        t += frame_dt
    event, payload, _visible, _failed = process_frame(
        detector, parallel, make_landmarks(False), "shoulder_abduction", None, t, required_roles,
    )
    assert event == "rep_complete"
    return payload


# --- visibility fractions ---------------------------------------------------

def test_elbow_wrist_fraction_counts_frames_where_both_clear_the_gate():
    payload = run_one_rep(1 / 30, low_wrist_frames=6)  # default gate doesn't require the wrist
    assert payload["elbow_wrist_visible_fraction"] == pytest.approx((REP_FRAMES + 1 - 6) / (REP_FRAMES + 1))
    assert payload["valid_fraction"] == 1.0


def test_elbow_wrist_fraction_is_one_with_every_frame_visible():
    assert run_one_rep(1 / 30)["elbow_wrist_visible_fraction"] == 1.0


def test_fractions_reset_between_reps():
    detector = RepDetector()
    parallel = {"elbow": [], "trunk": [], "total_frames": 0}
    payloads = []
    t = 0.0
    for wrist in (0.4, 0.9):
        for _ in range(REST_FRAMES):
            process_frame(detector, parallel, make_landmarks(False), "shoulder_abduction", None, t)
            t += 1 / 30
        for _ in range(REP_FRAMES):
            process_frame(detector, parallel, make_landmarks(True, wrist), "shoulder_abduction", None, t)
            t += 1 / 30
        _event, payload, _visible, _failed = process_frame(detector, parallel, make_landmarks(False), "shoulder_abduction", None, t)
        payloads.append(payload)
    assert payloads[0]["elbow_wrist_visible_fraction"] == pytest.approx(1 / (REP_FRAMES + 1))
    assert payloads[1]["elbow_wrist_visible_fraction"] == 1.0


# --- live rep timing --------------------------------------------------------

@pytest.mark.parametrize("loop_fps", [10.0, 17.0, 30.0])
def test_tempo_is_the_real_duration_at_any_loop_rate(loop_fps):
    """Regression: frames / the camera's nominal 30 fps halved a rep's tempo
    when the loop really ran at 15 fps."""
    payload = run_one_rep(1 / loop_fps)
    real_duration = REP_FRAMES / loop_fps
    assert payload["duration_s"] == pytest.approx(real_duration)
    rep = {**payload, "valid_fraction": 1.0}
    measured = rules_module.rep_summary_features(rep, live_rep_fps(payload, 30.0))["tempo_s"]
    assert measured == pytest.approx(real_duration)


def test_tempo_includes_frames_rejected_by_the_gate():
    roles = rules_module.required_joint_roles()
    payload = run_one_rep(1 / 15, low_wrist_frames=5, required_roles=roles)
    assert len(payload["angle_deg"]) == REP_FRAMES + 1 - 5
    measured = rules_module.rep_summary_features(payload, live_rep_fps(payload, 30.0))["tempo_s"]
    assert measured == pytest.approx(REP_FRAMES / 15)


def test_score_rep_reports_the_real_duration_to_the_tempo_rule():
    payload = run_one_rep(1 / 15)
    config = rules_module.load_config("shoulder_abduction")
    result = rules_module.score_rep({**payload, "valid_fraction": 1.0}, config, live_rep_fps(payload, 30.0))
    assert result["rules"]["tempo"]["measured"] == pytest.approx(2.0)


def test_live_rep_fps_falls_back_without_a_measured_duration():
    assert live_rep_fps({"angle_deg": [1.0] * 10, "duration_s": None}, 30.0) == 30.0
    assert live_rep_fps({"angle_deg": [1.0] * 10, "duration_s": 0.0}, 30.0) == 30.0
    assert live_rep_fps({"angle_deg": [1.0] * 10}, 30.0) == 30.0


def test_build_reference_times_each_rep_with_its_own_fps():
    rep = {"angle_deg": [90.0] * 30, "elbow_angle_deg": [170.0] * 30, "trunk_lean_deg": [5.0] * 30}
    reference = rules_module.build_reference("r", "shoulder_abduction", [{**rep, "fps": 15.0}], 30.0)
    assert reference["medians"]["tempo_s"] == pytest.approx(2.0)
    assert rules_module.build_reference("r", "shoulder_abduction", [rep], 30.0)["medians"]["tempo_s"] == pytest.approx(1.0)


def test_rep_start_timestamp_subtracts_the_duration():
    now = datetime(2026, 10, 30, 10, 0, 10, tzinfo=timezone.utc)
    assert rep_start_timestamp(4.0, now) == "2026-10-30T10:00:06+00:00"
    assert rep_start_timestamp(None, now) is None


# --- scoring mode -----------------------------------------------------------

REFERENCE = {"name": "r", "medians": {}}


@pytest.mark.parametrize("config, reference, expected", [
    ({}, None, "absolute"),
    ({}, REFERENCE, "absolute"),  # the live config: a selected reference has no effect
    ({"mode": "absolute"}, REFERENCE, "absolute"),
    ({"mode": "reference_relative"}, None, "absolute"),  # score_rep() falls back
    ({"mode": "reference_relative"}, REFERENCE, "reference_relative"),
])
def test_active_scoring_mode_reports_what_score_rep_applies(config, reference, expected):
    assert rules_module.active_scoring_mode(config, reference) == expected


def test_reference_ignored_message_only_when_a_selected_reference_is_not_applied():
    assert "NOT applied" in rules_module.reference_ignored_message({}, REFERENCE)
    assert rules_module.reference_ignored_message({"mode": "reference_relative"}, REFERENCE) is None
    assert rules_module.reference_ignored_message({}, None) is None


def test_shipped_live_config_does_not_apply_a_selected_reference():
    config = rules_module.load_config("shoulder_abduction")
    assert rules_module.active_scoring_mode(config, REFERENCE) == "absolute"


# --- session CSV ------------------------------------------------------------

def test_session_csv_carries_the_rule_measurements(tmp_path):
    tracker = SessionTracker(user_id="demo", exercise="shoulder_abduction")
    tracker.add_rep(RepRecord(
        exercise="shoulder_abduction", side="LEFT", angle_deg=88.0, status="fail", confidence=None,
        elbow_min_deg=127.5, trunk_max_deg=13.4, tempo_s=3.2, failed_rules="elbow_straight;tempo",
        scoring_mode="absolute", reference_name="demo_ref", valid_fraction=0.9,
        elbow_wrist_visible_fraction=0.8, n_valid_frames=55,
    ))
    with tracker.save(logs_dir=tmp_path).open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["elbow_min_deg"] == "127.5"
    assert row["trunk_max_deg"] == "13.4"
    assert row["tempo_s"] == "3.2"
    assert row["failed_rules"] == "elbow_straight;tempo"
    assert row["scoring_mode"] == "absolute"
    assert row["reference_name"] == "demo_ref"
    assert row["valid_fraction"] == "0.9"
    assert row["elbow_wrist_visible_fraction"] == "0.8"
    assert row["n_valid_frames"] == "55"
