"""Per-rep diagnostics and timing for the live path, on synthetic landmarks."""

import csv

import pytest

from src import rules as rules_module
from src.feature_extraction import LEFT
from src.pipeline import RepDetector, process_frame
from src.tracking import RepRecord, SessionTracker

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
        wrist = 0.4 if i < low_wrist_frames else 0.9
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
