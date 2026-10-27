"""Tests for src/pipeline.py's Session: IDLE -> CALIBRATING -> ACTIVE ->
STOPPED. Frames fed in while IDLE or STOPPED must never reach
RepDetector -- this is what app.py's Start/Stop controls and --span
(headless mode) both rely on.
"""

from src.feature_extraction import LEFT
from src.pipeline import Session, SessionState


def make_landmarks(overrides=None):
    landmarks = [(0.1 * i, 0.5, 0.0, 1.0) for i in range(33)]
    for idx, point in (overrides or {}).items():
        landmarks[idx] = point
    return landmarks


def test_no_reps_before_start():
    session = Session("shoulder_abduction")
    landmarks = make_landmarks()

    assert session.session_state == SessionState.IDLE
    for i in range(20):
        event, payload, visible, failed = session.process_frame(landmarks, t=float(i))
        assert event == "idle"
        assert payload is None
        assert visible is False
    assert session.session_state == SessionState.IDLE
    assert session.detector is None  # frames never even constructed a RepDetector


def test_calibration_begins_at_start():
    session = Session("shoulder_abduction")
    landmarks = make_landmarks()

    # frames before Start are ignored (not even buffered toward calibration)
    for i in range(5):
        session.process_frame(landmarks, t=float(i))
    assert session.session_state == SessionState.IDLE

    session.start()
    assert session.session_state == SessionState.CALIBRATING
    assert session.detector is not None
    assert session.detector.calib_attempt_frames == 0  # a fresh RepDetector, not primed by the ignored frames

    # CALIBRATION_FRAMES (15) consistent, visible frames -> CALIBRATING -> ACTIVE (RESTING)
    for i in range(15):
        session.process_frame(landmarks, t=float(i))
    assert session.session_state == SessionState.ACTIVE
    assert session.detector.state == "RESTING"


def test_nothing_after_stop():
    session = Session("shoulder_abduction")
    landmarks = make_landmarks()
    session.start()
    for i in range(15):
        session.process_frame(landmarks, t=float(i))
    assert session.session_state == SessionState.ACTIVE

    session.stop()
    assert session.session_state == SessionState.STOPPED

    detector_snapshot_state = session.detector.state
    for i in range(20):
        event, payload, visible, failed = session.process_frame(landmarks, t=float(15 + i))
        assert event == "idle"
        assert payload is None
        assert visible is False
    assert session.session_state == SessionState.STOPPED
    assert session.detector.state == detector_snapshot_state  # untouched by the ignored post-Stop frames


def test_start_is_a_noop_once_already_started_or_stopped():
    session = Session("shoulder_abduction")
    session.start()
    detector_first = session.detector
    session.start()  # already CALIBRATING -- must not replace the detector / reset calibration progress
    assert session.detector is detector_first

    session.stop()
    session.start()  # no restart after Stop
    assert session.session_state == SessionState.STOPPED
    assert session.detector is detector_first
