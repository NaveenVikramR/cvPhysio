"""Tests for app.py's RepDetector segmenter: the tracking-lost rescue
(PART C) and the discard reason codes (too_short, too_long,
lost_tracking), on synthetic angle sequences (no video).
"""

from conftest import load_app_rep_detector

RepDetector = load_app_rep_detector()

BASELINE = 20.0


def calibrate(detector) -> None:
    """Drives CALIBRATING -> RESTING at a fixed baseline, via 15 frames
    of a constant, visible angle (matches CALIBRATION_FRAMES=15,
    CALIBRATION_MAX_STD_DEG=6.0 in app.py)."""
    for _ in range(15):
        event, _payload = detector.update(True, BASELINE, "RIGHT")
        assert event == "none"
    assert detector.state == "RESTING"
    assert detector.baseline == BASELINE


def enter_moving(detector, first_angle: float) -> None:
    event, _payload = detector.update(True, first_angle, "RIGHT")
    assert event == "none"
    assert detector.state == "MOVING"


def test_tracking_lost_after_already_near_rest_band_is_emitted_not_discarded():
    """A rep that already climbed to a real peak and was heading back
    toward rest (last visible angle only 15deg from baseline -- outside
    the strict 10deg REST_BAND_DEG, so it didn't complete normally yet,
    but within the rescue's wider tolerance) must be emitted as
    rep_complete if tracking is then lost, not thrown away."""
    detector = RepDetector()
    calibrate(detector)
    enter_moving(detector, 60.0)  # |60-20| = 40, outside rest band -> MOVING

    # climb and come back down to near (but not within) the rest band --
    # enough frames/range to qualify as a real rep (>= MIN_REP_FRAMES=15,
    # >= MIN_REP_RANGE_DEG=25).
    angles = [65, 70, 75, 80, 85, 90, 95, 90, 80, 70, 60, 50, 40, 35]  # last: |35-20|=15
    for angle in angles:
        event, _payload = detector.update(True, float(angle), "RIGHT")
        assert event == "none"
    assert len(detector.rep_buffer) == 15  # 1 (entry) + 14
    assert abs(detector.rep_buffer[-1] - BASELINE) == 15  # outside REST_BAND_DEG(10), within RESCUE(20)

    # tracking drops for longer than CONFIDENCE_TOLERANCE_FRAMES (10)
    event, payload = None, None
    for _ in range(11):
        event, payload = detector.update(False, None, None)

    assert event == "rep_complete"
    assert payload["angle_deg"][-1] == 35.0
    assert detector.state == "RESTING"


def test_tracking_lost_before_returning_near_rest_is_discarded_as_lost_tracking():
    """If the last visible angle was still far from baseline (the rep
    genuinely had not returned to rest yet), losing tracking must
    discard it with reason_code='lost_tracking', not fabricate a
    completion."""
    detector = RepDetector()
    calibrate(detector)
    enter_moving(detector, 60.0)

    angles = [65, 70, 75, 80, 85, 90, 95, 90, 85, 80, 75, 70, 65, 62]  # last: |62-20|=42, far outside RESCUE(20) too
    for angle in angles:
        detector.update(True, float(angle), "RIGHT")
    assert len(detector.rep_buffer) == 15  # 1 (entry) + 14 -- same length as the rescued test, only the ending differs
    assert abs(detector.rep_buffer[-1] - BASELINE) == 42

    event, payload = None, None
    for _ in range(11):
        event, payload = detector.update(False, None, None)

    assert event == "rep_discarded"
    assert payload["reason_code"] == "lost_tracking"
    assert detector.state == "RESTING"


def test_discard_reason_too_short_on_brief_small_movement():
    detector = RepDetector()
    calibrate(detector)
    enter_moving(detector, 35.0)  # |35-20|=15, outside rest band -> MOVING

    # immediately comes back within the rest band -- too few frames / too
    # little range to count as a real rep.
    event, payload = detector.update(True, 22.0, "RIGHT")  # |22-20|=2 <= REST_BAND_DEG
    assert event == "rep_discarded"
    assert payload["reason_code"] == "too_short"
    assert detector.state == "RESTING"


def test_discard_reason_too_long_when_rep_runs_past_max_frames():
    detector = RepDetector()
    calibrate(detector)
    enter_moving(detector, 60.0)

    event, payload = None, None
    # stay well outside the rest band for more than MAX_REP_FRAMES (450) frames
    for _ in range(460):
        event, payload = detector.update(True, 60.0, "RIGHT")
        if event != "none":
            break

    assert event == "rep_discarded"
    assert payload["reason_code"] == "too_long"
    assert detector.state == "RESTING"
