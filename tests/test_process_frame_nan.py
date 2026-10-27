"""PART of item 1's test requirement: a degenerate (coincident-landmark)
frame producing a NaN angle must not crash app.py's process_frame(), the
RepDetector segmenter, or src.rules.score_rep() -- it must instead be
excluded, the same way a low-confidence frame is.
"""

from src.feature_extraction import LEFT
from src.pipeline import RepDetector, process_frame
from src.rules import score_rep


def make_landmarks(overrides=None):
    landmarks = [(0.1 * i, 0.5, 0.0, 1.0) for i in range(33)]
    for idx, point in (overrides or {}).items():
        landmarks[idx] = point
    return landmarks


def make_degenerate_landmarks():
    """Fully visible, but shoulder and elbow coincide -- angle is NaN."""
    return make_landmarks({
        LEFT["shoulder"]: (0.5, 0.5, 0.0, 1.0),
        LEFT["elbow"]: (0.5, 0.5, 0.0, 1.0),
    })


def test_process_frame_does_not_crash_on_degenerate_landmarks():
    detector = RepDetector()
    parallel = {"elbow": [], "trunk": [], "total_frames": 0}
    landmarks = make_degenerate_landmarks()

    # Should not raise, for any number of repeated degenerate frames.
    for i in range(20):
        event, payload, visible, failed = process_frame(
            detector, parallel, landmarks, "shoulder_abduction", None, float(i), None,
        )
        assert visible is False


def test_degenerate_frame_is_excluded_not_appended_to_parallel_buffers():
    detector = RepDetector()
    parallel = {"elbow": [], "trunk": [], "total_frames": 0}
    landmarks = make_degenerate_landmarks()

    for i in range(20):
        process_frame(detector, parallel, landmarks, "shoulder_abduction", None, float(i), None)

    assert parallel["elbow"] == []
    assert parallel["trunk"] == []
    assert detector.state == "CALIBRATING"  # never saw a single valid frame


def test_mixed_healthy_and_degenerate_frames_keep_buffers_equal_length():
    """A rep built from a mix of healthy and degenerate frames must still
    leave RepDetector's own angle buffer and the parallel elbow/trunk
    buffers at equal length -- score_rep() requires this."""
    detector = RepDetector()
    parallel = {"elbow": [], "trunk": [], "total_frames": 0}
    healthy = make_landmarks()
    degenerate = make_degenerate_landmarks()

    t = 0.0
    # calibrate on healthy frames
    for _ in range(15):
        process_frame(detector, parallel, healthy, "shoulder_abduction", None, t, None)
        t += 1 / 30
    assert detector.state == "RESTING"

    # a degenerate frame while resting must not crash and must not move the state
    process_frame(detector, parallel, degenerate, "shoulder_abduction", None, t, None)
    t += 1 / 30
    assert detector.state == "RESTING"

    payload = None
    for angle_x in [0.3, 0.4, 0.5, 0.3]:
        landmarks = make_landmarks({LEFT["elbow"]: (angle_x, 0.9, 0.0, 1.0)})
        event, payload, _visible, _failed = process_frame(
            detector, parallel, landmarks, "shoulder_abduction", None, t, None,
        )
        t += 1 / 30

    # whatever happened (rep still building, discarded, or completed),
    # the buffers must never have diverged in length.
    assert len(parallel["elbow"]) == len(parallel["trunk"])
    if payload and "angle_deg" in payload:
        assert len(payload["angle_deg"]) == len(payload.get("elbow_angle_deg", payload["angle_deg"]))


def test_score_rep_does_not_crash_if_a_nan_somehow_reaches_it():
    # Defensive: even if a NaN slipped past the gate, score_rep() itself
    # must not raise -- Python's max()/min() tolerate NaN (no exception,
    # just an unreliable result), so this should already hold.
    config = {
        "target_peak_deg": {"value": 90}, "tolerance_deg": {"value": 15},
        "min_elbow_straight_deg": {"value": 135}, "max_trunk_lean_deg": {"value": 15},
        "tempo_min_s": {"value": 1.0}, "tempo_max_s": {"value": 6.0},
        "min_valid_fraction": {"value": 0.7}, "cues": {},
    }
    rep = {
        "angle_deg": [10.0, float("nan"), 90.0, 10.0],
        "elbow_angle_deg": [170.0, 170.0, 170.0, 170.0],
        "trunk_lean_deg": [5.0, 5.0, 5.0, 5.0],
        "valid_fraction": 1.0,
    }
    result = score_rep(rep, config, fps=30.0)
    assert result["status"] in ("pass", "fail", "insufficient_data")
