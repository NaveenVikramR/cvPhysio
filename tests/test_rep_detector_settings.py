"""RepDetector's opt-in settings (start_frames, end_frames, baseline),
process_frame's buffer alignment with them, Session plumbing, and the
tuning script's selection rule."""

import math

import pandas as pd
import pytest

from scripts.tune_rep_detector import CURRENT, select_cell
from src.pipeline import (
    BASELINE_WINDOW,
    CALIBRATION_FRAMES,
    RepDetector,
    Session,
)
from tests.test_side_lock import body, rep_frames, run


def feed(detector, angles):
    events = []
    for a in angles:
        event, payload = detector.update(True, a, "RIGHT")
        if event != "none":
            events.append((event, payload))
    return events


def calibrated(**settings):
    d = RepDetector(**settings)
    feed(d, [10.0] * CALIBRATION_FRAMES)
    assert d.state == "RESTING"
    return d


def raise_once(peak=90.0, n=40, rest=10.0):
    return [rest + (peak - rest) * math.sin(math.pi * k / (n - 1)) for k in range(n)] + [rest] * 5


def test_defaults_are_the_original_detector():
    d = RepDetector()
    assert (d.start_frames, d.end_frames, d.baseline_mode) == (1, 1, "running")


def test_invalid_settings_refused():
    with pytest.raises(ValueError):
        RepDetector(start_frames=0)
    with pytest.raises(ValueError):
        RepDetector(baseline="drift")
    with pytest.raises(ValueError):
        Session("shoulder_abduction", detector={"end_frames": 0})


def test_start_debounce_ignores_a_short_flicker():
    flicker = [10.0] * 5 + [40.0] + [10.0] * 5
    assert [e for e, _ in feed(calibrated(), flicker)] == ["rep_discarded"]
    d = calibrated(start_frames=2)
    assert feed(d, flicker) == [] and d.state == "RESTING"


def test_start_debounce_keeps_the_held_frames_in_the_rep():
    angles = raise_once()
    original = feed(calibrated(), angles)
    debounced = feed(calibrated(start_frames=3), angles)
    assert [e for e, _ in original] == [e for e, _ in debounced] == ["rep_complete"]
    assert debounced[0][1]["angle_deg"] == original[0][1]["angle_deg"]


def test_end_debounce_bridges_a_one_frame_dip():
    angles = raise_once(n=40)
    angles[20] = 12.0  # mid-rep dip into the rest band
    assert [e for e, _ in feed(calibrated(), angles)] == ["rep_complete", "rep_complete"]  # split in two
    events = feed(calibrated(end_frames=2), angles)
    assert [e for e, _ in events] == ["rep_complete"]


def test_running_baseline_follows_a_slow_raise_fixed_and_stable_do_not():
    slow = [10.0 + 0.8 * k for k in range(100)]  # 0.8 deg/frame, 10 -> 89 deg
    running = calibrated()
    feed(running, slow)
    assert running.state == "RESTING" and running.baseline > 70  # dragged along: no rep starts
    for mode in ("fixed", "stable"):
        d = calibrated(baseline=mode)
        feed(d, slow)
        assert d.state == "MOVING" and d.baseline == pytest.approx(10.0)


def test_stable_baseline_updates_at_steady_rest():
    d = calibrated(baseline="stable")
    feed(d, [14.0] * BASELINE_WINDOW)
    assert d.baseline == pytest.approx(14.0)


@pytest.mark.parametrize("settings", [{"start_frames": 3}, {"start_frames": 5, "end_frames": 3},
                                      {"baseline": "fixed"}])
def test_session_buffers_stay_aligned(settings):
    session = Session("shoulder_abduction", side="RIGHT", detector=settings)
    session.start()
    completes = [p for e, p in run(session, rep_frames("right", n_reps=2)) if e == "rep_complete"]
    assert len(completes) == 2
    for p in completes:
        n = len(p["angle_deg"])
        assert len(p["elbow_angle_deg"]) == len(p["trunk_lean_deg"]) == len(p["frame_t"]) == n
        assert p["duration_s"] == pytest.approx(p["frame_t"][-1] - p["frame_t"][0])


def test_session_default_matches_explicit_defaults():
    def completes(detector):
        session = Session("shoulder_abduction", side="auto", detector=detector)
        session.start()
        return [p["angle_deg"] for e, p in run(session, rep_frames("right")) if e == "rep_complete"]

    assert completes(None) == completes(dict(CURRENT))


def cell_rows(settings, errors):
    return [{**settings, "recording": f"r{i}", "abs_error": e, "exact": int(e == 0)} for i, e in enumerate(errors)]


def test_select_rule():
    cur = cell_rows(CURRENT, [0, 1, 2])
    better = cell_rows({"baseline": "fixed", "start_frames": 2, "end_frames": 1}, [0, 0, 1])
    simpler_tie = cell_rows({"baseline": "running", "start_frames": 3, "end_frames": 1}, [0, 0, 1])
    worse_one = cell_rows({"baseline": "running", "start_frames": 2, "end_frames": 1}, [1, 0, 0])
    assert select_cell(pd.DataFrame(cur + better + simpler_tie + worse_one)) == \
        {"baseline": "running", "start_frames": 3, "end_frames": 1}
    assert select_cell(pd.DataFrame(cur + worse_one)) is None
    assert select_cell(pd.DataFrame(cur)) is None
