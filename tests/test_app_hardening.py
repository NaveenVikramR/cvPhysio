"""Tests for the Streamlit hardening items from this task that CAN be
tested without a real camera: open_webcam() raising a catchable
RuntimeError (app.py's new try/except around it), and that a rep
abandoned mid-MOVING (simulating "Stop" interrupting the script before
the rep completes) leaves no trace in a SessionTracker, which still
writes a valid header-only CSV. The Streamlit UI code itself (the
try/except, the button wiring, the live camera preview) cannot be
exercised here -- see the chat response for what remains untestable
without a camera.
"""

from dataclasses import fields

import pytest

from src.pose_estimation import open_webcam
from src.pipeline import Session
from src.tracking import RepRecord, SessionTracker


def test_open_webcam_raises_runtime_error_for_an_invalid_camera_index():
    """The exact exception type/behavior app.py's new try/except
    (around both `capture = open_webcam()` call sites) is built to
    catch -- proven directly, not assumed."""
    with pytest.raises(RuntimeError):
        open_webcam(camera_index=99)


def make_landmarks(overrides=None):
    landmarks = [(0.1 * i, 0.5, 0.0, 1.0) for i in range(33)]
    for idx, point in (overrides or {}).items():
        landmarks[idx] = point
    return landmarks


def test_rep_abandoned_mid_moving_leaves_no_trace_and_tracker_still_saves(tmp_path):
    """Simulates pressing Stop while a rep is in progress (MOVING, never
    reaching rep_complete): the Session/RepDetector that buffered it
    simply goes out of scope (app.py's `finally:` calls session.stop()
    and lets `session`/`detector` be garbage collected) -- nothing
    about that partial rep is ever added to the tracker. tracker.save()
    must still write a valid header-only CSV, exactly as app.py's
    `finally:` block now does unconditionally."""
    session = Session("shoulder_abduction")
    session.start()
    landmarks = make_landmarks()

    # calibrate (15 consistent frames), then move into MOVING with a real excursion --
    # never reaching rep_complete before we simulate "Stop".
    for i in range(15):
        session.process_frame(landmarks, t=float(i))
    assert session.detector.state == "RESTING"

    moving_landmarks = make_landmarks({13: (1.1, 2.0, 0.0, 1.0)})  # elbow moved -- angle 0 -> 90 deg
    event, _payload, _visible, _failed = session.process_frame(moving_landmarks, t=15.0)
    assert session.detector.state == "MOVING"
    assert event == "none"  # not yet a complete rep

    session.stop()  # "Stop session" interrupting the script, same as app.py's finally: block

    tracker = SessionTracker(user_id="demo", exercise="shoulder_abduction")
    # app.py's event handling only ever calls tracker.add_rep/add_discard on
    # "rep_complete"/"rep_discarded" -- neither fired above, so nothing to add:
    assert tracker.reps == []
    assert tracker.discards == []

    path = tracker.save(logs_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1  # header only
    assert lines[0] == ",".join(f.name for f in fields(RepRecord))
