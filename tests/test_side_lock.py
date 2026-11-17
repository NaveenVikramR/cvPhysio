"""Arm locking (docs/landmark_fidelity.md, finding 5): feature_extraction's
optional `side`, process_frame's `side`, and Session's side modes."""

import math

import pytest

from src.feature_extraction import (
    LEFT,
    RIGHT,
    compute_shoulder_abduction_angle,
    extract_features,
    side_joints,
)
from src.pipeline import Session

UPPER, FORE = 0.25, 0.2


def body(right_deg=10.0, left_deg=10.0, right_vis=0.9, left_vis=0.95):
    """Image-space pose facing the camera (y down): each arm straight at the
    given abduction angle. The subject's right arm is on the image left."""
    lm = [(0.5, 0.5, 0.0, 0.9)] * 33
    for joints, sx, outward, deg, vis in ((RIGHT, 0.4, -1, right_deg, right_vis),
                                          (LEFT, 0.6, +1, left_deg, left_vis)):
        a = math.radians(deg)
        dx, dy = outward * math.sin(a), math.cos(a)
        lm[joints["hip"]] = (sx, 0.6, 0.0, vis)
        lm[joints["shoulder"]] = (sx, 0.3, 0.0, vis)
        lm[joints["elbow"]] = (sx + UPPER * dx, 0.3 + UPPER * dy, 0.0, vis)
        lm[joints["wrist"]] = (sx + (UPPER + FORE) * dx, 0.3 + (UPPER + FORE) * dy, 0.0, vis)
    return lm


def run(session, frames):
    events = []
    for i, lm in enumerate(frames):
        event, payload, _visible, _failed = session.process_frame(lm, t=i / 30)
        if event in ("rep_complete", "rep_discarded"):
            events.append((event, payload))
    return events


def rep_frames(arm="right", n_reps=3, peak=90.0, rest=10.0, other=10.0):
    """Calibration rest, then n_reps smooth raises of one arm (60 frames each)."""
    frames = [body(rest, other) if arm == "right" else body(other, rest) for _ in range(30)]
    for _ in range(n_reps):
        for k in range(60):
            deg = rest + (peak - rest) * math.sin(math.pi * k / 59)
            frames.append(body(deg, other) if arm == "right" else body(other, deg))
        frames += [body(rest, other) if arm == "right" else body(other, rest) for _ in range(15)]
    return frames


# --- feature_extraction ---------------------------------------------------

def test_side_joints_default_and_locked():
    lm = body(right_vis=0.6, left_vis=0.95)
    assert side_joints(lm) is LEFT  # per-frame pick: the more visible arm
    assert side_joints(lm, "RIGHT") is RIGHT
    with pytest.raises(ValueError):
        side_joints(lm, "right")


def test_locked_side_measures_that_arm_even_when_less_visible():
    lm = body(right_deg=80.0, left_deg=10.0, right_vis=0.6, left_vis=0.95)
    assert compute_shoulder_abduction_angle(lm) == pytest.approx(10.0, abs=1e-6)
    assert compute_shoulder_abduction_angle(lm, "RIGHT") == pytest.approx(80.0, abs=1e-6)
    feats = extract_features(lm, "shoulder_abduction", "RIGHT")
    assert feats["side"] == "RIGHT" and feats["angle_deg"] == pytest.approx(80.0, abs=1e-6)


def test_extract_features_default_unchanged():
    lm = body(right_deg=80.0, left_deg=10.0, right_vis=0.6, left_vis=0.95)
    assert extract_features(lm, "shoulder_abduction")["side"] == "LEFT"


# --- Session ----------------------------------------------------------------

def test_invalid_side_mode_refused():
    with pytest.raises(ValueError):
        Session("shoulder_abduction", side="left")


def test_per_frame_misses_reps_on_the_less_visible_arm():
    # The moving right arm is less visible, so the per-frame pick watches the
    # idle left arm and sees nothing: the failure mode arm locking fixes.
    session = Session("shoulder_abduction")
    session.start()
    assert [e for e, _ in run(session, rep_frames("right"))].count("rep_complete") == 0


def test_fixed_side_counts_every_rep():
    session = Session("shoulder_abduction", side="RIGHT")
    session.start()
    events = run(session, rep_frames("right"))
    completes = [p for e, p in events if e == "rep_complete"]
    assert len(completes) == 3
    assert all(set(p["side"]) == {"RIGHT"} for p in completes)


def test_auto_locks_to_the_moving_arm_and_keeps_the_first_rep():
    session = Session("shoulder_abduction", side="auto")
    session.start()
    assert session.locked_side is None
    completes = [p for e, p in run(session, rep_frames("right")) if e == "rep_complete"]
    assert len(completes) == 3  # the rep that triggered the lock is reported, not lost
    assert session.locked_side == "RIGHT"
    assert all(set(p["side"]) == {"RIGHT"} for p in completes)


def test_auto_locks_left_when_the_left_arm_moves():
    session = Session("shoulder_abduction", side="auto")
    session.start()
    completes = [p for e, p in run(session, rep_frames("left")) if e == "rep_complete"]
    assert len(completes) == 3 and session.locked_side == "LEFT"


def test_auto_ignores_an_idle_arm_flicker_before_the_lock():
    frames = rep_frames("right")
    # one-frame 40 deg spikes on the idle left arm during rest, before the first rep
    for i in (20, 24, 28):
        frames[i] = body(10.0, 50.0)
    session = Session("shoulder_abduction", side="auto")
    session.start()
    events = run(session, frames)
    assert session.locked_side == "RIGHT"
    assert [e for e, _ in events].count("rep_complete") == 3
    assert all(e == "rep_complete" for e, _ in events[:1])  # no pre-lock discard reported


def test_auto_same_frame_finish_picks_the_larger_range():
    # Both arms leave and re-enter rest on the same frames (square wave), so
    # both complete together; the larger range of motion decides.
    frames = [body(10.0, 10.0) for _ in range(30)]
    frames += [body(90.0, 60.0) for _ in range(30)]
    frames += [body(10.0, 10.0) for _ in range(15)]
    session = Session("shoulder_abduction", side="auto")
    session.start()
    run(session, frames)
    assert session.locked_side == "RIGHT"


def test_auto_otherwise_the_first_completed_rep_wins():
    # Smooth raise of both arms: the smaller left movement re-enters its rest
    # band a few frames earlier, so it completes first and wins. Documented
    # behaviour for two-arm movement: either arm is doing the exercise.
    frames = [body(10.0, 10.0) for _ in range(30)]
    for k in range(60):
        s = math.sin(math.pi * k / 59)
        frames.append(body(10.0 + 80.0 * s, 10.0 + 50.0 * s))
    frames += [body(10.0, 10.0) for _ in range(15)]
    session = Session("shoulder_abduction", side="auto")
    session.start()
    run(session, frames)
    assert session.locked_side == "LEFT"


def test_auto_smooths_each_frame_once():
    calls = []

    class CountingSmoother:
        def update(self, landmarks, t):
            calls.append(t)
            return landmarks

    session = Session("shoulder_abduction", side="auto")
    session.start()
    frames = rep_frames("right", n_reps=1)
    for i, lm in enumerate(frames):
        session.process_frame(lm, CountingSmoother(), i / 30)
    assert len(calls) == len(frames)
