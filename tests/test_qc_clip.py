"""scripts/qc_clip.py's three checks on synthetic traces. No video or pose model,
except where stubbed, and no rule is ever scored."""

import ast
from pathlib import Path

import pytest

from scripts import qc_clip as qc

FPS = 30.0


def make_frames(seconds=10.0, rest=20.0, middle=90.0):
    """Arms down for the first and last 2 s, up in between."""
    frames = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        angle = rest if t < 2.0 or t >= seconds - 2.0 else middle
        frames.append({"t": t, "angle": angle, "visible": True})
    return frames


def make_trace(frames=None, rep_count=3, lost_tracking=0, seconds=10.0):
    return {"frames": frames if frames is not None else make_frames(seconds), "rep_count": rep_count,
            "lost_tracking": lost_tracking, "duration_s": seconds}


def set_range(frames, start, end, **values):
    for frame in frames:
        if start <= frame["t"] < end:
            frame.update(values)
    return frames


# --- arms_down ----------------------------------------------------------------

def test_arms_down_passes_with_rest_in_both_windows():
    assert qc.check_arms_down(make_frames(), 10.0)["passed"] is True


def test_arms_up_at_the_start_fails_the_way_the_first_run_did():
    frames = set_range(make_frames(), 0.0, 1.0, angle=85.0)
    result = qc.check_arms_down(frames, 10.0)
    assert result["passed"] is False
    assert "first 1.5 s: highest 85.0 deg" in result["detail"]


def test_arms_up_at_the_end_fails():
    assert qc.check_arms_down(set_range(make_frames(), 9.0, 10.0, angle=60.0), 10.0)["passed"] is False


def test_one_high_frame_in_a_window_fails():
    frames = make_frames()
    frames[10]["angle"] = 45.0
    assert qc.check_arms_down(frames, 10.0)["passed"] is False  # must be below 45


def test_just_below_the_limit_passes_and_outside_the_windows_is_ignored():
    frames = make_frames()
    frames[10]["angle"] = 44.9
    frames[int(1.6 * FPS)]["angle"] = 120.0  # 1.6 s is past the window
    assert qc.check_arms_down(frames, 10.0)["passed"] is True


@pytest.mark.parametrize("start, end, name", [(0.0, 1.5, "first"), (8.5, 10.0, "last")])
def test_a_window_with_no_measured_frames_fails(start, end, name):
    result = qc.check_arms_down(set_range(make_frames(), start, end, angle=None, visible=False), 10.0)
    assert result["passed"] is False
    assert f"{name} 1.5 s: no measured frames" in result["detail"]


# --- rep_count ----------------------------------------------------------------

@pytest.mark.parametrize("detected, passed", [(3, True), (2, False), (4, False), (0, False)])
def test_rep_count_must_equal_the_plan(detected, passed):
    assert qc.check_rep_count(detected, 3)["passed"] is passed


# --- tracking -----------------------------------------------------------------

def test_tracking_passes_with_every_frame_visible():
    assert qc.check_tracking(make_frames(), 0)["passed"] is True


def test_a_lost_tracking_discard_fails():
    assert qc.check_tracking(make_frames(), 1)["passed"] is False


def test_a_gated_run_at_the_detectors_limit_passes_and_one_more_fails():
    limit = qc.CONFIDENCE_TOLERANCE_FRAMES
    frames = make_frames()
    for frame in frames[100:100 + limit]:
        frame["visible"] = False
    assert qc.longest_gated_run(frames) == limit
    assert qc.check_tracking(frames, 0)["passed"] is True
    frames[100 + limit]["visible"] = False
    assert qc.check_tracking(frames, 0)["passed"] is False


def test_tracking_lost_while_resting_fails_even_with_no_discard():
    frames = set_range(make_frames(), 8.0, 10.0, visible=False, angle=None)
    assert qc.check_tracking(frames, 0)["passed"] is False


def test_scattered_gated_frames_do_not_add_up():
    frames = make_frames()
    for frame in frames[::5]:
        frame["visible"] = False
    assert qc.longest_gated_run(frames) == 1


# --- the script ---------------------------------------------------------------

def stub_trace(monkeypatch, trace):
    monkeypatch.setattr(qc, "collect_detector_trace", lambda path: trace)


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "abd_eval_correct_01.mp4"
    path.write_bytes(b"x")
    return path


def test_main_prints_each_check_and_exits_zero_on_a_pass(clip, capsys, monkeypatch):
    stub_trace(monkeypatch, make_trace())
    assert qc.main([str(clip), "--expected-reps", "3"]) == 0
    out = capsys.readouterr().out
    for name in ("arms_down", "rep_count", "tracking"):
        assert f"PASS  {name}" in out
    assert "Overall: PASS" in out


def test_main_reports_only_the_failing_check_and_exits_one(clip, capsys, monkeypatch):
    stub_trace(monkeypatch, make_trace(rep_count=2))
    assert qc.main([str(clip), "--expected-reps", "3"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  rep_count: detected 2, planned 3" in out
    assert "PASS  arms_down" in out and "PASS  tracking" in out
    assert "Overall: FAIL" in out


def test_planned_count_comes_from_the_manifest_row(clip, tmp_path, capsys, monkeypatch):
    manifest = tmp_path / "own_manifest.csv"
    manifest.write_text("clip,role,intended,expected_reps,subject,notes\n"
                        f"data/recordings/{clip.name},eval,correct,6,s,n\n")
    stub_trace(monkeypatch, make_trace(rep_count=6))
    assert qc.main([str(clip), "--manifest", str(manifest)]) == 0
    assert "planned 6" in capsys.readouterr().out


def test_a_clip_with_no_planned_count_cannot_be_checked(clip, tmp_path, capsys):
    assert qc.main([str(clip), "--manifest", str(tmp_path / "absent.csv")]) == 2
    assert "No planned rep count" in capsys.readouterr().out


def test_a_missing_clip_cannot_be_checked(tmp_path, capsys):
    assert qc.main([str(tmp_path / "absent.mp4"), "--expected-reps", "3"]) == 2
    assert "No clip at" in capsys.readouterr().out


def test_output_never_mentions_rule_outcomes(clip, capsys, monkeypatch):
    stub_trace(monkeypatch, make_trace())
    qc.main([str(clip), "--expected-reps", "3"])
    out = capsys.readouterr().out.lower()
    for word in ("peak", "elbow", "trunk", "tempo", "fail_rule", "failed_rules", "status"):
        assert word not in out


def test_the_script_never_calls_rule_scoring():
    tree = ast.parse(Path(qc.__file__).read_text())
    called = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not called & {"score_rep", "build_reference", "rep_summary_features", "apply_advisory_rules"}
