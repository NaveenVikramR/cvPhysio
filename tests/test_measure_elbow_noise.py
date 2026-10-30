"""Statistics in scripts/measure_elbow_noise.py, on synthetic readings. No camera."""

import pytest

from scripts import measure_elbow_noise as men
from src.feature_extraction import LEFT, RIGHT


def reading(left_deg=170.0, right_deg=170.0, selected="left", left_wrist_vis=0.9):
    return {
        "selected": selected,
        "left": {"elbow_deg": left_deg, "elbow_vis": 0.9, "wrist_vis": left_wrist_vis},
        "right": {"elbow_deg": right_deg, "elbow_vis": 0.9, "wrist_vis": 0.9},
    }


def frame(pose="arm_out", **kwargs):
    return {"pose": pose, "t": 0.0, "reading": reading(**kwargs)}


def test_summarize_reports_the_requested_statistics():
    values = [float(v) for v in range(130, 150)]
    s = men.summarize(values, 25)
    assert s["mean"] == pytest.approx(139.5)
    assert s["std"] == pytest.approx(5.766, abs=1e-3)
    assert s["p5"] == pytest.approx(130.95)
    assert s["p95"] == pytest.approx(148.05)
    assert s["frac_below_threshold"] == pytest.approx(5 / 20)
    assert s["gate_pass_fraction"] == pytest.approx(20 / 25)


def test_summarize_with_no_gated_frames_reports_none_not_zero():
    s = men.summarize([], 10)
    assert s["mean"] is None
    assert s["frac_below_threshold"] is None
    assert s["gate_pass_fraction"] == 0.0


def test_frames_failing_visibility_or_with_no_pose_are_excluded_but_counted():
    frames = [frame(left_deg=170.0), frame(left_deg=90.0, left_wrist_vis=0.4), {"pose": "arm_out", "t": 0.0, "reading": None}]
    values, considered = men.arm_values(frames, "left")
    assert values == [170.0]
    assert considered == 3


def test_selected_follows_the_side_chosen_each_frame():
    frames = [frame(left_deg=100.0, right_deg=170.0, selected="right"), frame(left_deg=110.0, right_deg=160.0, selected="left")]
    assert men.arm_values(frames, "selected")[0] == [170.0, 110.0]


def test_session_summary_separates_the_two_poses():
    frames = [frame("arm_out", left_deg=170.0), frame("arm_down", left_deg=120.0)]
    summary = men.summarize_session(frames)
    assert summary["arm_out"]["left"]["frac_below_threshold"] == 0.0
    assert summary["arm_down"]["left"]["frac_below_threshold"] == 1.0


def test_frame_reading_uses_each_arms_own_joints_and_the_scorers_side():
    points = [(0.0, 0.0, 0.0, 0.1)] * 33
    for joints, vis in ((LEFT, 0.9), (RIGHT, 0.6)):
        points[joints["hip"]] = (0.0, 1.0, 0.0, vis)
        points[joints["shoulder"]] = (0.0, 0.0, 0.0, vis)
        points[joints["elbow"]] = (1.0, 0.0, 0.0, vis)
    points[LEFT["wrist"]] = (2.0, 0.0, 0.0, 0.9)  # straight
    points[RIGHT["wrist"]] = (1.0, 1.0, 0.0, 0.6)  # bent 90
    result = men.frame_reading(points)
    assert result["selected"] == "left"
    assert result["left"]["elbow_deg"] == pytest.approx(180.0)
    assert result["right"]["elbow_deg"] == pytest.approx(90.0)
    assert result["right"]["wrist_vis"] == 0.6


def test_tag_selects_its_own_file_and_no_tag_keeps_the_default():
    assert men.results_path(None) == men.RESULTS_PATH
    assert men.RESULTS_PATH.name == "elbow_noise.json"
    assert men.results_path("rod") == men.RESULTS_PATH.with_name("elbow_noise_rod.json")
    assert men.results_path("rod") != men.results_path("free")


def test_tag_argument_is_parsed_and_defaults_to_none():
    assert men.parse_args([]).tag is None
    assert men.parse_args(["--tag", "rod-1_a"]).tag == "rod-1_a"


@pytest.mark.parametrize("tag", ["../escape", "a/b", "a b", "a.json", ""])
def test_tag_that_could_leave_the_results_folder_is_refused(tag):
    with pytest.raises(SystemExit):
        men.parse_args(["--tag", tag])


def test_format_summary_marks_an_empty_arm():
    frames = [{"pose": "arm_out", "t": 0.0, "reading": None}, frame("arm_down")]
    text = "\n".join(men.format_summary(men.summarize_session(frames)))
    assert "no gated frames" in text
