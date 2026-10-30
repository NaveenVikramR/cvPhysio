"""Report figure scripts on synthetic inputs. No camera, and every output path is
under tmp_path, so results/ is never touched."""

import json

import cv2
import numpy as np
import pytest

from scripts import plot_session_trace as pst
from scripts import render_landmark_figure as rlf
from src import pose_estimation
from src.feature_extraction import LEFT, RIGHT


def make_landmarks(visibility=0.9):
    """Image-space positions; right arm out straight, trunk upright."""
    points = [(0.5, 0.5, 0.0, 0.9) for _ in range(33)]
    for joints, x in ((RIGHT, 0.4), (LEFT, 0.6)):
        points[joints["shoulder"]] = (x, 0.3, 0.0, 0.9)
        points[joints["hip"]] = (x, 0.7, 0.0, 0.9)
        points[joints["elbow"]] = (x, 0.5, 0.0, 0.9)
        points[joints["wrist"]] = (x, 0.65, 0.0, 0.9)
    points[RIGHT["elbow"]] = (0.25, 0.3, 0.0, visibility)
    points[RIGHT["wrist"]] = (0.1, 0.3, 0.0, 0.9)
    return points


# --- landmark figure ----------------------------------------------------------

def test_values_come_from_the_landmarks():
    values = rlf.figure_values(make_landmarks())
    assert values["shoulder_deg"] == pytest.approx(90.0)
    assert values["elbow_deg"] == pytest.approx(180.0)
    assert values["trunk_deg"] == pytest.approx(0.0)
    assert values["scorer_side"] in ("left", "right")


def test_trunk_is_not_reported_when_the_other_side_is_below_the_gate():
    points = make_landmarks()
    x, y, z, _vis = points[LEFT["hip"]]
    points[LEFT["hip"]] = (x, y, z, 0.2)
    assert rlf.figure_values(points)["trunk_deg"] is None


def test_low_visibility_highlight_is_named():
    assert rlf.right_arm_problems(make_landmarks()) == []
    assert rlf.right_arm_problems(make_landmarks(visibility=0.4)) == ["R elbow (0.40)"]


def test_annotate_draws_on_a_copy():
    frame = np.zeros((480, 640, 3), np.uint8)
    points = make_landmarks()
    image = rlf.annotate(frame, points, rlf.figure_values(points))
    assert image.any()
    assert not frame.any()


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), np.zeros((480, 640, 3), np.uint8))
    return path


def test_no_pose_exits_clearly_and_saves_nothing(image_path, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pose_estimation, "get_pose_landmarks_pair", lambda frame: (None, None))
    output = tmp_path / "out" / "figure.png"
    assert rlf.main([str(image_path), "--output", str(output)]) == 1
    assert "No pose detected" in capsys.readouterr().out
    assert not output.exists()


def test_real_model_finds_no_pose_on_a_blank_image(image_path, tmp_path, capsys):
    output = tmp_path / "figure.png"
    assert rlf.main([str(image_path), "--output", str(output)]) == 1
    assert "No pose detected" in capsys.readouterr().out
    assert not output.exists()


def test_low_visibility_exits_and_saves_nothing(image_path, tmp_path, capsys, monkeypatch):
    points = make_landmarks(visibility=0.3)
    monkeypatch.setattr(pose_estimation, "get_pose_landmarks_pair", lambda frame: (points, points))
    output = tmp_path / "figure.png"
    assert rlf.main([str(image_path), "--output", str(output)]) == 1
    assert "Below the 0.5 visibility gate: R elbow (0.30)" in capsys.readouterr().out
    assert not output.exists()


def test_detected_pose_saves_a_figure(image_path, tmp_path, capsys, monkeypatch):
    points = make_landmarks()
    monkeypatch.setattr(pose_estimation, "get_pose_landmarks_pair", lambda frame: (points, points))
    output = tmp_path / "out" / "figure.png"
    assert rlf.main([str(image_path), "--output", str(output)]) == 0
    assert cv2.imread(str(output)).shape == (480, 640, 3)
    assert "Right shoulder 90.0 deg" in capsys.readouterr().out


def test_unreadable_image_exits_clearly(tmp_path, capsys):
    assert rlf.main([str(tmp_path / "absent.png"), "--output", str(tmp_path / "f.png")]) == 1
    assert "Could not read an image" in capsys.readouterr().out


# --- session trace ------------------------------------------------------------

BAND = (75.0, 105.0)


def trace(rep, angle):
    return {"rep": rep, "fps": 10.0, "duration_s": 1.0, "valid_fraction": 1.0,
            "frame_t": [i * 0.1 for i in range(len(angle))], "angle_deg": angle,
            "elbow_angle_deg": [170.0] * len(angle), "trunk_lean_deg": [5.0] * len(angle)}


TRACES = [trace(1, [10.0, 50.0, 90.0, 40.0]), trace(2, [12.0, 100.0, 120.0, 30.0])]
ROWS = [{"rep_start_timestamp": "2026-10-30T10:00:10+00:00"}, {"rep_start_timestamp": "2026-10-30T10:00:15+00:00"}]


def write_session(tmp_path, traces=TRACES, rows=ROWS, name="demo"):
    (tmp_path / f"{name}.traces.json").write_text(json.dumps({"exercise": "shoulder_abduction", "reps": traces}))
    with (tmp_path / f"{name}.csv").open("w", newline="") as f:
        f.write("rep_start_timestamp\n" + "".join(f"{r['rep_start_timestamp']}\n" for r in rows))
    return tmp_path / f"{name}.csv"


def test_series_keep_the_trace_values_and_place_reps_on_a_shared_clock():
    series = pst.rep_series(TRACES, ROWS)
    assert series[0]["angle"] == TRACES[0]["angle_deg"]
    assert series[1]["angle"] == TRACES[1]["angle_deg"]
    assert series[0]["t"][0] == 0.0
    assert series[1]["t"] == pytest.approx([5.0, 5.1, 5.2, 5.3])


def test_peak_is_the_reps_true_maximum():
    series = pst.rep_series(TRACES, ROWS)
    assert (series[0]["peak_angle"], series[0]["peak_t"]) == (90.0, pytest.approx(0.2))
    assert (series[1]["peak_angle"], series[1]["peak_t"]) == (120.0, pytest.approx(5.2))


def plotted_shoulder_lines(ax):
    return [line for line in ax.lines if len(line.get_xdata()) > 2]


def test_figure_plots_exactly_the_trace_with_no_points_added():
    series = pst.rep_series(TRACES, ROWS)
    fig, ax = pst.build_figure(series, BAND, "title")
    lines = plotted_shoulder_lines(ax)
    assert len(lines) == 2  # one per rep, so nothing is drawn across the gap
    for line, s in zip(lines, series):
        assert list(line.get_ydata()) == s["angle"]
        assert list(line.get_xdata()) == s["t"]
    pst.plt.close(fig)


def test_figure_has_band_boundaries_peaks_and_labelled_units():
    series = pst.rep_series(TRACES, ROWS)
    fig, ax = pst.build_figure(series, BAND, "title")
    assert "(s)" in ax.get_xlabel()
    assert "degrees" in ax.get_ylabel()
    assert len(ax.patches) == 1  # the band
    peaks = [line for line in ax.lines if line.get_marker() == "o"]
    assert [line.get_ydata()[0] if hasattr(line.get_ydata(), "__len__") else line.get_ydata() for line in peaks] == [90.0, 120.0]
    boundaries = [line for line in ax.lines if line.get_linestyle() != "-" and line.get_marker() != "o"]
    assert len(boundaries) == 4  # start and end of each rep
    pst.plt.close(fig)


def test_main_saves_a_figure_and_reports_peaks_against_the_shipped_band(tmp_path, capsys):
    output = tmp_path / "out" / "trace.png"
    assert pst.main([str(write_session(tmp_path)), "--output", str(output)]) == 0
    assert cv2.imread(str(output)) is not None
    assert "peaks in 75-105 deg band: 1/2" in capsys.readouterr().out


def test_main_accepts_the_traces_path_and_defaults_to_the_newest(tmp_path, capsys):
    write_session(tmp_path)
    output = tmp_path / "trace.png"
    assert pst.main([str(tmp_path / "demo.traces.json"), "--output", str(output)]) == 0
    assert pst.main(["--logs-dir", str(tmp_path), "--output", str(output)]) == 0
    assert output.exists()


@pytest.mark.parametrize("setup, message", [
    (lambda d: d / "absent.csv", "No traces at"),
    (lambda d: (write_session(d), (d / "demo.csv").unlink(), d / "demo.traces.json")[2], "No session CSV"),
    (lambda d: write_session(d, rows=ROWS[:1]), "no rep_start_timestamp"),
    (lambda d: write_session(d, rows=[{"rep_start_timestamp": ""}, ROWS[1]]), "no rep_start_timestamp"),
])
def test_main_refuses_rather_than_inventing_a_timeline(tmp_path, capsys, setup, message):
    output = tmp_path / "trace.png"
    assert pst.main([str(setup(tmp_path)), "--output", str(output)]) == 1
    assert message in capsys.readouterr().out
    assert not output.exists()


def test_main_with_no_sessions(tmp_path, capsys):
    assert pst.main(["--logs-dir", str(tmp_path), "--output", str(tmp_path / "t.png")]) == 1
    assert "No session with traces" in capsys.readouterr().out
