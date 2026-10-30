"""Per-rep traces and scripts/rescore_session.py, on synthetic reps. Every path
is under tmp_path, so logs/ is never touched."""

import csv
import json

import pytest

from scripts import rescore_session as rs
from src import rules as rules_module
from src.tracking import RepRecord, SessionTracker, rep_trace

FPS = 10.0
CONFIG = rules_module.load_config("shoulder_abduction")
ANGLE = [10.0, 30.0, 50.0, 70.0, 85.0, 90.0, 85.0, 70.0, 50.0, 30.0, 10.0] * 2


def make_rep(elbow=None, valid_fraction=1.0, angle=None):
    angle = angle or ANGLE
    return {
        "angle_deg": angle, "elbow_angle_deg": elbow or [170.0] * len(angle), "trunk_lean_deg": [5.0] * len(angle),
        "valid_fraction": valid_fraction, "frame_t": [100.0 + i / FPS for i in range(len(angle))],
    }


def record(status="pass"):
    return RepRecord(exercise="shoulder_abduction", side="LEFT", angle_deg=90.0, status=status, confidence=None)


def save_session(tmp_path, reps, statuses=None):
    tracker = SessionTracker(user_id="demo", exercise="shoulder_abduction")
    for i, rep in enumerate(reps):
        tracker.add_rep(record(statuses[i] if statuses else "pass"), trace=rep_trace(rep, FPS, 2.0))
    return tracker.save(logs_dir=tmp_path)


# --- traces -------------------------------------------------------------------

def test_trace_holds_every_gated_frame_and_times_from_the_rep_start():
    rep = make_rep()
    trace = rep_trace(rep, FPS, 2.1)
    assert trace["elbow_angle_deg"] == rep["elbow_angle_deg"]
    assert trace["angle_deg"] == rep["angle_deg"]
    assert trace["frame_t"][0] == 0.0
    assert trace["frame_t"][5] == pytest.approx(0.5)
    assert (trace["fps"], trace["duration_s"], trace["valid_fraction"]) == (FPS, 2.1, 1.0)


def test_session_saves_traces_numbered_like_the_csv(tmp_path):
    path = save_session(tmp_path, [make_rep(), make_rep([150.0] * len(ANGLE))])
    data = json.loads(path.with_suffix(".traces.json").read_text())
    assert [r["rep"] for r in data["reps"]] == [1, 2]
    assert data["reps"][1]["elbow_angle_deg"][0] == 150.0
    assert data["exercise"] == "shoulder_abduction"


def test_rep_numbers_stay_aligned_when_a_rep_has_no_trace(tmp_path):
    tracker = SessionTracker(user_id="demo", exercise="shoulder_abduction")
    tracker.add_rep(record())
    tracker.add_rep(record(), trace=rep_trace(make_rep(), FPS, 2.0))
    data = json.loads(tracker.save(logs_dir=tmp_path).with_suffix(".traces.json").read_text())
    assert [r["rep"] for r in data["reps"]] == [2]


def test_no_traces_file_without_traces(tmp_path):
    tracker = SessionTracker(user_id="demo", exercise="shoulder_abduction")
    tracker.add_rep(record())
    assert not tracker.save(logs_dir=tmp_path).with_suffix(".traces.json").exists()


def test_history_ignores_the_traces_file(tmp_path):
    from src.tracking import load_history
    save_session(tmp_path, [make_rep()])
    assert len(load_history("demo", logs_dir=tmp_path)) == 1


# --- statistics ---------------------------------------------------------------

def test_p10_matches_a_known_value():
    assert rs.p10([float(v) for v in range(101)]) == pytest.approx(10.0)
    assert rs.p10([5.0, 15.0]) == pytest.approx(6.0)
    assert rs.p10([7.0]) == 7.0


def test_each_statistic_reads_the_same_trace_differently():
    # One glitch frame, away from the peak at index 5.
    elbow = [170.0] * len(ANGLE)
    elbow[1] = 100.0
    trace = rep_trace(make_rep(elbow), FPS, 2.0)
    assert rs.elbow_value(trace, "min") == 100.0
    assert rs.elbow_value(trace, "hold_median") == 170.0
    assert rs.elbow_value(trace, "p10") == pytest.approx(170.0)


def test_rescoring_with_min_reproduces_live_scoring_exactly():
    for elbow in (170.0, 134.9, 135.0):
        rep = make_rep([elbow] * len(ANGLE))
        live = rules_module.score_rep(rep, CONFIG, FPS)
        rescored = rs.rescore_rep(json.loads(json.dumps(rep_trace(rep, FPS, 2.0))), CONFIG, "min")
        assert rescored["status"] == live["status"]
        assert rescored["failed_rules"] == live["failed_rules"]
        assert rescored["elbow"] == live["rules"]["elbow_straight"]["measured"]


def test_statistics_can_change_only_the_elbow_rule():
    elbow = [170.0] * len(ANGLE)
    elbow[1] = 100.0
    angle = [a * 1.5 for a in ANGLE]  # peak 135: fails peak_angle under every statistic
    trace = rep_trace(make_rep(elbow, angle=angle), FPS, 2.0)
    assert rs.rescore_rep(trace, CONFIG, "min")["failed_rules"] == ["peak_angle", "elbow_straight"]
    assert rs.rescore_rep(trace, CONFIG, "hold_median")["failed_rules"] == ["peak_angle"]


def test_insufficient_data_stays_insufficient_under_every_statistic():
    trace = rep_trace(make_rep(valid_fraction=0.5), FPS, 2.0)
    for statistic in rs.ELBOW_STATISTICS:
        assert rs.rescore_rep(trace, CONFIG, statistic)["status"] == "insufficient_data"


def test_summarize_counts_status_and_rule_failures():
    results = [{"status": "pass", "failed_rules": []}, {"status": "fail", "failed_rules": ["elbow_straight", "tempo"]},
               {"status": "insufficient_data", "failed_rules": ["elbow_straight"]}]
    summary = rs.summarize(results)
    assert (summary["n"], summary["pass"], summary["fail"], summary["insufficient_data"]) == (3, 1, 1, 1)
    assert summary["rule_failures"]["elbow_straight"] == 2


# --- the script ---------------------------------------------------------------

def files_snapshot(directory):
    return {p: p.read_bytes() for p in directory.iterdir()}


def test_main_reports_each_statistic_and_writes_nothing(tmp_path, capsys):
    glitch = [170.0] * len(ANGLE)
    glitch[1] = 100.0
    path = save_session(tmp_path, [make_rep(), make_rep(glitch)], statuses=["pass", "fail"])
    before = files_snapshot(tmp_path)
    assert rs.main([str(path)]) == 0
    assert files_snapshot(tmp_path) == before
    out = capsys.readouterr().out
    for statistic in rs.ELBOW_STATISTICS:
        assert statistic in out
    assert "all reps match" in out
    assert "aren't accuracy" in out


def test_main_flags_disagreement_with_the_logged_status(tmp_path, capsys):
    path = save_session(tmp_path, [make_rep()], statuses=["fail"])
    assert rs.main([str(path)]) == 0
    assert "DIFFERS on reps [1]" in capsys.readouterr().out


def test_main_defaults_to_the_newest_session_with_traces(tmp_path, capsys):
    path = save_session(tmp_path, [make_rep()])
    assert rs.find_latest_session(tmp_path) == path
    assert rs.main(["--logs-dir", str(tmp_path)]) == 0
    assert path.name in capsys.readouterr().out


def test_main_explains_a_session_without_traces(tmp_path, capsys):
    tracker = SessionTracker(user_id="demo", exercise="shoulder_abduction")
    tracker.add_rep(record())
    path = tracker.save(logs_dir=tmp_path)
    assert rs.main([str(path)]) == 1
    assert "can't be re-scored" in capsys.readouterr().out


def test_main_with_no_sessions_at_all(tmp_path, capsys):
    assert rs.main(["--logs-dir", str(tmp_path)]) == 1
    assert "No session with traces" in capsys.readouterr().out


def test_csv_rows_keep_their_order_for_comparison(tmp_path):
    path = save_session(tmp_path, [make_rep(), make_rep()], statuses=["pass", "fail"])
    with path.open(newline="") as f:
        assert [r["status"] for r in csv.DictReader(f)] == ["pass", "fail"]
