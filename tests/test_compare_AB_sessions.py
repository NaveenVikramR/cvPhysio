"""scripts/compare_AB_sessions.py on synthetic traces. Every path is under tmp_path."""

import json

import pytest

from scripts import compare_AB_sessions as ab
from scripts.rescore_session import elbow_value

FPS = 10.0
N_FRAMES = 30
ANGLE = [min(i, N_FRAMES - 1 - i) * 6.0 for i in range(N_FRAMES)]  # peaks at frames 14-15


def make_trace(rep, elbow):
    return {
        "rep": rep, "fps": FPS, "duration_s": N_FRAMES / FPS, "valid_fraction": 1.0,
        "frame_t": [i / FPS for i in range(N_FRAMES)], "angle_deg": list(ANGLE),
        "elbow_angle_deg": list(elbow), "trunk_lean_deg": [5.0] * N_FRAMES,
    }


def straight(rep, value=165.0):
    return make_trace(rep, [value] * N_FRAMES)


def bent(rep):
    """A sustained bend through the raise and hold, not just one frame."""
    return make_trace(rep, [165.0] * 5 + [115.0] * 15 + [165.0] * 10)


def glitch(rep):
    """A straight arm with one wild frame, away from the hold window."""
    elbow = [165.0] * N_FRAMES
    elbow[3] = 80.0
    return make_trace(rep, elbow)


def write_session(tmp_path, name, traces, discards=0):
    csv_path = tmp_path / f"{name}.csv"
    csv_path.write_text("exercise,status\n")
    csv_path.with_suffix(".traces.json").write_text(json.dumps({"user_id": "demo", "exercise": "shoulder_abduction", "reps": traces}))
    if discards:
        csv_path.with_suffix(".discards.csv").write_text(
            "exercise,reason_code,message,timestamp\n" + "shoulder_abduction,too_short,m,t\n" * discards)
    return csv_path


# --- the two behaviours this comparison exists to show ---------------------------

@pytest.mark.parametrize("statistic", ["min", "p10"])
def test_known_bent_rep_is_separated(statistic):
    report = ab.compare([straight(1), straight(2)], [straight(1)], [bent(2)])[statistic]
    assert report["auc"] == 1.0
    assert (report["bent_below"], report["bent_n"]) == (1, 1)
    assert (report["straight_below"], report["straight_n"]) == (0, 3)


def test_known_one_frame_glitch_fails_min_but_passes_hold_median_and_p10():
    trace = glitch(1)
    assert elbow_value(trace, "min") < ab.THRESHOLD_DEG
    assert elbow_value(trace, "hold_median") >= ab.THRESHOLD_DEG
    assert elbow_value(trace, "p10") >= ab.THRESHOLD_DEG


def test_a_glitch_counts_against_min_only_in_the_report():
    report = ab.compare([glitch(1)], [], [])
    assert report["min"]["straight_below"] == 1
    assert report["hold_median"]["straight_below"] == 0
    assert report["p10"]["straight_below"] == 0


# --- the statistic helpers --------------------------------------------------------

def test_auc_known_values():
    assert ab.auc_lower_is_bent([100.0, 110.0], [150.0, 160.0]) == 1.0
    assert ab.auc_lower_is_bent([170.0], [150.0]) == 0.0
    assert ab.auc_lower_is_bent([150.0], [150.0]) == 0.5
    assert ab.auc_lower_is_bent([100.0, 155.0], [150.0, 160.0]) == 0.75


def test_auc_without_one_group_is_none_not_a_number():
    assert ab.auc_lower_is_bent([], [150.0]) is None
    assert ab.auc_lower_is_bent([150.0], []) is None


def test_group_stats_values_and_threshold_is_strict():
    stats = ab.group_stats([130.0, 135.0, 140.0])
    assert (stats["n"], stats["median"], stats["low"], stats["high"], stats["below"]) == (3, 135.0, 130.0, 140.0, 1)
    assert ab.group_stats([])["median"] is None


def test_b_reps_not_listed_count_as_straight():
    traces = [straight(i) for i in range(1, 6)]
    b_straight, b_bent = ab.split_b(traces, {2, 4})
    assert [t["rep"] for t in b_bent] == [2, 4]
    assert [t["rep"] for t in b_straight] == [1, 3, 5]


def test_listing_a_rep_the_session_lacks_is_an_error():
    with pytest.raises(ValueError, match=r"\[10\]"):
        ab.split_b([straight(1), straight(2)], {2, 10})


@pytest.mark.parametrize("text", ["2,x", "", "0,1", "-1"])
def test_bad_bent_rep_lists_are_refused(text):
    with pytest.raises(SystemExit):
        ab.parse_args(["a.csv", "b.csv", "--bent-reps", text])


def test_rep_list_parses():
    assert ab.parse_rep_list("2,4, 6") == {2, 4, 6}


# --- the script -------------------------------------------------------------------

def snapshot(directory):
    return {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_main_reports_warns_and_writes_nothing(tmp_path, capsys):
    a = write_session(tmp_path, "a", [straight(i) for i in range(1, 11)])
    b = write_session(tmp_path, "b", [bent(i) if i % 2 == 0 else straight(i) for i in range(1, 11)], discards=2)
    before = snapshot(tmp_path)
    assert ab.main([str(a), str(b), "--bent-reps", "2,4,6,8,10", "--results-dir", str(tmp_path)]) == 0
    assert snapshot(tmp_path) == before
    out = capsys.readouterr().out
    assert out.count("EXPLORATORY ONLY") == 2
    assert "discarded segments: 2" in out
    for statistic in ("min", "hold_median", "p10"):
        assert f"--- {statistic} ---" in out
    assert "AUC bent vs straight: 1.000" in out
    assert "Bent below 135: 5/5" in out


@pytest.mark.parametrize("missing", ["a", "b"])
def test_main_exits_clearly_when_traces_are_missing(tmp_path, capsys, missing):
    paths = {name: write_session(tmp_path, name, [straight(1), straight(2)]) for name in ("a", "b")}
    paths[missing].with_suffix(".traces.json").unlink()
    assert ab.main([str(paths["a"]), str(paths["b"]), "--bent-reps", "2", "--results-dir", str(tmp_path)]) == 1
    assert "No traces at" in capsys.readouterr().out


def test_main_exits_clearly_when_a_bent_rep_is_absent(tmp_path, capsys):
    a = write_session(tmp_path, "a", [straight(1)])
    b = write_session(tmp_path, "b", [straight(1)])
    assert ab.main([str(a), str(b), "--bent-reps", "9", "--results-dir", str(tmp_path)]) == 1
    assert "lists [9]" in capsys.readouterr().out


# --- noise summaries --------------------------------------------------------------

def noise_summary():
    s = {"n_frames": 200, "n_gated": 190, "gate_pass_fraction": 0.95, "mean": 162.3, "std": 4.1, "p5": 155.0,
         "p95": 168.0, "frac_below_threshold": 0.02}
    empty = {"n_frames": 200, "n_gated": 0, "gate_pass_fraction": 0.0, "mean": None, "std": None, "p5": None,
             "p95": None, "frac_below_threshold": None}
    return {"arm_out": {"left": s, "right": empty}}


def test_noise_lines_print_each_tagged_run(tmp_path):
    for tag in ("rod", "free"):
        (tmp_path / f"elbow_noise_{tag}.json").write_text(json.dumps({"summary": noise_summary()}))
    text = "\n".join(ab.noise_lines(tmp_path))
    assert "elbow_noise_rod.json" in text and "elbow_noise_free.json" in text
    assert "162.3" in text and "155.0" in text and "168.0" in text and "2.0%" in text
    assert "no gated frames" in text


def test_noise_lines_say_when_there_are_none(tmp_path):
    assert "No noise runs found" in ab.noise_lines(tmp_path)[0]


def test_noise_lines_survive_an_unreadable_file(tmp_path):
    (tmp_path / "elbow_noise_bad.json").write_text("not json")
    assert "unreadable" in ab.noise_lines(tmp_path)[0]
