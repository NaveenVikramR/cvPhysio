"""scripts/qc_all.py with synthetic manifests and traces. No pose model runs and
nothing is scored; every path is under tmp_path."""

import csv

import pytest

from scripts import qc_all
from tests.test_qc_clip import make_trace

PLAN = [
    ("abd_reference_correct_01", "reference", "correct", 3),
    ("abd_calibration_correct_01", "calibration", "correct", 5),
    ("abd_eval_correct_01", "eval", "correct", 6),
    ("abd_eval_fault_lowarm_01", "eval", "fault_lowarm", 3),
    ("abd_eval_fault_trunk_03", "eval", "fault_trunk", 3),
    ("abd_eval_fault_elbow_01", "eval", "fault_elbow", 3),
    ("abd_eval_fault_fast_01", "eval", "fault_fast", 3),
]


def rows_for(plan=PLAN, directory="data/recordings"):
    return [{"clip": f"{directory}/{name}.mp4", "role": role, "intended": intended, "expected_reps": str(reps),
             "subject": "s", "notes": "n"} for name, role, intended, reps in plan]


def write_manifest(tmp_path, rows):
    path = tmp_path / "own_manifest.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "role", "intended", "expected_reps", "subject", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def session(tmp_path, monkeypatch):
    """Files on disk, with both runs stubbed to agree and pass."""
    for name, *_ in PLAN:
        (tmp_path / f"{name}.mp4").write_bytes(b"x")
    rows = rows_for(directory=tmp_path.as_posix())
    planned = {row["clip"]: int(row["expected_reps"]) for row in rows}
    monkeypatch.setattr(qc_all, "detected_alone", lambda clip, reps: reps)
    opened = []

    def combined(path):
        opened.append(path.name)
        return make_trace(rep_count=planned[path.as_posix()])

    monkeypatch.setattr(qc_all.qc_clip, "collect_detector_trace", combined)
    return {"manifest": write_manifest(tmp_path, rows), "opened": opened, "rows": rows}


# --- order --------------------------------------------------------------------

def test_order_matches_the_evaluator_whatever_the_manifest_order():
    # fault_trunk_03 was recorded first, so it heads this manifest.
    rows = rows_for([PLAN[4], PLAN[0], PLAN[1], PLAN[2], PLAN[3], PLAN[5], PLAN[6]])
    names = [r["clip"].split("/")[-1] for r in qc_all.evaluator_order(rows)]
    assert names == ["abd_reference_correct_01.mp4", "abd_calibration_correct_01.mp4", "abd_eval_fault_trunk_03.mp4",
                     "abd_eval_correct_01.mp4", "abd_eval_fault_lowarm_01.mp4", "abd_eval_fault_elbow_01.mp4",
                     "abd_eval_fault_fast_01.mp4"]


def test_order_uses_only_the_first_reference_like_the_evaluator():
    rows = rows_for([PLAN[0], ("abd_reference_correct_02", "reference", "correct", 3)])
    assert [r["clip"] for r in qc_all.evaluator_order(rows)] == ["data/recordings/abd_reference_correct_01.mp4"]


def test_evaluator_source_still_uses_this_order():
    source = open("scripts/evaluate_own_clips.py", encoding="utf-8").read()
    reference = source.index('process_clip_reps(Path(reference_rows[0]["clip"])')
    calibration = source.index("for row in calibration_rows:")
    evaluation = source.index("for row in eval_rows:")
    assert reference < calibration < evaluation


# --- the plan -----------------------------------------------------------------

def test_the_planned_set_has_no_problems():
    assert qc_all.manifest_problems(rows_for()) == []


def test_planned_counts_match_the_outcomes_script():
    from scripts.compute_primary_outcomes import PLANNED_SCORABLE
    assert {intended: reps for (role, intended), reps in qc_all.PLANNED_CLIPS.items() if role == "eval"} == PLANNED_SCORABLE


def test_a_missing_clip_is_a_problem():
    assert qc_all.manifest_problems(rows_for(PLAN[:-1])) == ["eval/fault_fast: 0 manifest rows, expected 1"]


def test_an_extra_take_left_in_the_manifest_is_a_problem():
    extra = ("abd_eval_fault_trunk_01", "eval", "fault_trunk", 3)
    assert qc_all.manifest_problems(rows_for(PLAN + [extra])) == ["eval/fault_trunk: 2 manifest rows, expected 1"]


def test_a_wrong_rep_plan_is_a_problem():
    wrong = [("abd_eval_correct_01", "eval", "correct", 10) if p[0] == "abd_eval_correct_01" else p for p in PLAN]
    assert "pre-registered 6" in qc_all.manifest_problems(rows_for(wrong))[0]


# --- comparison ---------------------------------------------------------------

def result(alone=3, combined=3, passed=True):
    return {"name": "c.mp4", "planned": 3, "alone": alone, "combined": combined,
            "checks": {"rep_count": {"passed": passed, "detail": "d"}}}


def test_agreeing_passing_clips_have_no_problems():
    assert qc_all.compare([result()]) == []


def test_a_count_differing_between_runs_is_reported():
    assert qc_all.compare([result(alone=3, combined=4)])[0] == "c.mp4: detected 3 alone but 4 combined"


def test_an_unreadable_alone_run_is_reported():
    assert "detected None alone" in qc_all.compare([result(alone=None)])[0]


def test_a_check_failing_only_in_the_combined_run_is_reported():
    assert "rep_count failed in the combined run" in qc_all.compare([result(passed=False)])[0]


# --- the script ---------------------------------------------------------------

def test_main_ok_to_score_when_all_agree_and_opens_clips_in_order(session, capsys):
    assert qc_all.main(["--manifest", str(session["manifest"])]) == 0
    out = capsys.readouterr().out
    assert "OK to score" in out
    assert session["opened"][0] == "abd_reference_correct_01.mp4"
    assert len(session["opened"]) == 7
    assert "abd_eval_fault_fast_01.mp4" in out and "planned" in out


def test_main_refuses_when_runs_disagree(session, capsys, monkeypatch):
    monkeypatch.setattr(qc_all, "detected_alone", lambda clip, reps: reps + 1)
    assert qc_all.main(["--manifest", str(session["manifest"])]) == 1
    out = capsys.readouterr().out
    assert "DO NOT SCORE" in out
    assert "OK to score" not in out


def test_main_refuses_a_manifest_off_plan_without_opening_a_clip(session, tmp_path, capsys):
    manifest = write_manifest(tmp_path, session["rows"][:-1])
    assert qc_all.main(["--manifest", str(manifest)]) == 1
    assert "DO NOT SCORE" in capsys.readouterr().out
    assert session["opened"] == []


def test_main_refuses_when_a_listed_file_is_missing(session, tmp_path, capsys):
    (tmp_path / "abd_eval_fault_fast_01.mp4").unlink()
    assert qc_all.main(["--manifest", str(session["manifest"])]) == 1
    assert "missing file" in capsys.readouterr().out


def test_main_with_no_manifest(tmp_path, capsys):
    assert qc_all.main(["--manifest", str(tmp_path / "absent.csv")]) == 1
    assert "No manifest" in capsys.readouterr().out


def test_output_never_mentions_rule_outcomes(session, capsys):
    qc_all.main(["--manifest", str(session["manifest"])])
    out = capsys.readouterr().out.lower()
    for word in ("peak", "elbow angle", "trunk lean", "tempo", "failed_rules"):
        assert word not in out
