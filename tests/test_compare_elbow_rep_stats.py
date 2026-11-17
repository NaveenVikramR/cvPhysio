"""scripts/compare_elbow_rep_stats.py: per-rep statistics, the bent-rep AUC
and the pre-declared selection rule."""

import math

import numpy as np
import pandas as pd

from scripts.compare_elbow_rep_stats import (
    CURRENT,
    STATISTICS,
    VARIANTS,
    bent_auc,
    rep_statistic,
    select,
)


def test_rep_statistics():
    values = np.array([170.0, 160.0, np.nan, 150.0, 140.0, 165.0, 175.0])
    shoulder = np.array([10.0, 40.0, 70.0, 90.0, 60.0, 30.0, 10.0])
    assert rep_statistic(values, shoulder, 30.0, "min") == 140.0
    assert rep_statistic(values, shoulder, 30.0, "median") == 162.5
    assert rep_statistic(values, shoulder, 30.0, "p10") == np.percentile([170, 160, 150, 140, 165, 175], 10)
    # 0.3 s at 10 fps = 3 frames either side of the peak (index 3); NaN skipped
    assert rep_statistic(values, shoulder, 10.0, "hold_median") == 162.5
    assert rep_statistic(values, shoulder, 1.0, "hold_median") == 150.0  # window = the peak frame only


def test_rep_statistic_needs_enough_gated_frames():
    assert math.isnan(rep_statistic(np.array([150.0, np.nan, 160.0]), np.zeros(3), 30.0, "min"))


def test_bent_auc():
    truth = pd.Series([120.0, 130.0, 160.0, 170.0])
    assert bent_auc(pd.Series([100.0, 110.0, 150.0, 160.0]), truth) == 1.0
    assert bent_auc(pd.Series([160.0, 150.0, 110.0, 100.0]), truth) == 0.0
    assert math.isnan(bent_auc(pd.Series([1.0, 2.0, 3.0, 4.0]), pd.Series([160.0] * 4)))


def reps_with(scores: dict, n=12):
    rng = np.random.default_rng(0)
    truth = np.linspace(120, 170, n)
    rows = []
    for i in range(n):
        row = {"view": "front", "truth_min": truth[i]}
        for v in VARIANTS:
            for s in STATISTICS:
                row[f"{v}:{s}"] = rng.normal()  # unrelated to the truth
        for key, quality in scores.items():
            row[key] = truth[i] if quality == "perfect" else -truth[i]
        rows.append(row)
    return pd.DataFrame(rows)


def test_select_picks_a_clearly_better_measure():
    reps = reps_with({"world_xy:median": "perfect", f"{CURRENT[0]}:{CURRENT[1]}": "inverse"})
    chosen, table = select(reps)
    assert chosen == ("world_xy", "median") and len(table) == len(VARIANTS) * len(STATISTICS)


def test_select_keeps_current_when_it_is_already_best():
    chosen, _ = select(reps_with({f"{CURRENT[0]}:{CURRENT[1]}": "perfect"}))
    assert chosen is None
