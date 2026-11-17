"""scripts/tune_smoothing_mocap.py: the lag measure and the pre-declared
selection rule."""

import math

import numpy as np
import pandas as pd

from scripts.tune_smoothing_mocap import RecordingSmoother, best_lag, select_cell


def wave(n=120, shift=0, offset=0.0):
    return np.array([offset + 40 * math.sin(2 * math.pi * (i - shift) / 60) for i in range(n)])


def test_best_lag_finds_a_trailing_signal():
    truth = wave()
    assert best_lag(wave(shift=0, offset=-5.0), truth) == 0  # constant offset ignored
    assert best_lag(wave(shift=4), truth) == 4
    assert best_lag(wave(shift=-2), truth) == -2


def test_best_lag_skips_nan_and_short_series():
    truth = wave()
    mp = wave(shift=3)
    mp[10:20] = np.nan
    assert best_lag(mp, truth) == 3
    assert math.isnan(best_lag(np.array([1.0, 2.0]), np.array([1.0, 2.0])))


def row(mc, beta, worse=0, lag=0.0, peak=0.0, mae=4.0):
    return {"min_cutoff": mc, "beta": beta, "recordings_count_worse": worse, "added_lag": lag,
            "peak_shift": peak, "front_shoulder_mae_debiased": mae}


def table(*rows):
    return pd.DataFrame([row(None, None, mae=4.2), *rows])


def test_select_picks_lowest_mae_among_eligible():
    t = table(row(1.0, 0.0, mae=3.9), row(2.0, 0.0, mae=3.5, worse=1),  # worse rep count: ineligible
              row(3.0, 0.0, mae=3.6, lag=4.0), row(5.0, 0.0, mae=3.7, peak=2.5), row(0.5, 1.0, mae=3.8))
    assert select_cell(t) == (0.5, 1.0)


def test_select_none_when_nothing_beats_off():
    assert select_cell(table(row(1.0, 0.0, mae=4.15))) is None
    assert select_cell(table(row(1.0, 0.0, mae=3.0, worse=2))) is None


def test_recording_smoother_passes_through_and_remembers():
    s = RecordingSmoother(None)
    lm = [(0.1, 0.2, 0.3, 0.9)] * 33
    assert s.update(lm, 0.0) is lm and s.last is lm
    assert s.update(None, 0.1) is None and s.last is None
