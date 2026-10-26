"""Tests for src/smoothing.py on synthetic 1D signals (no video)."""

import random

from src.smoothing import LandmarkSmoother, OneEuroFilter

FREQ = 30.0


def ts(n: int, freq: float = FREQ, start: int = 0) -> list[float]:
    """n timestamps, `start` frames into a freq-Hz stream."""
    return [(start + i) / freq for i in range(n)]


def moving_average_filter(values: list[float], window: int) -> list[float]:
    out = []
    buf = []
    for v in values:
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


def noisy_constant(n: int, level: float, noise_std: float, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [level + rng.gauss(0, noise_std) for _ in range(n)]


def noisy_step(n_before: int, n_after: int, before: float, after: float, noise_std: float, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    signal = [before] * n_before + [after] * n_after
    return [v + rng.gauss(0, noise_std) for v in signal]


def stdev(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def settle_frame(values: list[float], target: float, tolerance: float, start: int) -> int:
    """First index >= start from which `values` stays within `tolerance`
    of `target` for the rest of the sequence. len(values) if it never does."""
    for i in range(start, len(values)):
        if all(abs(v - target) <= tolerance for v in values[i:]):
            return i
    return len(values)


def test_one_euro_filter_reduces_jitter_on_noisy_constant_signal():
    # Scale matches real usage: MediaPipe's normalized landmark coords are
    # in [0, 1], with jitter on the order of a few percent -- not an
    # arbitrary unit choice, since the One Euro filter's cutoff/beta are
    # absolute constants in Hz, so the signal's own scale changes how much
    # smoothing the same params produce (verified: this test failed with
    # level=50/noise_std=3.0 before rescaling to match real landmark units).
    raw = noisy_constant(n=200, level=0.5, noise_std=0.03, seed=1)
    times = ts(len(raw))
    f = OneEuroFilter(min_cutoff=1.0, beta=0.3, d_cutoff=1.0)
    filtered = [f(v, t) for v, t in zip(raw, times)]

    raw_std = stdev(raw)
    filtered_std = stdev(filtered[20:])  # skip initial transient
    assert filtered_std < raw_std * 0.5


def test_one_euro_filter_lags_less_than_moving_average_of_similar_strength():
    # Moving-average window chosen so its steady-state noise std is in the
    # same ballpark as the One Euro filter's, so the comparison is fair.
    raw = noisy_step(n_before=60, n_after=140, before=0.0, after=1.0, noise_std=0.03, seed=2)
    times = ts(len(raw))

    euro = OneEuroFilter(min_cutoff=1.0, beta=0.3, d_cutoff=1.0)
    euro_out = [euro(v, t) for v, t in zip(raw, times)]

    ma_window = 9
    ma_out = moving_average_filter(raw, ma_window)

    euro_steady_std = stdev(euro_out[-40:])
    ma_steady_std = stdev(ma_out[-40:])
    # same ballpark -- not an exact match, just close enough to be a fair race
    assert abs(euro_steady_std - ma_steady_std) < 0.003

    tolerance = 0.05
    euro_settle = settle_frame(euro_out, target=1.0, tolerance=tolerance, start=60)
    ma_settle = settle_frame(ma_out, target=1.0, tolerance=tolerance, start=60)

    assert euro_settle < ma_settle


def test_one_euro_filter_uses_real_dt_not_a_fixed_assumed_rate():
    # Two calls spaced 1s apart (way slower than any plausible webcam/video
    # rate) should barely smooth at all -- a fixed-freq filter would wrongly
    # treat this as a normal 1/freq-spaced frame and over-smooth it.
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    f(0.0, t=0.0)
    out = f(1.0, t=1.0)  # 1 full second later, far slower than a live stream
    assert out > 0.8  # should track the jump mostly, not lag the way a
    # fixed-30Hz-assumption filter would (that would compute a smoothing
    # factor of ~0.17 for this same jump, landing near 0.17 instead)


def test_landmark_smoother_holds_last_good_value_during_short_dropout():
    smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.3, d_cutoff=1.0, max_hold_frames=3)
    landmarks = [(0.5, 0.5, 0.0, 1.0)] * 33

    first = smoother.update(landmarks, t=0.0)
    assert first is not None

    held_1 = smoother.update(None, t=1 / FREQ)
    held_2 = smoother.update(None, t=2 / FREQ)
    held_3 = smoother.update(None, t=3 / FREQ)
    assert held_1 == first
    assert held_2 == first
    assert held_3 == first


def test_landmark_smoother_invalidates_after_hold_budget_exhausted():
    smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.3, d_cutoff=1.0, max_hold_frames=3)
    landmarks = [(0.5, 0.5, 0.0, 1.0)] * 33

    smoother.update(landmarks, t=0.0)
    for i in range(1, 4):
        assert smoother.update(None, t=i / FREQ) is not None
    assert smoother.update(None, t=4 / FREQ) is None


def test_landmark_smoother_recovers_once_pose_returns():
    smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.3, d_cutoff=1.0, max_hold_frames=2)
    landmarks_a = [(0.5, 0.5, 0.0, 1.0)] * 33
    landmarks_b = [(0.6, 0.6, 0.0, 1.0)] * 33

    smoother.update(landmarks_a, t=0.0)
    smoother.update(None, t=1 / FREQ)
    smoother.update(None, t=2 / FREQ)
    assert smoother.update(None, t=3 / FREQ) is None  # hold budget exhausted

    recovered = smoother.update(landmarks_b, t=4 / FREQ)
    assert recovered is not None


def test_landmark_smoother_smooths_each_coordinate_independently():
    smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.0, d_cutoff=1.0, max_hold_frames=0)
    rng = random.Random(3)
    raw_sequence = [
        [(0.5 + rng.gauss(0, 0.02), 0.5 + rng.gauss(0, 0.02), 0.0, 1.0)] * 33
        for _ in range(50)
    ]
    times = ts(len(raw_sequence))
    outputs = [smoother.update(frame, t) for frame, t in zip(raw_sequence, times)]

    raw_x = [frame[0][0] for frame in raw_sequence]
    smoothed_x = [out[0][0] for out in outputs]
    assert stdev(smoothed_x[10:]) < stdev(raw_x[10:])
