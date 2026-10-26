"""Landmark smoothing for the live pipeline. Sits between pose_estimation
(raw per-frame landmarks) and feature_extraction (angles computed from
them) -- never imported by either, per claude.md's module-independence
rule; app.py wires it in between the two.

Two independent pieces:
  - OneEuroFilter: Casiez et al.'s "1 Euro Filter" (a low-pass filter
    whose cutoff adapts to the signal's own speed -- slow motion gets
    heavy smoothing, fast motion gets little, so it doesn't lag behind a
    real, fast repetition the way a fixed-window moving average would).
  - LandmarkSmoother: runs one OneEuroFilter per (landmark, x/y/z) pair
    (99 filters total for MediaPipe's 33 landmarks), plus a hold-last-
    good gate for when pose_estimation returns no landmarks at all for
    up to max_hold_frames consecutive frames, bridging brief tracking
    dropouts instead of flipping the frame to invalid immediately.
"""

import json
import math
from pathlib import Path

CONFIG_PATH = Path("config/smoothing.json")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def _exponential_smoothing(a: float, x: float, x_prev: float) -> float:
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    """min_cutoff: cutoff frequency (Hz) at zero speed -- lower means
    heavier smoothing at rest. beta: how much the cutoff rises with
    speed -- higher means less lag on fast motion. d_cutoff: cutoff used
    to smooth the speed estimate itself.

    Takes the real timestamp `t` (seconds) at each call instead of a
    fixed assumed frequency -- dt = t - t_prev, so a video file's actual
    per-frame spacing (frame_idx / fps) or a webcam's real, sometimes-
    uneven frame arrival (time.monotonic()) both smooth correctly,
    rather than assuming every frame arrived exactly 1/freq apart.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, x: float, t: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        if t_e <= 0:
            # duplicate or out-of-order timestamp -- hold, don't divide by
            # a zero/negative dt
            return self.x_prev

        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class LandmarkSmoother:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float, max_hold_frames: int):
        self.max_hold_frames = max_hold_frames
        self._filters = [
            [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(3)]
            for _ in range(33)
        ]
        self._last_good = None
        self._hold_count = 0

    def update(self, landmarks: list[tuple] | None, t: float) -> list[tuple] | None:
        """landmarks: this frame's raw 33-tuple (x, y, z, visibility) list
        from pose_estimation.get_pose_landmarks(), or None if it detected
        no pose at all. t: this frame's timestamp in seconds (frame_idx /
        fps for a video file, time.monotonic() for a live webcam).
        Returns the smoothed 33-tuple list, the held-over last-good list
        (while within max_hold_frames of a dropout), or None once the
        hold budget is exhausted -- the frame is then invalid, same as a
        raw None would have been.
        """
        if landmarks is None:
            if self._last_good is not None and self._hold_count < self.max_hold_frames:
                self._hold_count += 1
                return self._last_good
            return None

        self._hold_count = 0
        smoothed = [
            (self._filters[i][0](x, t), self._filters[i][1](y, t), self._filters[i][2](z, t), visibility)
            for i, (x, y, z, visibility) in enumerate(landmarks)
        ]
        self._last_good = smoothed
        return smoothed
