# Algorithms (pseudocode, with the actual constants in use)

Pseudocode below mirrors the real control flow line-for-line where it
matters (branch order, inclusive/exclusive comparisons) — it is not a
simplified summary. Exact source: `src/feature_extraction.py` (angle),
`app.py`'s `RepDetector` class (segmenter), `src/rules.py`'s `score_rep`
(scoring), `src/smoothing.py` (filter).

## compute_angle(a, b, c)

```
function compute_angle(a, b, c):
    a, b, c = a[:3], b[:3], c[:3]          # drop any 4th field (e.g. visibility)
    ba = a - b
    bc = c - b
    cos_theta = dot(ba, bc) / (norm(ba) * norm(bc))
    cos_theta = clip(cos_theta, -1.0, 1.0)  # guards float overshoot past +-1
    return degrees(arccos(cos_theta))       # angle AT b, between b->a and b->c
```

No special-casing for zero-length vectors — if `a == b` or `c == b`,
`norm(...)` is 0 and this divides by zero (not guarded; see
`docs/open_questions.md`).

## The rep segmenter (`RepDetector.update`, 3-state hysteresis machine)

Constants actually used (from `app.py`):

```
CALIBRATION_FRAMES        = 15     # visible frames to seed the baseline
CALIBRATION_MAX_STD_DEG    = 6.0    # reject the calibration window above this stdev
REST_BAND_DEG              = 10.0   # |angle - baseline| considered "at rest"
BASELINE_WINDOW            = 10     # rolling window for the live baseline
MIN_REP_FRAMES             = 15     # minimum frames for a real rep
MIN_REP_RANGE_DEG          = 25.0   # minimum angle range for a real rep
MAX_REP_FRAMES             = 450    # ceiling before "too_long"
CONFIDENCE_TOLERANCE_FRAMES = 10    # consecutive invisible frames tolerated mid-rep
CALIBRATION_STALL_FRAMES   = 150    # frames stuck calibrating before a hint fires
RESCUE_REST_BAND_DEG        = 2 * REST_BAND_DEG = 20.0   # see rescue branch below
```

```
state: one of CALIBRATING, RESTING, MOVING
baseline: float | None
calib_window: last up to 15 visible angle_deg values
rest_window: last up to 10 visible angle_deg values (while RESTING)
rep_buffer, side_votes: this rep's accumulated angle_deg / side, while MOVING
bad_confidence_streak: consecutive invisible frames while MOVING

function update(visible, angle_deg, side, failed):

    if state == CALIBRATING:
        if not visible:
            record failed landmarks for calibration_hint()
            return NONE
        append angle_deg to calib_window
        if len(calib_window) < 15: return NONE
        if stdev(calib_window) > 6.0:
            drop the oldest entry (NOT the whole window)   # retry, don't restart from empty
            return NONE
        baseline = mean(calib_window); clear calib_window
        state = RESTING
        return NONE

    if state == RESTING:
        if not visible: return NONE
        if |angle_deg - baseline| <= 10.0:
            baseline = mean(rest_window + angle_deg)        # live-updated while resting
            return NONE
        # left the rest band -> a rep has started
        rep_buffer = [angle_deg]; side_votes = [side]
        bad_confidence_streak = 0
        state = MOVING
        return NONE

    # state == MOVING
    if not visible:
        bad_confidence_streak += 1
        if bad_confidence_streak > 10:
            # --- tracking-lost rescue branch ---
            if rep_buffer is non-empty AND |rep_buffer[-1] - baseline| <= 20.0:
                range = max(rep_buffer) - min(rep_buffer)
                if len(rep_buffer) >= 15 AND range >= 25.0:
                    emit REP_COMPLETE(rep_buffer, side_votes)   # rescued, not discarded
                    baseline = mean(rest_window + rep_buffer[-1]); reset to RESTING
                    return (REP_COMPLETE, ...)
            reset to RESTING
            return (REP_DISCARDED, reason_code="lost_tracking")
        return NONE

    # visible, still MOVING
    bad_confidence_streak = 0
    append angle_deg, side to rep_buffer, side_votes

    if len(rep_buffer) > 450:
        reset to RESTING
        return (REP_DISCARDED, reason_code="too_long")

    if |angle_deg - baseline| > 10.0:
        return NONE                                          # still moving, nothing to report

    # back within the rest band
    range = max(rep_buffer) - min(rep_buffer)
    if len(rep_buffer) < 15 OR range < 25.0:
        reset to RESTING
        return (REP_DISCARDED, reason_code="too_short")

    emit REP_COMPLETE(rep_buffer, side_votes)
    baseline = mean(rest_window + angle_deg); reset to RESTING
    return (REP_COMPLETE, ...)
```

**Why `RESCUE_REST_BAND_DEG` ≠ `REST_BAND_DEG`**: any frame that lands
inside `REST_BAND_DEG` while still **visible** already triggers the
normal completion/`too_short` path above, in the same call — so by the
time the tracking-lost branch is reached, `rep_buffer[-1]` is, by
construction, always still *outside* `REST_BAND_DEG`. The rescue needs
a separate, wider threshold to ever fire at all (code comment in
`app.py` states this invariant explicitly).

## score_rep (rule scoring, 3-way status)

```
function score_rep(rep, config, fps):
    angle, elbow, trunk = rep.angle_deg, rep.elbow_angle_deg, rep.trunk_lean_deg
    valid_fraction = rep.valid_fraction
    assert len(angle) == len(elbow) == len(trunk) and len(angle) > 0   # else ValueError

    peak = max(angle); worst_elbow = min(elbow); worst_trunk = max(trunk)
    duration_s = len(angle) / fps

    rules.peak_angle     = { measured: peak,        allowed: [target_peak-tol, target_peak+tol], passed: allowed[0] <= peak <= allowed[1] }
    rules.elbow_straight = { measured: worst_elbow,  allowed: [min_elbow, 180.0],                 passed: worst_elbow >= min_elbow }
    rules.trunk_lean     = { measured: worst_trunk,  allowed: [0.0, max_trunk],                   passed: worst_trunk <= max_trunk }
    rules.tempo          = { measured: duration_s,   allowed: [tempo_min, tempo_max],              passed: tempo_min <= duration_s <= tempo_max }

    failed_rules = [id for id, r in rules if not r.passed]

    # ALL FOUR RULES ARE COMPUTED REGARDLESS OF valid_fraction -- the
    # insufficient_data check happens AFTER, as an override on top:
    if valid_fraction < config.min_valid_fraction:
        return { status: "insufficient_data", failed_rules, rules, valid_fraction,
                 cue: config.cues.insufficient_data }

    return { status: (failed_rules empty) ? "pass" : "fail",
             failed_rules, rules, valid_fraction, cue: None }
```

Note: `insufficient_data` is a status **override**, not a short-circuit
— a rep with too little valid data still has its 4 rules evaluated (so
the calibration-set report can use them), it just never gets to claim
`"pass"` or `"fail"`.

## One Euro filter (`OneEuroFilter.__call__`)

```
state per filter instance: x_prev, dx_prev, t_prev (all None initially)

function filter(x, t):
    if x_prev is None:
        x_prev, dx_prev, t_prev = x, 0.0, t
        return x                                   # first sample passes through unfiltered

    t_e = t - t_prev
    if t_e <= 0:
        return x_prev                              # duplicate/out-of-order timestamp: hold

    a_d = smoothing_factor(t_e, d_cutoff)
    dx = (x - x_prev) / t_e                         # estimated speed
    dx_hat = a_d * dx + (1 - a_d) * dx_prev          # smoothed speed

    cutoff = min_cutoff + beta * |dx_hat|            # speed-adaptive cutoff
    a = smoothing_factor(t_e, cutoff)
    x_hat = a * x + (1 - a) * x_prev                 # the actual output

    x_prev, dx_prev, t_prev = x_hat, dx_hat, t
    return x_hat

function smoothing_factor(t_e, cutoff):
    r = 2 * pi * cutoff * t_e
    return r / (r + 1)
```

`LandmarkSmoother` runs one independent instance of the above per
(landmark, axis) pair — 33 landmarks × 3 axes (x, y, z) = 99 filters;
`visibility` (the 4th tuple field) passes through unfiltered. Its
`update()` wraps this with a hold-last-good gate: if `landmarks is None`
and the hold budget (`max_hold_frames`) isn't exhausted, it returns the
last good (already-filtered) landmark list unchanged and increments a
hold counter; the counter resets to 0 on the next real frame.
