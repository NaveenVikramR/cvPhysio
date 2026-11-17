# Landmark fidelity baseline (F1)

Written 2026-11-16 from a run of `scripts/landmark_fidelity_report.py` on
the PC, at the commit that adds this file. The settings were those shipped
at the time: `lite` model, `world` landmarks, smoothing off, visibility
gate 0.5. The full tables are in `results/landmark_fidelity.md`
(gitignored); re-run the script to regenerate them.

**What it measures:** MediaPipe joint angles as the app computes them,
compared with the same angles from REHAB24-6 motion capture. The data is
Exercise 1 (right-arm abduction), every annotated rep, 54,884 frames,
subjects 1–9. It compares angles rather than point positions because the
two skeletons put their joints at different anatomical places.

**Subject groups:** calibration subjects 1, 2 and 6, evaluation subjects 3,
4, 5, 7, 8 and 9. Subject 6 has no front-view recording, so front-view
calibration figures come from subjects 1 and 2 only. That is a very small
group.

## Front view (the recommended setup for abduction)

| Angle | Group | Bias | MAE | MAE after removing bias | r | Slope |
|---|---|---|---|---|---|---|
| shoulder | calibration | −4.8° | 6.2° | 4.2° | 0.994 | 1.06 |
| shoulder | evaluation | −10.5° | 10.5° | 4.3° | 0.989 | 0.98 |
| elbow | calibration | −12.8° | 13.5° | 6.1° | 0.645 | 0.29 |
| elbow | evaluation | −18.1° | 19.7° | 8.5° | 0.391 | 0.18 |
| trunk lean | calibration | +5.2° | 5.7° | 3.7° | 0.271 | 0.18 |
| trunk lean | evaluation | +6.6° | 7.0° | 3.7° | 0.183 | 0.21 |

## Findings

1. **The shoulder angle tracks motion capture closely; most of the error is a constant offset.**
   - r is about 0.99 and the slope about 1. Once each group's own offset is removed, the error drops to about 4.3°.
   - **About 10° of the offset recorded in earlier reports came from how motion capture was measured, not from MediaPipe.** Earlier scripts used the motion-capture "Hips" root joint as the hip. Against the anatomical right hip, which is what MediaPipe's hip landmark represents, the bias falls from −15.2° to −4.8° (calibration) and from −20.9° to −10.5° (evaluation).
   - The remaining offset varies by subject, from −3.8° to −13.6°. A single correction fitted on calibration subjects would therefore leave some offset on evaluation subjects.
2. **The elbow angle barely follows the real elbow.**
   - The true elbow moves between about 140° and 178° during Ex1, but MediaPipe follows only a fifth to a quarter of that change (slope 0.18–0.29), and r is only 0.39–0.65. It also reads 13–18° too bent.
   - This independently confirms the decision in pre-registration Amendment 1 to make `elbow_straight` advisory.
3. **Trunk lean is about right on average but doesn't follow small changes.**
   - The true lean varies only about 1–12°. MediaPipe reads about 5–7° high in front view and follows little of the variation (r 0.18–0.27).
   - The `trunk_lean` rule therefore works best as a coarse check for large leans, as its 15° threshold already does.
4. **Camera view matters a great deal.**
   - Half-profile roughly triples the shoulder error after bias removal (about 12° against 4°) and doubles the jitter.
   - In the profile camera (cam18) the right arm never passes the confidence gate.
   - Front view is the right recording setup for abduction.
5. **The live app sometimes measures the wrong arm.**
   - In front view, `select_side()` picks the idle left arm on 10.9% of frames, almost all near rest (median true shoulder angle about 25°). 63 of 88 front-view reps contain at least one such frame.
   - Per-frame side switching adds noise at rep starts and ends, exactly where the rep detector makes its decisions.
6. **The confidence gate never triggers on this data:** 100% of front-view frames pass. Whether 0.5 is the right cutoff can't be judged from these recordings.

## F2 step 1 result: arm locking (2026-11-17)

**What changed.** `src.pipeline.Session` gained a `side` option:
- `"per_frame"` is the original behaviour and still the API default, so existing scripts reproduce.
- `"LEFT"`/`"RIGHT"` fixes the arm for the whole session.
- `"auto"` runs one detector per arm until the first rep completes on either, then locks to that arm.

The live app (Streamlit "Which arm?" selector, and `--side` with `--video`) now uses `auto` by default. Measured with `scripts/evaluate_side_lock.py` on every REHAB24-6 Ex1 recording; full table in `results/side_lock.md` (gitignored).

| View, group | Mode | Exact rep count | Mean signed error | Reps measured on both arms |
|---|---|---|---|---|
| front, calibration | per_frame → auto | 2/3 → 2/3 | −0.33 → −0.33 | 4/24 → 0/24 |
| front, evaluation | per_frame → auto | 1/6 → 2/6 | +1.50 → +0.33 | 8/123 → 0/116 |
| half-profile, calibration | per_frame → auto | 2/6 → 3/6 | −2.17 → +0.83 | 11/45 → 0/63 |
| half-profile, evaluation | per_frame → auto | 0/2 → 0/2 | +1.00 → +1.00 | 10/22 → 0/22 |

- **No rep is measured on a mix of arms any more.**
- **`auto` chose the right arm** in all 17 front and half-profile recordings. Its results there are identical to fixing the right arm by hand.
- **Rep counts improved slightly.** With 2–6 recordings per row, the 95% intervals overlap almost completely, so this is a direction, not a demonstrated gain. Most remaining counting errors come from the rep detector itself (the drifting baseline and missing debounce in `docs/rotation_diagnosis.md`, open question #14).
- **Profile view:** the right arm is hidden, so `auto` mostly locked the left. That view is unsupported, as finding 4 already says.
- **Not checked:** a live webcam session with the new selector (the page renders; Start was not pressed), and the new team recordings, which don't exist yet.

## F2 step 2 result: shoulder offset correction (2026-11-17)

**What was compared.** `scripts/fit_angle_correction.py` fitted four corrections of the shoulder (abduction) angle on front-view calibration subjects 1 and 2 only, and judged them on evaluation subjects 3, 4, 5, 7, 8 and 9. Full tables are in `results/angle_correction.md` (gitignored).

| Method | What it does | Front-view evaluation: bias | MAE | 90th pct |
|---|---|---|---|---|
| A none | raw angle | −10.50° | 10.53° | 17.74° |
| **B offset** | subtract −4.82° (calibration mean error) | −5.68° | **6.28°** | 12.92° |
| C linear | invert mediapipe = 1.060 × mocap − 9.37 | −5.56° | 6.51° | 13.02° |
| D rest-anchor | per recording, anchor the first 15 frames to the calibration subjects' mean true rest angle (34.08°) | +5.90° | 7.74° | 13.83° |

Per evaluation subject (front-view MAE, A → B): 3: 8.23 → 4.13, 4: 11.07 → 6.63, 5: 9.77 → 5.94, 7: 8.58 → 4.82, 8: 10.93 → 6.53, 9: 13.62 → 8.80.

**Chosen: B.**
- It lowers the error for all six evaluation subjects; front-view MAE falls by about 40%.
- C fits the calibration pair better (MAE 3.46° against 4.24°) but is slightly worse on new people, which suggests the slope is fitted to those two subjects.
- D is inconsistent: better for subjects 5, 8 and 9, clearly worse for 3 and 7. It assumes everyone rests at the same true angle. It was best in half-profile (11.07° against 15.05° for B), but front view is the recording setup.

**What changed.**
- `config/angle_correction.json` holds the offset (−4.82° for `shoulder_abduction` only) with its fitting provenance.
- `src.feature_extraction.extract_features(..., correction=None)` applies `raw − offset`, clipped to [0, 180]. `Session(..., correction=None)` passes it through. Elbow and trunk angles are never corrected. With the default `None` every analysis script reproduces, including the pre-registered evaluation (no pinned file changed).
- The live app (Streamlit and `--video`) turns it on for the rules scorer only; `--angle-correction off` gives raw angles. The classifier scorer always gets raw angles.
- New references store `angle_offset_deg`. Loading a reference recorded with a different offset (including every reference saved before this change, which counts as raw) shows a warning, because in reference-relative mode the mismatch shifts the peak comparison by about 4.8°.

**Limits.**
- Fitted on two subjects. Evaluation subjects still read about 5.7° low after correction, so the calibration pair read less low than the new people. A larger offset chosen by looking at evaluation subjects would not be a fair fit, so it isn't made.
- Reference-relative scoring cancels a constant offset, so it is unaffected. The gain is in the angles shown on screen and in the absolute `peak_angle` band (target 90° ± 15°).
- Rotation has no correction: REHAB24-6 Ex1 has no rotation ground truth.
- Rep detection is unchanged: the detector works relative to the calibrated rest baseline. Checked on a spare s01 calibration clip (not one of the seven evaluation clips): same reps, peaks 4.8° higher, every other measurement identical.

## Improvements to try in F2, ranked by expected gain against effort

All of these come after the abduction pre-registered evaluation has run, because each one changes abduction results. Each is fitted on calibration subjects only and reported on evaluation subjects.

1. **Lock the arm per session.** Choose the arm once, for example at calibration or from whichever arm moves, instead of every frame (finding 5). Cheap, and removes a real source of error.
2. **Offset and slope correction for the shoulder angle (finding 1).** Expect the error to fall toward the roughly 4° that remains once offsets are removed, with some offset left over for subjects unlike those in calibration.
3. **One Euro smoothing tuned against motion capture.** Target the jitter without adding lag around rep boundaries.
4. **Elbow measure:** keep it advisory, and measure whether a hold-window statistic or a 2D variant agrees better (`scripts/compare_elbow_variants.py` exists).
5. **Rotation:** the camera angle and a depth-free measure (`docs/rotation_diagnosis.md`). REHAB24-6 Ex1 has no rotation, so rotation accuracy needs the laptop test clips, or the REHAB24-6 Ex2 videos if downloaded.
