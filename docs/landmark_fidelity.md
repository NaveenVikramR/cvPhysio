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

## Improvements to try in F2, ranked by expected gain against effort

All of these come after the abduction pre-registered evaluation has run, because each one changes abduction results. Each is fitted on calibration subjects only and reported on evaluation subjects.

1. **Lock the arm per session.** Choose the arm once, for example at calibration or from whichever arm moves, instead of every frame (finding 5). Cheap, and removes a real source of error.
2. **Offset and slope correction for the shoulder angle (finding 1).** Expect the error to fall toward the roughly 4° that remains once offsets are removed, with some offset left over for subjects unlike those in calibration.
3. **One Euro smoothing tuned against motion capture.** Target the jitter without adding lag around rep boundaries.
4. **Elbow measure:** keep it advisory, and measure whether a hold-window statistic or a 2D variant agrees better (`scripts/compare_elbow_variants.py` exists).
5. **Rotation:** the camera angle and a depth-free measure (`docs/rotation_diagnosis.md`). REHAB24-6 Ex1 has no rotation, so rotation accuracy needs the laptop test clips, or the REHAB24-6 Ex2 videos if downloaded.
