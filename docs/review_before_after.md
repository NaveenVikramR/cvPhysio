# Joint-location accuracy: before and after (F1 → F3)

Written 2026-11-17 for the project review, which asked us to improve how
accurately joints are located. Every figure below comes from a run of the
named script on the PC. The full tables are in `results/` (gitignored);
re-run the scripts to regenerate them. Details of each step are in
`docs/landmark_fidelity.md`.

## How accuracy is measured

- **Ground truth:** REHAB24-6 Exercise 1 (right-arm shoulder abduction).
  Every frame has motion-capture joint positions recorded at the same time
  as the video.
- **What is compared:** joint *angles*, the shoulder (abduction), elbow and
  trunk lean, as the app computes them from MediaPipe, against the same
  angles from motion capture. Point positions can't be compared directly,
  because the two skeletons put their joints at different places.
- **Honest testing:** subjects 1, 2 and 6 are used to choose or fit
  anything (calibration). Subjects 3, 4, 5, 7, 8 and 9 are only used to
  check the result (evaluation). For every change, the rule for choosing
  was written into the script before it ran.
- **Camera view:** front view is the recording setup the app is designed
  for. Half-profile is reported for reference.

## Result: the live app before and after

`scripts/f3_before_after.py` runs the live pipeline both ways on the same
recordings. *Before* is the app at the start of this work (the arm picked
again on every frame, raw angle). *After* is the app now (the arm locked
for the session, the shoulder offset corrected).

**Shoulder angle, front view, evaluation subjects (9,291 frames):**

| | Before | After |
|---|---|---|
| Frames measured on the wrong (idle) arm | 13.3% | **0.0%** |
| Average error (MAE) | 10.87° | **6.28°** |
| Average offset (bias) | −10.84° | −5.68° |
| Error once each group's offset is removed | 4.54° | 4.33° |
| Correlation with the true angle (r) | 0.987 | 0.989 |

**Rep counting, evaluation subjects (8 recordings):**

| | Before | After |
|---|---|---|
| Recordings counted exactly | 1/8 | 2/8 |
| Reps measured on a mix of both arms | 18/145 | **0/138** |

On calibration subjects, front view, the average error went from 6.16° to
4.24° and wrong-arm frames from 6.1% to 0%.

## What changed in the app

1. **The arm is locked per session** (F2 step 1). The app used to choose
   the more visible arm on every frame, so at rest it sometimes measured
   the idle arm. It now locks to the arm that completes the first rep;
   the user can also pick the arm. In all 17 front and half-profile
   recordings it locked the correct arm.
2. **The shoulder angle's constant offset is corrected** (F2 step 2).
   MediaPipe reads the shoulder angle low by a fairly constant amount. A
   fixed correction of +4.82°, fitted on calibration subjects, lowered the
   error for all six evaluation subjects.

## What was tried and rejected

Each of these was tuned on calibration subjects with a pre-declared rule,
then checked once on evaluation subjects. Rejecting them is a result, not a
gap: each would have made the app worse or no better.

| Step | Idea | Outcome |
|---|---|---|
| F2 step 2 | Offset plus scale correction; per-person correction from the rest pose | Scale fitted the two calibration subjects but was slightly worse on new people (6.51° vs 6.28°). Per-person made two of six subjects clearly worse. |
| F2 step 3 | Smoothing the landmarks (One Euro filter, 36 settings) | Every setting raised the shoulder error (best 4.32° vs 4.24° with none). The remaining error isn't frame-to-frame shake. Smoothing stays off. |
| F2 step 4 | Steadier rep counting (start/end debounce, steadier rest angle; 48 settings) | Better on calibration (5/9 → 7/9 exact) but worse on evaluation (total count error 8 → 24). Not used. |
| F2 step 5 | A better per-rep elbow measure (12 candidates) | The calibration winner didn't beat the current measure on evaluation (rho 0.40 vs 0.42). The elbow check stays advisory. |

## Limits

- **Small groups.** Front-view fitting used two subjects (subject 6 has no
  front view). After correction, evaluation subjects still read about 5.7°
  low on average, because the calibration pair read less low than the new
  people.
- **The remaining error is mostly a per-person offset.** Once each
  group's offset is removed, about 4.3° is left in front view. That is
  close to the floor for a fixed correction.
- **The elbow angle is weak.** MediaPipe follows only a fifth to a quarter
  of the real elbow movement frame by frame. Per rep, it spots a truly
  bent rep about 79% of the time (AUC) in front view, which is too weak for
  a pass/fail rule.
- **Half-profile is much worse** (shoulder error about 13° even after
  removing the offset), which is why the setup is front view.
- **Rotation isn't covered.** REHAB24-6 Exercise 1 has no rotation, so
  rotation accuracy needs the team's camera-angle test clips.
- **One correction during this work.** Five evaluation recordings contain
  a block of front reps and a block of half-profile reps. An earlier table
  filed them under front only. Totals and decisions were unaffected; the
  labels were corrected (`docs/landmark_fidelity.md`, F2 step 1).

## Scripts

| Step | Script |
|---|---|
| F1 baseline | `scripts/landmark_fidelity_report.py` |
| F2 step 1, arm locking | `scripts/evaluate_side_lock.py` |
| F2 step 2, offset correction | `scripts/fit_angle_correction.py` |
| F2 step 3, smoothing | `scripts/tune_smoothing_mocap.py` |
| F2 step 4, rep counting | `scripts/tune_rep_detector.py` |
| F2 step 5, elbow measure | `scripts/compare_elbow_rep_stats.py` |
| F3 before/after | `scripts/f3_before_after.py` |
