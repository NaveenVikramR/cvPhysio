# Pre-registered evaluation of our own clips

Written and committed before `scripts/evaluate_own_clips.py` has been run on
any real clip, and before any eval clip has been recorded. Git history shows
the order. The evaluation below has no results yet.

## Question

Do the rule scorer's frozen thresholds accept correct shoulder abduction reps
and catch deliberate faults, on our own webcam clips?

## What is frozen

Versions are git blob ids (`git rev-parse <commit>:<path>`), which don't depend
on line endings.

| Item | Frozen as |
|---|---|
| Thresholds | `config/exercises/shoulder_abduction.json`, blob `79c406f505c73c2ea87671b3773ea81e99e16848` |
| Pose settings | `config/pose.json`, blob `c15ab25f8d524734c300a45fcde8285c82ea4411`: `pose_model` lite, `landmark_space` world |
| Elbow statistic | `min` (no `elbow_statistic` key in the thresholds file) |
| Rules | `src/rules.py`, blob `4e85c2b701f86b952e71a4106022739fa41ce8d0` |
| Evaluator | `scripts/evaluate_own_clips.py`, blob `09f2715fbb7bb986d654dac3e28042f81cb3bf54` |
| Recording | `docs/recording_protocol.md`, using `scripts/record_clip.py` |

`reference_relative` tolerances are derived inside the run from the calibration
clips, by the frozen evaluator. They are never edited by hand.

## Data

One subject, one sitting, one setup, recorded after this file is committed:

| Role | Intended | Reps |
|---|---|---|
| reference | correct | 3 |
| calibration | correct | 5 |
| eval | correct | 6 |
| eval | fault_lowarm, fault_trunk, fault_elbow, fault_fast | 3 each |

## Adoption criteria

Judged separately for each scoring mode the evaluator reports (the demo
`absolute` config and the derived `reference_relative` config). A mode passes
only if all of these hold. Both modes are reported whichever passes.

| # | Criterion | Threshold | With the planned reps |
|---|---|---|---|
| 1 | False-fail rate on correct eval reps | at most 20% | at most 1 of 6 |
| 2 | Each fault caught by its expected rule (`fault_lowarm`: `peak_angle`, `fault_trunk`: `trunk_lean`, `fault_elbow`: `elbow_straight`, `fault_fast`: `tempo`) | at least 70%, per fault | 3 of 3, per fault |
| 3 | Thresholds and the other frozen items unchanged | identical blob ids | checked against the run's commit |

Reps scored `insufficient_data` are left out of these denominators, as the
evaluator does, and reported separately. The evaluator's any-rule detection rate
is reported as secondary and can't satisfy criterion 2.

## Inconclusive runs

The result is **inconclusive**, not a pass or a fail, if any of these hold:

- More than 20% of all eval reps are `insufficient_data`.
- Any eval group has fewer scorable reps than planned.
- The evaluator aborts, for example with under 3 detected reference reps.

## Rules for running it

1. Run once, from a clean working tree, on the commit that holds this file or a
   later one where the frozen blob ids still match.
2. Don't delete, re-record or exclude a clip after seeing any score. A clip may
   be dropped only for a recording failure noticed before scoring (camera stopped,
   wrong intended label), and the reason must be logged below.
3. Report every criterion, pass or fail, with its k/n and Wilson 95% interval.
4. Evaluation subjects 3, 4, 5, 7, 8 and 9 of REHAB24-6 are not used for any
   tuning.

## Changes

Noise measurements and the lite-vs-heavy comparison may still lead to a change
of statistic, model or threshold. Before the run, that means committing an
amendment here with its reason and the new blob ids. After the run, any change
needs newly recorded eval clips, because these will have been seen.

| Date | Change | Reason | Commit |
|---|---|---|---|
| | | | |

## What a pass would and wouldn't show

Three reps per fault and one subject make wide intervals: 3 of 3 has a Wilson 95%
lower bound near 44%. A pass supports continuing with these settings for this
person and setup. It isn't evidence of accuracy across people, and it isn't
clinical validation. The thresholds remain provisional until a licensed
physiotherapist reviews them.

## Amendment 1 (2026-10-30)

Made before any frozen evaluation: `scripts/evaluate_own_clips.py` hasn't been
run on real clips. Where this section differs from the text above, this section
applies. Everything else above still stands, including the frozen thresholds and
the inconclusive-run rules.

### Changes

- `elbow_straight` is reported as advisory only. It is excluded from the
  pass/fail verdict (FORM).
- Primary outcomes, replacing criteria 1 and 2:

| Outcome | Rules counted | Threshold | With the planned reps |
|---|---|---|---|
| False-fail rate on correct eval reps | `peak_angle`, `trunk_lean`, `tempo` | at most 20% | at most 1 of 6 |
| Arm-too-low fault (`fault_lowarm`) detected by its expected rule | `peak_angle` | at least 70% | 3 of 3 |
| Trunk-lean fault (`fault_trunk`) detected by its expected rule | `trunk_lean` | at least 70% | 3 of 3 |
| Too-fast fault (`fault_fast`) detected by its expected rule | `tempo` | at least 70% | 3 of 3 |

- The bent-elbow fault (`fault_elbow`, expected rule `elbow_straight`) is a
  secondary exploratory outcome with no pass criterion. Its result is reported
  whatever it shows.

### How the outcomes are computed

The frozen evaluator is unchanged. These outcomes come from the failed-rule
columns of its per-rep output (`results/own_clips_eval.csv`), ignoring
`elbow_straight`. Its own all-rule false-fail figures are still reported, labelled
as not primary.

`scripts/compute_primary_outcomes.py` computes them from that file, with Wilson 95%
intervals, and applies the thresholds and inconclusive rules above. It isn't a pinned
file. It was committed before any eval clip was recorded.

### Rationale

Measured on REHAB24-6, front view, against mocap:

- 3D world elbow angle: slope 0.298, and bias -15.25 deg on straight arms (mocap
  above 160 deg).
- World x,y elbow angle compared with 3D: MAE 8.39 vs 13.09 deg, but r 0.469 vs 0.647.

These REHAB24-6 figures are **reported from the PC run; not reproduced on this
laptop**. They were computed on the PC from calibration subjects 1 and 2 only, camera
17 front, 3 recordings (PM_000, PM_001, PM_012), n = 6658 valid frames, with a 0.5
visibility gate on shoulder, elbow and wrist. The output file `results/elbow_variants.md`
exists only on that PC, because `results/` is gitignored. The script is reported as
commit `62e0290` on the PC's copy of this branch (cherry-picked from `da881fb`). When
this section was written, neither commit was in the laptop's repository or on the remote.

Measured on the laptop with one person:

- Held straight arm reads about 143-145 deg, with single-frame minima of 122-124 deg
  (run `free2`). An earlier run in the same setup (`free`) read about 96 deg on the
  left arm, and that wasn't repeated, so held-pose readings aren't fully stable.
- A/B sessions (5 bent, 15 straight reps) showed no separation of bent from straight
  under `min`, `p10` or `hold_median` (AUC 0.64, 0.47 and 0.37).

### Disclosure

This decision was made after viewing the exploratory A/B and noise data, but before
any frozen evaluation. The laptop data comes from one person, so this is a design
decision, not an accuracy claim. Dropping a rule from the verdict makes passing
easier, so a pass under this amendment says nothing about whether the scorer can
detect a bent elbow.

## Amendment 2 (2026-10-30)

Appended after the first run. Where this section differs from the text above, this
section applies. The frozen files, thresholds and Amendment 1's outcomes are unchanged.

### First run: INCONCLUSIVE

The evaluator ran once, on commit `5e8c0646971f695bf0a4bacfcff129fe863712e8` with a clean
tree. It stopped at the reference clip with 2 reps detected, where at least 3 are
needed, which is inconclusive under the rules above. The arms were raised at the start
of that clip, so the detector took an arms-up baseline and counted the arms-down
periods as reps.

- The evaluator opened no eval clip, and no rule scores were computed.
- `scripts/compute_primary_outcomes.py` was not run, because no per-rep CSV existed.
- Before that run, an integrity check opened every clip, including the eval clips, to
  read frame counts and decode the first and last frames. It used no pose model or scoring.
- After that run, the reference clip alone went through the pose model and rep detector
  to find the cause, viewing shoulder angles and rep events. No rule was scored.
- The seven clips and the manifest are kept in `data/recordings/discarded_run1/`
  (gitignored) and are not used again. All seven clips are re-recorded.

### New recording rules

- Arms down for 2 s before and after each clip, within the recording.
- Each clip ends by its `--seconds` limit, not by pressing `q`.
- Real lighting and clothing are filled in on every clip's notes.

### Quality control

`scripts/qc_clip.py` checks every clip before the evaluator runs. It uses the pose model
and rep detector only, never rule outcomes, and prints no peak, elbow, trunk or tempo
value. A clip must pass all three:

| Check | Passes when |
|---|---|
| Arms down | Every measured shoulder angle in the first 1.5 s and the last 1.5 s is below 45 deg, and each window has at least one measured frame |
| Rep count | Detected reps equal the planned count |
| Tracking | No `lost_tracking` discard, and no run of frames failing the visibility gate longer than the detector's tolerance (10 frames) |

A clip failing QC is re-recorded under the same rule for all clips. The first take that
passes QC is used, and later takes aren't recorded. A failed take is renamed with a
`spare_` prefix and its manifest row removed, because the evaluator reads only the manifest.
The number of failed takes per clip and the failing check are reported with the results.

### Limitations

- **Detector baseline:** the rep detector takes its rest baseline from the first frames it
  sees (15 frames, about 0.5 s), with no check that the arms are down. If they're raised
  there, it inverts rep detection without any error. This was not changed, because the
  detector isn't part of this amendment. The recorder's preview warning and the QC check
  reduce the risk for these clips only; live sessions still have this weakness.
- **QC selects takes:** it keeps only takes whose reps the detector can count. A fault
  movement the detector can't segment, such as a very brief rep, is re-recorded and not
  counted as missed, so the results apply only to reps the detector segmented.

## Amendment 3 (2026-10-30)

Appended before any rule was scored on these takes. Where this section differs from the
text above, this section applies. The frozen files, thresholds and Amendment 1's
outcomes are unchanged.

### Failed takes

These eight takes failed `scripts/qc_clip.py`, each run alone in its own process. All
eight are excluded from scoring: they're renamed with a `spare_` prefix in
`data/recordings/` and have no manifest row.

| Take | Failed checks | Rep count (detected / planned) |
|---|---|---:|
| `abd_reference_correct_01` | arms down, rep count, tracking | 0 / 3 |
| `abd_calibration_correct_01` | arms down, rep count, tracking | 6 / 5 |
| `abd_eval_correct_01` | arms down, rep count, tracking | 9 / 6 |
| `abd_eval_fault_lowarm_01` | rep count | 6 / 3 |
| `abd_eval_fault_trunk_01` | arms down, rep count | 7 / 3 |
| `abd_eval_fault_trunk_02` | arms down | 3 / 3 |
| `abd_eval_fault_elbow_01` | arms down, rep count, tracking | 1 / 3 |
| `abd_eval_fault_elbow_02` | arms down | 3 / 3 |

`abd_eval_fault_trunk_03` passed all three checks when run after other clips in one
process, but fails the tracking check when run alone, by one frame (longest gated-out run
11 frames, limit 10), on each of four runs. The alone result decides, so it is a ninth
failed take, excluded from scoring and re-recorded like the others. It is now
`spare_abd_eval_fault_trunk_03`.

### Takes that failed during re-recording

These ten later takes also failed `scripts/qc_clip.py`, run alone, and are excluded from
scoring in the same way:

| Take (file name now) | Failed checks | Rep count (detected / planned) |
|---|---|---:|
| `spare_abd_reference_correct_01_take2` | arms down | 3 / 3 |
| `spare_abd_calibration_correct_01_take2` | arms down, rep count, tracking | 4 / 5 |
| `spare_abd_calibration_correct_01_take3` | arms down | 5 / 5 |
| `spare_abd_eval_correct_01_take2` | arms down | 6 / 6 |
| `spare_abd_eval_fault_lowarm_01_take2` | rep count | 0 / 3 |
| `spare_abd_eval_fault_lowarm_01_take3` | rep count | 1 / 3 |
| `spare_abd_eval_fault_lowarm_01_take4` | tracking | 3 / 3 |
| `spare_abd_eval_fault_trunk_01_take3` | rep count | 4 / 3 |
| `spare_abd_eval_fault_trunk_01_take4` | rep count | 0 / 3 |
| `spare_abd_eval_fault_trunk_03_take2` | rep count | 2 / 3 |

That makes 19 failed takes in all, across the seven clips.

### Clips used

The first take of each clip to pass all three checks alone:

| Clip | Failed takes before it |
|---|---:|
| `abd_reference_correct_02` | 2 |
| `abd_calibration_correct_01` | 3 |
| `abd_eval_correct_01` | 2 |
| `abd_eval_fault_lowarm_01` | 4 |
| `abd_eval_fault_trunk_02` | 5 |
| `abd_eval_fault_elbow_01` | 2 |
| `abd_eval_fault_fast_01` | 0 |

`spare_abd_eval_fault_trunk_03_take2` was recorded after `abd_eval_fault_trunk_02` had been
recorded, before that take's QC result had been reviewed. It failed and isn't used.

This many failed takes means the clips used are the takes that met these recording
conditions, not a typical attempt, as the "QC selects takes" limitation in Amendment 2 says.

### Disclosure

During diagnosis, detector peak angles, rep ranges and rep timings, and per-second
shoulder angle ranges, were printed for the eight takes in the table, including the six
eval-role takes. No rule was scored. That output was not produced for
`abd_eval_fault_trunk_03`, for which only the three QC results were printed. Later, the
detector's state changes and discard reasons, with no angles, were printed for the take
now named `spare_abd_eval_fault_trunk_01_take4`. For every other take, including all seven
clips used, only the QC results were printed.

Recording guidance given after failed takes concerned framing, rest, timing and movement
size. It was based on QC results and these outputs, and on no rule score. The guidance for
the arm-too-low clip did follow the diagnosis output described above.

The clips were recorded with the `--countdown` change in the working tree, before it was
committed. The evaluator runs only after it and this amendment are committed.

Running every clip in one process gave different details from running each alone: for
`abd_eval_fault_elbow_01`, the arms-down check passed in one process and failed alone, and
`abd_eval_correct_01` had one more tracking loss. Pose tracking carries over between clips
in one process, as it does in the evaluator.

### Recording rules added

- The countdown between pressing SPACE and recording is 15 s (`--countdown`, default 15),
  so the subject is in position, arms down, before recording starts.
- The subject stops at the planned rep count and stands still, arms down, until the
  `--seconds` limit ends the clip.

### Final check before scoring

`scripts/qc_all.py` (not a pinned file) is run once, after every clip passes
`scripts/qc_clip.py` and before `scripts/evaluate_own_clips.py`. It checks that the manifest
holds exactly the planned clips, runs each clip alone, then runs all of them in one process
in the evaluator's order, and prints detected and planned reps for each. The evaluator is
not run if a clip fails in the combined run, or if any clip's rep count differs between the
two runs. It uses the pose model and rep detector only and scores no rule.

## Amendment 4 (2026-10-31)

Appended before any rule was scored on the seven clips used. Where this section differs from
the text above, this section applies. Thresholds, rules, pose settings and Amendment 1's
outcomes are unchanged.

### What happened

`scripts/qc_all.py` printed DO NOT SCORE: `abd_eval_correct_01` was detected with 6 reps alone
and 7 when run after other clips in one process, with its planned count being 6. The other
six clips agreed. As Amendment 3 requires, the evaluator was not run.

### Change

Pose tracking carried over between clips in one process, so a clip's detected reps depended
partly on the clip before it. `scripts/evaluate_own_clips.py` now resets pose tracking at the
start of each clip, using a new `reset_tracking()` in `src/pose_estimation.py`.
`scripts/qc_clip.py` does the same. The clips, manifest and recordings are unchanged, and no
clip was re-recorded for this.

This changes one pinned file:

| Item | Was | Now |
|---|---|---|
| Evaluator, `scripts/evaluate_own_clips.py` | blob `09f2715fbb7bb986d654dac3e28042f81cb3bf54` | blob `4382f70f5012da8ed6a8630ddcaa121e4f0617e2` |

The other three pins are unchanged. `PINNED_BLOBS` in `scripts/compute_primary_outcomes.py`
holds the new evaluator id.

### Disclosure

This change was made after seeing that QC result, to resolve it. Only detected rep counts and
QC check results were seen for this clip. No rule score, peak, elbow, trunk or tempo value
from any of the seven clips was viewed. Re-recording or reordering clips until the check passed
was rejected.

With the change in the working tree and not yet committed, `scripts/qc_all.py` was run once to
test it. All seven clips passed, with matching counts, including `abd_eval_correct_01` at 6 in
both runs. It is run once more on the committed code before the evaluator, and the evaluator
isn't run unless that run also prints OK to score.

The reset applies to offline runs only. A live session keeps tracking from frame to frame.
