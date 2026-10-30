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
