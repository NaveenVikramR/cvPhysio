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
