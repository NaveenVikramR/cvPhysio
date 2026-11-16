# Project handover

Written 2026-11-16, from the PC, at `elbow-accuracy` commit `dedd7c3`
(working tree clean, 482/482 tests passing). This file is a snapshot,
not a living doc -- re-verify the "current state" section against
`git log`/`git status` before trusting it, especially the pre-registered
evaluation's status, which is time-sensitive.

## What this is

cvPhysio: a real-time webcam rehab coach. Pose estimation (MediaPipe) ->
joint-angle features -> rule-based (or classifier) correctness scoring ->
feedback (visual + speech) -> session logging -> Streamlit UI. University
semester project. Full scope, architecture and hard rules are in
`CLAUDE.md` at the repo root -- read that first, it governs everything
below (exactly 2 exercises, 6-module boundaries, no fabricated metrics,
confidence gating, module independence, classical ML only).

Two people/machines have worked on this repo under Claude Code: this
machine ("the PC", commits signed `Claude Sonnet 5`) and another machine
referred to in commit messages and docs as "the laptop" (commits signed
`Claude Opus 5.5`). Several recent decisions explicitly distinguish
"measured on the PC" vs "measured on the laptop" -- that distinction is
load-bearing (see the pre-registered eval below), not incidental.

## Current state (verify before trusting)

- **`master` is stale** (`3f36d5b`, 2026-10-26-ish). It does **not**
  reflect current work. Do not treat it as "the real state of the
  project."
- **`elbow-accuracy` is the active integration branch** -- it already
  contains everything from `ui-theme` and `laptop-recording` (both
  merged in), plus its own name-sake elbow-angle work, plus four
  pre-registration amendments for an own-clip evaluation. 91 commits
  total since 2026-09-18. No PRs are used in this repo (`gh pr list`
  returns nothing, ever) -- everything lands via direct branch merges
  and pushes.
- `ui-theme` and `laptop-recording` still exist as branches on `origin`
  but are **fully subsumed** by `elbow-accuracy` -- their content is
  already there via merge commits (`ba9ae8d`, `4be13a3`). They're not
  "pending work still to integrate."
- There is also a local-only branch `elbow-variants-pc` (`da881fb`) --
  this is a renamed-not-deleted leftover from a branch-name collision
  (the PC and the laptop independently created a branch called
  `elbow-accuracy` with unrelated content; the PC's was renamed to
  `elbow-variants-pc` and its one commit cherry-picked onto the real,
  laptop-tracked `elbow-accuracy`). Safe to delete once you've confirmed
  you don't need it; nothing currently depends on it.
- 482 tests pass as of `dedd7c3`. If you see a different number, something
  changed underneath you -- `git log` to see what.
- `results/` and `models/*.task`/`*.pkl` are gitignored and **per-machine**.
  A results file reported in a doc (e.g. `results/elbow_variants.md`)
  may exist on only one of the two machines -- several docs explicitly
  say so (see Amendment 1's Rationale section in
  `docs/preregistered_own_clip_eval.md`).

## The one thing in progress right now: the pre-registered own-clip evaluation

This is the main active thread and the most important thing to read
before doing anything else: **`docs/preregistered_own_clip_eval.md`**,
in full, including all 4 amendments. Summary:

- **Question**: do the rule scorer's frozen thresholds accept correct
  shoulder-abduction reps and catch deliberate faults, on the laptop's
  own webcam clips?
- **Discipline**: thresholds, pose config, `src/rules.py`, and the
  evaluator script are all pinned by git blob id, frozen *before* any
  clip was scored. Every subsequent change (Amendments 1-4) is logged
  with its reason, committed before the relevant data was seen, and
  discloses exactly what was and wasn't looked at first.
- **Amendment 1**: `elbow_straight` demoted to advisory-only (excluded
  from pass/fail) based on exploratory noise/A-B data from the laptop
  and the PC's elbow-variant measurements -- cited explicitly as "not
  reproduced on this laptop" and "exists only on that PC."
- **Amendment 2**: first real run was **inconclusive** (bad baseline
  calibration because arms were raised at clip start) -- all 7 clips
  discarded and re-recorded under new rules (arms down 2s before/after,
  fixed clip length, lighting/clothing logged). Added
  `scripts/qc_clip.py` as a pre-scoring gate.
- **Amendment 3**: 19 (!) re-recorded takes failed QC before 7 finally
  passed. Added `scripts/qc_all.py` as a final manifest-wide gate (checks
  rep counts match both when a clip is run alone and when run after other
  clips in one process).
- **Amendment 4** (most recent, 2026-10-31): `qc_all.py` caught a real
  bug -- MediaPipe's tracking state was carrying over *between clips* in
  one process, so a clip's detected rep count depended on what ran before
  it. Fixed with a new `reset_tracking()` in `src/pose_estimation.py`,
  called at the start of each clip in `evaluate_own_clips.py` and
  `qc_clip.py` (offline-only; a live session still tracks continuously).
  This changed the evaluator's pinned blob id (now `4382f70f...`) --
  confirmed on this PC to still match `scripts/compute_primary_outcomes.py`'s
  `PINNED_BLOBS`.

**Exact next step, per Amendment 4's own text**: run `scripts/qc_all.py`
once more on the now-committed code; only if it prints OK to score for
all 7 clips, run `scripts/evaluate_own_clips.py` for the real, final,
binding result, then `scripts/compute_primary_outcomes.py` on its output
to get the actual pass/fail/inconclusive verdict against Amendment 1's
criteria. **This has not happened yet** as of this handover -- no
`results/own_clips_eval.csv` exists on this PC, and Amendment 4's text
itself says the evaluator "isn't run unless that run also prints OK to
score."

**This can only be done on the laptop.** The real recorded clips
(`abd_reference_correct_02`, `abd_calibration_correct_01`, etc.) are
gitignored and device-local; this PC has only a stale `data/own_manifest.csv`
(an example/template from early scaffolding, not the real 7-clip
manifest) and no real clips in `data/recordings/`. Confirmed by checking
`data/recordings/` on this machine -- only unrelated REHAB24 demo clips
are present.

Once there's a real result: rule #2 in the pre-registration ("don't
delete, re-record or exclude a clip after seeing any score") means the
result stands as-is, pass or fail or inconclusive -- the next step after
that is to actually write it up, not to re-run anything.

## Architecture quick-map

Don't restructure these boundaries (`CLAUDE.md`'s rule). Each arrow is
"output of left is the only input to right":

```
pose_estimation.py --> feature_extraction.py --> rules.py / classifier.py --> feedback.py --> tracking.py
                                                        |
                                                      app.py (Streamlit) --> ui_theme.py (presentation only)
```

- `src/pose_estimation.py` -- MediaPipe wrapper. Two models available
  (`lite` default, `heavy`), selected by `config/pose.json`'s
  `pose_model` key; `lite` tracks mocap far better and is ~3.5x faster
  (see `results/pose_model_comparison.md`, PC-local). `landmark_space`
  defaults to `"world"` (metric, not image-normalized) -- adopted after
  `results/diag_coords.md`/`diag_coords_C.md` found normalized
  image-space coordinates anisotropically distorted on this dataset's
  non-square frames.
- `src/feature_extraction.py` -- joint angles via one generic
  `compute_angle(a, b, c)` (3D, degrees, NaN on a degenerate/zero-length
  vector). `RIGHT`/`LEFT` landmark-index dicts; `select_side()` picks
  whichever side has higher average visibility. `MIN_VISIBILITY = 0.5`
  is the confidence-gate threshold used everywhere (live and offline).
- `src/rules.py` -- the live scorer. Two modes: `absolute` (fixed
  per-config thresholds) and `reference_relative` (tolerances derived
  from a patient's own recorded reference reps). `RULE_IDS = [peak_angle,
  elbow_straight, trunk_lean, tempo]`. Only `shoulder_abduction` has a
  rules config (`config/exercises/shoulder_abduction.json`) --
  `shoulder_rotation` has a trained classifier
  (`models/shoulder_rotation.pkl`) but **no rules config**, so
  `--scorer rules --exercise shoulder_rotation` always raises
  `FileNotFoundError`. This is documented (`docs/open_questions.md` #11)
  but not resolved -- unclear if it's intentional scoping or a gap.
- `src/pipeline.py` -- `Session`/`RepDetector` state machine
  (`IDLE -> CALIBRATING -> ACTIVE -> STOPPED`; `CALIBRATING -> RESTING <->
  MOVING` inside). New in this branch: `apply_advisory_rules()` -- an
  opt-in (`advisory_rules` config list, absent by default) mechanism so
  a rule can still report in feedback/chips/logs without affecting
  pass/fail. **Currently inert**: no shipped config sets the key, and it
  is deliberately decoupled from the pre-registered eval (which demotes
  `elbow_straight` in the *interpretation* layer,
  `compute_primary_outcomes.py`, not via this config option) -- see
  `docs/config_schema.md`'s `advisory_rules` row.
- `app.py` -- Streamlit UI + `--video` headless scorer (`run_headless()`,
  used for all offline/CI verification since it never touches a camera).
  Restyled on `ui-theme` (now merged in): tabs for Session/Reference/
  History/About, glass cards, a state-colored glow ring, native buttons
  re-skinned via CSS. `CVPHYSIO_PLAIN_UI=1` disables all of that and
  falls back to the pre-restyle plain Streamlit look -- the escape hatch
  for demo day.
- `src/ui_theme.py` -- presentation-only (CSS string + pure HTML-snippet
  functions), no Streamlit/cv2/mediapipe import, independently testable
  (`tests/test_ui_theme.py`, includes a from-scratch WCAG contrast check).
- `ui_preview.py` -- mock-data Streamlit page exercising every UI state
  (pass/fail/insufficient_data/idle/calibrating/error) with no camera,
  for visual QA.
- `src/tracking.py` -- `SessionTracker`, writes per-session CSV + a
  `.summary.json` sidecar (now includes actual camera resolution/fps and
  per-rep rule measurements, not just pass/fail).

Fuller references: `docs/module_interfaces.md`, `docs/data_flow.md`,
`docs/algorithms.md`, `docs/config_schema.md`, `docs/log_schema.md`,
`docs/module_table.md`.

## Datasets and evaluation discipline

Two mocap-backed sources, both used read-only with a strict
calibration/evaluation subject split that must never be crossed:

- **UI-PRMD** -- used for classifier training (`scripts/train_classifier.py`,
  `src/dataset_loader.py`). Cross-subject splits only.
- **REHAB24-6** -- used for scorer/pose-model diagnostics. Split lives in
  `data/rehab24_subject_split.json`: calibration `[1, 2, 6]`, evaluation
  `[3, 4, 5, 7, 8, 9]`. **Subject 6 has no cam17/front recording** in this
  dataset -- every "calibration subjects 1, 2" analysis (elbow-variant
  comparison, pose-model comparison) is really just subjects 1 and 2, and
  says so explicitly rather than silently treating 6 as present.
  Evaluation subjects are never opened by any of the diagnostic scripts
  -- several of them (`compare_elbow_variants.py`, `compare_pose_models.py`)
  hardcode the subject list independently of the split file and/or assert
  against it at runtime, specifically so a future split-file edit can't
  silently leak an evaluation subject into a "calibration-only" analysis.

Key measured facts already on record (don't re-derive, re-verify if
acting on them):
- World landmarks >> normalized image landmarks for angle fidelity on
  this (non-square-frame) dataset (`docs/diag_coords*.md`).
- Dropping MediaPipe's depth (z) from the elbow angle lowers MAE but
  also lowers correlation with mocap and roughly doubles frame-to-frame
  jitter -- not a clean win either way (`results/elbow_variants.md`,
  PC-local; cited in Amendment 1).
- `lite` pose model beats `heavy` on this data for shoulder angle fidelity
  (r=0.991 vs 0.814) and is ~3.5x faster; `heavy`'s elbow-angle
  correlation with mocap is actually *negative* in this sample
  (`results/pose_model_comparison.md`, PC-local).
- A held-straight-arm reading was unstable across two laptop sessions
  (~143-145 deg vs ~96 deg) and bent-vs-straight didn't separate under
  any of 3 elbow statistics tried (AUC 0.37-0.64) -- the empirical basis
  for demoting `elbow_straight` to advisory in Amendment 1.

## Known gaps / open questions

`docs/open_questions.md` has 19 items found by reading the code early on
(dated, written before this branch's work). Spot-checked against current
state while writing this handover:

- **#11 (no `shoulder_rotation.json`)**: still true, still unresolved.
- **#13 ("test coverage gaps... no tests for most modules")**: **stale/
  outdated** -- 482 tests now exist, covering `ui_theme.py`, `rules.py`,
  advisory rules, `qc_all`/`qc_clip`, `record_clip`, report figures,
  `reset_tracking`, and more. Don't trust this item's specific claim; the
  general spirit (always check test coverage before relying on untested
  code) still applies.
- The other 17 items (smoothing-before-gating order, `compute_angle`'s
  unguarded zero-length vector producing a silent NaN+RuntimeWarning,
  `RepDetector`'s undebounced RESTING->MOVING transition, etc.) were not
  individually re-verified for this handover -- treat them as still open
  unless you check otherwise.

## How to resume

1. `git fetch origin && git log --oneline -5 origin/elbow-accuracy` --
   confirm you're reading this from the actual current tip; the laptop
   may have pushed more since this was written.
2. `git switch elbow-accuracy && git pull --ff-only` (or clone fresh --
   see `docs/laptop_setup.md` for the exact fresh-clone steps, including
   the pose-model download and the Windows `MAX_PATH` gotcha for deep
   clone paths).
3. `python -m pytest -q` -- confirm the test count before changing
   anything; if it's not what this doc states, something changed.
4. For the own-clip evaluation specifically: that's laptop-only (see
   above) -- on the laptop, re-read all 4 amendments in
   `docs/preregistered_own_clip_eval.md`, run `scripts/qc_all.py`, and
   only on a clean OK run `scripts/evaluate_own_clips.py` then
   `scripts/compute_primary_outcomes.py`.
5. For anything else: check `config/pose.json`/`config/exercises/
   shoulder_abduction.json`'s blob ids against the pins table in
   `docs/preregistered_own_clip_eval.md` before touching either --
   changing them after the real evaluation runs invalidates it per the
   pre-registration's own "Changes" rule (needs newly recorded clips).

## Hard rules this project runs under (from `CLAUDE.md`, condensed)

- Exactly 2 exercises (shoulder abduction, shoulder internal/external
  rotation). No other exercises, no lower-body movements -- don't
  suggest expanding scope.
- Never report an accuracy/precision/recall number that wasn't just
  produced by actually running the code.
- Confidence gating: MediaPipe visibility < 0.5 on a required landmark ->
  don't classify that rep, say so instead.
- Keep the 6 modules independent (`feature_extraction.py` must not import
  `classifier.py`, etc.) -- each module's output is the next module's
  only input.
- Classical ML only (scikit-learn Random Forest) -- no PyTorch/TensorFlow,
  deliberately, not a gap to fill.
