# Development plan: both exercises working, on a measured foundation

Copied into the repo on 2026-11-17 so every team member and machine works from the same plan.
Update the status table when a step finishes. History is in `git log`.

## Status (2026-11-17)

| Step | Status | Where |
|---|---|---|
| Phase 0: abduction pre-registered evaluation | **waiting**, laptop only | `docs/HANDOVER.md` "How to resume", step 4 |
| Phase 1: rotation diagnosis | done | `docs/rotation_diagnosis.md` |
| F1: landmark-fidelity baseline | done | `docs/landmark_fidelity.md` |
| Team recording kit | done | `docs/RECORDING_GUIDE.md`, `scripts/record_clip.py`, `scripts/package_results.py` |
| Stage 1 camera-angle test clips | **next**, recorder (friend) | `docs/RECORDING_GUIDE.md` section 5 |
| F2 improvements, Phase 2 rotation fixes | blocked on Phase 0 (and Stage 1 for rotation) | below |
| Stage 3 team dataset, Phase 3 evaluation | after Stage 1 decides the camera angle | below |

**Change to Phase 3 (2026-11-17):** team recordings now happen *before* the rotation
pre-registration is written. To keep the evaluation honest, subject `s02` (role `eval`) is
recorded but nobody looks at or scores those clips until `docs/preregistered_rotation_eval.md`
is committed with the code frozen. Subject `s01` (role `calibration`) may be used for all
tuning. The write-up must say that there is one evaluation subject.


## Context

The abduction pipeline is complete, and its pre-registered own-clip evaluation is ready to run on the laptop. Shoulder rotation, the second of the two in-scope exercises, is only nominally supported:

- The UI-PRMD Random Forest is close to chance on held-out subjects (LOSO accuracy 0.635 ± 0.187, run today).
- The new `config/exercises/shoulder_rotation.json` accepts correct mocap reps but catches only 6/50 incorrect ones. This is because `peak_angle` checks only the outward end of the movement.
- On the three webcam demo clips, most reps peak at 118–170°, below the 172.9° floor, and many movements are thrown away as `too_short`.
- Nothing in the recording, QC or evaluation tooling handles rotation (`evaluate_own_clips.py` hardcodes `EXERCISE = "shoulder_abduction"`).

The user's priority is **both exercises working** by a milestone 2–4 weeks away. So this phase is about rotation: find out why it fails on the webcam, fix it, and evaluate it as rigorously as abduction, without disturbing the frozen abduction evaluation.

**Standing constraints:** CLAUDE.md scope, the module boundaries and no fabricated metrics. The pinned blobs stay untouched until the abduction evaluation has run: `rules.py`, `config/pose.json`, `shoulder_abduction.json` and `evaluate_own_clips.py`. After that run, any change must keep abduction scoring bit-identical and must be logged in the Changes table of `docs/preregistered_own_clip_eval.md`.

## Update 2026-11-16: landmark-fidelity foundation (review feedback)
The project review asked for better accuracy in locating joints. Phase 1 is done: in a front view the rotation angle fails because of MediaPipe's depth estimate. So the work below now comes **before** the Phase 2 rotation fixes and feeds into them. The pose model can't be retrained (classical ML only), so the improvement comes from what happens around MediaPipe. Accuracy is measured on **joint angles** against motion capture, not as raw point error, because the two skeletons and coordinate systems differ.
- **F1, measure (PC, now; no change to src/ or config):** `scripts/landmark_fidelity_report.py`.
  - Data: REHAB24-6 Ex1 against motion capture, per angle (shoulder abduction, elbow, trunk lean), per view (front, half-profile, profile) and per subject group (calibration 1, 2, 6 kept separate from evaluation 3, 4, 5, 7, 8, 9).
  - Metrics: bias, MAE, MAE after removing bias, r, slope, jitter, and the share of frames that pass the confidence gate.
  - Also checks whether part of the shoulder offset is a definitional mismatch: the motion-capture "Hips" root joint versus the anatomical right hip.
  - Output: `results/landmark_fidelity.md`, plus a committed summary in `docs/landmark_fidelity.md`.
- **F2, improve (after Phase 0, because each step changes abduction results).** Fit on calibration subjects only, report on evaluation subjects:
  - an offset/slope correction per angle
  - One Euro smoothing tuned against motion capture
  - confidence-gate checks
  - the camera angle for rotation, from the laptop test clips
  - a depth-free rotation measure
- **F3, re-measure** with the same report and show before and after. That before/after comparison is the deliverable for the review.
- **Optional:** download the REHAB24-6 Ex2 ("arm VW", which includes external rotation) videos, if the user approves, to get motion-capture ground truth for the rotation angle.

## Phase 0: in parallel, laptop, this week
- Run the abduction evaluation (`qc_all.py`, then `evaluate_own_clips.py`, then `compute_primary_outcomes.py`) and commit the result as it stands.
- This unblocks the changes in Phase 2 that touch `rules.py` and `pipeline.py`.

## Phase 1: diagnose rotation on the webcam (PC, days 1–3)
New script `scripts/diagnose_rotation.py`, measurement only, with output in `results/diag_rotation.md` plus plots. For each demo rotation clip in `data/recordings/` it will show:
- The per-frame rotation angle (`compute_shoulder_rotation_angle`, world landmarks), alongside the depth-free proxy `proxy_from_points` from `scripts/rotation_proxy_experiment.py`.
- The per-frame visibility of the elbow and wrist landmarks.
- `RepDetector` states, baseline, and every discard with its reason.

Reuse:
- the `process_clip` frame loop and plotting from `scripts/diagnose_misclassification.py`
- `scripts/landmark_cache.py` for caching landmarks
- `scripts/plot_rotation_reps.py` for plotting traces

It will answer three questions, each with an explicit yes/no in the report:
1. **Depth:** do the low peaks come from MediaPipe's depth (z) noise? Compare the angle and the proxy on the same reps.
2. **Rep segmentation:** does `RepDetector` split one out-and-in rotation into several short segments? In-and-out around a neutral baseline crosses the rest band twice per rep.
3. **Camera angle:** with the forearm pointing at a front camera, the neutral position is almost entirely depth, so would a 45° oblique view help? Propose two short laptop test clips, one front and one 45°, to settle this.

## Phase 2: rotation fixes (PC, days 3–8; the `src/` changes come after Phase 0)
Which items get built depends on the Phase 1 answers:
- **a. Per-exercise detector settings.** An optional `"detector"` block in `shoulder_rotation.json` (`rest_band_deg`, `min_rep_range_deg`, plus, if Phase 1 shows it is needed, a mode where a rep is out-and-back through neutral). `src/pipeline.py`'s `RepDetector` reads it, with the defaults equal to today's constants (`REST_BAND_DEG = 10`, `MIN_REP_RANGE_DEG = 25`). A test checks that abduction rep events don't change.
- **b. A rule for the inward end.** A new opt-in rule in `src/rules.py` (e.g. `min_angle`, which checks the rep's smallest rotation angle), active only when its config key is present. Abduction results must be bit-identical, checked by a test over the existing `test_rules*.py` fixtures. The change is logged in the pre-registration's Changes table.
- **c. If depth is the cause:** a `rotation_angle_method` option in `feature_extraction.py`, either the current 3D angle or a depth-free version, used only for rotation. Retrain the rotation classifier with `scripts/train_classifier.py` and report the new LOSO output from that run.
- **d.** Re-run `scripts/build_rotation_config.py` (extended for a, b and c) to regenerate `shoulder_rotation.json`, using the same s01–s05 / s06–s10 split.
- Update `docs/config_schema.md`, `docs/algorithms.md` and the tests.

## Phase 3: pre-register and record a rotation evaluation (PC writes it, laptop records it, week 2)
- `docs/preregistered_rotation_eval.md`, committed **before** any rotation clip is recorded. It mirrors the abduction pre-registration: pinned blob ids, criteria, Wilson intervals, an inconclusive rule, and no re-recording after a score has been seen.
- Fault clips:
  - range too short, outward end
  - range too short, inward end
  - elbow drifting away from the side
  - trunk twist
  - rushed tempo
- Tooling. `evaluate_own_clips.py` is pinned, so it isn't edited:
  - `scripts/evaluate_rotation_clips.py` imports `rate_with_ci` and the other helpers from `evaluate_own_clips.py`.
  - `record_clip.py` and `qc_clip.py` get an `--exercise` flag, with abduction as the default.
  - `docs/recording_protocol.md` gets a rotation section covering framing (the camera angle chosen in Phase 1) and the arms-down start rule.
- The laptop records the clips and runs the QC gate.

## Phase 4: evaluate, then write up and harden the demo (week 3)
- Run the rotation evaluation once and record the verdict as it comes out.
- Write up both exercises with the actual outputs: the abduction pre-registered result, the rotation result, classifier LOSO numbers, and limitations. The 5- and 10-subject groups are small, and the write-up says so.
- Demo:
  - In Streamlit, check rotation end to end: the exercise selector already exists, so cover reference recording, cues, voice and history.
  - Check rotation in `ui_preview.py`.
  - Check that the `CVPHYSIO_PLAIN_UI=1` fallback works.

## Stretch (week 4, only if time is left)
Scoring-path items still open in `docs/open_questions.md`:
- #1: smoothing runs before the confidence gate
- #14: the start of a rep isn't debounced
- #15: calibration recovery after a noisy window

Each one changes abduction behaviour, so it is a post-evaluation change and is disclosed as such.

## Critical files
- new: `scripts/diagnose_rotation.py`, `scripts/evaluate_rotation_clips.py`, `docs/preregistered_rotation_eval.md`
- modified (after Phase 0): `src/pipeline.py` (`RepDetector` settings), `src/rules.py` (opt-in rule), possibly `src/feature_extraction.py` (rotation angle option)
- modified: `scripts/build_rotation_config.py`, `config/exercises/shoulder_rotation.json`, `scripts/record_clip.py`, `scripts/qc_clip.py`, `docs/recording_protocol.md`, `docs/config_schema.md`
- never modified: `config/exercises/shoulder_abduction.json`, `config/pose.json`, `scripts/evaluate_own_clips.py`

## Verification
- `python -m pytest -q` passes after every step. New tests:
  - the abduction `RepDetector` events and `score_rep` outputs are unchanged
  - the opt-in rule does nothing when its key is absent
  - rotation config properties
- `test_rotation_config.py`'s blob check plus `git hash-object` on every pinned file, before each commit.
- `python app.py --video data/recordings/<rotation clip> --exercise shoulder_rotation --scorer rules`, compared before and after on the demo clips. Fewer `too_short` discards and plausible peaks count as exploratory evidence only, not accuracy.
- `python app.py --video data/recordings/shoulderAbduction1.mp4 --exercise shoulder_abduction` gives identical output before and after.
- The final rotation numbers come only from the single pre-registered run in Phase 4.
