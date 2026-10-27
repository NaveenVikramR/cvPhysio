# Figures

One line per figure: what it shows, and which code it was derived
from (file and function/class). Rendered from the `.mmd` Mermaid
sources in this directory; see each file's own header comment for the
full list of sources. SVG/PNG render status is reported in the commit
message and chat response for this task, not duplicated here.

- **`data_flow_live.mmd`** — the live per-frame data flow (frame →
  pose → landmark space → confidence gate → features → rep detector →
  per-rule scoring → feedback → session log), with data types on each
  arrow. Derived from `src/pose_estimation.py` (`get_pose_landmarks`),
  `src/feature_extraction.py` (`check_confidence`, `extract_features`,
  `compute_elbow_angle`, `compute_trunk_lean_deg`), `src/pipeline.py`
  (`Session.process_frame`, `process_frame`, `RepDetector.update`),
  `src/rules.py` (`score_rep`), `src/feedback.py`
  (`message_for_rules_result`, `draw_overlay`,
  `SpeechFeedback.speak`), `src/tracking.py`
  (`SessionTracker.add_rep`/`add_discard`).
- **`sequence_one_rep.mmd`** — sequence diagram of one repetition,
  including the `insufficient_data` path. Derived from the same
  functions as above, specifically `src/rules.py`'s `score_rep()`
  valid_fraction gate and `src/pipeline.py`'s `Session`/`RepDetector`.
- **`session_state_machine.mmd`** — the Session Start/Stop state
  machine (`IDLE → CALIBRATING → ACTIVE → STOPPED`), with
  `RepDetector`'s own nested `RESTING ⇄ MOVING` states shown inside
  `ACTIVE`. Derived from `src/pipeline.py`'s `SessionState`,
  `Session.start`/`stop`/`session_state`, and `RepDetector.update`
  (and its `CALIBRATION_FRAMES`/`REST_BAND_DEG`/`MIN_REP_FRAMES`/
  `MIN_REP_RANGE_DEG`/`MAX_REP_FRAMES` constants).
- **`offline_evaluation_protocol.mmd`** — the REHAB24-6 offline
  evaluation protocol: the calibration/evaluation subject split, what
  each of runs 1, 2, and the oracle run derives and evaluates, and the
  separate own-clip evaluation path. Derived from
  `data/rehab24_subject_split.json`, `scripts/build_benchmark_config.py`,
  `scripts/derive_reference_tolerances.py`, `scripts/benchmark_rehab24.py`
  (run 1), `scripts/evaluate_reference_scoring.py` (run 2),
  `scripts/evaluate_oracle_ceiling.py` (oracle run),
  `scripts/build_review3_summary.py`, `scripts/evaluate_own_clips.py`.
- **`reference_relative_scoring.mmd`** — how a Reference is recorded
  (K reps → per-rep features → median) and how a new rep is compared
  to it (lower/upper-bound-only or fractional tolerances, with the
  absolute-mode fallback when no reference is supplied). Derived from
  `src/rules.py`'s `rep_summary_features`, `build_reference`,
  `save_reference`, `load_reference`, `score_rep`, and
  `_reference_relative_bounds`.
