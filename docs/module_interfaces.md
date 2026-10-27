# Module interfaces

Generated from the code as it exists on disk (not from design intent).
Every row below is traceable to a specific function/class in the listed
file. Open questions raised while writing this are collected in
`docs/open_questions.md`, not inline here.

## src/pose_estimation.py

Purpose (module docstring): "Video capture & pose estimation. Output: a
33-tuple landmark list per frame (MediaPipe's PoseLandmark order),
nothing else." Uses MediaPipe's Tasks API (`PoseLandmarker`), `RunningMode.VIDEO`,
`min_pose_detection_confidence=0.5`, `min_tracking_confidence=0.5`. Model
file: `models/pose_landmarker_lite.task` (hardcoded path, not a config key).

MediaPipe's `PoseLandmarkerResult` carries TWO landmark sets per frame:
`pose_landmarks` (normalized image-space, x/y independently divided by
frame width/height) and `pose_world_landmarks` (metric meters,
hip-centered, independent of frame width/height). `get_pose_landmarks()`
returns whichever one `config/pose.json`'s `landmark_space` key selects
(see `docs/config_schema.md`) — this is the single place that selection
happens; every caller downstream is unaware of which space is active.

| Function | Signature | Input | Output | Units | Failure behaviour |
|---|---|---|---|---|---|
| `load_config` | `() -> dict` | — | parsed JSON from `config/pose.json` | — | Raises if the file is missing (plain `Path.read_text()`, no existence check). |
| `get_pose_landmarks_pair` | `(frame) -> (list[tuple]\|None, list[tuple]\|None)` | one BGR frame | `(normalized, world)` from ONE `detect_for_video()` call | normalized: x/y `[0,1]`, z roughly on x's scale (MediaPipe convention); world: meters, hip-centered; visibility `[0,1]`, identical between the two sets for the same joint (confirmed empirically, not just assumed) | Either or both sides `None` if MediaPipe detects no pose. Never raises for a normal frame. |
| `get_pose_landmarks` | `(frame) -> list[tuple[float,float,float,float]] \| None` | one BGR frame | 33 `(x, y, z, visibility)` tuples, MediaPipe `PoseLandmark` index order, in whichever space `load_config()["landmark_space"]` selects (`"world"` default, `"normalized"` to reproduce the pre-landmark_space behavior) | see `get_pose_landmarks_pair` above, depending on the selected space | Raises `ValueError` if `landmark_space` is neither `"world"` nor `"normalized"`. Returns `None` if MediaPipe detects no pose, or if the selected space's set wasn't available for this frame (never silently falls back to the other space). |
| `open_webcam` | `(camera_index: int = 0) -> cv2.VideoCapture` | camera index | an opened `cv2.VideoCapture` | n/a | Raises `RuntimeError` if the camera can't be opened. |

Module-level mutable state: `_next_timestamp_ms`, incremented by 33 on
every `get_pose_landmarks_pair()` call (a monotonic tick MediaPipe's
VIDEO mode requires — not real elapsed time; see `docs/open_questions.md`).

## src/feature_extraction.py

Purpose (module docstring): "Biomechanical feature extraction. Input:
pose_estimation's 33-tuple landmark list ... Output: feature dict, or
None if confidence gating fails. Must not import pose_estimation or
classifier." No config keys are read directly by this module — callers
pass in `exercise` strings and (optionally) a `required_roles` set.

| Function/constant | Signature | Input | Output | Units | Failure behaviour |
|---|---|---|---|---|---|
| `MIN_VISIBILITY` | constant `= 0.5` | — | — | visibility units `[0,1]` | — |
| `EXERCISES` | constant dict | — | `{"shoulder_abduction": "Standing Shoulder Abduction", "shoulder_rotation": "Standing Shoulder Internal-External Rotation"}` | — | — |
| `LEFT`, `RIGHT` | constant dicts | — | `{"shoulder": 11/12, "elbow": 13/14, "wrist": 15/16, "hip": 23/24}` (MediaPipe landmark indices) | — | — |
| `BASE_REQUIRED_ROLES` | constant dict | — | `{"shoulder_abduction": {"shoulder","elbow","hip"}, "shoulder_rotation": {"shoulder","elbow","wrist","hip"}}` — default joint roles for each exercise's own primary angle only | — | — |
| `compute_angle` | `(a, b, c) -> float` | 3 points, each `(x,y,z,...)`; only `[:3]` used | angle at `b` between `b->a` and `b->c` | degrees | None (pure math); `arccos` input is clipped to `[-1,1]` so it never raises on floating-point overshoot. |
| `select_side` | `(landmarks) -> dict[str,int]` | 33-tuple landmark list | `LEFT` or `RIGHT` role-dict, whichever side has higher mean visibility across its 4 roles | — | None; always returns one of the two dicts. |
| `compute_shoulder_abduction_angle` | `(landmarks) -> float` | 33-tuple landmark list | `compute_angle(hip, shoulder, elbow)` on the selected side | degrees (~0 = arm at side, ~90 = raised to horizontal) | Assumes landmarks is non-`None`; no internal confidence check. |
| `shoulder_rotation_angle_from_points` | `(shoulder, elbow, wrist, hip) -> float` | 4 raw `(x,y,z,...)` points (not a full landmark list — reused by `dataset_loader.py` on FK positions) | angle between forearm and torso-reference, both projected into the plane perpendicular to the humeral axis | degrees | None. |
| `compute_shoulder_rotation_angle` | `(landmarks) -> float` | 33-tuple landmark list | calls `shoulder_rotation_angle_from_points` on the selected side's 4 landmarks | degrees | Same as above. |
| `compute_elbow_angle` | `(landmarks) -> float` | 33-tuple landmark list | `compute_angle(shoulder, elbow, wrist)` on the selected side | degrees (~180 = straight arm) | None. Depends on the **wrist** landmark. |
| `compute_trunk_lean_deg` | `(landmarks) -> float` | 33-tuple landmark list | angle between the trunk axis (hip-midpoint→shoulder-midpoint, using **both** sides, not `select_side`) and straight up (`-Y`, since MediaPipe's Y increases downward) | degrees (0 = upright) | None. Docstring states it "must only be called on live landmarks, never on UI-PRMD's Y-up FK positions" — **not enforced in code**, see open questions. |
| `extract_rep_features` | `(sequence_df: pd.DataFrame, angle_col: str = "angle_deg") -> dict` | one rep's frame sequence with a numeric `angle_col` | `{"angle_max_deg", "angle_min_deg", "angle_mean_deg", "angle_range_deg", "angle_midpoint_deg", "angle_recovery_deg"}` | degrees | Will raise (pandas `IndexError`/`KeyError`) on an empty `sequence_df` — not guarded. `angle_recovery_deg = abs(angles.iloc[-1] - angles.iloc[0])`. |
| `check_confidence` | `(landmarks, side, exercise, required_roles=None) -> (bool, list[int])` | landmarks, a side role-dict, exercise name, optional override role set | `(ok, failed_landmark_indices)` | — | `roles = required_roles if given else BASE_REQUIRED_ROLES[exercise]`; a role fails if `landmarks[i][3] < MIN_VISIBILITY`. Raises `KeyError` if `exercise` isn't in `BASE_REQUIRED_ROLES` and no `required_roles` given. |
| `extract_features` | `(landmarks, exercise) -> dict \| None` | landmarks (or `None`), exercise name | `{"exercise", "side", "angle_deg"}` or `None` | degrees | Raises `ValueError` for an unknown exercise. Returns `None` if `landmarks is None` or `check_confidence` (using the **default**, not rules-derived, required roles) fails. Internally calls `check_confidence` a second time with no `required_roles` override — redundant with any outer gate, see open questions. |

## src/pipeline.py

The shared live-pipeline orchestration module: `RepDetector` (the
segmenter), `process_frame()`, `build_smoother()`,
`load_classifier_for_live_use()`, and `Session`/`SessionState` (the
Start/Stop lifecycle). `app.py` and every `scripts/*.py` diagnostic that
needs rep detection import these directly from here — previously each
independently `exec()`-sliced them out of `app.py`'s own source text
(dropped: every new name that sliced code referenced had to be
re-added to every loader's namespace by hand, and had already silently
broken one script before this module existed). No Streamlit, no cv2
window/webcam code, no work at import time.

### RepDetector (the "segmenter")

Purpose: turn a continuous per-frame angle signal into discrete
repetitions via a 3-state machine (`CALIBRATING → RESTING ⇄ MOVING`).
**Unmodified** by the Session lifecycle below — Session only decides
whether a frame reaches this state machine at all.

| Constant | Value | Meaning |
|---|---|---|
| `CALIBRATION_FRAMES` | 15 | visible frames needed to compute the initial rest baseline |
| `CALIBRATION_MAX_STD_DEG` | 6.0 | max allowed stdev (degrees) of those 15 frames before calibration restarts |
| `REST_BAND_DEG` | 10.0 | degrees from baseline still considered "at rest" |
| `BASELINE_WINDOW` | 10 | rolling window size for the live-updated rest baseline |
| `MIN_REP_FRAMES` | 15 | minimum frames for a buffered movement to count as a real rep |
| `MIN_REP_RANGE_DEG` | 25.0 | minimum angle range (max−min) for a real rep |
| `MAX_REP_FRAMES` | 450 | a rep buffered longer than this is discarded (`too_long`) |
| `CONFIDENCE_TOLERANCE_FRAMES` | 10 | consecutive invisible frames tolerated mid-rep before giving up |
| `CALIBRATION_STALL_FRAMES` | 150 | frames stuck in `CALIBRATING` before `calibration_hint()` returns non-`None` |
| `RESCUE_REST_BAND_DEG` | `2 * REST_BAND_DEG` = 20.0 | wider tolerance used only by the tracking-lost rescue (see `docs/algorithms.md`) |

| Method | Signature | Input | Output | Failure behaviour |
|---|---|---|---|---|
| `__init__` | `()` | — | fresh instance, `state="CALIBRATING"` | — |
| `update` | `(visible: bool, angle_deg: float\|None, side: str\|None, failed: list[int]\|None=None) -> (event, payload)` | one frame's confidence-gated angle/side, plus (while calibrating) the list of failed landmark indices | `("none", None)` / `("rep_complete", {"angle_deg": [...], "side": [...]})` / `("rep_discarded", {"reason_code": "too_short"\|"too_long"\|"lost_tracking", "message": str})` | No exceptions raised for any input combination observed in the code; `angle_deg`/`side` are only read when `visible` is `True`. |
| `_reset_to_resting` | `() -> None` | — | clears `rep_buffer`/`side_votes`/`bad_confidence_streak`, sets `state="RESTING"` | — |
| `calibration_hint` | `() -> str \| None` | — | a hint string once stalled `CALIBRATING` past `CALIBRATION_STALL_FRAMES`, else `None` | Called from both `app.py`'s Streamlit path and `run_headless()`. |

### process_frame, build_smoother, load_classifier_for_live_use

| Function | Signature | Input | Output | Failure behaviour |
|---|---|---|---|---|
| `process_frame` | `(detector: RepDetector, parallel: dict, landmarks, exercise: str, smoother=None, t: float\|None=None, required_roles: set[str]\|None=None) -> (event, payload, visible, failed)` | one frame's raw landmarks + shared `RepDetector`/accumulator state | event tuple from `RepDetector.update`, with `elbow_angle_deg`/`trunk_lean_deg`/`valid_fraction` added to the payload on `rep_complete` | `parallel` must be pre-seeded as `{"elbow": [], "trunk": [], "total_frames": 0}`; no internal guard if it isn't. |
| `build_smoother` | `() -> LandmarkSmoother \| None` | — | `None` if `config/smoothing.json`'s `"enabled"` is false/missing, else a configured `LandmarkSmoother` | Raises whatever `smoothing_module.load_config()` raises if the file is missing. |
| `load_classifier_for_live_use` | `(exercise: str, models_dir: Path = MODELS_DIR) -> (CorrectnessClassifier\|None, dict\|None)` | exercise name, optional override dir (for tests) | `(None, None)` if no `{exercise}.pkl` exists yet; else `(classifier, correct_feature_ranges)` | Raises `RuntimeError` if the sidecar `{exercise}.metadata.json` is missing, or its `feature_names`/`feature_source` don't match the live path's `FEATURE_COLUMNS`/`LIVE_FEATURE_SOURCE` exactly. |

### SessionState / Session (Start/Stop lifecycle)

Purpose: gate whether a frame reaches `RepDetector` at all, via
`IDLE → CALIBRATING → ACTIVE → STOPPED`. `CALIBRATING`/`ACTIVE` are
*derived* from `RepDetector.state` (`CALIBRATING` mirrors
`detector.state == "CALIBRATING"`; `ACTIVE` covers both `RESTING` and
`MOVING`) — `Session` adds no segmentation logic of its own. Drives
both `app.py`'s Streamlit Start/Stop buttons and headless mode's
`--span START END`.

| Member | Signature | Behaviour |
|---|---|---|
| `SessionState.IDLE/CALIBRATING/ACTIVE/STOPPED` | string constants | the 4 state names, as returned by `Session.session_state` |
| `Session.__init__` | `(exercise: str)` | `session_state == IDLE`; `detector`/`parallel` are `None` until `start()` |
| `Session.session_state` | property `-> str` | `STOPPED` if `stop()` was ever called; else `IDLE` if `detector is None`; else `CALIBRATING`/`ACTIVE` per `detector.state` |
| `Session.start` | `() -> None` | `IDLE → CALIBRATING`: constructs a fresh `RepDetector` + `parallel` dict. No-op if already started or stopped (no restart after Stop). |
| `Session.stop` | `() -> None` | One-way, idempotent transition to `STOPPED`. |
| `Session.process_frame` | `(landmarks, smoother=None, t: float\|None=None, required_roles: set[str]\|None=None) -> (event, payload, visible, failed)` | While `IDLE`/`STOPPED`: returns `("idle", None, False, [])` without calling `RepDetector`/`process_frame()` at all. Otherwise delegates straight to the module-level `process_frame()`. |

## src/rules.py

Purpose (module docstring): "Rule-based live scorer. Checks measured
per-rep features against physiotherapist-tunable thresholds in
`config/exercises/{exercise}.json` ... Must not import classifier.py or
pose_estimation.py." Config keys read: see `docs/config_schema.md`.

| Function/class | Signature | Input | Output | Units | Failure behaviour |
|---|---|---|---|---|---|
| `RULE_IDS` | constant list | — | `["peak_angle", "elbow_straight", "trunk_lean", "tempo"]` | — | — |
| `REQUIRED_ROLES_BY_RULE` | constant dict | — | role sets per rule (see `docs/algorithms.md`) | — | — |
| `required_joint_roles` | `() -> set[str]` | — | union of all 4 rules' role sets = `{"shoulder","elbow","wrist","hip"}` | — | — |
| `load_config` | `(exercise: str) -> dict` | exercise name | parsed JSON from `config/exercises/{exercise}.json` | — | Raises `FileNotFoundError` with an explanatory message if that file doesn't exist (true today for every exercise except `shoulder_abduction`). |
| `score_rep` | `(rep: dict, config: dict, fps: float) -> dict` | `rep = {"angle_deg": [...], "elbow_angle_deg": [...], "trunk_lean_deg": [...], "valid_fraction": float}`; `config` from `load_config`; `fps` of the source | `{"status": "pass"\|"fail"\|"insufficient_data", "failed_rules": [...], "rules": {...}, "valid_fraction": float, "cue": str\|None}` | degrees / seconds | Raises `ValueError` if the three feature lists aren't equal-length and non-empty. See `docs/algorithms.md` for the exact gating order. |
| `CueCooldown.__init__` | `(cooldown_s: float)` | cooldown window in seconds | instance | seconds | — |
| `CueCooldown.should_fire` | `(cue_id: str, now: float\|None=None) -> bool` | cue id, optional injected clock | whether this cue may fire now | — | `now` defaults to `time.monotonic()` |
| `CueCooldown.record` | `(cue_id: str, now: float\|None=None) -> None` | — | — | — | — |
| `CueCooldown.cues_to_speak` | `(failed_rules: list[str], cue_text: dict, now=None) -> list[str]` | failed rule ids this rep, the config's `cues` dict | cue strings not currently in cooldown (and marks them fired) | — | Skips any `rule_id` not present in `cue_text`. |

## src/smoothing.py

Purpose (module docstring): "Landmark smoothing for the live pipeline.
Sits between pose_estimation ... and feature_extraction ... never
imported by either." Config keys read: see `docs/config_schema.md`
(only by `app.py`'s `build_smoother()`, not by this module itself,
except `load_config()`'s file path).

| Function/class | Signature | Input | Output | Units | Failure behaviour |
|---|---|---|---|---|---|
| `load_config` | `() -> dict` | — | parsed JSON from `config/smoothing.json` | — | Raises if the file is missing (no explicit check — plain `Path.read_text()`). |
| `OneEuroFilter.__init__` | `(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)` | filter params (Hz / unitless gain / Hz) | instance | — | — |
| `OneEuroFilter.__call__` | `(x: float, t: float) -> float` | raw value, timestamp (seconds) | filtered value | same units as `x`; `t` in seconds | First call returns `x` unchanged (initializes state). If `t - t_prev <= 0` (duplicate/out-of-order timestamp), returns the previous filtered value unchanged rather than dividing by zero. |
| `LandmarkSmoother.__init__` | `(min_cutoff, beta, d_cutoff, max_hold_frames)` | filter params + hold budget (frames) | instance holding 33×3 = 99 independent `OneEuroFilter`s | — | — |
| `LandmarkSmoother.update` | `(landmarks: list[tuple]\|None, t: float) -> list[tuple]\|None` | one frame's raw landmarks (or `None`) + timestamp | smoothed landmarks, the held-over last-good list (visibility passed through unchanged, only x/y/z filtered), or `None` once `max_hold_frames` is exceeded | — | None raised; `None` is a valid, documented return. |

## src/feedback.py

Purpose (module docstring): "Real-time feedback engine. Input:
classifier's result dict (or a confidence-gating failure). Output:
annotated frame + spoken feedback." No config keys read.

| Function/class | Signature | Input | Output | Units | Failure behaviour |
|---|---|---|---|---|---|
| `CORRECT_MESSAGE`, `INCORRECT_MESSAGE` | constants | — | fixed strings | — | — |
| `LANDMARK_LABELS` | constant dict | — | 8 MediaPipe indices (11-16, 23-24) → human-readable names | — | — |
| `low_visibility_message` | `(failed_landmarks: list[int]) -> str` | indices | a sentence naming them (via `LANDMARK_LABELS`, or the raw index if unknown) | — | — |
| `draw_overlay` | `(frame, text: str, is_correct: bool\|None)` | a frame, a message, a 3-way correctness flag | the same frame with `cv2.putText` burned in | — | Color: green if truthy, red if `False`, orange if `None` (also used for `insufficient_data`, since `app.py` maps that status to `None`). |
| `SpeechFeedback.__init__` | `()` | — | wraps a `pyttsx3` engine + a `threading.Lock` | — | — |
| `SpeechFeedback.speak` | `(text: str) -> None` | message | spawns a daemon thread that speaks it | — | Non-blocking; no return value, no explicit error handling if TTS fails. |
| `message_for_result` | `(result: dict\|None, failed_landmarks=None) -> str` | a **classifier**-style result (`{"is_correct": bool}`) or `None` | the low-visibility message, or `CORRECT_MESSAGE`/`INCORRECT_MESSAGE` | — | This is the **classifier** path's message builder only — the rules path builds its own strings inline in `app.py` (see `docs/open_questions.md`). |

## src/tracking.py

Purpose (module docstring): "Progress tracking. Input: one rep record
per completed rep ... Output: session log files + summaries." No config
keys read.

| Function/class | Signature | Input | Output | Units | Failure behaviour |
|---|---|---|---|---|---|
| `RepRecord` (dataclass) | fields: `exercise: str, side: str, angle_deg: float, status: str, confidence: float\|None, timestamp: str` (auto UTC ISO) | — | one CSV row per completed rep | degrees | `status` is a free string, not validated against `{"pass","fail","insufficient_data"}` at the dataclass level. |
| `DiscardRecord` (dataclass) | fields: `exercise: str, reason_code: str, message: str, timestamp: str` | — | one CSV row per discarded segment | — | Same lack of enum validation on `reason_code`. |
| `SessionTracker.__init__` | `(user_id: str, exercise: str)` | — | instance with empty `reps`/`discards` lists | — | — |
| `SessionTracker.add_rep` / `add_discard` | `(rep: RepRecord)` / `(discard: DiscardRecord)` | — | appends | — | — |
| `SessionTracker.summary` | `() -> dict` | — | `{"user_id","exercise","total_reps","passed_reps","failed_reps","insufficient_data_reps","accuracy_pct","insufficient_data_rate_pct","discarded_segments"}` | — | `accuracy_pct`/`insufficient_data_rate_pct` are `None` when their denominator (`scored`/`total`) is 0. `accuracy_pct = passed / (passed+failed)` — **excludes `insufficient_data` from the denominator**. |
| `SessionTracker.save` | `(logs_dir=LOGS_DIR) -> Path` | — | writes `{user}_{exercise}_{stamp}.csv`, a sibling `.discards.csv` (only if any discards), and `.summary.json` | — | Writes an empty (header-less) reps CSV if `self.reps` is empty but `self.discards` is not — see `docs/open_questions.md`. |
| `load_history` | `(user_id, logs_dir=LOGS_DIR) -> list[dict]` | — | every `*.summary.json` for that user, sorted by filename | — | Returns `[]` if none exist. |

## src/classifier.py

Purpose (module docstring): "Correctness classification. Input:
feature_extraction's feature dict ... Output: label + confidence.
Trained on UI-PRMD movements 7 and 9 only, with cross-subject splits."
No `config/` keys read — reads `models/{exercise}.pkl` (the model
itself) and, only via `app.py`, the sibling `models/{exercise}.metadata.json`.

| Constant/function/class | Signature | Input | Output | Units | Failure behaviour |
|---|---|---|---|---|---|
| `FEATURE_COLUMNS` | constant list | — | `["angle_max_deg","angle_min_deg","angle_mean_deg","angle_range_deg","angle_midpoint_deg","angle_recovery_deg"]` | — | — |
| `LABEL_COLUMN`, `GROUP_COLUMN` | constants | — | `"is_correct"`, `"subject_id"` | — | — |
| `CorrectnessClassifier.__init__` | `(n_estimators=200, random_state=42)` | — | wraps an untrained `RandomForestClassifier` | — | — |
| `.train` | `(features: pd.DataFrame, labels: pd.Series) -> None` | — | fits the forest | — | — |
| `.predict` | `(features: dict) -> dict` | one rep's aggregate feature dict (must contain all `FEATURE_COLUMNS`) | `{"is_correct": bool, "confidence": float}` | confidence `[0,1]` (max class probability) | Raises `RuntimeError` if called before `train`/`load`. Raises `KeyError` if a required column is missing from `features`. |
| `.predict_batch` | `(features: pd.DataFrame) -> pd.DataFrame` | many reps' features | `{"is_correct", "confidence"}` per row, same index | — | Same `RuntimeError` guard. |
| `.save` / `.load` | `(path)` | — | `joblib.dump`/`joblib.load` of the sklearn model only (not the wrapper) | — | `.load` sets `_is_fitted=True` unconditionally — does not verify the loaded object is actually a fitted model. |
| `compute_correct_feature_ranges` | `(dataset: pd.DataFrame) -> dict` | a labeled training dataset | `{col: [min, max]}` per feature, computed over rows where `is_correct` is true | degrees | — |
| `is_within_correct_range` | `(features: dict, correct_feature_ranges: dict) -> bool` | one rep's features + the ranges above | whether every present feature falls inside its training-correct range | — | Silently skips any column in `correct_feature_ranges` not present in `features` (via `if col in features`). |
| `cross_subject_split` | `(dataset, test_size=0.2, random_state=42) -> (train_df, test_df)` | — | a `GroupShuffleSplit` by `subject_id` | — | Prints (not raises) a warning if fewer than 5 subjects are present. |

## app.py

Purpose (module docstring): "Streamlit UI: wires pose_estimation →
feature_extraction → classifier → feedback → tracking into a live
webcam session." Also runs headless via `--video`. Imports
`RepDetector`/`process_frame`/`build_smoother`/
`load_classifier_for_live_use`/`Session` from `src/pipeline.py` (see
above) instead of defining them itself. All CLI/Streamlit work runs
inside `if __name__ == "__main__":` — importing `app` does no work and
parses no CLI args (`tests/test_script_imports.py` asserts this).
Config keys read: see `docs/config_schema.md`.

| Function | Signature | Input | Output | Failure behaviour |
|---|---|---|---|---|
| `parse_args` | `() -> Namespace` | `sys.argv[1:]` | `--scorer {rules,classifier}` (default `rules`), `--video` (default `None`), `--exercise` (default `shoulder_abduction`, choices from `EXERCISES`), `--span START END` (two ints, default `None`, only meaningful with `--video`) | argparse's own usage-error exit on bad input. |
| `run_headless` | `(video_path: str, exercise: str, scorer: str, span: tuple[int,int]\|None=None) -> None` | CLI args | prints per-rep results to stdout; no return value | Prints a message and returns early (no exception) if the video can't be opened, the rules config is missing, or no classifier model file exists. Internally drives a `src.pipeline.Session`: with `span=None`, `session.start()` is called before frame 0 and never stopped (every frame reaches the rep detector — the pre-`--span` behavior, confirmed byte-identical on the project's real test clip); with `span=(start,end)`, `start()` fires at `frame_idx==start` and `stop()` right after `frame_idx==end`. |
| (module-level Streamlit script body) | — | webcam frames via `open_webcam()` | live overlay + spoken cues + `logs/*.csv`/`.discards.csv`/`.summary.json` on stop | `st.stop()` if `--scorer rules` and no config exists for the selected exercise. "Start session"/"Stop session" buttons drive `Session.start()`/`Session.stop()` (the latter in a `finally:` block, since Streamlit's own `st.session_state.running` flag — unchanged — is what actually breaks the per-frame `while` loop). |

Entry point: `if __name__ == "__main__": ARGS = parse_args(); if ARGS.video:
run_headless(...); raise SystemExit(0) ...` — so `streamlit run app.py`
and `python app.py --video ...` are two disjoint code paths through the
same file; everything after the `--video` check is Streamlit-only.
