# Config schema

Every key in `config/exercises/shoulder_abduction.json` and
`config/smoothing.json`, as they exist on disk today. "Where read"
cites the exact call site found via `grep` over `src/`, `app.py`, and
`scripts/`.

## config/exercises/shoulder_abduction.json

No `config/exercises/shoulder_rotation.json` (or any other exercise)
exists — `src.rules.load_config()` raises `FileNotFoundError` for any
exercise but `shoulder_abduction`. This file is the **only** rules
config in the repo.

| Key | Type | Unit | Meaning | Current value | Provisional? | Where read |
|---|---|---|---|---|---|---|
| `exercise` | string | — | which exercise this file is for | `"shoulder_abduction"` | not marked | **Not read by any code** — purely descriptive. |
| `notice` | string | — | human-readable disclaimer that every value below is a placeholder | (long string) | not marked (it IS the provisional notice) | **Not read by any code.** |
| `target_peak_deg.value` | number | degrees | the abduction angle (hip-shoulder-elbow) a rep's peak should reach | `90` | yes | `src/rules.py` `score_rep()` via `_value(config, "target_peak_deg")`. |
| `tolerance_deg.value` | number | degrees | +/- allowed around `target_peak_deg` for the `peak_angle` rule | `15` | yes | `src/rules.py` `score_rep()`. |
| `min_elbow_straight_deg.value` | number | degrees | minimum elbow angle (shoulder-elbow-wrist) allowed anywhere in the rep | `135` | yes | `src/rules.py` `score_rep()`. |
| `min_elbow_straight_deg.comment` | string | — | rationale for the 150→135 change (cites UI-PRMD measurement) | (long string) | — | **Not read by any code** — this key only exists on `min_elbow_straight_deg`, not the other thresholds. |
| `max_trunk_lean_deg.value` | number | degrees | maximum trunk-lean angle allowed anywhere in the rep | `15` | yes | `src/rules.py` `score_rep()`. |
| `tempo_min_s.value` | number | seconds | minimum rep duration (frame count / fps) | `1.0` | yes | `src/rules.py` `score_rep()`. |
| `tempo_max_s.value` | number | seconds | maximum rep duration | `6.0` | yes | `src/rules.py` `score_rep()`. |
| `cue_cooldown_s.value` | number | seconds | minimum gap between repeating the same spoken cue | `8` | yes | `app.py`, Streamlit body: `rules_module.CueCooldown(rules_config["cue_cooldown_s"]["value"])`. **Not read in `run_headless()`** — the headless path never constructs a `CueCooldown` at all. |
| `min_valid_fraction.value` | number | fraction `[0,1]` | minimum share of a rep's frames that must pass the confidence gate for it to be scored pass/fail at all | `0.7` | yes | `src/rules.py` `score_rep()` via `_value(config, "min_valid_fraction")`. |
| `cues.peak_angle` | string | — | spoken/displayed cue when `peak_angle` fails | "Raise your arm closer to the target height." | n/a (text, not numeric) | `app.py` Streamlit body, passed into `cue_cooldown.cues_to_speak(result["failed_rules"], rules_config["cues"])`. |
| `cues.elbow_straight` | string | — | cue when `elbow_straight` fails | "Keep your elbow straighter as you raise your arm." | n/a | same as above. |
| `cues.trunk_lean` | string | — | cue when `trunk_lean` fails | "Keep your torso upright -- try not to lean." | n/a | same as above. |
| `cues.tempo` | string | — | cue when `tempo` fails | "Move at a steadier, more controlled pace." | n/a | same as above. |
| `cues.insufficient_data` | string | — | cue when the rep's status is `insufficient_data` | "Keep your whole arm in view." | n/a | `src/rules.py` `score_rep()` (`config["cues"].get("insufficient_data")`, placed into the returned `cue` field), consumed in `app.py` Streamlit body. |

Every `*_deg`/`*_s`/`min_valid_fraction` value above is explicitly
marked `"status": "provisional"` in the file (the `status` field was
omitted from the table above per-row since it's identical for all of
them: `"provisional"`).

## config/smoothing.json

| Key | Type | Unit | Meaning | Current value | Provisional? | Where read |
|---|---|---|---|---|---|---|
| `enabled` | boolean | — | whether `app.py`'s `build_smoother()` constructs a `LandmarkSmoother` at all | `false` | n/a (not a threshold; see `enabled_comment`) | `app.py` `build_smoother()`: `if not config.get("enabled"): return None`. |
| `enabled_comment` | string | — | rationale for leaving it off (too few calibration clips to tune safely; prior fixed params measured to break rep detection on the one real test clip) | (long string) | — | **Not read by any code.** |
| `min_cutoff` | number | Hz | One Euro filter cutoff frequency at zero speed (lower = heavier smoothing at rest) | `1.0` | not marked `provisional` (no `status` field on this file at all — unlike the exercise config, these are plain numeric keys, not `{value, status}` objects) | `app.py` `build_smoother()`; also `scripts/measure_smoothing_effect.py`, `scripts/tune_smoothing.py` (construct their own `LandmarkSmoother` directly from this file for comparison runs). |
| `beta` | number | unitless gain | how much the cutoff rises with speed (higher = less lag on fast motion) | `0.3` | not marked | same call sites as `min_cutoff`. |
| `d_cutoff` | number | Hz | cutoff used to smooth the speed estimate itself | `1.0` | not marked | same call sites. |
| `max_hold_frames` | integer | frames | how many consecutive missing-landmark frames `LandmarkSmoother` bridges by holding the last good value before returning `None` | `5` | not marked | same call sites. |

Note the structural difference between the two config files:
`config/exercises/shoulder_abduction.json` wraps every numeric value in
`{"value": ..., "status": "provisional"}`, but `config/smoothing.json`
stores plain numbers directly (`config["min_cutoff"]`, not
`config["min_cutoff"]["value"]`) — confirmed from both files' literal
contents and from the differing access patterns in the code
(`_value(config, key)` helper in `rules.py` vs. direct `config[key]` in
`app.py`/the smoothing scripts). This inconsistency is in
`docs/open_questions.md`.
