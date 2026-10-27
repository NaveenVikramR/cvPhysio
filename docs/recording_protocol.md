# Recording protocol for our own clips

For `data/own_manifest.csv` / `scripts/evaluate_own_clips.py` (see that
script's docstring for how these clips are used). All clips for one
subject should be recorded in one sitting, same spot, same setup --
the whole point of a personal reference/calibration is that it only
has to generalize across THIS person's own future sessions, not across
subjects or cameras the way the REHAB24-6 work had to.

## Camera setup

- Camera at **chest height**, approximately **2 m** from the subject.
- **Front view**: the camera faces the subject directly, not an angle
  (REHAB24-6's cam18/profile view was found to have far worse MediaPipe
  tracking fidelity and rep-detection reliability than its front view
  -- see `results/review3_summary.md` table 1/2 -- front view is the
  only view this project's live pipeline is validated against).
- Frame the shot so the **full upper body and both arms stay in frame**
  for the entire range of motion, including at the top of the
  repetition (a raised arm leaving the frame is a common framing
  mistake with a chest-height camera at 2 m -- check this on a live
  preview before recording, not just on the first frame).
- **Plain background** (no patterned wallpaper, no clutter behind the
  subject) and a **shirt/sleeve colour that contrasts with the wall**
  behind them -- both reduce the chance MediaPipe's segmentation
  confuses the arm's silhouette with the background.
- **Same spot for all clips** (reference, calibration, and every eval
  clip) -- same camera position, same distance, same background, same
  lighting. A reference/calibration derived from one setup and then
  evaluated against footage from a different one would conflate
  "did the subject's movement change" with "did the camera/lighting
  change," which this protocol is specifically trying to avoid.

## Video settings

- **1080p** (1920x1080), **30 fps**. Matches the frame rate every
  timing-based threshold in this project (tempo_min_s/tempo_max_s,
  tol_tempo_frac) was derived and tested against; a different fps
  would need those thresholds re-validated, not just re-measured.

## Per-clip metadata

Record one metadata row per clip (see `data/own_manifest.csv`'s
`subject`/`notes` columns, and keep your own more detailed log if you
want more than those two columns) covering:

- **camera**: make/model or device used.
- **distance**: measured distance from camera to subject (should be
  ~2 m per clip; note any clip that deviates and by how much).
- **lighting**: brief description (e.g. "overhead fluorescent," "window
  light, overcast," "lamp only") -- lighting changes between clips are
  a common unannounced confound in pose-estimation fidelity.
- **subject**: who is in the clip (a name or id, consistent across
  that subject's reference/calibration/eval clips).
- **clothing**: sleeve length and colour (relevant to the
  contrast-with-background point above, and to whether the elbow/wrist
  landmarks stay visible).

## Clip roles and filenames

Filename convention: `abd_<role>_<intended>_<nn>.mp4` (`nn` = a
2-digit clip index, for when a role/intended combination needs more
than one clip -- e.g. a eval session split across two files).

| role | intended | reps | purpose |
|---|---|---|---|
| `reference` | `correct` | 3 | The "therapist" recording -- the first 3 reps become the per-patient Reference (`src.rules.build_reference()`). |
| `calibration` | `correct` | 8 | Used to derive `reference_relative` tolerances (deviation of each rep from the reference median) -- see `scripts/derive_reference_tolerances.py`'s method, applied to these clips by `scripts/evaluate_own_clips.py`. |
| `eval` | `correct` | 10 | Scored against both the demo (absolute) and the derived `reference_relative` config -- used for the false-fail rate. |
| `eval` | `fault_lowarm` | 5 | Arm raised deliberately too low -- expected to fail `peak_angle`. |
| `eval` | `fault_trunk` | 5 | Deliberate excessive trunk lean -- expected to fail `trunk_lean`. |
| `eval` | `fault_elbow` | 5 | Deliberate elbow bend during the raise -- expected to fail `elbow_straight`. |
| `eval` | `fault_fast` | 5 | Deliberately rushed tempo -- expected to fail `tempo`. |

Example filenames: `abd_reference_correct_01.mp4`,
`abd_calibration_correct_01.mp4`, `abd_eval_correct_01.mp4`,
`abd_eval_fault_lowarm_01.mp4`, `abd_eval_fault_trunk_01.mp4`,
`abd_eval_fault_elbow_01.mp4`, `abd_eval_fault_fast_01.mp4`.

Each row above corresponds to one row in `data/own_manifest.csv` (see
that file's header/example rows for the exact columns).
