# Offline pipeline: UI-PRMD → forward kinematics → features → LOSO training

Source: `src/dataset_loader.py` (loader + FK) and `scripts/train_classifier.py`
(training/evaluation driver). Build-time only — neither module is
imported by the 6 runtime modules or `app.py`.

## 1. Discovery and parsing (`discover_rep_files`, `parse_filename`)

- Scans `data/ui_prmd/` (or whatever `raw_dir` is passed) recursively for
  files matching `m{MM}_s{SS}_e{EE}_{positions|angles}(_inc)?.txt`.
- `movement` (`MM`), `subject_id` (`s{SS}`), `rep_id` (`e{EE}`),
  `is_correct` (`True` unless the `_inc` suffix is present), `source`
  (`"positions"` or `"angles"`) are parsed directly from the filename —
  **not** from the containing directory name (the module docstring notes
  the real download folders are "Segmented Movements"/"Incorrect
  Segmented Movements", neither literally "correct"/"incorrect", so a
  path-based check would not have worked).
- `modality` (`"kinect"` here; `"vicon"` is defined but explicitly
  out of scope) is read from the path itself (`_modality_from_path`:
  looks for `"kinect"` or `"vicon"` as a path component, case-insensitively).
- Returns `[]` if `raw_dir` doesn't exist — treated as expected (no
  download yet), not an error.

## 2. Side selection (`select_side_for_rep`)

Mocap data carries no per-frame visibility score, so which arm performed
the rep is inferred by comparing each side's candidate Angles-file
column range across the rep: whichever side's value swept a larger
range (`max - min`) is picked. The docstring states this was verified
against real data: for confirmed right-arm subjects, the right column's
range was 20-30x the left column's.

## 3. Forward kinematics (`forward_kinematics_frame`, `euler_zyx_to_matrix`)

Per the module docstring, this reconstructs **absolute** joint positions
from UI-PRMD's raw Positions file (parent-relative offsets) + Angles
file (per-joint ZYX Euler angles in degrees), replicating the dataset
author's own MATLAB `Animation.m`:

- Rotation matrix: `R = Rz(axis2) @ Ry(axis1) @ Rx(axis0)`.
- Torso chain (`Waist → Spine → Chest → Neck → Head → HeadTip`): each
  child's absolute rotation is the accumulated product of every
  ancestor's own rotation; position is `rot[child] @ offset[child] +
  pos[parent]`.
- **Verified quirk, not a simplification**: the left/right arm chains
  (`{L,R}_Collar → {L,R}_UpperArm → {L,R}_Forearm → {L,R}_Hand`) and leg
  chains (`{L,R}_UpperLeg → ... `) do **not** inherit the full torso
  rotation chain — `L_Collar`/`R_Collar` rotate by `Chest`'s own angle
  alone (not the accumulated `rot[Chest]`), and `{L,R}_UpperLeg` rotate
  by `Waist`'s own angle alone. This is implemented exactly as coded in
  `Animation.m`, per the docstring.
- Why this matters: a naive 3-point angle computed directly on the raw
  (un-FK'd) Positions data was measured (module docstring) to differ
  from the Angles-file value by a mean of 151.5° (range 95.0-208.2°)
  across one 63-frame rep — the raw positions barely moved (156-158°)
  while the true exercise swung ~115°. FK is therefore load-bearing, not
  optional polish.

## 4. Packing into MediaPipe-shaped landmarks (`_fk_frame_to_landmarks`)

The FK output (a dict of joint name → absolute `(x,y,z)`) is packed into
a 33-slot list shaped exactly like `pose_estimation.get_pose_landmarks()`'s
output, using the role mapping `shoulder=UpperArm, elbow=Forearm,
wrist=Hand, hip=UpperLeg` (mirrors `feature_extraction.py`'s MediaPipe
roles). The active side's 4 landmarks get `visibility=1.0`; the other
side's get `0.0` and are otherwise zero-filled — so
`feature_extraction.select_side()` deterministically re-derives the same
side already chosen by `select_side_for_rep()`. This lets
`compute_shoulder_abduction_angle`/`compute_shoulder_rotation_angle` run
**unmodified** on dataset data — the stated reason being to keep
train-time and inference-time angle math identical rather than
maintaining two implementations.

## 5. Two feature sources (`feature_source` parameter)

| `feature_source` | Per-frame `angle_deg` comes from | Status |
|---|---|---|
| `"fk_positions"` (default) | `compute_shoulder_abduction_angle`/`compute_shoulder_rotation_angle` applied to the FK output above | What `app.py`'s saved models actually use. |
| `"angles_file"` (legacy) | A single column read directly from the Angles file (`JOINT_COLUMNS[modality][exercise][side]`, e.g. `col_35` = `R_UpperArm` axis 2 for abduction) | Kept only for side-by-side comparison; `scripts/train_classifier.py` does not save a model from this path unless `--features angles_file` is explicitly passed. |

## 6. Per-rep aggregation (`extract_rep_features`, in `feature_extraction.py`)

Each rep's frame-by-frame `angle_deg` series (from either source) is
reduced to 6 scalars — see `docs/module_interfaces.md` for the exact
formulas (`angle_max_deg`, `angle_min_deg`, `angle_mean_deg`,
`angle_range_deg`, `angle_midpoint_deg`, `angle_recovery_deg`). This is
the **same function** `app.py` calls at inference time on a live rep's
angle series — not a separate training-time implementation.

## 7. LOSO training/evaluation (`scripts/train_classifier.py`)

- For each of `shoulder_abduction`/`shoulder_rotation`: loads the full
  rep-level dataset via `load_dataset_for_exercise`, then runs
  **leave-one-subject-out** cross-validation — one fold per subject,
  that subject held out as the test set, every other subject as training.
- Per fold, 3 models are compared on the held-out subject: the actual
  `RandomForestClassifier` (`CorrectnessClassifier`, `n_estimators=200,
  random_state=42`), a majority-class baseline (fit on the fold's train
  labels only), and a single-threshold-on-peak-angle baseline (threshold
  = midpoint between the two classes' mean `angle_max_deg`, direction
  decided from train-fold class means only — never from the held-out
  subject).
- Per-fold metrics: accuracy, precision/recall/F1 for both classes
  (`sklearn.metrics.precision_recall_fscore_support`), plus a pooled
  confusion matrix summed across all folds.
- After LOSO, a **final** model is trained on the **entire** dataset
  (all subjects, no held-out subject) — this is the model actually
  saved to `models/{exercise}.pkl`, not any individual fold's model.
- `save_model_with_metadata` also computes
  `compute_correct_feature_ranges` (per-feature min/max across only the
  `is_correct==True` training rows) and writes it, plus
  `feature_names`, `feature_source`, `training_date`, `n_subjects`,
  `subjects`, and the LOSO summary (mean/std per metric), into
  `models/{exercise}.metadata.json`. This is what `app.py` reads back
  (via `is_within_correct_range`) to flag a live classifier prediction as
  outside anything the model was actually trained on as "correct."
- CLI: `python scripts/train_classifier.py [--features fk_positions|angles_file|both]`
  (default `both` — runs LOSO for each source and only saves a model for
  `fk_positions`, or for `angles_file` only if that's the sole value
  passed; `--features both` explicitly skips saving the `angles_file`
  model, per its own printed message).
