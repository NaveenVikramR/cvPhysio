# Open questions

Everything below was found by reading the code while writing the other
5 documents. Each item names the exact file/function it comes from so
it can be checked without re-deriving it.

1. **Smoothing runs before the confidence gate, not after.**
   `app.py`'s `process_frame()` calls `smoother.update(landmarks, t)`
   first, then `check_confidence(...)` on the result. If any design
   intent says "gate, then smooth," the code does the opposite.

2. **`compute_elbow_angle`/`compute_trunk_lean_deg` run unconditionally**,
   on every frame that has landmarks at all, *regardless of whether the
   confidence gate passed* — only whether the result gets **appended**
   into the per-rep buffers (and therefore ever reaches `score_rep`) is
   gated by `visible`. The values themselves are computed on
   possibly-low-confidence landmarks either way.

3. **`compute_trunk_lean_deg`'s "live landmarks only" invariant is not
   enforced in code.** Its docstring says it "must only be called on
   live landmarks, never on UI-PRMD's Y-up FK positions" (it assumes Y
   increases downward). Today that's true only because
   `dataset_loader.py` happens to never call it — there's no assertion,
   type distinction, or runtime guard that would catch a future caller
   doing so.

4. **`extract_features()` re-runs `check_confidence()` with the default
   (non-rules) role set, even when the caller already ran a stricter
   check.** In `process_frame`, `check_confidence(..., required_roles)`
   is called once, and then (only if that passed) `extract_features()`
   calls `check_confidence()` *again* internally with no `required_roles`
   override, i.e. against `BASE_REQUIRED_ROLES`. Today this is
   logically redundant (the rules' role set is always a superset of the
   base set for `shoulder_abduction`), but the two tables
   (`BASE_REQUIRED_ROLES` and `REQUIRED_ROLES_BY_RULE`) are maintained
   independently in two files — nothing stops them from drifting apart.

5. **`run_headless()` never surfaces `RepDetector.calibration_hint()`.**
   If calibration never completes (confirmed to happen on a real clip in
   a prior task, when wrist visibility was too low almost throughout),
   `--video` mode just prints "No completed reps detected in this clip."
   at the end with no diagnosis of *why* — the hint mechanism that would
   explain it only runs in the Streamlit path.

6. **`run_headless()` never constructs a `CueCooldown` or reads
   `cue_cooldown_s`/`cues` from config at all.** It prints the raw rule
   breakdown instead of cue text. This means those two config keys are
   entirely unexercised by `--video` runs.

7. **Two independent message-building implementations for the same 3
   statuses.** `src/feedback.py`'s `message_for_result()` is shaped for
   the classifier's `{"is_correct": bool}` result and is never called
   from the rules path. The rules path builds its own `status_text`
   strings inline, once in `app.py`'s Streamlit body and again,
   differently, in `run_headless()`'s print statements. There is no
   single shared "format a rules verdict as text" function.

8. **`RepRecord.status` and `DiscardRecord.reason_code` are plain `str`
   fields**, not a `Literal`/enum, in `src/tracking.py`. Nothing stops a
   future caller from writing a typo'd status string into the CSV.

9. **`SessionTracker.save()` can write a 0-byte, header-less reps CSV.**
   If `self.reps` is empty but `self.discards` is not, the reps CSV is
   still opened and written with `fieldnames=[]` and no header row
   (since `if self.reps: writer.writeheader()` never fires) — a 0-byte
   file, rather than either a header-only file or no file at all. The
   discards CSV, by contrast, is only created when non-empty.

10. **The two config files use different schemas for the same kind of
    value.** `config/exercises/shoulder_abduction.json` wraps every
    numeric key as `{"value": ..., "status": "provisional"}`;
    `config/smoothing.json` stores plain numbers
    (`config["min_cutoff"]`, not `config["min_cutoff"]["value"]`). Both
    are read by code in the same project; the access pattern differs
    accordingly (`src/rules.py`'s `_value()` helper vs. direct
    `dict[key]` elsewhere).

11. **`config/exercises/shoulder_rotation.json` does not exist.**
    `src.rules.load_config("shoulder_rotation")` raises
    `FileNotFoundError`. A trained classifier *and* metadata exist for
    `shoulder_rotation` (`models/shoulder_rotation.pkl`/`.metadata.json`),
    but `--scorer rules --exercise shoulder_rotation` is permanently
    unusable until that config file is written — this may be intentional
    project scoping (abduction-first), but nothing in the code says so.

12. **`compute_angle()` has no zero-length-vector guard.** If any two of
    the three input points coincide exactly, `norm(ba)` or `norm(bc)` is
    0, producing `0/0 = nan` (NumPy emits a `RuntimeWarning`, does not
    raise) rather than a caught, explicit error. Not reproduced or
    tested here — flagged from reading the function, not from running it.

13. **Test coverage gaps.** Only `src/rules.py`, `src/smoothing.py`, and
    `app.py`'s `RepDetector` (via `tests/test_segmenter.py`'s exec-sliced
    extraction) have dedicated tests. No tests exist for
    `feature_extraction.py`, `pose_estimation.py`, `feedback.py`,
    `tracking.py`, `classifier.py`, `dataset_loader.py`, `app.py`'s
    `process_frame()`/`run_headless()` themselves, or the Streamlit body.

14. **`RepDetector`'s RESTING→MOVING transition has no debounce.** A
    single frame whose angle exceeds `REST_BAND_DEG` from baseline
    immediately starts a new `rep_buffer` (state becomes `MOVING`) —
    there's no requirement that the excursion persist for more than one
    frame before being treated as the start of a real rep. Whether a
    single noisy frame can therefore spuriously start (and then, via
    `too_short`, silently discard) a rep is not addressed in any comment.

15. **Calibration recovery pops one sample, not the whole window, on a
    high-stdev window.** `if stdev(calib_window) > 6.0: calib_window.popleft()`
    removes only the oldest sample and retries — if noisy samples keep
    arriving as fast as they're trimmed, there's no fallback (e.g. a
    max-retry count) that would force a full restart or surface a
    different hint; `calibration_hint()`'s 150-frame stall threshold is
    the only backstop, and it only fires in the Streamlit path (see #5).

16. **`OneEuroFilter`'s parameters are unvalidated.** `min_cutoff`,
    `beta`, `d_cutoff` accept any float, including zero or negative
    values, with no bounds check — the smoothing-factor formula's
    behavior at e.g. `cutoff <= 0` isn't guarded against or documented.

17. **`LandmarkSmoother` never filters `visibility`.** Each landmark's
    4th field (`visibility`) is passed through from the raw (or
    held-over) frame unchanged; only x/y/z go through a `OneEuroFilter`.
    This is a reasonable design (visibility isn't a spatial coordinate)
    but isn't stated as a deliberate choice anywhere in the module
    docstring — it's only inferable from the tuple-construction line in
    `update()`.

18. **`app.py` reads `models/{exercise}.metadata.json` with no schema
    check.** `correct_feature_ranges` is pulled out via a plain
    `.get()` call; if `scripts/train_classifier.py`'s `FEATURE_COLUMNS`
    ever changes without the metadata file being regenerated, `app.py`
    would silently use stale ranges rather than detecting the mismatch.

19. **`select_side()`'s tie-break is `>=` (left wins on an exact tie).**
    A minor, undocumented convention: `LEFT if left_visibility >=
    right_visibility else RIGHT` — not wrong, just never stated as
    intentional anywhere outside the code itself.
