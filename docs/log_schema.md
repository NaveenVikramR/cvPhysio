# Log schema

Both CSVs come from `src/tracking.py`'s dataclasses, written via
`SessionTracker.save()` as `csv.DictWriter(fieldnames=list(asdict(...).keys()))`
— so the column order below is exactly the dataclass field order, not
alphabetical or otherwise reordered.

## Session CSV: `logs/{user_id}_{exercise}_{stamp}.csv`

One row per completed repetition (`RepDetector` event `rep_complete`
that was then scored). From `RepRecord` in `src/tracking.py`.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `exercise` | string | — | the exercise name passed to `SessionTracker(exercise=...)` (`"shoulder_abduction"` or `"shoulder_rotation"`) |
| `side` | string | — | `"LEFT"` or `"RIGHT"` — the majority vote across the rep's frames (`Counter(payload["side"]).most_common(1)[0][0]`, computed in `app.py`, not in `tracking.py` itself) |
| `angle_deg` | float | degrees | **rules scorer**: `max(rep["angle_deg"])` (the rep's peak abduction angle). **classifier scorer**: `agg["angle_max_deg"]` (same quantity, via `extract_rep_features`). Either way it is the rep's peak angle, never its full per-frame series. |
| `status` | string | — | `"pass"`, `"fail"`, or `"insufficient_data"` (rules scorer, from `score_rep()`'s `status`), or `"pass"`/`"fail"` only (classifier scorer, mapped from `result["is_correct"]` — the classifier never produces `insufficient_data`). Not validated as an enum at the dataclass level. |
| `confidence` | float or empty | probability `[0,1]` | **classifier scorer**: `result["confidence"]` (max class probability from the RandomForest). **rules scorer**: always `None` (written as an empty CSV field) — the rule scorer has no notion of confidence. |
| `timestamp` | string | ISO-8601 UTC | `datetime.now(timezone.utc).isoformat()`, captured at `RepRecord` construction time (i.e. when the rep was scored, not when it started). |

If a session has zero completed reps, `self.reps` is empty and
`SessionTracker.save()` still opens the CSV path and writes it with an
empty fieldname list — producing a 0-byte file, not a file with just a
header row (see `docs/open_questions.md`).

## Discards CSV: `logs/{user_id}_{exercise}_{stamp}.discards.csv`

One row per segment the segmenter started buffering but discarded
(`RepDetector` event `rep_discarded`). From `DiscardRecord` in
`src/tracking.py`. **Only written if `self.discards` is non-empty** —
unlike the session CSV, there is no empty placeholder file.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `exercise` | string | — | same as above |
| `reason_code` | string | — | one of `"too_short"` (buffered movement didn't reach `MIN_REP_FRAMES`/`MIN_REP_RANGE_DEG`), `"too_long"` (exceeded `MAX_REP_FRAMES`), `"lost_tracking"` (confidence lost for more than `CONFIDENCE_TOLERANCE_FRAMES` consecutive frames, and the tracking-lost rescue didn't apply). Not validated as an enum at the dataclass level — see `docs/open_questions.md`. |
| `message` | string | — | the matching human-readable string: `"Movement too brief/small -- discarded."`, `"Movement took too long -- discarded."`, or `"Rep lost -- tracking interrupted."` (all three defined inline in `app.py`'s `RepDetector.update()`, not in a shared constants table). |
| `timestamp` | string | ISO-8601 UTC | capture time of the `DiscardRecord`, i.e. when the discard was detected, not when the discarded segment started. |

Only the **Streamlit** path ever calls `tracker.add_discard(...)` —
confirmed by `grep`: `run_headless()` only `print()`s discard events to
stdout and never touches a `SessionTracker`, so **no discards CSV is
ever produced by `--video` runs**, only by interactive webcam sessions.
Same asymmetry applies to the session CSV: `run_headless()` has no
`SessionTracker` at all.

## `.summary.json` (sibling to the session CSV, always written)

Not itself a CSV, but produced by the same `SessionTracker.save()`
call, from `SessionTracker.summary()`:

| Key | Type | Unit | Meaning |
|---|---|---|---|
| `user_id` | string | — | — |
| `exercise` | string | — | — |
| `total_reps` | int | count | `len(self.reps)` |
| `passed_reps` | int | count | reps with `status == "pass"` |
| `failed_reps` | int | count | reps with `status == "fail"` |
| `insufficient_data_reps` | int | count | reps with `status == "insufficient_data"` |
| `accuracy_pct` | float or `null` | percent | `100 * passed / (passed + failed)`, rounded to 1 decimal; `null` if `passed + failed == 0` (i.e. every rep was `insufficient_data`, or there were no reps) |
| `insufficient_data_rate_pct` | float or `null` | percent | `100 * insufficient / total_reps`; `null` if `total_reps == 0` |
| `discarded_segments` | int | count | `len(self.discards)` |
