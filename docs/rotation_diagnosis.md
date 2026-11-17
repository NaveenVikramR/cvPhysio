# Rotation diagnosis (Phase 1)

Written 2026-11-16 from a run of `scripts/diagnose_rotation.py`, on the PC,
at the commit that adds this file. The full tables and plots are in
`results/diag_rotation.md` and `results/diag/rotation_*.png`, which are
gitignored and exist only on the PC. Re-run the script to regenerate them.

**What this covers:** the three demo rotation clips in `data/recordings/`.
They are unlabeled and were recorded by other people with unknown framing.
Every statement below describes measurements, not accuracy. Three clips is
also far too few to generalise from.

## Measurements

| Clip | Forearm z share at rest (median) | Spearman r, 3D angle vs depth-free image proxy | Detector segments (completed / discarded) |
|---|---|---|---|
| ShoulderInternalExternal1 (front view) | 0.87 | -0.26 | 3 / 16 |
| ShoulderInteralExternal2 | 0.35 | 0.76 | 4 / 7 |
| physitrack_standing_rotation | 0.32 | 0.40 | 6 / 17 |

- **z share** is |z| of the elbow-to-wrist vector divided by its 3D length, in world landmarks: 1 means the forearm points straight at the camera.
- **Image proxy** is the wrist's horizontal offset from the elbow divided by the 2D forearm length, in pixels. It uses no depth.

## Verdicts

**Q1. Does the rotation angle fail because of MediaPipe's depth estimate? Yes, in a front view.**
- With the arm at the side, rotating about the upper arm is mostly motion toward or away from a front camera. The current angle (`compute_shoulder_rotation_angle`) therefore can't avoid depending on z. An image-plane version of the same angle is degenerate: it only ever reads 0° or 180°.
- In the front-view clip, the image proxy shows four clean full rotations, swinging from about +0.9 to −0.9. Over the same frames the 3D angle moves only about 20–40°, and MediaPipe keeps the forearm pointing at the camera (z share 0.8–1.0) throughout.
- This explains why webcam peaks sit far below the 172.9° floor that came from motion capture.

**Q2. Does `RepDetector` split or lose rotation reps? Yes.**
- **Drifting baseline.** The resting baseline is a running mean of in-band frames, so a slow rotation drags it along. In ShoulderInteralExternal2 the baseline ended up at the outward end (about 160°). After t = 9.3 s the angle never came back within ±10°, and no further reps were detected even though the rotations continued.
- **Split reps.** Reps are recorded as excursions on one side of the baseline, so a single out-and-back rotation can come out as separate "outward" and "inward" segments. 24 of the 40 discarded segments were 1–3 frames long, mostly 1-frame flickers across the edge of the rest band. That is open question #14, the missing debounce on rep start.

**Q3. Would a different camera angle help? Possibly. This needs clips recorded for the purpose.**
- The two clips where the resting forearm lies more in the image plane (z share about 0.3) agree better with the image proxy (r = 0.76 and 0.40) than the front view does (r = −0.26).
- That fits "a front camera is the worst case", but these clips differ in more than camera angle. It needs a controlled test.

## What this means for Phase 2 (`docs/dev_plan.md`)

1. **Main fix: the rotation signal (2c).** Add a rotation measure that works in the image for a front camera. One example is the wrist's lateral offset from the elbow, normalised by a body length that stays visible, such as the upper arm. Choose between it and the 3D angle by camera setup. Retrain the rotation classifier on the same measure computed from UI-PRMD positions projected to the frontal plane.
2. **Detector (2a), per exercise for rotation only:**
   - fix the baseline at calibration instead of letting it drift
   - debounce rep start (#14)
   - count a rep as out and back through neutral
3. **Laptop test clips before Phase 3 framing is fixed:** the same person doing about 5 rotations each in front view, 45° view and side view, all unscored. Rerun `scripts/diagnose_rotation.py` on them to choose the recording protocol's camera angle.
