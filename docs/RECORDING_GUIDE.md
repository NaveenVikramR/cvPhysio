# Recording guide for the team dataset

This guide is for the team member who records videos. You don't need to
read or change any code. Follow the steps in order, copy the commands as
they are, and ask if anything looks different from what's described here.

**What you're doing:** recording short webcam clips of the two project
exercises, done correctly and with specific mistakes on purpose. We use
the clips to develop and then test the app. The recorder shows a live
preview with a skeleton and warnings, so you can see when a take is set up
properly before you record.

---

## 0. Before anything else: the rules

1. **Consent.** Anyone who appears in a clip agrees in writing (a message
   is fine) that the clips will be used for this university project only,
   kept private, and deleted at the end of the course.
2. **Never put videos in git.** The recordings folder is already excluded,
   so don't force-add anything from `data/`. Videos travel only through
   our private shared drive (step 6).
3. **Never use `--dataset preregistered`.** That writes into a frozen
   evaluation with its own rules. Every command in this guide leaves it
   out, which is correct.
4. **Use subject IDs, not names:**

| ID | Who | Group | What that means |
|---|---|---|---|
| `s01` | the developer | `calibration` | We look at and tune on these clips |
| `s02` | you | `eval` | **Nobody looks at or scores these until the code is frozen.** Don't play them back for the developers or send them screenshots |

5. **Don't delete takes.** If a take goes wrong, record it again; the
   recorder gives the new take the next number automatically. Write the
   reason in `--notes`, e.g. `--notes "retake of 01, walked out of frame"`.

## 1. One-time setup

Follow `docs/laptop_setup.md` steps 1–4: clone, environment, pose model,
then the check. Then get the current branch:

```bash
git switch elbow-accuracy
```

```bash
git pull --ff-only
```

Check that the recorder starts. It should print its options:

```bash
python scripts/record_clip.py --help
```

## 2. Setting up the room (the same for every clip)

- **Camera:** laptop or webcam at chest height, on a table or stand,
  **2–2.5 m** from where the person stands.
- **Background:** plain wall, no clutter, no one else in the picture.
- **Clothing:** short sleeves, in a colour that stands out from the wall.
- **Light:** even light from the front or overhead. No bright window
  behind the person.
- **Same spot, same setup for a whole session.** Note what you used in
  `--notes`, e.g. `--notes "overhead LED, black t-shirt, 2.2 m"`.

**Camera angle (`--view`).** The person always moves their **right arm**,
and the camera goes on the **right-arm side** for the angled views. Seen
from above:

```
  front                 45                    side
  (camera faces         (camera half-way      (camera level with
   the person)           round to the right)   the right shoulder)

      [P]                   [P]                  [P] <--- [C]
       ^                       \
       |                        \
      [C]                       [C]
```

`[P]` = person facing up the page, `[C]` = camera. In the side view the
right arm must face the camera, not be hidden behind the body.

## 3. Recording one take

1. Type the command for the take (from step 4 or 5) and press Enter. A
   preview window opens.
2. Check the preview:
   - a green skeleton is drawn over the person
   - no red **LOW VISIBILITY** or **NO POSE DETECTED** text
   - no orange **OUT OF BOX** text: everything from the hips up, including
     the hands at the top of each movement, stays inside the white box
3. Press **SPACE**. You get a 15-second countdown to walk into place.
4. **Stand still with your arms down for 2 seconds**, then do the
   repetitions. Stop at the planned number.
5. **Stand still with your arms down again** until the recording stops by
   itself (that's what `--seconds` is for). Don't press `q`.
6. **ESC** at any point throws the take away and saves nothing.

If a red bar says **ARMS UP – LOWER YOUR ARMS NOW** or **KEEP YOUR UPPER
ARM AT YOUR SIDE** at the start, you weren't in the start position. Press
ESC and try again.

## 4. How to do each exercise

Aim for about 2–3 seconds per repetition, smooth and controlled.

**Abduction (`--exercise abduction`).** Arm straight at your side, palm
facing in. Raise it out to the side up to shoulder height, then lower it
back down. That is one rep.

| `--intended` | What to do on purpose |
|---|---|
| `correct` | As described above |
| `fault_lowarm` | Stop well short of shoulder height, about halfway |
| `fault_trunk` | Lean your upper body clearly to the opposite side as you raise the arm |
| `fault_elbow` | Bend the elbow clearly during the raise |
| `fault_fast` | Rush: about 1 second per rep or less |

**Rotation (`--exercise rotation`).** Upper arm against your side, elbow
bent at a right angle, forearm pointing forward, thumb up. Keeping the
elbow tucked in:
1. swing the forearm **outward**, away from your belly, as far as is comfortable
2. back through the start position
3. **inward** toward your belly
4. back to the start

That is one rep.

| `--intended` | What to do on purpose |
|---|---|
| `correct` | As described above |
| `fault_short_out` | Only go about halfway on the outward swing |
| `fault_short_in` | Only go about halfway on the inward swing |
| `fault_elbow_drift` | Let the elbow drift clearly away from your side during the movement |
| `fault_trunk_twist` | Twist your upper body to help the arm move |
| `fault_fast` | Rush: about 1 second per rep or less |

## 5. What to record, in stages

### Stage 1: camera-angle test (do this first, about 15 minutes)

These three clips decide which camera angle we use for rotation. **Subject
`s01` (the developer)**, rotation, correct, 5 reps, one clip per angle:

```bash
python scripts/record_clip.py --exercise rotation --view front --role test --intended correct --expected-reps 5 --subject s01 --seconds 30 --notes "camera angle test"
```

```bash
python scripts/record_clip.py --exercise rotation --view 45 --role test --intended correct --expected-reps 5 --subject s01 --seconds 30 --notes "camera angle test"
```

```bash
python scripts/record_clip.py --exercise rotation --view side --role test --intended correct --expected-reps 5 --subject s01 --seconds 30 --notes "camera angle test"
```

Then run the analysis on them and package the results (step 6):

```bash
python scripts/diagnose_rotation.py data/recordings/team/rot_front_test_correct_s01_01.mp4 data/recordings/team/rot_45_test_correct_s01_01.mp4 data/recordings/team/rot_side_test_correct_s01_01.mp4
```

```bash
python scripts/package_results.py camera-angle-test --machine laptop results/diag_rotation.md results/diag/rotation_rot_front_test_correct_s01_01.png results/diag/rotation_rot_45_test_correct_s01_01.png results/diag/rotation_rot_side_test_correct_s01_01.png
```

If the recorder named a take `_02` because of a retake, use that name in
both commands instead.

**Then stop and wait.** The developers will tell you which angle to use
for Stage 3 and update this guide.

### Stage 2: the frozen abduction evaluation (only on the laptop that has the 7 original clips)

Skip this if you don't have that laptop. If you do, follow
`docs/HANDOVER.md`, section "How to resume", step 4, exactly. It runs
`qc_all.py`, then `evaluate_own_clips.py` (only if QC says OK for all 7),
then `compute_primary_outcomes.py`. **Run each one once.** Package the
outputs with `scripts/package_results.py preregistered-abduction-eval ...`
and push.

### Stage 3: the full dataset (after Stage 1 is decided)

For **each subject** (`s01` with `--role calibration`, `s02` with
`--role eval`), at the chosen camera angle, record:

| Exercise | `--intended` | Reps | Takes | `--seconds` |
|---|---|---|---|---|
| abduction (front view) | `correct` | 10 | 2 | 45 |
| abduction (front view) | each of the 4 faults | 5 | 1 each | 30 |
| rotation (chosen view) | `correct` | 10 | 2 | 50 |
| rotation (chosen view) | each of the 5 faults | 5 | 1 each | 35 |

That's 13 takes per subject. Template, with the angle-brackets filled in:

```bash
python scripts/record_clip.py --exercise <abduction|rotation> --view <front|45|side> --role <calibration|eval> --intended <label> --expected-reps <5|10> --subject <s01|s02> --seconds <from table> --notes "<light, clothing, distance>"
```

## 6. Sending things back

- **Videos:** copy the whole `data/recordings/team/` folder **and**
  `data/team_manifest.csv` to our private shared drive folder. They never
  go into git.
- **Small results** (reports, plots, CSVs, no video) go into git through
  `team_results/`:

```bash
python scripts/package_results.py <short-label> --machine <laptop|pc> <files...>
```

This copies the files into `team_results/<date>_<label>/` with a
`provenance.json` recording the git commit, so we know exactly which code
produced them. Then commit and push that folder:

```bash
git add team_results
```

```bash
git commit -m "Add <label> results"
```

```bash
git push
```

## 7. If something goes wrong

| Problem | What to do |
|---|---|
| Black or frozen preview | Close other apps using the camera. Try `--camera 1` |
| NO POSE DETECTED | Step back so you're fully in view, and improve the light |
| LOW VISIBILITY: right wrist | Short sleeves, contrasting colour, more light, and keep the hand in front of the body, not behind it |
| OUT OF BOX | Move the camera back or tilt it, until the hands stay inside the box at the top of each rep |
| You did the wrong thing during a take | Let it finish, then record it again with `--notes "retake of NN, reason"` |
| Any error text in the terminal | Copy the whole message and send it to the developers. Don't try to fix the code |
