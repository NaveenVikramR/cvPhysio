# data/ui_prmd — UI-PRMD dataset

## Source

The official dataset site (`https://webpages.uidaho.edu/ui-prmd/`, linked
from `claude.md`) is down — it 301-redirects to the University of Idaho's
generic 404 page. The data in `data/ui_prmd/` was recovered from a cached
capture of that site in the [Internet Archive Wayback
Machine](http://web.archive.org/web/20250424230129/https://www.webpages.uidaho.edu/ui-prmd/)
(snapshot dated 2025-04-24).

That capture itself turned out to be **truncated** — archive.org silently
cuts off very large files, and the two dataset zips ("Segmented Movements",
"Incorrect Segmented Movements") were cut off partway through. The
truncation point happened to fall after the entire **Kinect** portion had
downloaded intact, and only partway into **Vicon** (movements 1-2 only).
The zips' central directories were missing as a result (standard
`unzip`/`zipfile` tools reported "not a zip file"), so the files here were
recovered by manually parsing local zip file headers and extracting each
entry's data directly, verifying size/decompression as we went.

**Net result: Kinect data only.** Vicon is not present and would need a
separate, complete download to add.

## What's here

Movements 7 and 9 only (per `claude.md`'s scope), both correct and
incorrect, all 10 subjects, Kinect positions and angles:

```
data/ui_prmd/
  Segmented Movements/Kinect/Angles/m{07,09}_s{01-10}_e{01-10}_angles.txt
  Segmented Movements/Kinect/Positions/m{07,09}_s{01-10}_e{01-10}_positions.txt
  Incorrect Segmented Movements/Kinect/Angles/m{07,09}_s{01-10}_e{01-10}_angles_inc.txt
  Incorrect Segmented Movements/Kinect/Positions/m{07,09}_s{01-10}_e{01-10}_positions_inc.txt
```

- **800 files total**: 400 correct + 400 incorrect (200 positions + 200
  angles each).
- m07 = Standing Shoulder Abduction, m09 = Standing Shoulder
  Internal-External Rotation.
- Each file is one already-segmented repetition (10 subjects × 10 reps ×
  2 movements × 2 correctness × 2 file types = 800).
- Plain text, comma-delimited, no header row. See `src/dataset_loader.py`'s
  module docstring for the full column layout and everything verified
  about this format.

This folder (and `data/raw/`, `data/processed/`, `data/recordings/`) is
gitignored — only this README is tracked.

## Citation

A. Vakanski, H-P. Jun, D. Paul, and R. Baker, "A data set of human body
movements for physical rehabilitation exercises," *Data*, vol. 3, no. 2,
p. 2, 2018. https://doi.org/10.3390/data3010002

# data/rehab24_6 — REHAB24-6 dataset

## Source

Zenodo record 13305826: https://zenodo.org/records/13305826
(DOI 10.5281/zenodo.13305826), fetched via the Zenodo REST API
(`https://zenodo.org/api/records/13305826`).

## Licence

**CC BY-NC 4.0** (`cc-by-nc-4.0`). Copied verbatim from the Zenodo
record's own "License" section:

> This dataset is for academic or non-profit organization noncomercial
> research use only. By using you agree to appropriately reference the
> paper above in any publication making of its use. For comercial
> purposes contact us at info@visioncraft.ai

**This project's use of it (benchmarking an academic course project's
pipeline) is academic/non-commercial, consistent with that restriction.**
Not to be used for any commercial purpose without contacting the dataset
authors.

## Citation

Copied verbatim from the Zenodo record's "Citation" section:

> Černek, A., Sedmidubsky, J., Budikova P.: REHAB24-6: Physical Therapy
> Dataset for Analyzing Pose Estimation Methods. 17th International
> Conference on Similarity Search and Applications (SISAP). Springer,
> 14 pages, 2024.

## Files used

Only these 5 of the record's 9 files were downloaded (the rest --
`3d_markers.zip`, `2d_markers.zip`, `2d_joints.zip`, `marker_names.txt`
-- were not needed and not fetched):

| file | size | md5 |
|---|---|---|
| `videos.zip` | 2,651,613,914 B | `ed183a245a1b171638e422f7a288e5a8` |
| `3d_joints.zip` | 551,298,977 B | `c75ae3fc13fcf16d7f4ca36b93849c74` |
| `Segmentation.csv` | 53,290 B | `90b8fbd7445dd050bf27b17126c78fbe` |
| `Segmentation.txt` | 1,729 B | `5f2a5b886c6f794f03e1a8f642738c86` |
| `joints_names.txt` | 395 B | `3f04a37cce43086b1ea1fa49f1e4514f` |

All 5 md5s verified against the Zenodo API's reported checksums after
download. Extracted to `data/rehab24_6/` (gitignored, like every other
`data/` subdirectory in this project -- only this README is tracked).
