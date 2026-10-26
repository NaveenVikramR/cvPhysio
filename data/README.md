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
