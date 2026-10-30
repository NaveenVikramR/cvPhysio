# Laptop setup: from clone to running the app

Exact commands, verified against a real fresh clone (see the commit
that introduced this file for the exact pass/fail counts and Python
version that clone produced).

## 1. Clone

```
git clone https://github.com/NaveenVikramR/cvPhysio.git
cd cvPhysio
```

**Windows note**: clone to a short path (e.g. `C:\cvPhysio`), not deep
inside `AppData\Local\Temp\...`. One dependency (`streamlit`) ships a
deeply-nested asset folder that can exceed Windows' 260-character
`MAX_PATH` limit if your clone path is already long -- `pip install`
will fail partway through with an `OSError` on a path like
`...\site-packages\streamlit\.agents\skills\...\pyproject.toml` if so.

## 2. Create a virtual environment and install dependencies

```
python -m venv .venv
.venv\Scripts\activate          # Windows (cmd/PowerShell)
# source .venv/bin/activate     # macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt
pip install pytest              # NOT in requirements.txt -- needed to run the test suite
```

## 3. Download the MediaPipe pose-landmarker model (gitignored, required)

The app cannot start without this file -- `src/pose_estimation.py`
raises a clear error naming this exact URL and path if it's missing.

```
mkdir -p models
curl -L -o models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

(~5.8 MB / 5,777,746 bytes. On Windows without `curl`, download the
URL in a browser and save it to `models\pose_landmarker_lite.task`.)

## 4. Verify

```
python -m pytest -q
```

Should report all tests passing with no collection errors (a
collection error naming `pose_landmarker_lite.task` means step 3
didn't complete).

## 5. Run the app

Streamlit (live webcam UI):
```
streamlit run app.py
```

Headless, against a recorded video file (no webcam needed):
```
python app.py --video path/to/clip.mp4 --scorer rules --exercise shoulder_abduction
```

## Camera permissions

The Streamlit UI opens your webcam (`cv2.VideoCapture`) the first time
you click "Start session". If the camera can't be opened, `open_webcam()`
raises a clear `RuntimeError` that the Streamlit UI now catches and
shows via `st.error()` (see `app.py`) -- but the OS still has to grant
camera access to begin with:

- **Windows**: Settings → Privacy & security → Camera → turn on "Camera
  access" and "Let apps access your camera", then make sure "Let
  desktop apps access your camera" is also on. If you still get a
  camera error, check Device Manager for a disabled/missing camera
  driver.
- **macOS**: System Settings → Privacy & Security → Camera → enable
  access for the app you're launching Python from (Terminal, iTerm,
  or your IDE) -- Python itself isn't listed until the first time it
  actually tries to open the camera and macOS prompts you.

## Every gitignored file needed at runtime

| File | Required for | Where to get it |
|---|---|---|
| `models/pose_landmarker_lite.task` | **Everything** -- `src/pose_estimation.py` fails to import without it | Step 3 above |
| `models/{exercise}.pkl` + `models/{exercise}.metadata.json` | Only `--scorer classifier` (the rules scorer, the default, doesn't need these) | Train locally: `python scripts/train_classifier.py` (needs the UI-PRMD dataset under `data/ui_prmd/` -- see `data/README.md`); not published anywhere to download |
| `data/references/<name>.json` | Only `reference_relative` scoring mode | Created by recording a reference: Streamlit's "Record reference" mode, or `src.rules.build_reference()`/`save_reference()` programmatically -- see `docs/figures/reference_relative_scoring.mmd` |
| `logs/*.csv`, `logs/*.summary.json` | Nothing -- these are session **output**, written by `SessionTracker.save()` | Created automatically each session; nothing to provide |
| `results/*` | Nothing -- these are diagnostic/benchmark script **output** | Created automatically when you run a `scripts/*.py` diagnostic |
| `data/clips_manifest.csv`, `data/own_manifest.csv` | Only `scripts/evaluate_rules.py` / `scripts/evaluate_own_clips.py` | You create these yourself, from the templates described in their own docstrings / `docs/recording_protocol.md` |
| `data/rehab24_6/`, `data/rehab24_manifest.csv`, `data/rehab24_subject_split.json` | Only the REHAB24-6 benchmark scripts (`scripts/benchmark_rehab24.py` and friends) | `scripts/download_rehab24.py` + `scripts/build_rehab24_manifest.py` (requires a REHAB24-6 dataset license from its authors -- not redistributed here) |

Everything else the app needs at runtime (`config/*.json`, including
`config/pose.json` and `config/exercises/*.json`) is tracked in git,
not gitignored.
