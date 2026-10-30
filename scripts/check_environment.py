"""Environment sanity check. Run this after `pip install -r
requirements.txt` (or `requirements-dev.txt`) and downloading the
MediaPipe pose model (see docs/laptop_setup.md), before trying to use
the live app -- it's meant to catch setup problems in one pass instead
of one cryptic error at a time.

Does NOT need anything under data/, and needs only ONE file under
models/ (models/pose_landmarker_lite.task) -- no trained classifier,
no dataset, nothing else.

Checks: Python version (warns if not 3.13.x); every package listed in
requirements.txt, installed vs pinned version; whether the MediaPipe
pose model file exists (prints the download URL if not); cv2.
VideoCapture on indices 0-2 (which open, and the resolution/fps each
actually reports); one real pose-model inference on a blank frame;
pyttsx3 init + speaking a test phrase. Prints a PASS/FAIL/WARN list
with a one-line fix for anything not PASS.
"""

import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
REQUIREMENTS_PATH = ROOT / "requirements.txt"
EXPECTED_PYTHON = (3, 13)

_REQ_LINE_RE = re.compile(r"^([A-Za-z0-9_.-]+)\s*(==\s*([A-Za-z0-9_.+-]+))?")


def check_python_version() -> dict:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == EXPECTED_PYTHON:
        return {"name": "Python version", "status": "PASS", "detail": detail, "fix": ""}
    return {
        "name": "Python version", "status": "WARN", "detail": detail,
        "fix": f"Expected Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}.x -- install that version if you hit import errors.",
    }


def parse_requirements(path: Path) -> list[tuple[str, str | None]]:
    """[(package_name, pinned_version_or_None), ...] -- skips blank
    lines, comments, and `-r other_file.txt` lines."""
    reqs = []
    if not path.exists():
        return reqs
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        m = _REQ_LINE_RE.match(line)
        if not m:
            continue
        reqs.append((m.group(1), m.group(3)))
    return reqs


def check_packages() -> list[dict]:
    results = []
    for name, pin in parse_requirements(REQUIREMENTS_PATH):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            results.append({
                "name": f"Package: {name}", "status": "FAIL",
                "detail": f"installed=NOT INSTALLED pinned={pin or 'unpinned'}",
                "fix": "pip install -r requirements.txt",
            })
            continue
        if pin is not None and installed != pin:
            results.append({
                "name": f"Package: {name}", "status": "WARN",
                "detail": f"installed={installed} pinned={pin}",
                "fix": f"pip install {name}=={pin}",
            })
        else:
            results.append({
                "name": f"Package: {name}", "status": "PASS",
                "detail": f"installed={installed} pinned={pin or 'unpinned'}", "fix": "",
            })
    return results


def check_model_file() -> dict:
    if MODEL_PATH.exists():
        size = MODEL_PATH.stat().st_size
        return {"name": "Pose model file", "status": "PASS", "detail": f"{MODEL_PATH} ({size:,} bytes)", "fix": ""}
    return {
        "name": "Pose model file", "status": "FAIL", "detail": f"missing: {MODEL_PATH}",
        "fix": f"Download from {MODEL_URL} and save to {MODEL_PATH}",
    }


def check_cameras() -> tuple[dict, list[dict]]:
    """Returns (summary_check, per_index_details). The summary check is
    PASS iff at least one of indices 0-2 opened AND returned a
    readable frame -- individual indices beyond the first real camera
    are expected to fail to open on most laptops, so they're reported
    as detail only, not as separate FAILs."""
    import cv2

    details = []
    any_working = False
    for idx in range(3):
        cap = cv2.VideoCapture(idx)
        opened = cap.isOpened()
        if opened:
            ok, _frame = cap.read()
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if ok:
                any_working = True
                details.append({"index": idx, "opened": True, "read_ok": True,
                                 "resolution": f"{int(w)}x{int(h)}", "fps": round(fps, 1)})
            else:
                details.append({"index": idx, "opened": True, "read_ok": False,
                                 "resolution": None, "fps": None})
        else:
            details.append({"index": idx, "opened": False, "read_ok": False, "resolution": None, "fps": None})
        cap.release()

    if any_working:
        summary = {"name": "Camera (at least one of indices 0-2)", "status": "PASS",
                    "detail": "see per-index detail above", "fix": ""}
    else:
        summary = {
            "name": "Camera (at least one of indices 0-2)", "status": "FAIL",
            "detail": "no camera opened and returned a readable frame on indices 0-2",
            "fix": "Check a camera is connected, not in use by another app, and OS camera permission is granted (see docs/laptop_setup.md).",
        }
    return summary, details


def check_pose_model_inference() -> dict:
    try:
        import numpy as np
        from src.pose_estimation import get_pose_landmarks
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = get_pose_landmarks(blank)
        note = "no pose detected on a blank frame, as expected" if landmarks is None else "a pose was (unexpectedly) detected on a blank frame"
        return {"name": "Pose model inference", "status": "PASS", "detail": f"ran one inference -- {note}", "fix": ""}
    except Exception as exc:
        return {"name": "Pose model inference", "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}", "fix": "See the detail above."}


def check_pyttsx3() -> dict:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say("Environment check test phrase.")
        engine.runAndWait()
        return {"name": "Text-to-speech (pyttsx3)", "status": "PASS", "detail": "initialized and spoke a test phrase", "fix": ""}
    except Exception as exc:
        return {
            "name": "Text-to-speech (pyttsx3)", "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}",
            "fix": "Check your OS has a TTS engine available (Windows: SAPI5, built in; macOS: NSSpeechSynthesizer, built in; Linux: install espeak).",
        }


def main() -> None:
    checks = [check_python_version()]
    checks.extend(check_packages())
    checks.append(check_model_file())

    camera_summary, camera_details = check_cameras()
    print("Cameras (indices 0-2):")
    for d in camera_details:
        if d["opened"] and d["read_ok"]:
            print(f"  index {d['index']}: opened, {d['resolution']} @ {d['fps']}fps")
        elif d["opened"]:
            print(f"  index {d['index']}: opened but could not read a frame")
        else:
            print(f"  index {d['index']}: could not open")
    checks.append(camera_summary)

    checks.append(check_pose_model_inference())
    checks.append(check_pyttsx3())

    print("\n" + "=" * 60)
    print("PASS/FAIL SUMMARY")
    print("=" * 60)
    for c in checks:
        line = f"[{c['status']}] {c['name']}: {c['detail']}"
        print(line)
        if c["status"] in ("FAIL", "WARN") and c["fix"]:
            print(f"    fix: {c['fix']}")

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    n_pass = len(checks) - n_fail - n_warn
    print(f"\n{n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
