"""Records one clip with a guided live preview, following
docs/recording_protocol.md (and docs/RECORDING_GUIDE.md for team recordings).

Two datasets, chosen with --dataset:
  team (default): any exercise (--exercise abduction|rotation) and camera
    view (--view front|45|side). Writes data/recordings/team/<clip>.mp4/.json
    and a row in data/team_manifest.csv. Never touches the pre-registered
    manifest -- save_take() refuses to.
  preregistered: the original own-clip set for scripts/evaluate_own_clips.py
    (front-view abduction only). Writes data/recordings/<clip>.mp4/.json and
    a row in data/own_manifest.csv. That evaluation's clips are final, so
    this is kept only for completeness and must be asked for explicitly.

Capture goes through src.pose_estimation.open_webcam(), the same call the
live app makes -- it requests no resolution or fps, so the clip has whatever
the camera's default mode delivers. The actual values are what the sidecar
records.

Flow: live preview (framing box, skeleton, warnings) -> SPACE starts the
countdown (--countdown, default 15 s) -> records until 'q' or --seconds. ESC at any point
discards the take and writes nothing.

The saved video holds the clean camera frames, never the preview overlays.
Frames are spooled uncompressed while recording and encoded afterwards,
because an mp4's fps is fixed when its writer is opened and the measured fps
isn't known until recording ends. Pose runs on a separate thread so inference
can't lower the recording frame rate.

Everything it writes is gitignored. No network use.
"""

import argparse
import csv
import json
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.feature_extraction import LEFT, RIGHT, compute_angle

RECORDINGS_DIR = REPO_ROOT / "data" / "recordings"
MANIFEST_PATH = REPO_ROOT / "data" / "own_manifest.csv"

ROLES = ("reference", "calibration", "eval")
INTENDED = ("correct", "fault_lowarm", "fault_trunk", "fault_elbow", "fault_fast")
MANIFEST_COLUMNS = ["clip", "role", "intended", "expected_reps", "subject", "notes"]

# Team dataset (docs/RECORDING_GUIDE.md).
TEAM_RECORDINGS_DIR = RECORDINGS_DIR / "team"
TEAM_MANIFEST_PATH = REPO_ROOT / "data" / "team_manifest.csv"
TEAM_MANIFEST_COLUMNS = ["clip", "exercise", "view", "role", "intended", "expected_reps", "subject", "notes"]
DATASETS = ("team", "preregistered")
EXERCISES = ("abduction", "rotation")
EXERCISE_PREFIX = {"abduction": "abd", "rotation": "rot"}
VIEWS = ("front", "45", "side")
# Team roles name the subject's group, not the take's purpose:
#   test        camera-angle and practice takes, correct reps only, never scored
#   calibration a subject whose clips may be looked at and tuned on
#   eval        a held-out subject: nobody looks at or scores these until the
#               code is frozen and the evaluation is pre-registered
TEAM_ROLES = ("test", "calibration", "eval")
INTENDED_BY_EXERCISE = {
    "abduction": INTENDED,
    "rotation": ("correct", "fault_short_out", "fault_short_in", "fault_elbow_drift", "fault_trunk_twist",
                 "fault_fast"),
}
SUBJECT_RE = re.compile(r"[A-Za-z0-9-]+")  # goes into the filename; no "_" so names stay parseable
START_WARNING = {"abduction": "ARMS UP - LOWER YOUR ARMS NOW", "rotation": "KEEP YOUR UPPER ARM AT YOUR SIDE"}

DEFAULT_CAMERA_INDEX = 0  # open_webcam()'s default, which is what app.py calls
DEFAULT_COUNTDOWN_S = 15.0  # time to walk from the keyboard to the recording spot
MIN_VISIBILITY = 0.5  # same threshold as the confidence gate
FRAME_MARGIN_FRAC = 0.05  # framing box inset, as a fraction of each dimension
ARMS_UP_DEG = 45.0  # same limit as scripts/qc_clip.py's arms_down check
ARMS_DOWN_START_S = 1.5  # how far into the recording the arms-up warning stays active

WRISTS = {"left wrist": 15, "right wrist": 16}
# Everything the rules scorer needs (src.rules.REQUIRED_ROLES_BY_RULE).
FRAMING_LANDMARKS = {
    "left shoulder": 11, "right shoulder": 12, "left elbow": 13, "right elbow": 14,
    "left wrist": 15, "right wrist": 16, "left hip": 23, "right hip": 24,
}
SKELETON_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]


# ---------------------------------------------------------------------------
# Naming, sidecar and manifest (no camera, no MediaPipe)
# ---------------------------------------------------------------------------

def clip_stem(role: str, intended: str, index: int) -> str:
    return f"abd_{role}_{intended}_{index:02d}"


def team_stem_base(exercise: str, view: str, role: str, intended: str, subject: str) -> str:
    return f"{EXERCISE_PREFIX[exercise]}_{view}_{role}_{intended}_{subject}"


def next_index(recordings_dir: Path, stem_base: str) -> int:
    """One past the highest index already used for stem_base, so a deleted
    earlier take never gets its number reused. Counts sidecars too, in case
    an .mp4 was removed by hand and its .json left behind."""
    pattern = re.compile(rf"{re.escape(stem_base)}_(\d{{2,}})\.(?:mp4|json)")
    used = [
        int(match.group(1))
        for match in (pattern.fullmatch(p.name) for p in recordings_dir.glob(f"{stem_base}_*"))
        if match
    ]
    return max(used, default=0) + 1


def next_clip_index(recordings_dir: Path, role: str, intended: str) -> int:
    return next_index(recordings_dir, f"abd_{role}_{intended}")


def manifest_clip_path(video_path: Path, root: Path = REPO_ROOT) -> str:
    """Repo-relative, because evaluate_own_clips.py opens the manifest's
    clip column relative to the repo root it runs from."""
    return video_path.relative_to(root).as_posix()


def check_manifest_header(manifest_path: Path, columns: list[str] = MANIFEST_COLUMNS) -> None:
    """Raises rather than appending misaligned columns to an existing file."""
    if not manifest_path.exists():
        return
    with manifest_path.open(newline="") as f:
        header = next(csv.reader(f), None)
    if header != columns:
        raise ValueError(f"{manifest_path} has header {header}; expected {columns}")


def append_manifest_row(manifest_path: Path, row: dict, columns: list[str] = MANIFEST_COLUMNS) -> None:
    check_manifest_header(manifest_path, columns)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not manifest_path.exists() or manifest_path.stat().st_size == 0
    if not is_new and not manifest_path.read_bytes().endswith(b"\n"):
        with manifest_path.open("ab") as f:  # hand-edited file with no final newline
            f.write(b"\r\n")
    with manifest_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def measured_fps(timestamps: list[float]) -> float:
    """Frame intervals over elapsed time between the first and last frame."""
    if len(timestamps) < 2 or timestamps[-1] <= timestamps[0]:
        raise ValueError("Need at least 2 timestamped frames to measure fps")
    return (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])


def encode_spool(spool_path: Path, video_path: Path, width: int, height: int, fps: float) -> int:
    """Encodes a raw BGR spool file to mp4 at fps. Returns frames written."""
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for {video_path}")
    frame_bytes = width * height * 3
    n_frames = 0
    try:
        with spool_path.open("rb") as spool:
            while len(data := spool.read(frame_bytes)) == frame_bytes:
                writer.write(np.frombuffer(data, np.uint8).reshape(height, width, 3))
                n_frames += 1
    finally:
        writer.release()
    return n_frames


def dataset_paths(dataset: str) -> tuple[Path, Path]:
    """(recordings_dir, manifest_path) for a dataset."""
    if dataset == "preregistered":
        return RECORDINGS_DIR, MANIFEST_PATH
    return TEAM_RECORDINGS_DIR, TEAM_MANIFEST_PATH


def save_take(spool_path: Path, timestamps: list[float], width: int, height: int, args: argparse.Namespace,
              started_at: str, recordings_dir: Path | None = None, manifest_path: Path | None = None,
              root: Path = REPO_ROOT) -> Path:
    """mp4, then sidecar, then manifest row -- the row goes last so the
    manifest never lists a clip that failed to save. A team take is refused
    outright if it would land in the pre-registered manifest."""
    dataset = getattr(args, "dataset", "preregistered")
    default_dir, default_manifest = dataset_paths(dataset)
    recordings_dir = recordings_dir or default_dir
    manifest_path = manifest_path or default_manifest
    team = dataset == "team"
    if team and (manifest_path.resolve() == MANIFEST_PATH.resolve() or manifest_path.name == MANIFEST_PATH.name):
        raise ValueError(f"Team recordings must never be written to the pre-registered {MANIFEST_PATH.name}")
    recordings_dir.mkdir(parents=True, exist_ok=True)
    fps = measured_fps(timestamps)
    if team:
        base = team_stem_base(args.exercise, args.view, args.role, args.intended, args.subject)
        stem = f"{base}_{next_index(recordings_dir, base):02d}"
    else:
        stem = clip_stem(args.role, args.intended, next_clip_index(recordings_dir, args.role, args.intended))
    video_path = recordings_dir / f"{stem}.mp4"
    n_frames = encode_spool(spool_path, video_path, width, height, fps)
    if n_frames != len(timestamps):
        raise RuntimeError(f"Encoded {n_frames} frames but captured {len(timestamps)}")
    sidecar = {
        "clip": video_path.name,
        "measured_fps": fps,
        "resolution": [width, height],
        "frames": n_frames,
        "start_time": started_at,
        "camera_index": args.camera,
        "notes": args.notes,
    }
    row = {
        "clip": manifest_clip_path(video_path, root),
        "role": args.role,
        "intended": args.intended,
        "expected_reps": args.expected_reps,
        "subject": args.subject,
        "notes": args.notes,
    }
    if team:
        sidecar.update(dataset="team", exercise=args.exercise, view=args.view, subject=args.subject,
                       role=args.role, intended=args.intended, expected_reps=args.expected_reps)
        row.update(exercise=args.exercise, view=args.view)
    (recordings_dir / f"{stem}.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    append_manifest_row(manifest_path, row, TEAM_MANIFEST_COLUMNS if team else MANIFEST_COLUMNS)
    return video_path


# ---------------------------------------------------------------------------
# Preview warnings and drawing
# ---------------------------------------------------------------------------

def low_wrist_warnings(landmarks: list | None) -> list[str]:
    if landmarks is None:
        return ["NO POSE DETECTED"]
    return [
        f"LOW VISIBILITY: {name} {landmarks[i][3]:.2f}"
        for name, i in WRISTS.items() if landmarks[i][3] < MIN_VISIBILITY
    ]


def arms_up(landmarks: list | None) -> bool:
    """Either arm at or above ARMS_UP_DEG. The rep detector takes its rest baseline
    from the clip's first frames, so a raised arm there inverts rep detection.
    Either arm counts, because the scorer may select either side."""
    if landmarks is None:
        return False
    return any(
        compute_angle(landmarks[joints["hip"]], landmarks[joints["shoulder"]], landmarks[joints["elbow"]]) >= ARMS_UP_DEG
        for joints in (LEFT, RIGHT)
    )


def outside_box(landmarks: list | None) -> list[str]:
    """Required landmarks outside the framing box. Checked separately from
    visibility because MediaPipe can report a usable visibility score for a
    joint that is off-image altogether."""
    if landmarks is None:
        return []
    lo, hi = FRAME_MARGIN_FRAC, 1.0 - FRAME_MARGIN_FRAC
    return [
        name for name, i in FRAMING_LANDMARKS.items()
        if not (lo <= landmarks[i][0] <= hi and lo <= landmarks[i][1] <= hi)
    ]


def draw_preview(frame, landmarks: list | None, status: str, status_colour: tuple, arms_up_warning: bool = False,
                 warning_text: str = START_WARNING["abduction"]) -> np.ndarray:
    view = frame.copy()
    height, width = view.shape[:2]
    margin_x, margin_y = int(width * FRAME_MARGIN_FRAC), int(height * FRAME_MARGIN_FRAC)
    cv2.rectangle(view, (margin_x, margin_y), (width - margin_x, height - margin_y), (255, 255, 255), 1)
    if landmarks is not None:
        points = {i: (int(landmarks[i][0] * width), int(landmarks[i][1] * height)) for i in FRAMING_LANDMARKS.values()}
        for a, b in SKELETON_EDGES:
            cv2.line(view, points[a], points[b], (0, 255, 0), 2)
        for point in points.values():
            cv2.circle(view, point, 4, (0, 255, 255), -1)
    y = 24
    cv2.putText(view, status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_colour, 2)
    for warning in low_wrist_warnings(landmarks):
        y += 24
        cv2.putText(view, warning, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    missing = outside_box(landmarks)
    if missing:
        y += 24
        cv2.putText(view, "OUT OF BOX: " + ", ".join(missing), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    if arms_up_warning:
        cv2.rectangle(view, (0, height - 44), (width, height), (0, 0, 255), -1)
        cv2.putText(view, warning_text, (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return view


class PoseWorker(threading.Thread):
    """Runs pose on the newest frame only, skipping any it can't keep up with."""

    def __init__(self, estimate):
        super().__init__(daemon=True)
        self._estimate = estimate
        self._frame = None
        self._wake = threading.Event()
        self._stopping = False
        self.landmarks = None  # image space, for drawing
        self.angle_landmarks = None  # the configured landmark space, for angles

    def submit(self, frame) -> None:
        self._frame = frame
        self._wake.set()

    def run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stopping:
                return
            self.landmarks, self.angle_landmarks = self._estimate(self._frame)

    def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        self.join()


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

WINDOW = "record_clip"
KEY_ESC = 27


def show(frame, worker: PoseWorker, status: str, colour: tuple, check_arms: bool = True,
         exercise: str = "abduction") -> int:
    """The start-position check is the same for both exercises -- the upper
    arm hangs at the side (hip-shoulder-elbow below ARMS_UP_DEG) -- only the
    wording differs, since a rotation starts with the elbow bent."""
    worker.submit(frame)
    warning = check_arms and arms_up(worker.angle_landmarks)
    cv2.imshow(WINDOW, draw_preview(frame, worker.landmarks, status, colour, warning,
                                    warning_text=START_WARNING[exercise]))
    return cv2.waitKey(1) & 0xFF


def wait_for_start(capture, worker: PoseWorker, countdown_s: float = DEFAULT_COUNTDOWN_S,
                   exercise: str = "abduction") -> bool:
    """Framing preview, then countdown. False if the take was discarded."""
    while True:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Camera stopped delivering frames")
        key = show(frame, worker, "SPACE = start countdown, ESC = quit", (255, 255, 255), exercise=exercise)
        if key == KEY_ESC:
            return False
        if key == ord(" "):
            break
    deadline = time.monotonic() + countdown_s
    while (remaining := deadline - time.monotonic()) > 0:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Camera stopped delivering frames")
        if show(frame, worker, f"Starting in {int(remaining) + 1}... (ESC = quit)", (0, 255, 255),
                exercise=exercise) == KEY_ESC:
            return False
    return True


def record_frames(capture, worker: PoseWorker, spool_path: Path, seconds: float | None,
                  exercise: str = "abduction"):
    """Returns (timestamps, width, height), or None if discarded with ESC."""
    timestamps = []
    width = height = None
    with spool_path.open("wb") as spool:
        while seconds is None or not timestamps or timestamps[-1] - timestamps[0] < seconds:
            ok, frame = capture.read()
            if not ok:
                break
            now = time.monotonic()
            if width is None:
                height, width = frame.shape[:2]
            spool.write(frame.tobytes())
            timestamps.append(now)
            elapsed = now - timestamps[0]
            status = f"REC {elapsed:5.1f}s" + (f" / {seconds:g}s" if seconds else "") + "  q = stop, ESC = discard"
            key = show(frame, worker, status, (0, 0, 255), check_arms=elapsed < ARMS_DOWN_START_S, exercise=exercise)
            if key == KEY_ESC:
                return None
            if key == ord("q"):
                break
    return timestamps, width, height


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one clip with a guided preview (docs/RECORDING_GUIDE.md).")
    parser.add_argument("--dataset", choices=DATASETS, default="team",
                        help="team (default) or preregistered (the frozen abduction own-clip set)")
    parser.add_argument("--exercise", choices=EXERCISES, default="abduction")
    parser.add_argument("--view", choices=VIEWS, default="front", help="camera angle to the subject")
    parser.add_argument("--role", required=True, choices=sorted(set(ROLES) | set(TEAM_ROLES)))
    parser.add_argument("--intended", required=True,
                        choices=sorted({i for values in INTENDED_BY_EXERCISE.values() for i in values}))
    parser.add_argument("--expected-reps", required=True, type=int)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--seconds", type=float, default=None, help="stop automatically after this long")
    parser.add_argument("--countdown", type=float, default=DEFAULT_COUNTDOWN_S,
                        help=f"seconds between SPACE and recording (default {DEFAULT_COUNTDOWN_S:g}; the old default was 5)")
    args = parser.parse_args(argv)
    if args.countdown < 0:
        parser.error("--countdown can't be negative")
    if args.dataset == "preregistered":
        if args.exercise != "abduction" or args.view != "front":
            parser.error("--dataset preregistered is front-view abduction only")
        if args.role not in ROLES:
            parser.error(f"--dataset preregistered roles are {ROLES}")
    else:
        if args.role not in TEAM_ROLES:
            parser.error(f"team roles are {TEAM_ROLES}")
        if args.intended not in INTENDED_BY_EXERCISE[args.exercise]:
            parser.error(f"--intended for {args.exercise} must be one of {INTENDED_BY_EXERCISE[args.exercise]}")
        if not SUBJECT_RE.fullmatch(args.subject):
            parser.error("--subject may use only letters, digits and '-' (it goes into the filename)")
        # In the team set the role is the subject's group: calibration takes
        # (tuning allowed) need faults too, so only "test" is correct-only.
        if args.role == "test" and args.intended != "correct":
            parser.error("--role test must use --intended correct (docs/RECORDING_GUIDE.md)")
    if args.dataset == "preregistered" and args.role != "eval" and args.intended != "correct":
        parser.error(f"--role {args.role} must use --intended correct (docs/recording_protocol.md)")
    if args.expected_reps < 1:
        parser.error("--expected-reps must be at least 1")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    if not (args.subject + args.notes).isascii():
        # evaluate_own_clips.py reads the manifest in the locale's encoding.
        parser.error("--subject and --notes must be ASCII")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    recordings_dir, manifest_path = dataset_paths(args.dataset)
    columns = MANIFEST_COLUMNS if args.dataset == "preregistered" else TEAM_MANIFEST_COLUMNS
    check_manifest_header(manifest_path, columns)
    recordings_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "preregistered":
        print("WARNING: --dataset preregistered appends to the frozen own-clip manifest "
              "(docs/preregistered_own_clip_eval.md). Press ESC in the preview to back out.")

    # Imported here because it loads the pose model at import time.
    from src.pose_estimation import get_pose_landmarks_pair, load_config, open_webcam

    landmark_space = load_config().get("landmark_space", "world")

    def estimate(frame):
        normalized, world = get_pose_landmarks_pair(frame)
        return normalized, world if landmark_space == "world" else normalized

    capture = open_webcam(args.camera)
    worker = PoseWorker(estimate)
    worker.start()
    spool_path = recordings_dir / ".record_clip_spool.raw"
    try:
        if not wait_for_start(capture, worker, args.countdown, args.exercise):
            print("Discarded before recording. Nothing was saved.")
            return 1
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        recorded = record_frames(capture, worker, spool_path, args.seconds, args.exercise)
        capture.release()
        cv2.destroyAllWindows()
        if recorded is None:
            print("Take discarded. Nothing was saved.")
            return 1
        timestamps, width, height = recorded
        video_path = save_take(spool_path, timestamps, width, height, args, started_at)
    finally:
        worker.stop()
        capture.release()
        cv2.destroyAllWindows()
        spool_path.unlink(missing_ok=True)

    print(f"Saved {video_path.relative_to(REPO_ROOT)}: {len(timestamps)} frames, {width}x{height}, "
          f"{measured_fps(timestamps):.2f} fps measured. Manifest row added to {manifest_path.relative_to(REPO_ROOT)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
