"""Report figure A1.2: one real frame with the prototype's detected landmarks and
skeleton drawn on it. Uses src.pose_estimation, so the model and landmark_space
are the same as the app's. Read-only apart from the output image.

    python scripts/render_landmark_figure.py                    # camera, with preview
    python scripts/render_landmark_figure.py path/to/image.png  # a saved image

Camera: SPACE starts a 5 s countdown, then one frame is taken. ESC quits. Only that
annotated frame is saved, to results/figures/A1_2_landmarks.png (gitignored).

The right shoulder, elbow, wrist and hip are highlighted, and the printed angles
are the right arm's, so raise the right arm. The arm the scorer would select is
stated too, because it picks by visibility and may not be the right one.

If no pose is detected, or a highlighted landmark is below the 0.5 visibility gate,
the script says so and saves nothing. It never draws anything synthetic.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.feature_extraction import LEFT, MIN_VISIBILITY, RIGHT, compute_angle, compute_trunk_lean_deg, select_side

OUTPUT_PATH = REPO_ROOT / "results" / "figures" / "A1_2_landmarks.png"
COUNTDOWN_S = 5
KEY_ESC = 27

# MediaPipe's pose connections for the body, without the face detail.
SKELETON_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]
HIGHLIGHT_NAMES = {"shoulder": "R shoulder", "elbow": "R elbow", "wrist": "R wrist", "hip": "R hip"}
HIGHLIGHT_COLOUR = (0, 215, 255)
EDGE_COLOUR = (0, 255, 0)
POINT_COLOUR = (255, 255, 255)


def right_arm_problems(landmarks: list) -> list[str]:
    """Highlighted landmarks below the visibility gate, as "name (0.42)"."""
    return [
        f"{HIGHLIGHT_NAMES[role]} ({landmarks[RIGHT[role]][3]:.2f})"
        for role in HIGHLIGHT_NAMES if landmarks[RIGHT[role]][3] < MIN_VISIBILITY
    ]


def figure_values(landmarks: list) -> dict:
    """Right-arm angles and trunk lean, through the prototype's own functions, from
    the configured landmark space. Trunk lean is None if either side's shoulder
    or hip is below the gate, because it uses both sides."""
    shoulder, elbow, wrist, hip = (landmarks[RIGHT[role]] for role in ("shoulder", "elbow", "wrist", "hip"))
    trunk_points = [landmarks[side[role]] for side in (LEFT, RIGHT) for role in ("shoulder", "hip")]
    return {
        "shoulder_deg": compute_angle(hip, shoulder, elbow),
        "elbow_deg": compute_angle(shoulder, elbow, wrist),
        "trunk_deg": compute_trunk_lean_deg(landmarks) if all(p[3] >= MIN_VISIBILITY for p in trunk_points) else None,
        "scorer_side": "left" if select_side(landmarks) is LEFT else "right",
    }


def _pixel(point, width: int, height: int) -> tuple[int, int]:
    return int(point[0] * width), int(point[1] * height)


def annotate(frame, normalized: list, values: dict):
    """A copy of frame with the skeleton, highlights and values drawn on it.
    normalized supplies image positions only; the values are already computed."""
    image = frame.copy()
    height, width = image.shape[:2]
    for a, b in SKELETON_EDGES:
        if normalized[a][3] >= MIN_VISIBILITY and normalized[b][3] >= MIN_VISIBILITY:
            cv2.line(image, _pixel(normalized[a], width, height), _pixel(normalized[b], width, height), EDGE_COLOUR, 2, cv2.LINE_AA)
    for point in normalized:
        if point[3] >= MIN_VISIBILITY:
            cv2.circle(image, _pixel(point, width, height), 3, POINT_COLOUR, -1, cv2.LINE_AA)
    for role, name in HIGHLIGHT_NAMES.items():
        x, y = _pixel(normalized[RIGHT[role]], width, height)
        cv2.circle(image, (x, y), 8, HIGHLIGHT_COLOUR, 2, cv2.LINE_AA)
        cv2.putText(image, name, (x + 11, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, name, (x + 11, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, HIGHLIGHT_COLOUR, 1, cv2.LINE_AA)

    trunk = "n/a (below gate)" if values["trunk_deg"] is None else f"{values['trunk_deg']:.1f} deg"
    lines = [
        f"Right shoulder angle: {values['shoulder_deg']:.1f} deg",
        f"Right elbow angle: {values['elbow_deg']:.1f} deg",
        f"Trunk lean: {trunk}",
        f"Arm the scorer would select: {values['scorer_side']}",
    ]
    cv2.rectangle(image, (0, 0), (350, 24 * len(lines) + 8), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (8, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def capture_frame(pose_estimation, camera_index: int):
    """Preview, SPACE, countdown, one frame. None if ESC was pressed."""
    from scripts.record_clip import draw_preview

    capture = pose_estimation.open_webcam(camera_index)
    deadline = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames")
            if deadline is not None and time.monotonic() >= deadline:
                return frame
            normalized, _world = pose_estimation.get_pose_landmarks_pair(frame)
            status = ("Raise right arm to shoulder height. SPACE = start, ESC = quit" if deadline is None
                      else f"Hold still... {int(deadline - time.monotonic()) + 1}")
            cv2.imshow("render_landmark_figure", draw_preview(frame, normalized, status, (0, 255, 255)))
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESC:
                return None
            if key == ord(" ") and deadline is None:
                deadline = time.monotonic() + COUNTDOWN_S
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render report figure A1.2 from one real frame.")
    parser.add_argument("image", nargs="?", type=Path, default=None, help="saved image (default: capture from the camera)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from src import pose_estimation  # loads the pose model at import time

    if args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            print(f"Could not read an image from {args.image}. Nothing saved.")
            return 1
    else:
        frame = capture_frame(pose_estimation, args.camera)
        if frame is None:
            print("Quit. Nothing saved.")
            return 1

    normalized, world = pose_estimation.get_pose_landmarks_pair(frame)
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    landmarks = world if landmark_space == "world" else normalized
    if normalized is None or landmarks is None:
        print("No pose detected in this frame. Nothing saved.")
        return 1
    problems = right_arm_problems(landmarks)
    if problems:
        print(f"Below the {MIN_VISIBILITY} visibility gate: {', '.join(problems)}. Nothing saved. "
              "Check the framing and light, then try again.")
        return 1

    values = figure_values(landmarks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotate(frame, normalized, values)):
        print(f"Could not write {args.output}.")
        return 1
    trunk = "n/a" if values["trunk_deg"] is None else f"{values['trunk_deg']:.1f}"
    print(f"Model: {pose_estimation.configured_pose_model()}  landmark_space: {landmark_space}")
    print(f"Right shoulder {values['shoulder_deg']:.1f} deg, right elbow {values['elbow_deg']:.1f} deg, "
          f"trunk lean {trunk} deg. Scorer would select: {values['scorer_side']}.")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
