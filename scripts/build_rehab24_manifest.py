"""PART 2 of the REHAB24-6 benchmark task: builds data/rehab24_manifest.
csv for Exercise 1 (arm abduction) reps, and a fixed, by-subject
calibration/evaluation split, saved to data/rehab24_subject_split.json.
Both outputs are gitignored (generated from the gitignored dataset
download) -- only this script is committed.

Run after scripts/download_rehab24.py has extracted the dataset into
data/rehab24_6/.
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("data/rehab24_6")
SEGMENTATION_CSV = DATA_DIR / "Segmentation.csv"
MANIFEST_PATH = Path("data/rehab24_manifest.csv")
SPLIT_PATH = Path("data/rehab24_subject_split.json")
EXERCISE_ID_ABDUCTION = 1
N_CALIBRATION_SUBJECTS = 3
SPLIT_SEED = 42

# cam17_orientation -> cam18 view, per Segmentation.txt's own mapping
# table (its "side" value is this dataset's actual "profile" value --
# quoted and reconciled in the PART 1 report, not assumed here).
CAM17_TO_CAM18_VIEW = {
    "front": "profile",
    "half-profile": "half-profile",
    "profile": "front",
}


def video_filename(video_id: str, camera: str) -> str:
    """video_id + camera ("cam17"/"cam18") -> the actual filename inside
    videos.zip -- verified against the real extracted archive listing:
    "Ex1/{video_id}-Camera17-30fps.mp4" and "Ex1/{video_id}-Camera18-
    30fps-transposed.mp4" ("transposed": Camera18 is the vertically-
    mounted camera, per the Zenodo description -- its video is rotated
    so the subject appears upright)."""
    if camera == "cam17":
        return f"{video_id}-Camera17-30fps.mp4"
    return f"{video_id}-Camera18-30fps-transposed.mp4"


def build_manifest() -> pd.DataFrame:
    df = pd.read_csv(SEGMENTATION_CSV, sep=";")
    ex1 = df[df["exercise_id"] == EXERCISE_ID_ABDUCTION].copy()

    rows = []
    for _, row in ex1.iterrows():
        for camera, view in (
            ("cam17", row["cam17_orientation"]),
            ("cam18", CAM17_TO_CAM18_VIEW[row["cam17_orientation"]]),
        ):
            rows.append({
                "subject": int(row["person_id"]),
                "video_id": row["video_id"],
                "repetition_number": int(row["repetition_number"]),
                "camera": camera,
                "view": view,
                "video": video_filename(row["video_id"], camera),
                "start_frame": int(row["first_frame"]),
                "end_frame": int(row["last_frame"]),
                "label": "correct" if row["correctness"] == 1 else "incorrect",
                "lights_on": bool(row["lights_on"]),
                "mocap_erroneous": bool(row["mocap_erroneous"]),
            })
    return pd.DataFrame(rows)


def build_split(subjects: list[int]) -> dict:
    rng = random.Random(SPLIT_SEED)
    subjects = sorted(subjects)
    calibration = sorted(rng.sample(subjects, N_CALIBRATION_SUBJECTS))
    evaluation = sorted(s for s in subjects if s not in calibration)

    assert set(calibration).isdisjoint(evaluation), "calibration/evaluation subjects must not overlap"
    assert set(calibration) | set(evaluation) == set(subjects), "every subject must be in exactly one split"

    return {"seed": SPLIT_SEED, "calibration_subjects": calibration, "evaluation_subjects": evaluation}


def main() -> None:
    if not SEGMENTATION_CSV.exists():
        print(f"Missing {SEGMENTATION_CSV} -- run the download step first.")
        return

    manifest = build_manifest()
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"Wrote {MANIFEST_PATH} ({len(manifest)} rows, {manifest['subject'].nunique()} subjects)")

    split = build_split(manifest["subject"].unique().tolist())
    SPLIT_PATH.write_text(json.dumps(split, indent=2))
    print(f"Wrote {SPLIT_PATH}: calibration={split['calibration_subjects']}  evaluation={split['evaluation_subjects']}")


if __name__ == "__main__":
    main()
