"""Trains one RandomForestClassifier per movement (7, 9) on UI-PRMD,
cross-subject split, reports metrics on validation then test, saves models
to models/{exercise}.pkl for app.py to load.

Usage: python scripts/train_classifier.py
Requires UI-PRMD downloaded into data/ui_prmd/ -- see src/dataset_loader.py
for the (currently unverified) expected layout.
"""

import sys
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import GROUP_COLUMN, LABEL_COLUMN, CorrectnessClassifier, cross_subject_split
from src.dataset_loader import load_dataset_for_exercise

RAW_DIR = "data/ui_prmd"
MODELS_DIR = Path("models")
EXERCISES = ["shoulder_abduction", "shoulder_rotation"]
RANDOM_STATE = 42


def split_train_val_test(dataset, random_state: int = RANDOM_STATE):
    """~10 subjects -> ~6 train / 2 val / 2 test, grouped by subject_id.
    Reuses cross_subject_split() twice: first carve off ~20% as test, then
    split the remainder ~75/25 into train/val.
    """
    train_val_df, test_df = cross_subject_split(dataset, test_size=0.2, random_state=random_state)
    train_df, val_df = cross_subject_split(train_val_df, test_size=0.25, random_state=random_state)
    return train_df, val_df, test_df


def report_split(name: str, split_df, clf: CorrectnessClassifier) -> None:
    preds = clf.predict_batch(split_df)
    print(f"\n[{name}] {split_df[GROUP_COLUMN].nunique()} subjects, {len(split_df)} reps")
    print(classification_report(split_df[LABEL_COLUMN], preds["is_correct"]))
    print("Confusion matrix (rows=true [False,True], cols=pred [False,True]):")
    print(confusion_matrix(split_df[LABEL_COLUMN], preds["is_correct"]))


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for exercise in EXERCISES:
        print(f"\n{'=' * 60}\n{exercise}\n{'=' * 60}")
        dataset = load_dataset_for_exercise(raw_dir=RAW_DIR, exercise=exercise, modality="kinect")
        train_df, val_df, test_df = split_train_val_test(dataset)

        print(f"Train subjects ({train_df[GROUP_COLUMN].nunique()}): {sorted(train_df[GROUP_COLUMN].unique())}")
        print(f"Val subjects   ({val_df[GROUP_COLUMN].nunique()}): {sorted(val_df[GROUP_COLUMN].unique())}")
        print(f"Test subjects  ({test_df[GROUP_COLUMN].nunique()}): {sorted(test_df[GROUP_COLUMN].unique())}")

        if val_df[GROUP_COLUMN].nunique() <= 2 or test_df[GROUP_COLUMN].nunique() <= 2:
            print(
                f"CAUTION: validation ({val_df[GROUP_COLUMN].nunique()} subjects) and/or "
                f"test ({test_df[GROUP_COLUMN].nunique()} subjects) groups are very small "
                "(only ~10 subjects total in UI-PRMD). Treat the metrics below as noisy "
                "point estimates, not reliable generalization measures."
            )

        clf = CorrectnessClassifier(random_state=RANDOM_STATE)
        clf.train(train_df, train_df[LABEL_COLUMN])

        report_split("validation", val_df, clf)
        report_split("test", test_df, clf)

        model_path = MODELS_DIR / f"{exercise}.pkl"
        clf.save(model_path)
        print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()
