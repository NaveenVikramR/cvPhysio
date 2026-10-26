"""Trains a RandomForestClassifier per movement (7, 9) on UI-PRMD using
leave-one-subject-out cross-validation (10 folds, one subject held out per
fold -- claude.md requires cross-subject splits, and with only 10 subjects
LOSO uses every subject as a held-out test once instead of wasting most of
them in a single small val/test split).

Compares two feature sources (src/dataset_loader.py's feature_source):
  - fk_positions (default): forward-kinematics-derived absolute joint
    positions fed through the SAME feature_extraction.py functions the
    live app uses. This is what app.py's saved models should come from.
  - angles_file (legacy): a direct Angles-file column. Kept only so the
    two can be compared side by side; models from this path are not what
    app.py needs (see dataset_loader.py's module docstring for why).

Also reports two baselines in the same table: majority-class, and a
single-threshold rule on peak angle (angle_max_deg), with the threshold
chosen on each fold's training subjects only (never on the held-out one).

Usage: python scripts/train_classifier.py [--features fk_positions|angles_file|both]
Requires UI-PRMD downloaded into data/ui_prmd/.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import FEATURE_COLUMNS, GROUP_COLUMN, LABEL_COLUMN, CorrectnessClassifier
from src.dataset_loader import load_dataset_for_exercise

RAW_DIR = "data/ui_prmd"
MODELS_DIR = Path("models")
EXERCISES = ["shoulder_abduction", "shoulder_rotation"]
RANDOM_STATE = 42
PEAK_ANGLE_COLUMN = "angle_max_deg"


def majority_baseline_predict(train_labels, n_test: int) -> np.ndarray:
    majority = train_labels.mode().iloc[0]
    return np.full(n_test, majority)


def threshold_baseline_fit_predict(train_df, test_df, feature_col: str = PEAK_ANGLE_COLUMN):
    """Threshold chosen ONLY from train_df: midpoint between the two
    classes' mean peak angle. Direction (which side of the threshold means
    "correct") is also decided from train_df's class means.
    """
    mean_correct = train_df.loc[train_df[LABEL_COLUMN], feature_col].mean()
    mean_incorrect = train_df.loc[~train_df[LABEL_COLUMN], feature_col].mean()
    threshold = (mean_correct + mean_incorrect) / 2
    correct_is_above = mean_correct > mean_incorrect
    values = test_df[feature_col].to_numpy()
    return (values > threshold) if correct_is_above else (values < threshold)


def class_metrics(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[False, True], zero_division=0
    )
    accuracy = (np.asarray(y_true) == np.asarray(y_pred)).mean()
    return {
        "accuracy": accuracy,
        "precision_incorrect": precision[0], "recall_incorrect": recall[0], "f1_incorrect": f1[0],
        "precision_correct": precision[1], "recall_correct": recall[1], "f1_correct": f1[1],
    }


def summarize(fold_metrics: list[dict]) -> dict:
    keys = fold_metrics[0].keys()
    return {k: (np.mean([m[k] for m in fold_metrics]), np.std([m[k] for m in fold_metrics])) for k in keys}


def print_summary_table(name: str, fold_metrics: list[dict]) -> None:
    summary = summarize(fold_metrics)
    print(f"  {name}:")
    print(f"    accuracy            = {summary['accuracy'][0]:.3f} +/- {summary['accuracy'][1]:.3f}")
    print(f"    recall (incorrect)  = {summary['recall_incorrect'][0]:.3f} +/- {summary['recall_incorrect'][1]:.3f}   <- safety-relevant: catching bad form")
    print(f"    precision(incorrect)= {summary['precision_incorrect'][0]:.3f} +/- {summary['precision_incorrect'][1]:.3f}")
    print(f"    f1       (incorrect)= {summary['f1_incorrect'][0]:.3f} +/- {summary['f1_incorrect'][1]:.3f}")
    print(f"    recall   (correct)  = {summary['recall_correct'][0]:.3f} +/- {summary['recall_correct'][1]:.3f}")
    print(f"    precision(correct)  = {summary['precision_correct'][0]:.3f} +/- {summary['precision_correct'][1]:.3f}")
    print(f"    f1       (correct)  = {summary['f1_correct'][0]:.3f} +/- {summary['f1_correct'][1]:.3f}")


def run_loso(exercise: str, feature_source: str) -> CorrectnessClassifier:
    dataset = load_dataset_for_exercise(RAW_DIR, exercise, modality="kinect", feature_source=feature_source)
    subjects = sorted(dataset[GROUP_COLUMN].unique())
    print(f"\n[{exercise} / {feature_source}] {len(dataset)} reps across {len(subjects)} subjects: {subjects}")
    if len(subjects) < 10:
        print(f"  CAUTION: only {len(subjects)} of the expected 10 UI-PRMD subjects are present.")

    rf_metrics, majority_metrics, threshold_metrics = [], [], []
    rf_cm_total = np.zeros((2, 2), dtype=int)

    for held_out in subjects:
        train_df = dataset[dataset[GROUP_COLUMN] != held_out]
        test_df = dataset[dataset[GROUP_COLUMN] == held_out]

        clf = CorrectnessClassifier(random_state=RANDOM_STATE)
        clf.train(train_df, train_df[LABEL_COLUMN])
        preds = clf.predict_batch(test_df)["is_correct"]

        rf_metrics.append(class_metrics(test_df[LABEL_COLUMN], preds))
        rf_cm_total += confusion_matrix(test_df[LABEL_COLUMN], preds, labels=[False, True])

        maj_preds = majority_baseline_predict(train_df[LABEL_COLUMN], len(test_df))
        majority_metrics.append(class_metrics(test_df[LABEL_COLUMN], maj_preds))

        thr_preds = threshold_baseline_fit_predict(train_df, test_df)
        threshold_metrics.append(class_metrics(test_df[LABEL_COLUMN], thr_preds))

        print(f"  fold held_out={held_out}: RF acc={rf_metrics[-1]['accuracy']:.2f}  "
              f"majority acc={majority_metrics[-1]['accuracy']:.2f}  "
              f"threshold acc={threshold_metrics[-1]['accuracy']:.2f}")

    print(f"\n  --- {exercise} / {feature_source}: LOSO summary (mean +/- std over {len(subjects)} folds) ---")
    print_summary_table("RandomForest", rf_metrics)
    print_summary_table("Baseline: majority class", majority_metrics)
    print_summary_table("Baseline: single threshold on peak angle", threshold_metrics)

    print(f"\n  Pooled confusion matrix (RandomForest, summed over {len(subjects)} folds):")
    print(f"                 pred_incorrect  pred_correct")
    print(f"    true_incorrect   {rf_cm_total[0,0]:>10d}    {rf_cm_total[0,1]:>10d}")
    print(f"    true_correct     {rf_cm_total[1,0]:>10d}    {rf_cm_total[1,1]:>10d}")

    final_clf = CorrectnessClassifier(random_state=RANDOM_STATE)
    final_clf.train(dataset, dataset[LABEL_COLUMN])
    return final_clf, summarize(rf_metrics), subjects


def save_model_with_metadata(clf: CorrectnessClassifier, exercise: str, feature_source: str,
                              loso_summary: dict, subjects: list) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{exercise}.pkl"
    clf.save(model_path)

    metadata = {
        "exercise": exercise,
        "feature_names": FEATURE_COLUMNS,
        "feature_source": feature_source,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "loso_summary": {k: {"mean": v[0], "std": v[1]} for k, v in loso_summary.items()},
    }
    metadata_path = MODELS_DIR / f"{exercise}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"\n  Saved model to {model_path}")
    print(f"  Saved metadata to {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=["fk_positions", "angles_file", "both"], default="both")
    args = parser.parse_args()

    sources = ["fk_positions", "angles_file"] if args.features == "both" else [args.features]

    for exercise in EXERCISES:
        print(f"\n{'=' * 70}\n{exercise}\n{'=' * 70}")
        for feature_source in sources:
            clf, loso_summary, subjects = run_loso(exercise, feature_source)
            if feature_source == "fk_positions" or args.features == "angles_file":
                save_model_with_metadata(clf, exercise, feature_source, loso_summary, subjects)
            else:
                print(f"\n  (not saving a model for {feature_source} -- app.py needs fk_positions; "
                      f"this run was for comparison only)")


if __name__ == "__main__":
    main()
