"""Evaluation utilities: the six binary metrics, confusion matrix, classification
report and the five-model comparison table.

Used by ``src/train.py`` (to score the models after training) and by ``app.py``
(to score the saved models on an uploaded CSV). The same functions are used in
both places, so the app reproduces the numbers stored in ``model/metadata.json``.

Run ``python -m src.evaluate`` to reload the saved models, re-score them on
``test_data.csv`` and check the results against ``metadata.json``.

Positive class: ``yes`` (encoded as 1) for every metric.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.preprocess import (
    CLASS_LABELS,
    METADATA_PATH,
    MODEL_DIR,
    TEST_DATA_PATH,
    prepare_features_and_target,
    read_csv_auto,
)

# Display name -> saved file (order = order shown in tables and the app)
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
METRIC_KEYS = {"Accuracy": "accuracy", "AUC": "auc", "Precision": "precision",
               "Recall": "recall", "F1": "f1", "MCC": "mcc"}


# --------------------------------------------------------------------------- #
# Scores and predictions
# --------------------------------------------------------------------------- #
def get_scores(model, X: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return a ranking score for the positive class and where it came from.

    * models with ``predict_proba``: P(y = yes)
    * SVC (trained without ``probability=True``): ``decision_function`` distance.
      ROC-AUC depends only on the ranking of the scores, so this is valid for AUC.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1], "predict_proba"
    return model.decision_function(X), "decision_function"


def predict_from_scores(scores: np.ndarray, score_source: str) -> np.ndarray:
    """Turn scores into 0/1 labels exactly as ``model.predict`` does.

    * probabilities: argmax of [P(no), P(yes)], i.e. yes only when P(yes) > 0.5
    * binary SVC: yes when decision_function > 0
    Deriving the labels from the scores avoids a second (slow) pass through the SVM.
    """
    threshold = 0.5 if score_source == "predict_proba" else 0.0
    return (np.asarray(scores) > threshold).astype(int)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """The six required metrics, with 'yes' (=1) as the positive class."""
    y_true = np.asarray(y_true)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(auc),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def confusion_matrix_dict(y_true, y_pred) -> dict[str, int]:
    """TN / FP / FN / TP counts (rows = actual, columns = predicted, order no, yes)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    """sklearn's classification report as a DataFrame (rows: no, yes, accuracy, averages)."""
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_LABELS,
        output_dict=True, zero_division=0,
    )
    return pd.DataFrame(report).T


def evaluate_model(model, X: pd.DataFrame, y_true) -> dict:
    """Score one fitted pipeline on (X, y). Never refits anything."""
    scores, source = get_scores(model, X)
    y_pred = predict_from_scores(scores, source)
    return {
        "metrics": compute_metrics(y_true, y_pred, scores),
        "confusion_matrix": confusion_matrix_dict(y_true, y_pred),
        "report": classification_report_df(y_true, y_pred),
        "y_pred": y_pred,
        "scores": scores,
        "score_source": source,
    }


def comparison_table(results: dict[str, dict], decimals: int | None = 4) -> pd.DataFrame:
    """Build the assignment's comparison table from ``{model name: evaluate_model result}``."""
    rows = []
    for name, res in results.items():
        metrics = res["metrics"] if "metrics" in res else res
        rows.append({"ML Model Name": name, **{m: metrics[METRIC_KEYS[m]] for m in METRIC_NAMES}})
    table = pd.DataFrame(rows).set_index("ML Model Name")
    return table.round(decimals) if decimals is not None else table


# --------------------------------------------------------------------------- #
# Loading saved artifacts
# --------------------------------------------------------------------------- #
def load_models(model_dir: str | Path = MODEL_DIR) -> dict:
    """Load the five saved pipelines (raises FileNotFoundError if any is missing)."""
    model_dir = Path(model_dir)
    missing = [f for f in MODEL_FILES.values() if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model file(s) in {model_dir}: {', '.join(missing)}. Run `python -m src.train` first.")
    return {name: joblib.load(model_dir / fname) for name, fname in MODEL_FILES.items()}


def load_metadata(path: str | Path = METADATA_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# CLI: reload saved models and verify against metadata.json
# --------------------------------------------------------------------------- #
def main(tolerance: float = 1e-9) -> int:
    try:
        models = load_models()
        metadata = load_metadata()
    except FileNotFoundError as exc:
        print(f"[evaluate] {exc}")
        return 1

    X_test, y_test = prepare_features_and_target(read_csv_auto(TEST_DATA_PATH))
    print(f"Re-scoring {len(models)} saved models on {TEST_DATA_PATH.name}: "
          f"{len(X_test):,} rows, {int(y_test.sum()):,} positive ('yes')\n")

    results, all_ok = {}, True
    for name, model in models.items():
        res = evaluate_model(model, X_test, y_test)
        results[name] = res
        same_as_predict = np.array_equal(res["y_pred"], np.asarray(model.predict(X_test)))
        stored = metadata["models"][name]["test_metrics"]
        diffs = {k: abs(res["metrics"][k] - stored[k]) for k in stored}
        ok = same_as_predict and max(diffs.values()) <= tolerance
        all_ok &= ok
        cm = res["confusion_matrix"]
        print(f"{'OK ' if ok else 'BAD'} {name:<24} score={res['score_source']:<17} "
              f"predict() match={same_as_predict}  max|diff| vs metadata={max(diffs.values()):.1e}  "
              f"TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")

    print("\nComparison table (held-out test set, positive class = yes):")
    print(comparison_table(results).to_string())
    print("\nAll saved models reproduce metadata.json." if all_ok else "\nMISMATCH between saved models and metadata.json!")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
