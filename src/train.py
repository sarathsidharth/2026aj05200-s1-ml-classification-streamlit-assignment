"""Train, evaluate and save the five required classifiers.

Run from the project root:

    python -m src.train

Steps
1. Load data/bank-full.csv (45,211 rows) and make the stratified 80:20 split
   (random_state=42). Every model uses this same split.
2. For each classifier build Pipeline([preprocessor, classifier]) and fit it on
   the TRAINING split only. The preprocessor (imputers, scaler, one-hot encoder)
   is part of the pipeline, so nothing is ever fitted on test data.
3. Score each pipeline on the held-out 20% with the six metrics (src/evaluate.py).
4. Save each fitted pipeline to model/<name>.pkl, write model/metadata.json and
   export the untouched held-out rows to test_data.csv for the Streamlit app.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import MODEL_FILES, comparison_table, evaluate_model
from src.preprocess import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    EXCLUDED_FEATURES,
    FEATURES,
    METADATA_PATH,
    MODEL_DIR,
    NEGATIVE_LABEL,
    NUMERICAL_FEATURES,
    ORIGINAL_COLUMNS,
    POSITIVE_LABEL,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET,
    TEST_DATA_PATH,
    TEST_SIZE,
    build_preprocessor,
    encode_target,
    load_dataset,
    prepare_features_and_target,
    split_dataset,
)

# --------------------------------------------------------------------------- #
# The five classifiers and why each hyperparameter was chosen
# --------------------------------------------------------------------------- #
MODEL_SPECS = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "rationale": (
            "Linear baseline on standardised + one-hot features. max_iter=1000 guarantees the lbfgs "
            "solver converges on 50 columns; class_weight='balanced' up-weights the 11.7% 'yes' class "
            "so the model does not simply predict 'no'. Default L2 regularisation (C=1.0)."
        ),
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                            class_weight="balanced", random_state=RANDOM_STATE),
        "rationale": (
            "Captures non-linear interactions (e.g. poutcome x month). max_depth=8 and "
            "min_samples_leaf=20 limit tree growth so it does not memorise the training data "
            "(an unpruned tree overfits badly); class_weight='balanced' handles the imbalance."
        ),
    },
    "KNN": {
        "estimator": KNeighborsClassifier(n_neighbors=15),
        "rationale": (
            "Instance-based model on standardised features (scaling is essential for distances). "
            "k=15 smooths the noisy decisions of small k and gives finer-grained probabilities for "
            "AUC. KNN has no class_weight option, so with uniform voting the majority 'no' class "
            "dominates most neighbourhoods; expect high accuracy but low recall."
        ),
    },
    "Naive Bayes": {
        "estimator": GaussianNB(),
        "rationale": (
            "Gaussian Naive Bayes on the dense preprocessed matrix (hence dense one-hot output). "
            "Fast probabilistic baseline with default var_smoothing=1e-9. It assumes features are "
            "independent and Gaussian, which the skewed and one-hot features violate, so its "
            "probabilities are poorly calibrated. It has no class_weight; priors come from the training data."
        ),
    },
    "Support Vector Machine": {
        "estimator": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE),
        "rationale": (
            "RBF kernel for a non-linear boundary; C=1 and gamma='scale' are well-behaved defaults on "
            "standardised data; class_weight='balanced' handles the imbalance. probability=True is "
            "not used (deprecated in scikit-learn 1.9 and ~5x slower); AUC uses decision_function "
            "scores, which give the same ranking."
        ),
    },
}


def build_pipeline(estimator) -> Pipeline:
    """Unfitted preprocessing + classifier pipeline."""
    return Pipeline([("preprocessor", build_preprocessor()), ("classifier", estimator)])


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _class_distribution(y: pd.Series) -> dict:
    n_pos = int(y.sum())
    return {"rows": int(len(y)), NEGATIVE_LABEL: int(len(y) - n_pos), POSITIVE_LABEL: n_pos,
            "positive_rate": round(n_pos / len(y), 6)}


def train_all() -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data and the single shared split
    data = load_dataset(DATA_PATH)
    train_df, test_df = split_dataset(data)
    X_train, y_train = prepare_features_and_target(train_df)
    X_test, y_test = prepare_features_and_target(test_df)
    print(f"Data: {len(data):,} rows | train {len(X_train):,} | test {len(X_test):,} | "
          f"{len(FEATURES)} features (excluded: {', '.join(EXCLUDED_FEATURES)})\n")

    # 2-4. Fit, evaluate, save
    test_results, model_meta = {}, {}
    for name, spec in MODEL_SPECS.items():
        pipe = build_pipeline(spec["estimator"])
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)                    # fitted on TRAIN only
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        test_res = evaluate_model(pipe, X_test, y_test)
        score_s = time.perf_counter() - t0
        train_res = evaluate_model(pipe, X_train, y_train)
        test_results[name] = test_res

        path = MODEL_DIR / MODEL_FILES[name]
        joblib.dump(pipe, path, compress=3)

        clf = pipe.named_steps["classifier"]
        model_meta[name] = {
            "file": f"model/{MODEL_FILES[name]}",
            "estimator": type(clf).__name__,
            "params": {k: _json_safe(v) for k, v in sorted(clf.get_params().items())},
            "rationale": spec["rationale"],
            "score_source": test_res["score_source"],
            "fit_seconds": round(fit_s, 2),
            "test_scoring_seconds": round(score_s, 2),
            "file_size_mb": round(path.stat().st_size / 1e6, 2),
            "test_metrics": test_res["metrics"],
            "train_metrics": train_res["metrics"],
            "test_confusion_matrix": test_res["confusion_matrix"],
        }
        m = test_res["metrics"]
        print(f"{name:<24} fit {fit_s:6.1f}s | test acc {m['accuracy']:.4f} auc {m['auc']:.4f} "
              f"f1 {m['f1']:.4f} mcc {m['mcc']:.4f} | saved {path.relative_to(PROJECT_ROOT)}")

    # Held-out rows, all original columns, original yes/no labels, comma-separated
    test_df[ORIGINAL_COLUMNS].to_csv(TEST_DATA_PATH, index=False)

    table = comparison_table(test_results, decimals=None)
    best = {metric: table[metric].idxmax() for metric in table.columns}

    metadata = {
        "project": "2026aj05200-s1-ml-classification-streamlit-assignment",
        "student_id": "2026AJ05200",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "name": "UCI Bank Marketing (bank-full.csv)",
            "source": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
            "file": "data/bank-full.csv",
            "separator": ";",
            "rows": int(len(data)),
            "original_input_features": len(ORIGINAL_COLUMNS) - 1,
            "model_input_features": len(FEATURES),
            "class_distribution": _class_distribution(encode_target(data[TARGET])),
        },
        "features": {"numerical": NUMERICAL_FEATURES, "categorical": CATEGORICAL_FEATURES, "all": FEATURES},
        "target": {"column": TARGET, "positive_label": POSITIVE_LABEL, "negative_label": NEGATIVE_LABEL,
                   "encoding": {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}},
        "excluded_features": EXCLUDED_FEATURES,
        "split": {
            "method": "train_test_split, stratified on y, shuffle=True",
            "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
            "train": _class_distribution(y_train), "test": _class_distribution(y_test),
            "test_data_file": "test_data.csv",
        },
        "preprocessing": {
            "numerical": "SimpleImputer(strategy='median') -> StandardScaler()",
            "categorical": "SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "unknown_category": "'unknown' kept as its own category",
            "fitted_on": "training split only (inside each model Pipeline)",
            "output_columns": int(build_preprocessor().fit(X_train).transform(X_train.head(1)).shape[1]),
        },
        "metric_notes": {
            "positive_class": POSITIVE_LABEL,
            "auc": "roc_auc_score on predict_proba[:, 1]; SVC uses decision_function (same ranking)",
            "threshold": "default: P(yes) > 0.5, or decision_function > 0 for SVC",
            "majority_baseline_accuracy": round(1 - float(y_test.mean()), 6),
        },
        "models": model_meta,
        "best_model_by_metric": best,
        "versions": {
            "python": platform.python_version(),
            **{pkg: version(pkg) for pkg in ("scikit-learn", "numpy", "pandas", "joblib",
                                             "matplotlib", "seaborn", "altair", "streamlit")},
        },
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print("\nComparison table (held-out test set, n = "
          f"{len(X_test):,}, positive class = '{POSITIVE_LABEL}'):")
    print(table.round(4).to_string())
    print(f"\nBest per metric: {best}")
    print(f"Wrote {METADATA_PATH.relative_to(PROJECT_ROOT)} and {TEST_DATA_PATH.relative_to(PROJECT_ROOT)} "
          f"({len(test_df):,} rows)")
    return metadata


if __name__ == "__main__":
    train_all()
    sys.exit(0)
