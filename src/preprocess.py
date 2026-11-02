"""Data definitions, loading, validation and the leakage-safe preprocessing pipeline.

Everything that both training (``src/train.py``) and the Streamlit app (``app.py``)
need to agree on lives here, so the features seen at inference time are exactly
the features the models were trained on.

Key decisions (see notebooks/eda.ipynb for the evidence):
* ``duration`` is dropped: it is only known after the call ends (target leakage).
* ``"unknown"`` is kept as its own category; it carries signal.
* Outliers are kept (they are genuine); numerical features are scaled instead.
* The ColumnTransformer returned by :func:`build_preprocessor` is **unfitted**. It is
  placed inside each model's Pipeline and fitted on the training split only.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import IO

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "bank-full.csv"
MODEL_DIR = PROJECT_ROOT / "model"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
METADATA_PATH = MODEL_DIR / "metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "y"
POSITIVE_LABEL = "yes"   # positive class for every metric (encoded as 1)
NEGATIVE_LABEL = "no"    # encoded as 0
CLASS_LABELS = [NEGATIVE_LABEL, POSITIVE_LABEL]

NUMERICAL_FEATURES = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
]
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES  # 15 model inputs

EXCLUDED_FEATURES = {
    "duration": (
        "Length of the last call in seconds. It is only known after the call has "
        "ended, when the outcome is also known, so using it would be target leakage. "
        "The UCI dataset notes recommend discarding it for a realistic predictive model."
    ),
}

# Column order of the original UCI file (used when exporting test_data.csv)
ORIGINAL_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous",
    "poutcome", TARGET,
]


class DataValidationError(ValueError):
    """Raised when a dataset cannot be used; ``errors`` lists every problem found."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def detect_separator(text: str) -> str:
    """Return ';' or ',' depending on which appears more often in the header line."""
    header = text.lstrip("﻿").splitlines()[0] if text.strip() else ""
    return ";" if header.count(";") > header.count(",") else ","


def read_csv_auto(source: str | Path | bytes | IO) -> pd.DataFrame:
    """Read a CSV that may be ';'- or ','-separated.

    ``source`` can be a file path, raw bytes, or a file-like object (e.g. a
    Streamlit ``UploadedFile``).
    """
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif isinstance(source, bytes):
        raw = source
    else:
        raw = source.getvalue() if hasattr(source, "getvalue") else source.read()
    if not raw or not raw.strip():
        raise DataValidationError(["The file is empty."])

    text = raw.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep=detect_separator(text))
    df.columns = [str(c).strip().strip('"').strip("'").lower() for c in df.columns]
    return df


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the full UCI Bank Marketing dataset (all original columns)."""
    return read_csv_auto(path)


# --------------------------------------------------------------------------- #
# Validation and cleaning
# --------------------------------------------------------------------------- #
def validate_dataframe(df: pd.DataFrame, require_target: bool = True) -> list[str]:
    """Return a list of human-readable problems (an empty list means the data is usable).

    Checks: non-empty, all 15 input features present, target present with only
    yes/no values, and numerical columns actually numeric.
    """
    errors: list[str] = []
    if df is None or df.empty:
        return ["The file contains no data rows."]

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        errors.append(f"Missing required feature column(s): {', '.join(missing)}.")

    if require_target:
        if TARGET not in df.columns:
            errors.append(f"Missing target column '{TARGET}' (values must be 'yes'/'no').")
        else:
            labels = df[TARGET].dropna().astype(str).str.strip().str.lower()
            bad = sorted(set(labels.unique()) - set(CLASS_LABELS))
            if bad:
                shown = ", ".join(repr(b) for b in bad[:5])
                errors.append(f"Target column '{TARGET}' has invalid value(s): {shown}. Only 'yes'/'no' are allowed.")
            if df[TARGET].isna().any():
                errors.append(f"Target column '{TARGET}' has {int(df[TARGET].isna().sum())} empty value(s).")

    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            n_bad = int((converted.isna() & df[col].notna()).sum())
            if n_bad:
                errors.append(f"Numerical column '{col}' has {n_bad} non-numeric value(s).")
    return errors


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a validated dataframe into the model's expected format.

    * numerical columns -> numeric dtype (true blanks stay NaN for the imputer)
    * categorical columns -> stripped, lower-case strings (blanks stay NaN)
    * target (if present) -> stripped, lower-case 'yes'/'no'
    Extra columns such as ``duration`` are kept here and ignored by the
    ColumnTransformer, which only selects the 15 model features.
    """
    out = df.copy()
    for col in NUMERICAL_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_FEATURES + [TARGET]:
        if col in out.columns:
            s = out[col]
            out[col] = s.where(s.isna(), s.astype(str).str.strip().str.lower())
    return out


def encode_target(y: pd.Series) -> pd.Series:
    """Map 'yes' -> 1 and 'no' -> 0."""
    return y.astype(str).str.strip().str.lower().eq(POSITIVE_LABEL).astype(int)


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate + clean, then return (X with the 15 features, y encoded 0/1)."""
    errors = validate_dataframe(df, require_target=True)
    if errors:
        raise DataValidationError(errors)
    df = clean_dataframe(df)
    return df[FEATURES], encode_target(df[TARGET])


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def split_dataset(df: pd.DataFrame,
                  test_size: float = TEST_SIZE,
                  random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split of the *full* rows (all original columns kept).

    Returning whole rows lets train.py export the untouched held-out rows as
    test_data.csv, so the app is evaluated on exactly the same 20% as the models.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df[TARGET], shuffle=True,
    )
    return train_df, test_df


# --------------------------------------------------------------------------- #
# Preprocessing pipeline (unfitted)
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Return the UNFITTED ColumnTransformer used inside every model Pipeline.

    numerical  : median imputation  -> StandardScaler
    categorical: most-frequent imputation -> OneHotEncoder(handle_unknown="ignore")
    The one-hot output is dense because GaussianNB does not accept sparse input.
    Any other column (e.g. ``duration``) is dropped.
    """
    numerical = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numerical, NUMERICAL_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


if __name__ == "__main__":
    data = load_dataset()
    train_df, test_df = split_dataset(data)
    print(f"Dataset        : {data.shape[0]:,} rows x {data.shape[1]} columns")
    print(f"Model features : {len(FEATURES)} ({len(NUMERICAL_FEATURES)} numerical, {len(CATEGORICAL_FEATURES)} categorical)")
    print(f"Excluded       : {', '.join(EXCLUDED_FEATURES)}")
    for name, part in (("Train", train_df), ("Test", test_df)):
        print(f"{name:<6}: {len(part):>6,} rows, positive rate {encode_target(part[TARGET]).mean():.4f}")
    X_train, _ = prepare_features_and_target(train_df)
    n_out = build_preprocessor().fit(X_train).transform(X_train.head(1)).shape[1]
    print(f"Columns after preprocessing (fitted on train only): {n_out}")
