"""Preprocessing and feature-engineering module.

Applies the same transformations at training time and at inference time so
that the model always receives identically-shaped feature vectors.
"""

from __future__ import annotations

import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "Churn"

# ---------------------------------------------------------------------------
# Column groups (adapt to your actual dataset)
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    """Read a CSV file from *path* and return a DataFrame."""
    logger.info("Loading data from %s", path)
    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    """Return a scikit-learn ColumnTransformer for numeric + categorical cols."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def encode_target(series: pd.Series) -> np.ndarray:
    """Encode the target column to 0/1.

    Handles both 'Yes'/'No' strings and already-numeric values.
    """
    if series.dtype == object:
        le = LabelEncoder()
        return le.fit_transform(series)
    return series.values


def preprocess(
    input_path: str,
    output_path: str,
    preprocessor_path: str,
    fit: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load, clean and transform data.

    Parameters
    ----------
    input_path:
        Path to the raw CSV.
    output_path:
        Where to write the processed features CSV.
    preprocessor_path:
        Where to save (fit=True) or load (fit=False) the fitted preprocessor.
    fit:
        When True the preprocessor is fitted then saved.  When False an
        existing preprocessor is loaded (inference mode).

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Processed feature matrix and target array.
    """
    df = load_data(input_path)

    # Coerce TotalCharges to numeric (telecom datasets often have spaces)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop customer ID if present
    df = df.drop(columns=["customerID"], errors="ignore")

    y = np.array([])
    if TARGET_COLUMN in df.columns:
        y = encode_target(df[TARGET_COLUMN])
        df = df.drop(columns=[TARGET_COLUMN])

    if fit:
        preprocessor = build_preprocessor()
        X = preprocessor.fit_transform(df)
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)
        logger.info("Preprocessor saved to %s", preprocessor_path)
    else:
        preprocessor = joblib.load(preprocessor_path)
        logger.info("Preprocessor loaded from %s", preprocessor_path)
        X = preprocessor.transform(df)

    feature_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X, columns=feature_names)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_df.to_csv(output_path, index=False)
    logger.info("Processed features saved to %s", output_path)

    return X_df, y


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess churn data.")
    parser.add_argument(
        "--input", default="data/raw/churn.csv", help="Path to raw CSV."
    )
    parser.add_argument(
        "--output",
        default="data/processed/features.csv",
        help="Path to save processed features.",
    )
    parser.add_argument(
        "--preprocessor",
        default="models/preprocessor.joblib",
        help="Path to save/load preprocessor artifact.",
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="Load an existing preprocessor instead of fitting a new one.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess(
        input_path=args.input,
        output_path=args.output,
        preprocessor_path=args.preprocessor,
        fit=not args.no_fit,
    )
