"""Unit tests for src/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    encode_target,
    load_data,
)


def _make_df(n: int = 10) -> pd.DataFrame:
    """Return a minimal DataFrame that matches the expected schema."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "tenure": rng.integers(1, 72, size=n).astype(float),
            "MonthlyCharges": rng.uniform(20, 110, size=n),
            "TotalCharges": rng.uniform(50, 8000, size=n),
            "gender": rng.choice(["Male", "Female"], size=n),
            "SeniorCitizen": rng.choice(["0", "1"], size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Dependents": rng.choice(["Yes", "No"], size=n),
            "PhoneService": rng.choice(["Yes", "No"], size=n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size=n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                size=n,
            ),
            TARGET_COLUMN: rng.choice(["Yes", "No"], size=n),
        }
    )
    return df


def test_load_data(tmp_path):
    """load_data should read a CSV into a DataFrame."""
    df = _make_df()
    path = str(tmp_path / "test.csv")
    df.to_csv(path, index=False)
    loaded = load_data(path)
    assert len(loaded) == len(df)
    assert set(loaded.columns) == set(df.columns)


def test_build_preprocessor_transforms():
    """The preprocessor should transform data without raising."""
    df = _make_df(20)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == 20
    assert X_transformed.shape[1] > len(NUMERIC_FEATURES)  # OHE expands columns


def test_encode_target_string():
    """encode_target should handle 'Yes'/'No' strings."""
    series = pd.Series(["Yes", "No", "Yes", "No"])
    encoded = encode_target(series)
    assert set(encoded).issubset({0, 1})
    assert len(encoded) == 4


def test_encode_target_numeric():
    """encode_target should pass through already-numeric targets."""
    series = pd.Series([1, 0, 1, 0])
    encoded = encode_target(series)
    np.testing.assert_array_equal(encoded, [1, 0, 1, 0])


def test_preprocessor_no_nans():
    """Transformed output should not contain NaN values."""
    df = _make_df(30)
    # Introduce some NaNs
    df.loc[0, "TotalCharges"] = np.nan
    df.loc[1, "tenure"] = np.nan
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    assert not np.isnan(X_transformed).any()
