"""Inference / prediction module.

Loads the saved preprocessor and model artifacts and exposes a ``predict``
function that accepts a raw dict (as it would arrive from the API) and
returns a churn probability and binary label.
"""

from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
DEFAULT_PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.joblib")

# ---------------------------------------------------------------------------
# Artifact loading (lazy, module-level cache)
# ---------------------------------------------------------------------------

_model = None
_preprocessor = None


def load_artifacts(
    model_path: str = DEFAULT_MODEL_PATH,
    preprocessor_path: str = DEFAULT_PREPROCESSOR_PATH,
) -> None:
    """Load model and preprocessor from disk into the module-level cache."""
    global _model, _preprocessor
    logger.info("Loading model from %s", model_path)
    _model = joblib.load(model_path)
    logger.info("Loading preprocessor from %s", preprocessor_path)
    _preprocessor = joblib.load(preprocessor_path)


def _ensure_loaded() -> None:
    if _model is None or _preprocessor is None:
        load_artifacts()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict(features: dict) -> dict:
    """Run inference on a single customer record.

    Parameters
    ----------
    features:
        A dict whose keys correspond to the raw feature columns (before
        preprocessing).  For example::

            {
                "tenure": 24,
                "MonthlyCharges": 65.5,
                "TotalCharges": 1572.0,
                "Contract": "Month-to-month",
                ...
            }

    Returns
    -------
    dict
        ``{"churn": bool, "churn_probability": float}``
    """
    _ensure_loaded()

    df = pd.DataFrame([features])
    X = _preprocessor.transform(df)
    prob: float = float(_model.predict_proba(X)[0, 1])
    label: bool = bool(_model.predict(X)[0])

    return {"churn": label, "churn_probability": round(prob, 4)}


def predict_batch(records: list[dict]) -> list[dict]:
    """Run inference on a list of customer records.

    Parameters
    ----------
    records:
        A list of feature dicts (same schema as ``predict``).

    Returns
    -------
    list[dict]
        A list of ``{"churn": bool, "churn_probability": float}`` dicts.
    """
    _ensure_loaded()

    df = pd.DataFrame(records)
    X = _preprocessor.transform(df)
    probs: np.ndarray = _model.predict_proba(X)[:, 1]
    labels: np.ndarray = _model.predict(X)

    return [
        {"churn": bool(label), "churn_probability": round(float(prob), 4)}
        for label, prob in zip(labels, probs)
    ]
