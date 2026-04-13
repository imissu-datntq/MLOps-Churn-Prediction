"""Model training module.

Trains a classification model (Random Forest, XGBoost, or LightGBM),
logs all hyperparameters and metrics to MLflow, and saves the best model
to the models/ directory.
"""

import argparse
import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "random_forest": RandomForestClassifier,
}

try:
    from xgboost import XGBClassifier

    MODELS["xgboost"] = XGBClassifier
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier

    MODELS["lightgbm"] = LGBMClassifier
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_params(params_path: str = "params.yaml") -> dict:
    """Load hyperparameters from a YAML file."""
    with open(params_path) as fh:
        return yaml.safe_load(fh)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Return a dict of evaluation metrics."""
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train(
    features_path: str,
    target_path: str,
    model_output_path: str,
    params_path: str = "params.yaml",
) -> dict:
    """Train a model, log to MLflow, and save to disk.

    Parameters
    ----------
    features_path:
        Path to the processed features CSV.
    target_path:
        Path to a single-column CSV containing the encoded target.
    model_output_path:
        Where to save the trained model artifact (.joblib).
    params_path:
        Path to the YAML file with model hyperparameters.

    Returns
    -------
    dict
        Evaluation metrics on the held-out test split.
    """
    params = load_params(params_path)
    train_params = params.get("train", {})
    model_name = train_params.get("model", "random_forest")
    model_params = params.get(model_name, {})
    test_size = train_params.get("test_size", 0.2)
    random_state = train_params.get("random_state", 42)

    logger.info("Loading features from %s", features_path)
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}"
        )

    ModelClass = MODELS[model_name]
    model = ModelClass(**model_params)

    mlflow.set_experiment("churn-prediction")
    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_params(model_params)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        logger.info("Training %s ...", model_name)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        logger.info("Metrics: %s", metrics)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    logger.info("Model saved to %s", model_output_path)

    return metrics


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn prediction model.")
    parser.add_argument(
        "--features",
        default="data/processed/features.csv",
        help="Path to processed features CSV.",
    )
    parser.add_argument(
        "--target",
        default="data/processed/target.csv",
        help="Path to encoded target CSV.",
    )
    parser.add_argument(
        "--model-output",
        default="models/model.joblib",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Path to hyperparameter YAML file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        features_path=args.features,
        target_path=args.target,
        model_output_path=args.model_output,
        params_path=args.params,
    )
