"""Unit tests for src/train.py."""

import os

import numpy as np
import pandas as pd
import pytest
import yaml

from src.train import compute_metrics, load_params


def test_load_params(tmp_path):
    """load_params should read a YAML file and return a dict."""
    params = {"train": {"model": "random_forest", "test_size": 0.2}}
    path = str(tmp_path / "params.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(params, fh)
    loaded = load_params(path)
    assert loaded["train"]["model"] == "random_forest"
    assert loaded["train"]["test_size"] == pytest.approx(0.2)


def test_compute_metrics_perfect():
    """compute_metrics should return 1.0 for a perfect classifier."""
    y = np.array([0, 1, 0, 1, 1])
    metrics = compute_metrics(y_true=y, y_pred=y, y_prob=y.astype(float))
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_compute_metrics_keys():
    """compute_metrics should always return the expected keys."""
    y = np.array([0, 1])
    prob = np.array([0.2, 0.8])
    metrics = compute_metrics(y_true=y, y_pred=y, y_prob=prob)
    assert set(metrics.keys()) == {"f1", "precision", "recall", "roc_auc"}


def test_compute_metrics_range():
    """All metric values should be in [0, 1]."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=50)
    y_pred = rng.integers(0, 2, size=50)
    y_prob = rng.uniform(0, 1, size=50)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    for key, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"
