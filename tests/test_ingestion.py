"""Unit tests for src/ingestion.py."""

import os
import tempfile

import pandas as pd
import pytest

from src.ingestion import ingest


@pytest.fixture()
def sample_csv(tmp_path):
    """Create a small CSV file for testing."""
    df = pd.DataFrame(
        {
            "customerID": ["111-AAA", "222-BBB"],
            "tenure": [12, 36],
            "MonthlyCharges": [50.0, 75.0],
            "Churn": ["No", "Yes"],
        }
    )
    path = str(tmp_path / "sample.csv")
    df.to_csv(path, index=False)
    return path


def test_ingest_returns_dataframe(sample_csv, tmp_path):
    """ingest() should return a non-empty DataFrame."""
    output = str(tmp_path / "output.csv")
    df = ingest(sample_csv, output)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_ingest_saves_output_file(sample_csv, tmp_path):
    """ingest() should write the CSV to the specified output path."""
    output = str(tmp_path / "raw" / "churn.csv")
    ingest(sample_csv, output)
    assert os.path.exists(output)


def test_ingest_preserves_columns(sample_csv, tmp_path):
    """ingest() should preserve all original columns."""
    output = str(tmp_path / "output.csv")
    df = ingest(sample_csv, output)
    assert "customerID" in df.columns
    assert "tenure" in df.columns
    assert "Churn" in df.columns
