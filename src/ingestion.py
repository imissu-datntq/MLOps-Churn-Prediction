"""Data ingestion module.

Downloads or copies raw customer churn data into data/raw/ so that DVC
can version it.  The module accepts a CSV path/URL via the command line
or the ``ingest`` function.
"""

import argparse
import logging
import os
import shutil

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest(source: str, output_path: str) -> pd.DataFrame:
    """Load data from *source* and save it to *output_path*.

    Parameters
    ----------
    source:
        A local file path or a URL to a CSV file.
    output_path:
        Destination path where the raw CSV will be stored.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame (before any processing).
    """
    logger.info("Ingesting data from %s", source)

    if source.startswith("http://") or source.startswith("https://"):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if source != output_path:
        if source.startswith("http://") or source.startswith("https://"):
            df.to_csv(output_path, index=False)
        else:
            shutil.copy(source, output_path)

    logger.info("Saved %d rows to %s", len(df), output_path)
    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest raw churn data.")
    parser.add_argument("--source", required=True, help="Path or URL to source CSV.")
    parser.add_argument(
        "--output",
        default="data/raw/churn.csv",
        help="Destination path for the raw CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest(args.source, args.output)
