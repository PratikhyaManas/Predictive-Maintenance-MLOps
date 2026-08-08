"""Run batch predictions for a CSV of sensor readings and write results to disk.

Usage:
    python scripts/04_predict.py --input data/machine_sensors.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pm_mlops.config import ProjectConfig  # noqa: E402
from pm_mlops.models import FailureClassifier  # noqa: E402
from pm_mlops.utils import get_logger  # noqa: E402

logger = get_logger(__name__)
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "project_config.yml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", type=str, required=True, help="CSV with feature columns")
    parser.add_argument("--output", type=str, default="predictions.csv")
    args = parser.parse_args()

    config = ProjectConfig.from_yaml(args.config)
    model = FailureClassifier(config)
    model.load()

    input_df = pd.read_csv(args.input)
    x = input_df[config.features.all]

    predictions = model.predict(x)
    proba = model.predict_proba(x)

    output_df = input_df.copy()
    output_df["predicted_machine_failure"] = predictions
    output_df["failure_probability"] = proba[config.target.positive_label]
    output_df.to_csv(args.output, index=False)

    logger.info(f"Wrote {len(output_df)} predictions to {args.output}")
    high_risk = (output_df["failure_probability"] >= 0.7).sum()
    logger.info(f"{high_risk} readings flagged as high risk (probability >= 0.7)")


if __name__ == "__main__":
    main()
