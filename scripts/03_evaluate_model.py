"""Evaluate a previously saved model artifact against the test split.

Useful in CI, or to re-evaluate a model without retraining it.

Usage:
    python scripts/03_evaluate_model.py [--config path/to/project_config.yml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pm_mlops.config import ProjectConfig  # noqa: E402
from pm_mlops.data_processor import DataProcessor  # noqa: E402
from pm_mlops.models import FailureClassifier  # noqa: E402
from pm_mlops.utils import get_logger  # noqa: E402

logger = get_logger(__name__)
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "project_config.yml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = ProjectConfig.from_yaml(args.config)
    processor = DataProcessor(config)

    test_df = pd.read_csv(config.data.test_path)
    x_test, y_test = processor.get_features_and_target(test_df)

    model = FailureClassifier(config, data_processor=processor)
    model.load()

    metrics = model.evaluate(x_test, y_test)
    print(json.dumps(metrics, indent=2))

    importances = model.feature_importances()
    if importances is not None:
        logger.info(f"Top feature importances:\n{importances.head(10)}")


if __name__ == "__main__":
    main()
