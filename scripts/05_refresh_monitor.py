"""Compare a reference dataset against fresh sensor data and write a drift report.

In production `current` would be newly-collected telemetry since the last
run (e.g. this week's readings pulled from the historian); here it
defaults to the test split for a self-contained demo.

Usage:
    python scripts/05_refresh_monitor.py [--current path/to/new_readings.csv]
"""

from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import DEFAULT_CONFIG_PATH

from pm_mlops.config import ProjectConfig  # noqa: E402
from pm_mlops.monitoring import DriftMonitor  # noqa: E402
from pm_mlops.utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--current", type=str, default=None)
    args = parser.parse_args()

    config = ProjectConfig.from_yaml(args.config)

    reference_df = pd.read_csv(config.data.train_path)
    current_df = pd.read_csv(args.current or config.data.test_path)

    monitor = DriftMonitor(config)
    report = monitor.compute_report(reference_df, current_df)
    monitor.save_report(report)

    if report["any_feature_drifted"]:
        logger.warning("Drift detected — consider retraining the model or inspecting sensors.")
    else:
        logger.info("No drift detected — model is safe to keep serving.")


if __name__ == "__main__":
    main()
