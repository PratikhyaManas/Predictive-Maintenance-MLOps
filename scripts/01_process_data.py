"""Load raw sensor data, clean it, and write train/test splits to disk.

Usage:
    python scripts/01_process_data.py [--config path/to/project_config.yml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pm_mlops.config import ProjectConfig  # noqa: E402
from pm_mlops.data_processor import DataProcessor  # noqa: E402
from pm_mlops.utils import get_logger  # noqa: E402

logger = get_logger(__name__)
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "project_config.yml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = ProjectConfig.from_yaml(args.config)
    processor = DataProcessor(config)

    raw_df = processor.load_data()
    clean_df = processor.clean_data(raw_df)
    train_df, test_df = processor.split_data(clean_df)
    processor.save_splits(train_df, test_df)

    logger.info("Data processing complete.")


if __name__ == "__main__":
    main()
