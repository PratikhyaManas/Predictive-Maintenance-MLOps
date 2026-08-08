"""Train the FailureClassifier and save a versioned artifact.

Methodology: the training split is further divided into a sub-train set
(used to fit the model) and a validation set (used only to tune the
cost-aware decision threshold). The test split is touched exactly once,
for final reporting — never for threshold selection — to avoid leaking
information from the evaluation set into a tuned parameter.

Usage:
    python scripts/02_train_model.py [--config path/to/project_config.yml]
"""

from __future__ import annotations

import argparse
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


def _maybe_log_to_mlflow(config: ProjectConfig, metrics: dict, model_path: Path) -> None:
    if not config.mlflow.enabled:
        return
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow.enabled=true but mlflow isn't installed; skipping tracking.")
        return

    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.log_params(config.model.params)
        mlflow.log_param("model_type", config.model.type)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path))
    logger.info("Logged run to MLflow")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = ProjectConfig.from_yaml(args.config)
    processor = DataProcessor(config)

    train_df = pd.read_csv(config.data.train_path)
    test_df = pd.read_csv(config.data.test_path)

    model = FailureClassifier(config, data_processor=processor)

    if config.model.threshold_tuning.enabled:
        sub_train_df, val_df = processor.split_train_validation(train_df)
        x_sub_train, y_sub_train = processor.get_features_and_target(sub_train_df)
        x_val, y_val = processor.get_features_and_target(val_df)

        # Fit on the sub-train slice only, so the validation set used for
        # threshold tuning is genuinely unseen by the model.
        model.train(x_sub_train, y_sub_train)
        tuning_result = model.tune_threshold(x_val, y_val)
        logger.info(f"Threshold tuning result: {tuning_result}")

        # Refit on the *full* training split (sub_train + validation) for
        # the final artifact — more data for the model to learn from, while
        # the tuned threshold (stored in metadata, untouched by train())
        # carries over unchanged.
        x_train, y_train = processor.get_features_and_target(train_df)
        model.train(x_train, y_train)
    else:
        x_train, y_train = processor.get_features_and_target(train_df)
        model.train(x_train, y_train)

    x_test, y_test = processor.get_features_and_target(test_df)
    metrics = model.evaluate(x_test, y_test)
    model_path = model.save()

    _maybe_log_to_mlflow(config, metrics, model_path)

    logger.info(f"Training complete. Artifact saved to {model_path}")
    logger.info(f"Decision threshold: {model.threshold:.3f}")
    logger.info(f"Held-out metrics: {metrics}")


if __name__ == "__main__":
    main()
