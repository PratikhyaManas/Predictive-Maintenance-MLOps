from __future__ import annotations

import pandas as pd

from pm_mlops.config import ProjectConfig
from pm_mlops.data_processor import DataProcessor


def test_clean_data_fills_missing_categoricals(config: ProjectConfig, sample_df: pd.DataFrame):
    sample_df.loc[0, "product_type"] = None
    processor = DataProcessor(config)

    cleaned = processor.clean_data(sample_df)

    assert cleaned["product_type"].isna().sum() == 0
    assert cleaned.loc[0, "product_type"] == "UNKNOWN"


def test_clean_data_clips_out_of_range_sensor_values(config: ProjectConfig, sample_df: pd.DataFrame):
    sample_df.loc[0, "torque_nm"] = 999.0  # physically implausible
    processor = DataProcessor(config)

    cleaned = processor.clean_data(sample_df)

    assert cleaned.loc[0, "torque_nm"] <= 100.0


def test_clean_data_raises_on_missing_columns(config: ProjectConfig, sample_df: pd.DataFrame):
    processor = DataProcessor(config)
    broken = sample_df.drop(columns=["torque_nm"])

    try:
        processor.clean_data(broken)
        raised = False
    except ValueError:
        raised = True

    assert raised


def test_split_data_is_stratified(config: ProjectConfig, sample_df: pd.DataFrame):
    processor = DataProcessor(config)
    cleaned = processor.clean_data(sample_df)

    train_df, test_df = processor.split_data(cleaned)

    assert len(train_df) + len(test_df) == len(cleaned)
    overall_rate = cleaned["machine_failure"].mean()
    train_rate = train_df["machine_failure"].mean()
    test_rate = test_df["machine_failure"].mean()
    assert abs(train_rate - overall_rate) < 0.15
    assert abs(test_rate - overall_rate) < 0.15


def test_build_preprocessor_transforms_expected_columns(
    config: ProjectConfig, sample_df: pd.DataFrame
):
    processor = DataProcessor(config)
    cleaned = processor.clean_data(sample_df)
    x, _y = processor.get_features_and_target(cleaned)

    preprocessor = processor.build_preprocessor()
    transformed = preprocessor.fit_transform(x)

    assert transformed.shape[0] == len(x)
    assert transformed.shape[1] > len(config.features.numerical)  # one-hot expands columns


def test_get_features_and_target_returns_expected_shapes(
    config: ProjectConfig, sample_df: pd.DataFrame
):
    processor = DataProcessor(config)
    cleaned = processor.clean_data(sample_df)

    x, y = processor.get_features_and_target(cleaned)

    assert list(x.columns) == config.features.all
    assert len(x) == len(y) == len(cleaned)
