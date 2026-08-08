from __future__ import annotations

from pathlib import Path

import pandas as pd

from pm_mlops.config import ProjectConfig
from pm_mlops.monitoring import DriftMonitor


def test_no_drift_when_distributions_match(config: ProjectConfig, sample_df: pd.DataFrame):
    monitor = DriftMonitor(config)
    report = monitor.compute_report(sample_df, sample_df)

    assert report["any_feature_drifted"] is False


def test_drift_detected_on_shifted_sensor_distribution(
    config: ProjectConfig, sample_df: pd.DataFrame
):
    shifted = sample_df.copy()
    shifted["torque_nm"] = shifted["torque_nm"] + 40  # simulate a miscalibrated sensor

    monitor = DriftMonitor(config)
    report = monitor.compute_report(sample_df, shifted)

    assert report["features"]["torque_nm"]["drifted"] is True
    assert report["any_feature_drifted"] is True


def test_save_report_writes_json_file(
    config: ProjectConfig, sample_df: pd.DataFrame, tmp_path: Path
):
    monitor = DriftMonitor(config)
    report = monitor.compute_report(sample_df, sample_df)

    out_path = monitor.save_report(report, path=tmp_path)

    assert out_path.exists()
    assert out_path.suffix == ".json"
