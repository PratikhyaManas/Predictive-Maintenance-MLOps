from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pm_mlops.config import ProjectConfig


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def config(project_root: Path) -> ProjectConfig:
    return ProjectConfig.from_yaml(project_root / "project_config.yml")


@pytest.fixture
def sample_df(config: ProjectConfig) -> pd.DataFrame:
    """Small synthetic sensor dataset with genuine, learnable failure signal."""
    rng = np.random.default_rng(0)
    n = 300

    product_type = rng.choice(["L", "M", "H"], n, p=[0.6, 0.3, 0.1])
    air_temperature_k = rng.normal(300.0, 2.0, n)
    process_temperature_k = air_temperature_k + rng.normal(10.0, 1.0, n)
    torque_nm = np.clip(rng.normal(40.0, 10.0, n), 3.0, 80.0)
    rotational_speed_rpm = np.clip(rng.normal(1500.0, 200.0, n), 800, 3000)
    tool_wear_min = rng.uniform(0, 200, n)

    # Failure correlates with high torque * tool wear (overstrain-like signal)
    strain = tool_wear_min * torque_nm
    machine_failure = (strain > np.quantile(strain, 0.9)).astype(int)

    df = pd.DataFrame(
        {
            "product_type": product_type,
            "air_temperature_k": air_temperature_k,
            "process_temperature_k": process_temperature_k,
            "rotational_speed_rpm": rotational_speed_rpm,
            "torque_nm": torque_nm,
            "tool_wear_min": tool_wear_min,
            "machine_failure": machine_failure,
        }
    )
    return df
