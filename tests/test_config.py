from __future__ import annotations

from pathlib import Path

from pm_mlops.config import ProjectConfig


def test_config_loads_from_yaml(project_root: Path):
    config = ProjectConfig.from_yaml(project_root / "project_config.yml")

    assert config.project_name == "pm-mlops"
    assert config.target.column == "machine_failure"
    assert "torque_nm" in config.features.numerical
    assert "product_type" in config.features.categorical


def test_config_resolves_paths_relative_to_yaml_location(project_root: Path):
    config = ProjectConfig.from_yaml(project_root / "project_config.yml")

    assert Path(config.data.raw_path).is_absolute()
    assert str(project_root) in config.data.raw_path


def test_feature_config_all_combines_numerical_and_categorical(project_root: Path):
    config = ProjectConfig.from_yaml(project_root / "project_config.yml")

    assert config.features.all == config.features.numerical + config.features.categorical
