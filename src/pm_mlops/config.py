"""Centralized, typed configuration for the Predictive Maintenance pipeline.

Every other module reads settings through :class:`ProjectConfig` instead of
hard-coding paths or hyperparameters, so the whole pipeline can be
reconfigured by editing ``project_config.yml`` alone.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Raw-YAML cache, keyed by resolved file path. Caching the *parsed dict*
# (not the constructed ProjectConfig) avoids repeated disk I/O + YAML
# parsing across the many `from_yaml()` calls a pipeline run makes, while
# still handing back a fresh, independently-mutable ProjectConfig instance
# on every call — important since tests and scripts routinely tweak a
# loaded config in place.
_RAW_YAML_CACHE: dict[str, dict[str, Any]] = {}


class DataConfig(BaseModel):
    raw_path: str
    processed_dir: str
    train_path: str
    test_path: str
    test_size: float = Field(gt=0, lt=1, default=0.2)
    validation_size: float = Field(
        gt=0,
        lt=1,
        default=0.15,
        description=(
            "Fraction of the *training* split held out for threshold tuning. "
            "Kept separate from the test set so the decision threshold is "
            "never tuned against the data used for final evaluation."
        ),
    )


class TargetConfig(BaseModel):
    column: str
    positive_label: int = 1


class FeatureConfig(BaseModel):
    numerical: list[str]
    categorical: list[str]

    @property
    def all(self) -> list[str]:
        return [*self.numerical, *self.categorical]


class ThresholdTuningConfig(BaseModel):
    """Business-cost-aware decision threshold tuning.

    Industrial failure prediction is asymmetric: a missed failure (false
    negative) typically costs far more than an unnecessary inspection
    (false positive). The default 10:1 ratio is a reasonable starting
    assumption — tune it to your actual downtime/inspection costs.
    """

    enabled: bool = True
    false_negative_cost: float = Field(gt=0, default=10.0)
    false_positive_cost: float = Field(gt=0, default=1.0)
    search_steps: int = Field(gt=1, default=99)


class ModelConfig(BaseModel):
    artifact_dir: str
    artifact_name: str
    type: str = "random_forest"
    params: dict[str, Any] = Field(default_factory=dict)
    threshold_tuning: ThresholdTuningConfig = Field(default_factory=ThresholdTuningConfig)

    @property
    def artifact_path(self) -> Path:
        return Path(self.artifact_dir) / self.artifact_name


class MlflowConfig(BaseModel):
    enabled: bool = False
    experiment_name: str = "/pm-mlops/default"
    tracking_uri: str | None = None


class ServingConfig(BaseModel):
    # Binds all interfaces by design: this is the container's listen address,
    # meant to be reached from outside the container (via Docker's port
    # mapping) or from within an internal cluster network — not exposed
    # directly to the public internet. Explicitly suppressed rather than
    # rule-disabled repo-wide, since 0.0.0.0 elsewhere would be worth flagging.
    host: str = "0.0.0.0"  # nosec B104 -- container listen address, not public-internet-facing
    port: int = 8000


class MonitoringConfig(BaseModel):
    report_dir: str = "monitoring_reports"
    drift_threshold: float = 0.1


class ProjectConfig(BaseModel):
    """Root configuration object, loaded once and passed around explicitly."""

    project_name: str
    random_seed: int = 42
    data: DataConfig
    target: TargetConfig
    features: FeatureConfig
    model: ModelConfig
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @classmethod
    def from_yaml(cls, path: str | Path = "project_config.yml") -> ProjectConfig:
        """Load configuration from a YAML file, resolving relative paths.

        Paths inside the YAML file are resolved relative to the config
        file's own directory, so the pipeline works regardless of the
        current working directory it's invoked from. The parsed YAML is
        cached per-path (see :data:`_RAW_YAML_CACHE`); each call still
        returns a brand-new, independently mutable ``ProjectConfig``.
        """
        config_path = Path(path).resolve()
        cache_key = str(config_path)

        if cache_key not in _RAW_YAML_CACHE:
            with config_path.open("r") as f:
                _RAW_YAML_CACHE[cache_key] = yaml.safe_load(f)

        raw: dict[str, Any] = copy.deepcopy(_RAW_YAML_CACHE[cache_key])

        root = config_path.parent
        data = raw.get("data", {})
        for key in ("raw_path", "processed_dir", "train_path", "test_path"):
            if key in data:
                data[key] = str(root / data[key])

        model = raw.get("model", {})
        if "artifact_dir" in model:
            model["artifact_dir"] = str(root / model["artifact_dir"])

        monitoring = raw.get("monitoring", {})
        if "report_dir" in monitoring:
            monitoring["report_dir"] = str(root / monitoring["report_dir"])

        return cls(**raw)
