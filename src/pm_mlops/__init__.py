"""pm_mlops: a modular, reusable pipeline for industrial predictive maintenance."""

from pm_mlops.config import ProjectConfig
from pm_mlops.data_processor import DataProcessor
from pm_mlops.models import FailureClassifier

__all__ = ["ProjectConfig", "DataProcessor", "FailureClassifier"]
__version__ = "0.1.0"
