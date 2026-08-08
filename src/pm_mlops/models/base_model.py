"""Abstract base class defining the contract every model wrapper implements.

Keeping a single interface (`train` / `predict` / `evaluate` / `save` /
`load`) means scripts, the serving API, and tests never need to know which
concrete algorithm is behind a model — they only depend on this contract.

Artifacts are versioned: `save()`/`load()` persist not just the fitted
sklearn pipeline but a metadata envelope (decision threshold, training
timestamp, package version, held-out metrics). This means a deployed
model's decision threshold travels with the artifact itself rather than
living as a separate, easy-to-desync constant in the serving code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pm_mlops.config import ProjectConfig
from pm_mlops.utils import get_logger

logger = get_logger(__name__)

DEFAULT_THRESHOLD = 0.5


class BaseModel(ABC):
    """Common interface + shared persistence logic for all model wrappers."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.pipeline: Any = None  # set by subclasses (typically an sklearn Pipeline)
        self.metadata: dict[str, Any] = {"threshold": DEFAULT_THRESHOLD}

    @property
    def threshold(self) -> float:
        """The decision threshold used to turn probabilities into hard labels."""
        return float(self.metadata.get("threshold", DEFAULT_THRESHOLD))

    @abstractmethod
    def build_pipeline(self) -> Any:
        """Construct the (unfitted) end-to-end sklearn pipeline."""

    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> BaseModel:
        """Fit the pipeline on training data. Returns self for chaining."""

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Return hard class predictions for the given features."""

    @abstractmethod
    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities for the given features."""

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Compute standard classification metrics on a held-out set.

        For rare-event industrial failure prediction, precision/recall/F1
        and ROC-AUC matter far more than accuracy alone — a model that
        never predicts failure can still be >95% "accurate" on an
        imbalanced fleet while being operationally useless.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_pred = self.predict(x_test)
        y_proba = self.predict_proba(x_test)
        positive_col = self.config.target.positive_label

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba[positive_col]),
        }
        logger.info(f"Evaluation metrics (threshold={self.threshold:.3f}): {metrics}")
        return metrics

    def tune_threshold(self, x_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Pick the decision threshold that minimizes expected business cost.

        Scans a grid of thresholds and, for each, computes
        ``false_negative_cost * FN + false_positive_cost * FP`` on the
        validation set — a proxy for real operating cost, since in
        predictive maintenance a missed failure is typically far more
        expensive than an unnecessary inspection. The winning threshold is
        stored on ``self.metadata`` (and therefore persisted by `save()`).

        Must be called with a validation set that is disjoint from
        whatever set `evaluate()` is later called on, to avoid tuning the
        threshold against the same data used to report final metrics.
        """
        from sklearn.metrics import confusion_matrix

        tuning_cfg = self.config.model.threshold_tuning
        positive_col = self.config.target.positive_label
        proba = self.predict_proba(x_val)[positive_col].to_numpy()
        y_true = y_val.to_numpy()

        candidates = np.linspace(0.01, 0.99, tuning_cfg.search_steps)
        best_threshold, best_cost = DEFAULT_THRESHOLD, float("inf")

        for t in candidates:
            preds = (proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            cost = tuning_cfg.false_negative_cost * fn + tuning_cfg.false_positive_cost * fp
            if cost < best_cost:
                best_cost, best_threshold = cost, float(t)

        tuning_result = {
            "best_cost": float(best_cost),
            "false_negative_cost": tuning_cfg.false_negative_cost,
            "false_positive_cost": tuning_cfg.false_positive_cost,
            "n_validation_rows": len(y_val),
        }
        self.metadata["threshold"] = best_threshold
        self.metadata["threshold_tuning"] = tuning_result

        logger.info(
            f"Tuned decision threshold to {best_threshold:.3f} "
            f"(expected cost {best_cost:.1f} on {len(y_val)} validation rows)"
        )
        return {"threshold": best_threshold, **tuning_result}

    def save(self, path: str | Path | None = None) -> Path:
        """Serialize the fitted pipeline + metadata envelope to disk as one artifact."""
        if self.pipeline is None:
            raise RuntimeError("Cannot save a model that hasn't been trained yet.")

        self.metadata.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
        self.metadata.setdefault("model_type", self.config.model.type)

        payload = {"pipeline": self.pipeline, "metadata": self.metadata}
        out_path = Path(path or self.config.model.artifact_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, out_path)
        logger.info(f"Saved model artifact to {out_path} (threshold={self.threshold:.3f})")
        return out_path

    def load(self, path: str | Path | None = None) -> BaseModel:
        """Load a previously-saved artifact from disk. Returns self.

        Transparently handles artifacts saved by an older version of this
        code that stored a bare pipeline (no metadata envelope), falling
        back to the default 0.5 threshold in that case.
        """
        in_path = Path(path or self.config.model.artifact_path)
        if not in_path.exists():
            raise FileNotFoundError(f"No model artifact found at {in_path}")

        payload = joblib.load(in_path)
        if isinstance(payload, dict) and "pipeline" in payload:
            self.pipeline = payload["pipeline"]
            self.metadata = payload.get("metadata", {"threshold": DEFAULT_THRESHOLD})
        else:
            logger.warning(
                f"Artifact at {in_path} predates the metadata envelope; "
                "falling back to default threshold 0.5."
            )
            self.pipeline = payload
            self.metadata = {"threshold": DEFAULT_THRESHOLD}

        logger.info(f"Loaded model artifact from {in_path} (threshold={self.threshold:.3f})")
        return self
