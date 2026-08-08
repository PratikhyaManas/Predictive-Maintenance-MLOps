"""Concrete model: predicts whether a machine reading indicates failure risk.

Wraps preprocessing + estimator in a single sklearn ``Pipeline`` so the
saved artifact is self-contained — the serving layer only needs the raw
sensor readings, never hand-rolled feature engineering, which matters in
industrial settings where the model may be deployed on edge gateways far
from the training environment.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from pm_mlops.config import ProjectConfig
from pm_mlops.data_processor import DataProcessor
from pm_mlops.models.base_model import BaseModel
from pm_mlops.utils import get_logger

logger = get_logger(__name__)

_ESTIMATORS = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}

# Estimators that support sklearn's n_jobs parallelism. Applied as a default
# (overridable via project_config.yml -> model.params.n_jobs) so training
# uses all available cores without every user having to know to set it.
_SUPPORTS_N_JOBS = {"random_forest"}


class FailureClassifier(BaseModel):
    """Binary classifier: does this sensor reading indicate machine failure?"""

    def __init__(self, config: ProjectConfig, data_processor: DataProcessor | None = None):
        super().__init__(config)
        # Reuses DataProcessor purely for its (unfitted) preprocessing pipeline,
        # so feature engineering logic lives in exactly one place.
        self.data_processor = data_processor or DataProcessor(config)

    def build_pipeline(self) -> Pipeline:
        model_type = self.config.model.type
        if model_type not in _ESTIMATORS:
            raise ValueError(
                f"Unknown model type '{model_type}'. Available: {list(_ESTIMATORS)}"
            )
        estimator_cls = _ESTIMATORS[model_type]
        params = dict(self.config.model.params)
        if model_type in _SUPPORTS_N_JOBS:
            params.setdefault("n_jobs", -1)  # use all cores unless the config overrides it
        estimator = estimator_cls(
            random_state=self.config.random_seed,
            **params,
        )
        preprocessor = self.data_processor.build_preprocessor()
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("estimator", estimator),
            ]
        )
        return self.pipeline

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> FailureClassifier:
        if self.pipeline is None:
            self.build_pipeline()
        logger.info(
            f"Training {self.config.model.type} on {len(x_train)} rows, "
            f"{len(x_train.columns)} features "
            f"(failure rate: {y_train.mean():.2%})"
        )
        self.pipeline.fit(x_train, y_train)
        return self

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Hard predictions using the *tuned* decision threshold (see `tune_threshold`).

        Deliberately does not call the underlying estimator's own
        `.predict()`, which always applies an implicit 0.5 cutoff — this
        model's threshold is a first-class, persisted property that may
        differ from 0.5 once cost-aware tuning has run.
        """
        if self.pipeline is None:
            raise RuntimeError("Model must be trained or loaded before predicting.")
        proba = self.predict_proba(x)
        positive_label = self.config.target.positive_label
        preds = (proba[positive_label] >= self.threshold).astype(int)
        preds.name = "prediction"
        return preds

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Model must be trained or loaded before predicting.")
        proba = self.pipeline.predict_proba(x)
        classes = self.pipeline.named_steps["estimator"].classes_
        return pd.DataFrame(proba, index=x.index, columns=classes)

    def feature_importances(self) -> pd.Series | None:
        """Return feature importances mapped back to human-readable names, if supported."""
        estimator = self.pipeline.named_steps["estimator"]
        if not hasattr(estimator, "feature_importances_"):
            return None
        feature_names = self.pipeline.named_steps["preprocessor"].get_feature_names_out()
        return pd.Series(
            estimator.feature_importances_, index=feature_names
        ).sort_values(ascending=False)
