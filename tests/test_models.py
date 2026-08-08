from __future__ import annotations

from pathlib import Path

import pandas as pd

from pm_mlops.config import ProjectConfig
from pm_mlops.data_processor import DataProcessor
from pm_mlops.models import FailureClassifier
from pm_mlops.models.base_model import DEFAULT_THRESHOLD


def _train_small_model(config: ProjectConfig, sample_df: pd.DataFrame) -> FailureClassifier:
    processor = DataProcessor(config)
    cleaned = processor.clean_data(sample_df)
    train_df, _test_df = processor.split_data(cleaned)
    x_train, y_train = processor.get_features_and_target(train_df)

    model = FailureClassifier(config, data_processor=processor)
    model.train(x_train, y_train)
    return model


def test_train_produces_fitted_pipeline(config: ProjectConfig, sample_df: pd.DataFrame):
    model = _train_small_model(config, sample_df)
    assert model.pipeline is not None


def test_new_model_has_default_threshold(config: ProjectConfig):
    model = FailureClassifier(config)
    assert model.threshold == DEFAULT_THRESHOLD


def test_predict_returns_binary_series(config: ProjectConfig, sample_df: pd.DataFrame):
    model = _train_small_model(config, sample_df)
    processor = DataProcessor(config)
    cleaned = processor.clean_data(sample_df)
    x, _y = processor.get_features_and_target(cleaned)

    predictions = model.predict(x)

    assert set(predictions.unique()).issubset({0, 1})
    assert len(predictions) == len(x)


def test_predict_proba_rows_sum_to_one(config: ProjectConfig, sample_df: pd.DataFrame):
    model = _train_small_model(config, sample_df)
    processor = DataProcessor(config)
    cleaned = processor.clean_data(sample_df)
    x, _y = processor.get_features_and_target(cleaned)

    proba = model.predict_proba(x)

    assert proba.sum(axis=1).round(5).eq(1.0).all()


def test_evaluate_returns_expected_metric_keys(config: ProjectConfig, sample_df: pd.DataFrame):
    processor = DataProcessor(config)
    cleaned = processor.clean_data(sample_df)
    train_df, test_df = processor.split_data(cleaned)
    x_train, y_train = processor.get_features_and_target(train_df)
    x_test, y_test = processor.get_features_and_target(test_df)

    model = FailureClassifier(config, data_processor=processor)
    model.train(x_train, y_train)
    metrics = model.evaluate(x_test, y_test)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_tune_threshold_sets_metadata_and_returns_diagnostics(
    config: ProjectConfig, sample_df: pd.DataFrame
):
    processor = DataProcessor(config)
    cleaned = processor.clean_data(sample_df)
    train_df, _test_df = processor.split_data(cleaned)
    sub_train_df, val_df = processor.split_train_validation(train_df)
    x_sub_train, y_sub_train = processor.get_features_and_target(sub_train_df)
    x_val, y_val = processor.get_features_and_target(val_df)

    model = FailureClassifier(config, data_processor=processor)
    model.train(x_sub_train, y_sub_train)
    result = model.tune_threshold(x_val, y_val)

    assert 0.0 < result["threshold"] < 1.0
    assert model.threshold == result["threshold"]
    assert "threshold_tuning" in model.metadata
    assert model.metadata["threshold_tuning"]["n_validation_rows"] == len(y_val)


def test_tune_threshold_prefers_lower_threshold_when_false_negatives_costly(
    config: ProjectConfig, sample_df: pd.DataFrame
):
    """A high false-negative cost should push the tuned threshold below 0.5,
    since catching more real failures becomes worth more false alarms."""
    processor = DataProcessor(config)
    cleaned = processor.clean_data(sample_df)
    train_df, _test_df = processor.split_data(cleaned)
    sub_train_df, val_df = processor.split_train_validation(train_df)
    x_sub_train, y_sub_train = processor.get_features_and_target(sub_train_df)
    x_val, y_val = processor.get_features_and_target(val_df)

    config.model.threshold_tuning.false_negative_cost = 50.0
    config.model.threshold_tuning.false_positive_cost = 1.0

    model = FailureClassifier(config, data_processor=processor)
    model.train(x_sub_train, y_sub_train)
    result = model.tune_threshold(x_val, y_val)

    assert result["threshold"] <= 0.5


def test_predict_respects_tuned_threshold(config: ProjectConfig, sample_df: pd.DataFrame):
    model = _train_small_model(config, sample_df)
    processor = DataProcessor(config)
    cleaned = processor.clean_data(sample_df)
    x, _y = processor.get_features_and_target(cleaned)

    proba = model.predict_proba(x)[config.target.positive_label]

    # Force a threshold so extreme it should flag ~nothing as failure.
    model.metadata["threshold"] = 0.999999
    predictions_strict = model.predict(x)
    assert predictions_strict.sum() <= (proba >= 0.999999).sum()

    # And a threshold so low it should flag everything.
    model.metadata["threshold"] = 0.0
    predictions_lenient = model.predict(x)
    assert predictions_lenient.sum() == len(x)


def test_save_and_load_roundtrip_preserves_threshold(
    config: ProjectConfig, sample_df: pd.DataFrame, tmp_path: Path
):
    model = _train_small_model(config, sample_df)
    model.metadata["threshold"] = 0.37
    artifact_path = tmp_path / "model.joblib"
    model.save(artifact_path)

    reloaded = FailureClassifier(config)
    reloaded.load(artifact_path)

    assert reloaded.threshold == 0.37
    assert "saved_at" in reloaded.metadata
    assert reloaded.metadata["model_type"] == config.model.type

    processor = DataProcessor(config)
    cleaned = processor.clean_data(sample_df)
    x, _y = processor.get_features_and_target(cleaned)

    original_preds = model.predict(x)
    reloaded_preds = reloaded.predict(x)
    assert original_preds.equals(reloaded_preds)


def test_load_handles_legacy_artifact_without_metadata_envelope(
    config: ProjectConfig, sample_df: pd.DataFrame, tmp_path: Path
):
    """Artifacts saved before the metadata envelope was introduced were a
    bare sklearn pipeline; loading one should fall back to threshold 0.5
    instead of crashing."""
    import joblib

    model = _train_small_model(config, sample_df)
    legacy_path = tmp_path / "legacy_model.joblib"
    joblib.dump(model.pipeline, legacy_path)  # simulate old bare-pipeline artifact

    reloaded = FailureClassifier(config)
    reloaded.load(legacy_path)

    assert reloaded.threshold == DEFAULT_THRESHOLD
    assert reloaded.pipeline is not None


def test_unknown_model_type_raises(config: ProjectConfig):
    config.model.type = "not_a_real_model"
    model = FailureClassifier(config)

    try:
        model.build_pipeline()
        raised = False
    except ValueError:
        raised = True

    assert raised


def test_random_forest_defaults_to_all_cores(config: ProjectConfig):
    config.model.type = "random_forest"
    model = FailureClassifier(config)
    pipeline = model.build_pipeline()

    assert pipeline.named_steps["estimator"].n_jobs == -1


def test_n_jobs_override_is_respected(config: ProjectConfig):
    config.model.type = "random_forest"
    config.model.params = {**config.model.params, "n_jobs": 2}
    model = FailureClassifier(config)
    pipeline = model.build_pipeline()

    assert pipeline.named_steps["estimator"].n_jobs == 2
