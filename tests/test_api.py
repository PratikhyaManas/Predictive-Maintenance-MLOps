from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pm_mlops.config import ProjectConfig
from pm_mlops.data_processor import DataProcessor
from pm_mlops.models import FailureClassifier

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pm_mlops.serving import api  # noqa: E402

_SAMPLE_PAYLOAD = {
    "air_temperature_k": 300.5,
    "process_temperature_k": 310.2,
    "rotational_speed_rpm": 1450.0,
    "torque_nm": 42.8,
    "tool_wear_min": 108.0,
    "product_type": "M",
}


@pytest.fixture
def trained_model(config: ProjectConfig, sample_df: pd.DataFrame, tmp_path: Path) -> FailureClassifier:
    processor = DataProcessor(config)
    cleaned = processor.clean_data(sample_df)
    train_df, val_df = processor.split_train_validation(cleaned)
    x_train, y_train = processor.get_features_and_target(train_df)
    x_val, y_val = processor.get_features_and_target(val_df)

    model = FailureClassifier(config, data_processor=processor)
    model.train(x_train, y_train)
    model.tune_threshold(x_val, y_val)
    model.save(tmp_path / "model.joblib")
    return model


@pytest.fixture(autouse=True)
def _reset_metrics():
    api._metrics.update(
        {"total_predictions_served": 0, "high_risk_predictions": 0, "probability_sum": 0.0}
    )
    yield


def test_health_endpoint():
    with TestClient(api.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_503_without_model(monkeypatch):
    monkeypatch.setattr(api, "_state", {})
    with TestClient(api.app) as client:
        api._state["model"] = None
        response = client.post("/predict", json=_SAMPLE_PAYLOAD)
    assert response.status_code == 503


def test_predict_endpoint_returns_prediction(config: ProjectConfig, trained_model: FailureClassifier):
    with TestClient(api.app) as client:
        api._state["model"] = trained_model
        api._state["config"] = config
        response = client.post("/predict", json=_SAMPLE_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert "machine_failure_predicted" in body
    assert body["risk_level"] in {"low", "medium", "high"}
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["decision_threshold"] == pytest.approx(trained_model.threshold)


def test_predict_endpoint_echoes_machine_id(config: ProjectConfig, trained_model: FailureClassifier):
    payload = {**_SAMPLE_PAYLOAD, "machine_id": "press-014"}
    with TestClient(api.app) as client:
        api._state["model"] = trained_model
        api._state["config"] = config
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["machine_id"] == "press-014"


def test_predict_batch_endpoint(config: ProjectConfig, trained_model: FailureClassifier):
    with TestClient(api.app) as client:
        api._state["model"] = trained_model
        api._state["config"] = config
        response = client.post(
            "/predict/batch",
            json={"readings": [_SAMPLE_PAYLOAD, {**_SAMPLE_PAYLOAD, "machine_id": "m-2"}]},
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["high_risk_count"] >= 0


def test_predict_batch_endpoint_rejects_empty_list(
    config: ProjectConfig, trained_model: FailureClassifier
):
    with TestClient(api.app) as client:
        api._state["model"] = trained_model
        api._state["config"] = config
        response = client.post("/predict/batch", json={"readings": []})

    assert response.status_code == 400


def test_metrics_endpoint_tracks_predictions(config: ProjectConfig, trained_model: FailureClassifier):
    with TestClient(api.app) as client:
        api._state["model"] = trained_model
        api._state["config"] = config
        client.post("/predict", json=_SAMPLE_PAYLOAD)
        client.post("/predict", json=_SAMPLE_PAYLOAD)
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["total_predictions_served"] == 2
    assert body["model_loaded"] is True
    assert body["model_threshold"] == pytest.approx(trained_model.threshold)
