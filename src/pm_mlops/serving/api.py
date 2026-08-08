"""FastAPI application serving the trained FailureClassifier.

Run locally with:
    uvicorn pm_mlops.serving.api:app --reload --port 8000

The model is loaded once at startup (lifespan hook) rather than per-request,
and all business logic stays in `pm_mlops.models`, so this file is a thin
transport layer only — the kind that would sit behind an OPC-UA/MQTT
gateway or a shop-floor dashboard in a real deployment.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from fastapi import FastAPI, HTTPException

from pm_mlops import __version__
from pm_mlops.config import ProjectConfig
from pm_mlops.models import FailureClassifier
from pm_mlops.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    MetricsResponse,
    PredictionResponse,
    SensorReading,
)
from pm_mlops.utils import get_logger

if TYPE_CHECKING:
    from pm_mlops.models.base_model import BaseModel

logger = get_logger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[3] / "project_config.yml"

_state: dict[str, "BaseModel | ProjectConfig"] = {}

# In-memory serving metrics. Deliberately dependency-free (no prometheus_client)
# so the API has zero extra requirements; swap for a real metrics backend
# (Prometheus, CloudWatch, ...) in production — the /metrics endpoint below
# is the single seam where that would plug in.
_metrics = {
    "total_predictions_served": 0,
    "high_risk_predictions": 0,
    "probability_sum": 0.0,
}


def _risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.3:
        return "medium"
    return "low"


def _record_prediction(probability: float) -> None:
    _metrics["total_predictions_served"] += 1
    _metrics["probability_sum"] += probability
    if _risk_level(probability) == "high":
        _metrics["high_risk_predictions"] += 1


@asynccontextmanager
async def lifespan(_: FastAPI):
    config = ProjectConfig.from_yaml(CONFIG_PATH)
    model = FailureClassifier(config)
    try:
        model.load()
        logger.info(f"Model artifact loaded successfully (threshold={model.threshold:.3f})")
    except FileNotFoundError:
        logger.warning(
            "No trained model artifact found. Run scripts/02_train_model.py first; "
            "/predict will fail until then."
        )
    _state["config"] = config
    _state["model"] = model
    yield
    _state.clear()


app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts machine failure risk from real-time sensor telemetry.",
    version=__version__,
    lifespan=lifespan,
)


def _get_model() -> "BaseModel":
    model = _state.get("model")
    if model is None or model.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first (scripts/02_train_model.py).",
        )
    return model


def _score(model: "BaseModel", readings: list[SensorReading]) -> list[PredictionResponse]:
    df = pd.DataFrame([r.model_dump() for r in readings])
    df["product_type"] = df["product_type"].astype(str).str.upper()
    feature_cols = [c for c in df.columns if c != "machine_id"]

    try:
        proba = model.predict_proba(df[feature_cols])
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Prediction failed: {exc}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    positive_col = 1 if 1 in proba.columns else proba.columns[-1]
    responses = []
    for i, reading in enumerate(readings):
        failure_prob = float(proba[positive_col].iloc[i])
        _record_prediction(failure_prob)
        responses.append(
            PredictionResponse(
                machine_id=reading.machine_id,
                machine_failure_predicted=bool(failure_prob >= model.threshold),
                failure_probability=round(failure_prob, 4),
                risk_level=_risk_level(failure_prob),
                decision_threshold=model.threshold,
                model_version=__version__,
            )
        )
    return responses


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading) -> PredictionResponse:
    model = _get_model()
    return _score(model, [reading])[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score an entire fleet in one call — far cheaper than N single requests
    when polling hundreds or thousands of machines on a fixed interval."""
    model = _get_model()
    if not request.readings:
        raise HTTPException(status_code=400, detail="`readings` must contain at least one item.")
    predictions = _score(model, request.readings)
    high_risk_count = sum(1 for p in predictions if p.risk_level == "high")
    return BatchPredictionResponse(predictions=predictions, high_risk_count=high_risk_count)


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Lightweight in-process serving metrics (resets on restart)."""
    model = _state.get("model")
    total = _metrics["total_predictions_served"]
    avg_prob = (_metrics["probability_sum"] / total) if total else None
    return MetricsResponse(
        total_predictions_served=total,
        high_risk_predictions=_metrics["high_risk_predictions"],
        average_failure_probability=round(avg_prob, 4) if avg_prob is not None else None,
        model_threshold=model.threshold if model else 0.5,
        model_loaded=bool(model and model.pipeline is not None),
    )
