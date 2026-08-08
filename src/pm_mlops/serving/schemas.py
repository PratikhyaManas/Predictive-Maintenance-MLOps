"""Request/response schemas for the failure-prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """A single real-time sensor reading from a machine on the shop floor."""

    machine_id: str | None = Field(default=None, examples=["press-014"])
    air_temperature_k: float = Field(ge=250, le=340, examples=[300.5])
    process_temperature_k: float = Field(ge=260, le=350, examples=[310.2])
    rotational_speed_rpm: float = Field(ge=500, le=3500, examples=[1450.0])
    torque_nm: float = Field(ge=0, le=100, examples=[42.8])
    tool_wear_min: float = Field(ge=0, le=300, examples=[108.0])
    product_type: str = Field(examples=["M"], description="Product quality variant: L, M, or H")


class PredictionResponse(BaseModel):
    machine_id: str | None = None
    machine_failure_predicted: bool
    failure_probability: float
    risk_level: str
    decision_threshold: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Score an entire fleet of machines in one call instead of N round trips."""

    readings: list[SensorReading]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    high_risk_count: int


class MetricsResponse(BaseModel):
    total_predictions_served: int
    high_risk_predictions: int
    average_failure_probability: float | None
    model_threshold: float
    model_loaded: bool
