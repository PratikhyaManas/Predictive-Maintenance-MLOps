# pm-mlops — Industrial Predictive Maintenance

A complete, modular end-to-end MLOps pipeline that predicts **machine
failure risk from real-time sensor telemetry** (air/process temperature,
rotational speed, torque, tool wear) — structured the same way you'd
structure a production ML project: config-driven, testable, and framework
agnostic (no cloud/platform lock-in).

The feature/target schema mirrors the widely-used **AI4I 2020 Predictive
Maintenance** dataset (UCI), a standard reference for industrial ML —
swap the synthetic data generator for a real historian/SCADA export and
the rest of the pipeline needs no changes.

## What's optimized here (and why)

| Optimization | Where | Why it matters |
|---|---|---|
| **Cost-aware decision threshold** | `models/base_model.py::tune_threshold` | A missed failure typically costs far more than a false alarm. Instead of a hard-coded 0.5 cutoff, the threshold is chosen to minimize `false_negative_cost × FN + false_positive_cost × FP` (configurable in YAML, default 10:1). |
| **Leak-free threshold tuning** | `scripts/02_train_model.py` | The threshold is tuned on a validation slice carved out of *training* data, never on the test set — so reported test metrics stay an honest, untuned estimate of real-world performance. The model is then refit on the full training split (sub-train + validation) before saving, so no data goes to waste. |
| **Versioned model artifacts** | `models/base_model.py::save/load` | Artifacts store a metadata envelope (tuned threshold, training timestamp, model type) alongside the sklearn pipeline — the threshold travels with the model instead of living as a separate, easy-to-desync constant in serving code. Old bare-pipeline artifacts still load fine (fallback to 0.5). |
| **Parallel training** | `models/classifier.py` | RandomForest defaults to `n_jobs=-1` (all cores), overridable per-run via `project_config.yml`. |
| **Config parse caching** | `config.py::from_yaml` | Repeated `from_yaml()` calls across a pipeline run reuse a cached parsed YAML dict instead of re-reading/re-parsing the file, while still returning an independently-mutable config object each time. |
| **Batch scoring endpoint** | `serving/api.py::/predict/batch` | Scoring a fleet of machines is one HTTP round trip instead of N — meaningful at typical fleet-monitoring polling intervals. |
| **Lightweight `/metrics` endpoint** | `serving/api.py` | Zero-dependency in-process counters (prediction volume, high-risk rate) as an integration seam for a real metrics backend. |
| **Containerized serving** | `Dockerfile` | Multi-stage build (slim runtime image, non-root user, healthcheck) for deploying the API to a shop-floor gateway or cloud service. |
| **SAST + SCA scanning** | `.github/workflows/security.yml`, `SECURITY.md` | Bandit + ruff's bandit ruleset catch insecure code patterns; pip-audit + Dependabot catch known-vulnerable dependencies; Trivy scans the built container image. Runs on every push/PR, weekly on a schedule, and locally via `make security`. |

## Why this structure

| Concern | Where it lives | Why |
|---|---|---|
| All tunable settings | `project_config.yml` | Change behavior without touching code |
| Typed config access | `src/pm_mlops/config.py` | Fail fast on bad config, autocomplete in your IDE |
| Data loading/cleaning/splitting | `src/pm_mlops/data_processor.py` | One reusable class; clips out-of-range sensor readings, stratified splits so rare failures aren't lost |
| Model contract | `src/pm_mlops/models/base_model.py` | Swap algorithms without touching callers |
| Concrete model | `src/pm_mlops/models/classifier.py` | Preprocessing + estimator saved as one artifact |
| Drift monitoring | `src/pm_mlops/monitoring/drift.py` | Lightweight PSI/TVD checks — catches sensor miscalibration or regime change, zero extra infra |
| REST serving | `src/pm_mlops/serving/api.py` | FastAPI app, loads the same artifact scripts produce, returns a risk tier |
| Orchestration | `scripts/0N_*.py` | Thin, numbered, CLI-runnable pipeline stages |
| Tests | `tests/` | One test module per package module |

## Project layout

```
predictive-maintenance-mlops/
├── .github/workflows/ci.yml       # lint + test + pipeline smoke test on every push
├── data/                          # raw & processed data (gitignored, regenerate locally)
├── notebooks/                     # exploratory analysis
├── scripts/
│   ├── 00_generate_sample_data.py # synthetic sensor-fleet generator (swap for real data)
│   ├── 01_process_data.py         # clean + split
│   ├── 02_train_model.py          # train + evaluate + save (+ optional MLflow logging)
│   ├── 03_evaluate_model.py       # re-evaluate a saved artifact
│   ├── 04_predict.py              # batch inference on a CSV of readings
│   └── 05_refresh_monitor.py      # drift report: reference vs current sensor data
├── src/pm_mlops/
│   ├── config.py                  # ProjectConfig (pydantic), loaded from project_config.yml
│   ├── data_processor.py          # DataProcessor: load/clean/split/preprocess
│   ├── models/
│   │   ├── base_model.py          # BaseModel ABC: train/predict/evaluate/save/load
│   │   └── classifier.py          # FailureClassifier(BaseModel)
│   ├── serving/
│   │   ├── api.py                 # FastAPI app
│   │   └── schemas.py             # request/response pydantic models
│   ├── monitoring/
│   │   └── drift.py               # DriftMonitor (PSI + categorical shift)
│   └── utils/logging.py           # shared logger factory
├── tests/                         # pytest suite mirroring the src/ layout
├── project_config.yml             # single source of truth for the whole pipeline
├── pyproject.toml
└── Makefile
```

## Architecture diagram

```mermaid
flowchart LR
  A[Raw Sensor Data CSV\ndata/machine_sensors.csv]
  B[scripts/00_generate_sample_data.py\nSynthetic data generation]
  C[src/pm_mlops/data_processor.py\nClean, validate, stratified split]
  D[data/processed/train.csv\ndata/processed/test.csv]

  E[scripts/02_train_model.py\nTraining pipeline]
  F[src/pm_mlops/models/classifier.py\nFailureClassifier + preprocessing]
  G[models/model.pkl\nModel artifact + threshold metadata]

  H[scripts/03_evaluate_model.py\nOffline evaluation]
  I[Metrics\nPrecision/Recall/F1/ROC-AUC]

  J[scripts/05_refresh_monitor.py]
  K[src/pm_mlops/monitoring/drift.py\nDrift report PSI/TVD]

  L[src/pm_mlops/serving/api.py\nFastAPI service]
  M[/predict and /predict/batch\nInference responses]
  N[/metrics\nServing counters]

  B --> A
  A --> C --> D
  D --> E --> F --> G
  G --> H --> I
  D --> J --> K
  G --> L
  L --> M
  L --> N
```

## The problem being modeled

Each row is one sensor reading off a machine. `machine_failure` is 1 if
the machine failed around that reading. The synthetic generator encodes
simplified versions of real, documented industrial failure mechanisms so
the signal is genuine rather than random noise:

- **Heat dissipation failure** — process/air temperature difference too
  small combined with low rotational speed
- **Power failure** — mechanical power (torque × angular velocity) outside
  a safe operating band
- **Overstrain failure** — cumulative `tool_wear × torque` exceeds a
  threshold that depends on tooling quality (`product_type`: L/M/H)
- **Tool wear failure** — tool nearing end-of-life, probabilistic

Failures are intentionally rare (a few percent of readings), which is why
the pipeline stratifies splits and reports precision/recall/F1/ROC-AUC
rather than leaning on accuracy alone — a model that never predicts
failure can look >95% "accurate" on an imbalanced fleet while being
operationally useless.

## Quickstart

```bash
# 1. Install (uses uv, https://docs.astral.sh/uv/)
uv sync --extra dev --extra serving --extra tracking

# 2. Generate a synthetic sensor dataset shaped like the real AI4I dataset
#    (swap this out for real data by pointing project_config.yml -> data.raw_path
#    at your own CSV with the same columns)
python scripts/00_generate_sample_data.py

# 3. Run the full pipeline
python scripts/01_process_data.py
python scripts/02_train_model.py
python scripts/03_evaluate_model.py
python scripts/05_refresh_monitor.py

# or, equivalently:
make pipeline
```

## Serving predictions

```bash
uvicorn pm_mlops.serving.api:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "press-014",
    "air_temperature_k": 300.5,
    "process_temperature_k": 305.0,
    "rotational_speed_rpm": 1350.0,
    "torque_nm": 65.0,
    "tool_wear_min": 210.0,
    "product_type": "L"
  }'
```

```json
{
  "machine_id": "press-014",
  "machine_failure_predicted": true,
  "failure_probability": 0.83,
  "risk_level": "high",
  "decision_threshold": 0.34,
  "model_version": "0.1.0"
}
```

Score an entire fleet in one call:

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"readings": [{"air_temperature_k": 300.5, "process_temperature_k": 305.0, "rotational_speed_rpm": 1350.0, "torque_nm": 65.0, "tool_wear_min": 210.0, "product_type": "L"}]}'
```

Check serving metrics:

```bash
curl http://localhost:8000/metrics
```

### Running in Docker

```bash
make pipeline        # train a model artifact first — the image bundles it
make docker-build
make docker-run       # serves on http://localhost:8000
```

## Batch predictions

```bash
python scripts/04_predict.py --input data/machine_sensors.csv --output predictions.csv
```

## Testing & linting

```bash
pytest              # full test suite with coverage
ruff check .         # lint (includes flake8-bandit "S" security rules)
ruff format .        # format
pre-commit install   # enable git hooks locally (lint, format, SAST, SCA)
```

## Security

SAST (`bandit`) and SCA (`pip-audit` + Dependabot) run on every push/PR and
weekly on a schedule (see `.github/workflows/security.yml`), plus a Trivy
scan of the built Docker image. Run the same checks locally:

```bash
uv sync --extra security
make security        # bandit + pip-audit
```

See [`SECURITY.md`](SECURITY.md) for the full breakdown and the reasoning
behind a couple of deliberate, documented suppressions.

## Swapping in real sensor data

Replace the synthetic CSV with a real export that has these columns, then
re-run the pipeline — no code changes needed:

`product_type, air_temperature_k, process_temperature_k, rotational_speed_rpm, torque_nm, tool_wear_min, machine_failure`

To model a different industrial problem entirely (e.g. remaining useful
life, a specific failure-mode classifier, quality-defect prediction), edit
`project_config.yml`'s `features`/`target` blocks — `DataProcessor` and
`FailureClassifier` are generic enough to follow.

## Extending

- **New model type**: add a class implementing `BaseModel` in `models/`,
  register it in `_ESTIMATORS` in `classifier.py`, and set `model.type` in
  `project_config.yml`.
- **Experiment tracking**: set `mlflow.enabled: true` in the config; training
  runs are logged automatically if `mlflow` is installed
  (`uv sync --extra tracking`).
- **Production monitoring**: `monitoring/drift.py` is intentionally
  dependency-light; swap in `evidently` or a historian-integrated monitor
  by implementing the same `compute_report`/`save_report` interface.
- **Alerting**: the `/predict` endpoint already returns a `risk_level`
  tier (`low`/`medium`/`high`) — wire `high` results to a maintenance
  ticketing system or shop-floor alert in your own integration layer.
