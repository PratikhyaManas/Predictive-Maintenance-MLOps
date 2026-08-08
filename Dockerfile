# syntax=docker/dockerfile:1

# --- Builder stage: install dependencies with uv (fast, reproducible) ---
# Pinned to a specific patch tag rather than a floating "3.12-slim" tag to
# reduce supply-chain drift; for production, pin further to an immutable
# digest (`python:3.12.5-slim@sha256:...`) and let Dependabot (see
# .github/dependabot.yml) open a PR whenever a new patch is released.
FROM python:3.12.5-slim AS builder

RUN pip install --no-cache-dir "uv==0.4.*"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package + serving extras into a self-contained venv.
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --python /opt/venv/bin/python ".[serving]"

# --- Runtime stage: slim image, no build toolchain ---
FROM python:3.12.5-slim AS runtime

# Security patches for OS packages present in the base image (defense in
# depth alongside the Trivy image scan in CI, which flags anything this
# misses or that lands after the image is built).
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src ./src
COPY project_config.yml ./
# The trained artifact must exist before building the image — run
# `make pipeline` locally first, or mount it at runtime with -v instead.
COPY models ./models

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "pm_mlops.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
