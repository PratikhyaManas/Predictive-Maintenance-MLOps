# Security

## Scanning in this repo

| Category | Tool | Where it runs |
|---|---|---|
| **SAST** (static code analysis) | [`bandit`](https://bandit.readthedocs.io/) | `.github/workflows/security.yml` on every push/PR + weekly; also in `.pre-commit-config.yaml` |
| **SAST** (fast, in-editor) | `ruff` with the `S` (flake8-bandit) ruleset | Every `ruff check .` invocation — CI, pre-commit, and locally |
| **SCA** (dependency vulnerabilities) | [`pip-audit`](https://github.com/pypa/pip-audit) | `.github/workflows/security.yml` on every push/PR + weekly; also in `.pre-commit-config.yaml` |
| **SCA** (dependency freshness) | GitHub Dependabot | `.github/dependabot.yml` — weekly PRs for pip, Docker base image, and GitHub Actions dependencies |
| **Container image scanning** | [`trivy`](https://github.com/aquasecurity/trivy) | `.github/workflows/security.yml`, scans the built Docker image for OS + application-layer CVEs (fails on CRITICAL/HIGH) |

Run everything locally before pushing:

```bash
uv sync --extra security
make security      # = make sast + make sca
```

## Design choices worth knowing about

- **Bandit config** lives in `[tool.bandit]` in `pyproject.toml` and excludes
  `tests/` (test code legitimately uses `assert`, which Bandit flags as
  B101 in production code but is pytest's normal idiom in tests).
- **`0.0.0.0` bind address** in `ServingConfig` (`src/pm_mlops/config.py`) is
  an intentional, explicitly-suppressed finding (`# nosec B104`) — it's the
  container's listen address, reached via Docker's port mapping or an
  internal cluster network, not exposed directly to the public internet.
- **Docker base image** is pinned to a specific patch tag
  (`python:3.12.5-slim`) rather than a floating tag, and the runtime stage
  runs `apt-get upgrade` for OS-level security patches. For production,
  pin further to an immutable digest and let Dependabot open the bump PRs.
- **Model artifacts** are loaded with `joblib.load`, which deserializes via
  pickle — only ever load artifacts your own pipeline produced
  (`scripts/02_train_model.py`), never an artifact from an untrusted
  source, since pickle deserialization can execute arbitrary code.

## Reporting a vulnerability

This is a demo/reference project, not a maintained public package. If
you're adapting it for a real deployment and find a security issue in
*this scaffolding itself*, open an issue describing it. For issues in a
dependency, `pip-audit`/Dependabot will generally already have flagged it —
check `.github/workflows/security.yml` run history first.
