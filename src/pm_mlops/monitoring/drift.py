"""Lightweight drift monitoring — no heavyweight dependencies required.

In an industrial setting, sensor drift can mean a genuinely different
regime (a new product line, a recalibrated sensor, a machine reaching a
new life stage) or simply a fouled probe. Either way it's a signal that
the model's training distribution may no longer match reality.

Computes Population Stability Index (PSI) for numerical features and a
simple distribution-shift check for categoricals, comparing a reference
(training) dataset against new incoming data. This keeps the pipeline
runnable anywhere while still giving a genuine, actionable drift signal;
swap in `evidently` or a proper historian-integrated monitor for production.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pm_mlops.config import ProjectConfig
from pm_mlops.utils import get_logger

logger = get_logger(__name__)


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two numeric distributions.

    Rule of thumb: PSI < 0.1 -> no significant shift, 0.1-0.25 -> moderate
    shift worth investigating, > 0.25 -> significant shift.
    """
    edges = np.histogram_bin_edges(reference.dropna(), bins=bins)
    ref_counts, _ = np.histogram(reference.dropna(), bins=edges)
    cur_counts, _ = np.histogram(current.dropna(), bins=edges)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-4, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-4, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _categorical_shift(reference: pd.Series, current: pd.Series) -> float:
    """Total variation distance between two categorical distributions (0-1)."""
    ref_freq = reference.value_counts(normalize=True)
    cur_freq = current.value_counts(normalize=True)
    categories = set(ref_freq.index) | set(cur_freq.index)
    diff = sum(abs(ref_freq.get(c, 0) - cur_freq.get(c, 0)) for c in categories)
    return float(diff / 2)


class DriftMonitor:
    """Compares a reference dataset against new data and flags drift."""

    def __init__(self, config: ProjectConfig):
        self.config = config

    def compute_report(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> dict:
        """Compute per-feature drift scores and an overall pass/fail flag."""
        threshold = self.config.monitoring.drift_threshold
        results: dict[str, dict] = {}

        for col in self.config.features.numerical:
            score = _psi(reference_df[col], current_df[col])
            results[col] = {"type": "numerical", "psi": round(score, 4), "drifted": score > threshold}

        for col in self.config.features.categorical:
            score = _categorical_shift(reference_df[col], current_df[col])
            results[col] = {
                "type": "categorical",
                "distance": round(score, 4),
                "drifted": score > threshold,
            }

        any_drift = any(v["drifted"] for v in results.values())
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_reference_rows": len(reference_df),
            "n_current_rows": len(current_df),
            "drift_threshold": threshold,
            "any_feature_drifted": any_drift,
            "features": results,
        }

        if any_drift:
            logger.warning("Drift detected in one or more sensor features")
        else:
            logger.info("No significant drift detected")

        return report

    def save_report(self, report: dict, path: str | Path | None = None) -> Path:
        out_dir = Path(path or self.config.monitoring.report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"drift_report_{stamp}.json"
        out_path.write_text(json.dumps(report, indent=2))
        logger.info(f"Saved drift report to {out_path}")
        return out_path
