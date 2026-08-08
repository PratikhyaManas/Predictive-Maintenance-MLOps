"""Generate a synthetic, AI4I-shaped industrial sensor dataset for local demo/dev use.

This repo ships without a proprietary sensor log. Swap this out for a real
export from your historian/SCADA system (or the public
`AI4I 2020 Predictive Maintenance <https://archive.ics.uci.edu/dataset/601>`_
dataset) by pointing ``project_config.yml -> data.raw_path`` at it — the
column names below match that dataset's schema (snake_cased).

The failure label isn't random noise: it's generated from simplified
versions of the same physical failure mechanisms documented for that
dataset (heat dissipation, power, overstrain, tool wear), so the downstream
classifier has genuine, learnable, physically-motivated signal.

Usage
-----
    python scripts/00_generate_sample_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pm_mlops.config import ProjectConfig  # noqa: E402
from pm_mlops.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

# Overstrain thresholds (tool_wear_min * torque_nm) by product quality variant —
# lower-quality tooling (L) fails at a lower cumulative strain than premium (H).
# Calibrated to roughly the 97th percentile of each variant's strain distribution.
_OVERSTRAIN_THRESHOLD = {"L": 4200, "M": 5200, "H": 6200}


def generate_sensor_dataset(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic fleet of machine sensor readings with realistic failures."""
    rng = np.random.default_rng(seed)

    product_type = rng.choice(["L", "M", "H"], size=n_rows, p=[0.6, 0.3, 0.1])

    air_temperature_k = rng.normal(300.0, 2.0, n_rows)
    process_temperature_k = air_temperature_k + rng.normal(10.0, 1.0, n_rows)

    torque_nm = np.clip(rng.normal(40.0, 10.0, n_rows), 3.0, 80.0)
    # Typical industrial spindle speed range, independent of torque so that
    # mechanical power (torque * angular velocity) varies realistically.
    rotational_speed_rpm = np.clip(rng.normal(1700.0, 250.0, n_rows), 800, 3000)

    # Tool wear accumulates over a tool's life; premium (H) tooling lasts longer
    # before being swapped, so its wear distribution skews a bit higher.
    wear_scale = {"L": 90, "M": 110, "H": 140}
    tool_wear_min = np.array(
        [rng.uniform(0, wear_scale[t]) for t in product_type]
    )

    # --- Failure mechanisms (simplified from AI4I 2020 documentation) --- #
    temp_diff = process_temperature_k - air_temperature_k
    heat_dissipation_failure = (temp_diff < 8.6) & (rotational_speed_rpm < 1380)

    power_w = torque_nm * rotational_speed_rpm * (2 * np.pi / 60)
    power_failure = (power_w < 1200) | (power_w > 12500)

    overstrain = tool_wear_min * torque_nm
    overstrain_threshold = np.array([_OVERSTRAIN_THRESHOLD[t] for t in product_type])
    overstrain_failure = overstrain > overstrain_threshold

    tool_wear_failure = (tool_wear_min > 0.92 * np.array([wear_scale[t] for t in product_type])) & (
        rng.random(n_rows) < 0.3
    )

    # Small baseline random-failure rate for unmodelled causes.
    random_failure = rng.random(n_rows) < 0.003

    machine_failure = (
        heat_dissipation_failure
        | power_failure
        | overstrain_failure
        | tool_wear_failure
        | random_failure
    ).astype(int)

    df = pd.DataFrame(
        {
            "product_type": product_type,
            "air_temperature_k": air_temperature_k.round(2),
            "process_temperature_k": process_temperature_k.round(2),
            "rotational_speed_rpm": rotational_speed_rpm.round(1),
            "torque_nm": torque_nm.round(2),
            "tool_wear_min": tool_wear_min.round(1),
            "machine_failure": machine_failure,
        }
    )
    return df


def main() -> None:
    config = ProjectConfig.from_yaml(Path(__file__).resolve().parents[1] / "project_config.yml")
    df = generate_sensor_dataset()

    out_path = Path(config.data.raw_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    logger.info(f"Wrote {len(df)} synthetic sensor readings to {out_path}")
    logger.info(f"Failure rate: {df['machine_failure'].mean():.2%}")


if __name__ == "__main__":
    main()
