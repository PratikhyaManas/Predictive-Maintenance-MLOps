# Exploratory Data Analysis — Predictive Maintenance

This is a placeholder for exploratory analysis. Convert to a `.ipynb` in
Jupyter, or open the CSV directly:

```python
import pandas as pd
from pm_mlops.config import ProjectConfig

config = ProjectConfig.from_yaml("../project_config.yml")
df = pd.read_csv(config.data.raw_path)

df.describe()
df["machine_failure"].value_counts(normalize=True)
df.groupby("machine_failure")[config.features.numerical].mean()
```

Useful starting points:
- Compare sensor distributions (`torque_nm`, `tool_wear_min`, `rotational_speed_rpm`,
  temperature differential) between failed and healthy readings.
- Check class imbalance (`machine_failure` is intentionally rare, ~5-7%).
- Look at failure rate by `product_type` (tooling quality variant) — lower-grade
  tooling should show a modestly higher failure rate.
- Plot `tool_wear_min * torque_nm` (a proxy for cumulative mechanical strain)
  against failure status — this is the single strongest predictive signal
  in the synthetic data.
