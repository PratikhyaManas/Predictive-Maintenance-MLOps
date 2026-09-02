"""Data loading, cleaning, splitting, and preprocessing for sensor telemetry.

Design mirrors a typical production MLOps pattern: a single
:class:`DataProcessor` owns the full path from raw sensor-log CSV to
model-ready train/test splits and a fitted ``sklearn`` preprocessing
pipeline, so the same class is reusable from notebooks, scripts, tests, or
a serving layer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pm_mlops.config import ProjectConfig
from pm_mlops.utils import get_logger

logger = get_logger(__name__)

# Physically implausible sensor readings are treated as data-quality issues,
# not as ground truth. Bounds are intentionally generous around realistic
# operating ranges for the underlying (synthetic) machine fleet.
_SENSOR_BOUNDS = {
    "air_temperature_k": (250.0, 340.0),
    "process_temperature_k": (260.0, 350.0),
    "rotational_speed_rpm": (500.0, 3500.0),
    "torque_nm": (0.0, 100.0),
    "tool_wear_min": (0.0, 300.0),
}


class DataProcessor:
    """Loads, validates, cleans, splits, and preprocesses telemetry data."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.df: pd.DataFrame | None = None

    def _resolve_df(self, df: pd.DataFrame | None, *, context: str) -> pd.DataFrame:
        """Return an explicit dataframe or the processor-held one, with a clear error."""
        resolved = df if df is not None else self.df
        if resolved is None:
            raise ValueError(
                f"No dataframe available for {context}. "
                "Pass `df=` explicitly or call `load_data()` first."
            )
        return resolved

    # ------------------------------------------------------------------ #
    # Loading & cleaning
    # ------------------------------------------------------------------ #
    def load_data(self, path: str | Path | None = None) -> pd.DataFrame:
        """Load the raw sensor-log CSV into memory."""
        data_path = Path(path or self.config.data.raw_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Raw data not found at {data_path}. "
                "Run `python scripts/00_generate_sample_data.py` first, "
                "or point project_config.yml at your own sensor log export."
            )
        logger.info(f"Loading raw sensor data from {data_path}")
        self.df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
        return self.df

    def clean_data(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Clean sensor telemetry: dedupe, coerce types, clip out-of-range readings."""
        df = self._resolve_df(df, context="cleaning").copy()

        before = len(df)
        df = df.drop_duplicates()
        if len(df) != before:
            logger.info(f"Dropped {before - len(df)} duplicate rows")

        required = {self.config.target.column, *self.config.features.all}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")

        for col in self.config.features.numerical:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col in _SENSOR_BOUNDS:
                lo, hi = _SENSOR_BOUNDS[col]
                out_of_range = ~df[col].between(lo, hi) & df[col].notna()
                if out_of_range.any():
                    logger.warning(
                        f"{out_of_range.sum()} out-of-range readings clipped for '{col}' "
                        f"(valid range [{lo}, {hi}])"
                    )
                    df[col] = df[col].clip(lower=lo, upper=hi)

        df[self.config.features.numerical] = df[self.config.features.numerical].fillna(
            df[self.config.features.numerical].median(numeric_only=True)
        )

        for col in self.config.features.categorical:
            df[col] = df[col].fillna("unknown").astype(str).str.upper()

        df[self.config.target.column] = df[self.config.target.column].astype(int)

        self.df = df
        return df

    # ------------------------------------------------------------------ #
    # Splitting
    # ------------------------------------------------------------------ #
    def _stratified_split(
        self, df: pd.DataFrame, test_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Shared stratified split helper used for test/validation slicing."""
        target = self.config.target.column
        return train_test_split(
            df,
            test_size=test_size,
            random_state=self.config.random_seed,
            stratify=df[target],
        )

    def split_data(
        self, df: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Stratified train/test split on the target column.

        Stratification matters here specifically because machine failures
        are rare events (typically a few percent of readings); a random
        split without stratification risks a test set with zero failures.
        """
        df = self._resolve_df(df, context="splitting")
        train_df, test_df = self._stratified_split(df, self.config.data.test_size)
        logger.info(f"Split into train={len(train_df)} rows, test={len(test_df)} rows")
        return train_df, test_df

    def save_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Persist train/test splits to the paths configured in YAML."""
        Path(self.config.data.processed_dir).mkdir(parents=True, exist_ok=True)
        train_df.to_csv(self.config.data.train_path, index=False)
        test_df.to_csv(self.config.data.test_path, index=False)
        logger.info(
            f"Saved splits to {self.config.data.train_path} and {self.config.data.test_path}"
        )

    def split_train_validation(
        self, train_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Carve a stratified validation slice out of the training split.

        Used exclusively for cost-aware decision-threshold tuning
        (`BaseModel.tune_threshold`), kept disjoint from the test set so
        the threshold is never tuned against the data used for final
        reported metrics.
        """
        sub_train_df, val_df = self._stratified_split(
            train_df, self.config.data.validation_size
        )
        logger.info(
            f"Split training data into sub_train={len(sub_train_df)} rows, "
            f"validation={len(val_df)} rows"
        )
        return sub_train_df, val_df

    # ------------------------------------------------------------------ #
    # Preprocessing pipeline (feature engineering)
    # ------------------------------------------------------------------ #
    def build_preprocessor(self) -> ColumnTransformer:
        """Build (but do not fit) the sklearn preprocessing pipeline.

        Returned unfitted so it can be embedded directly inside a model
        :class:`~sklearn.pipeline.Pipeline`, keeping preprocessing and the
        estimator serialized together as a single artifact — critical for
        industrial deployments where the serving environment must never
        drift out of sync with how the model was trained.
        """
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, self.config.features.numerical),
                ("cat", categorical_pipeline, self.config.features.categorical),
            ]
        )

    def get_features_and_target(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Split a dataframe into model-ready X, y."""
        x = df[self.config.features.all]
        y = df[self.config.target.column]
        return x, y
