"""
app/ml/preprocessing/config.py — Normalization configuration for the ML pipeline.

These XGBoost models (xgb_criticality_regressor.joblib, xgb_priority_classifier.joblib)
are tree-based models. XGBoost is invariant to monotonic feature transformations —
it does not require and was confirmed to not have a fitted scaler artifact accompanying it.

SCALER ARTIFACT STATUS: NOT FOUND / NOT REQUIRED
  A search of the entire project found zero scaler or preprocessing artifacts.
  The MLNormalizer therefore applies a passthrough (identity) transformation
  that validates feature completeness and finiteness before inference.

UPGRADING TO A FITTED SCALER:
  If a future model is trained on normalized data and a scaler artifact is provided,
  drop the scaler file (e.g. scaler.joblib) into models/xgboost/ and update
  SCALER_PATH below. The MLNormalizer will automatically load and apply it.
  Do NOT refit the scaler during inference.
"""

from pathlib import Path

# Path to an optional fitted scaler artifact.
# Set to None if no scaler is used (current state).
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SCALER_PATH: Path | None = None  # e.g. PROJECT_DIR / "models" / "xgboost" / "scaler.joblib"

# Whether the pipeline currently applies normalization.
NORMALIZATION_ENABLED: bool = SCALER_PATH is not None
