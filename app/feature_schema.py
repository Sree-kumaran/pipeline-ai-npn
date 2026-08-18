"""
app/feature_schema.py — Backward-compatible export of the canonical ML feature schema.

Authoritative source: app/ml/feature_schema.py
"""

from app.ml.feature_schema import ML_MODEL_FEATURES as MODEL_FEATURES, PRIORITY_CLASS_LABELS

__all__ = ["MODEL_FEATURES", "PRIORITY_CLASS_LABELS"]