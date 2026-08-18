"""
app/ml/preprocessing/normalizer.py — ML feature normalization.

Responsibility:
    Accept a raw ML feature dictionary (produced by MLFeatureBuilder),
    apply the training-time transformation, and return a new, isolated
    normalized dictionary. The input is never mutated.

Current state:
    No scaler artifact is present. The models are tree-based XGBoost
    models that do not require normalization. The normalizer applies
    a passthrough (identity) transformation while enforcing the full
    feature contract (all features present, all values finite).

Upgrade path:
    If a fitted scaler is provided in the future, set SCALER_PATH in
    app/ml/preprocessing/config.py. This class will load it once and
    apply it at inference time without refitting.

Data isolation guarantee:
    This class operates exclusively on the ML feature dictionary.
    It never receives or modifies the original authorization JSON.
"""

import math
import logging
from functools import lru_cache
from pathlib import Path

from app.ml.feature_schema import ML_MODEL_FEATURES
from app.ml.preprocessing.config import SCALER_PATH, NORMALIZATION_ENABLED

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_scaler():
    """
    Load the fitted scaler artifact once and cache it.
    Returns None if no scaler is configured or normalization is disabled.
    """
    if not NORMALIZATION_ENABLED or SCALER_PATH is None:
        logger.debug("ML normalizer: normalization disabled or no scaler configured — passthrough mode active.")
        return None

    if not Path(SCALER_PATH).exists():
        raise FileNotFoundError(
            f"Scaler artifact configured but not found: {SCALER_PATH}"
        )

    import joblib
    scaler = joblib.load(SCALER_PATH)
    logger.info(
        "ML normalizer: loaded scaler '%s' from %s",
        type(scaler).__name__,
        SCALER_PATH,
    )
    return scaler


class MLNormalizer:
    """
    Transforms a raw ML feature dictionary into the normalized representation
    expected by the XGBoost models.

    IMPORTANT:
        - Input dict is never mutated.
        - Returns a new dict with the same keys and transformed values.
        - Raises MLNormalizationError on missing or invalid features.
        - Normalized values are ONLY used by the ML model; they must never
          be forwarded to the LLM, RAG pipeline, Rule Engine, or UI.
    """

    def transform(self, raw_features: dict[str, float]) -> dict[str, float]:
        """
        Validate and transform raw ML features.

        Args:
            raw_features: Dict mapping feature name → raw numeric value.
                          Produced by MLFeatureBuilder; derived from the
                          original authorization JSON without mutation.

        Returns:
            A new dict with transformed feature values in the exact
            model-expected order. Original dict is untouched.

        Raises:
            MLNormalizationError: If any required feature is missing or
                                  contains a non-finite value.
        """
        self._validate(raw_features)
        scaler = _load_scaler()

        if scaler is None:
            # Passthrough: copy values preserving exact feature order.
            normalized = {f: float(raw_features[f]) for f in ML_MODEL_FEATURES}
            logger.debug(
                "ML normalization (passthrough): %d features validated.",
                len(normalized),
            )
        else:
            import pandas as pd
            df = pd.DataFrame(
                [[raw_features[f] for f in ML_MODEL_FEATURES]],
                columns=ML_MODEL_FEATURES,
            )
            scaled_array = scaler.transform(df)
            normalized = {
                f: float(scaled_array[0, i])
                for i, f in enumerate(ML_MODEL_FEATURES)
            }
            logger.debug(
                "ML normalization (%s): %d features transformed.",
                type(scaler).__name__,
                len(normalized),
            )

        return normalized

    @staticmethod
    def _validate(raw_features: dict[str, float]) -> None:
        """Enforce the full 61-feature contract before transformation."""
        received = set(raw_features.keys())
        required = set(ML_MODEL_FEATURES)

        missing = sorted(required - received)
        if missing:
            raise MLNormalizationError(
                f"Missing required ML features: {', '.join(missing)}"
            )

        extra = sorted(received - required)
        if extra:
            # Extra features are not fatal — they are simply ignored.
            logger.warning(
                "ML normalizer: ignoring unexpected features: %s",
                extra,
            )

        invalid_values = []
        for name in ML_MODEL_FEATURES:
            if name in raw_features:
                val = raw_features[name]
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                    invalid_values.append(name)

        if invalid_values:
            raise MLNormalizationError(
                f"Non-finite or invalid numeric value in ML features: {', '.join(invalid_values)}"
            )


class MLNormalizationError(ValueError):
    """Raised when ML feature normalization cannot be completed safely."""
