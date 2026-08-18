"""
app/ml/preprocessing/feature_builder.py — ML feature extraction from authorization data.

Responsibility:
    Convert the incoming authorization feature dictionary (pre-validated by the API
    layer) into a raw ML feature dictionary using the exact 61-feature contract
    required by the XGBoost models.

Current pipeline context:
    The API currently receives a pre-built numeric feature dict from the external
    backend (the caller already computes the 61 feature values before calling this
    inference service). The MLFeatureBuilder therefore performs a clean extraction
    and validation pass — it maps the caller-supplied dict to the canonical feature
    order, applies type coercion, and provides a clear error for any missing field.

Data isolation guarantee:
    - Input dict (raw_features) is NEVER mutated.
    - Returns a new dict containing ONLY the 61 ML features.
    - Text/clinical/categorical fields NOT in ML_MODEL_FEATURES are silently ignored.
    - The original authorization JSON remains unchanged for the LLM, RAG, and
      Rule Engine paths.

Feature contract (61 features, in order):
    See app/ml/feature_schema.py for the authoritative ordered list.

Feature source documentation:
    Feature               | Source                         | Type  | Default
    ----------------------|--------------------------------|-------|--------
    FIPS                  | features["FIPS"]               | float | 0.0
    ZIP                   | features["ZIP"]                | float | 0.0
    LON                   | features["LON"]                | float | 0.0
    HEALTHCARE_EXPENSES   | features["HEALTHCARE_EXPENSES"]| float | 0.0
    HEALTHCARE_COVERAGE   | features["HEALTHCARE_COVERAGE"]| float | 0.0
    INCOME                | features["INCOME"]             | float | 0.0
    encounter_count       | features[...]                  | float | 0.0
    ... (all 61 features passed through from the validated API request)
    age                   | features["age"]                | float | 0.0
    coverage_expense_ratio| features[...]                  | float | 0.0
    claim_per_encounter   | features[...]                  | float | 0.0

    One-hot / binary features (PREFIX_*, SUFFIX_*, MARITAL_*, RACE_*,
    ETHNICITY_*, GENDER_M, COUNTY_*) are expected as 0.0/1.0 floats
    as provided by the external feature engineering pipeline.
"""

import logging

from app.ml.feature_schema import ML_MODEL_FEATURES

logger = logging.getLogger(__name__)


class MLFeatureBuilder:
    """
    Extracts the exact 61 ML features from the incoming validated feature dict.

    IMPORTANT:
        - Input dict is never mutated.
        - Only features in ML_MODEL_FEATURES are included in the output.
        - Missing features with a registered default are filled and logged.
        - Missing features without a default raise MLFeatureExtractionError.
    """

    # Default values for features that may be absent in the incoming dict.
    # These defaults are safe for XGBoost (tree-based, NaN-tolerant).
    FEATURE_DEFAULTS: dict[str, float] = {
        feature: 0.0 for feature in ML_MODEL_FEATURES
    }

    def build(self, raw_features: dict[str, float]) -> dict[str, float]:
        """
        Extract and validate ML features from the pre-computed feature dict.

        Args:
            raw_features: Validated feature dict from PredictionRequest.
                          Contains at least all 61 required features as floats.
                          This dict is NOT mutated.

        Returns:
            A new dict mapping each of the 61 ML feature names to its float value,
            in the canonical ML_MODEL_FEATURES order.

        Raises:
            MLFeatureExtractionError: If a required feature has no value and
                                      no registered default.
        """
        ml_features: dict[str, float] = {}
        missing_without_default: list[str] = []

        for feature_name in ML_MODEL_FEATURES:
            if feature_name in raw_features:
                ml_features[feature_name] = float(raw_features[feature_name])
            elif feature_name in self.FEATURE_DEFAULTS:
                default_val = self.FEATURE_DEFAULTS[feature_name]
                ml_features[feature_name] = default_val
                logger.warning(
                    "ML feature '%s' not in input — using default %.4f",
                    feature_name,
                    default_val,
                )
            else:
                missing_without_default.append(feature_name)

        if missing_without_default:
            raise MLFeatureExtractionError(
                f"Required ML features missing with no default: "
                f"{', '.join(missing_without_default)}"
            )

        logger.debug(
            "ML feature extraction complete: %d features extracted.",
            len(ml_features),
        )
        return ml_features


class MLFeatureExtractionError(ValueError):
    """Raised when required ML features cannot be extracted from the input."""
