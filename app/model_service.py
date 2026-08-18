from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from app.ml.feature_schema import ML_MODEL_FEATURES, PRIORITY_CLASS_LABELS
from app.ml.preprocessing.feature_builder import MLFeatureBuilder
from app.ml.preprocessing.normalizer import MLNormalizer


PROJECT_DIR = Path(__file__).resolve().parent.parent
XGBOOST_DIR = PROJECT_DIR / "models" / "xgboost"

PRIORITY_MODEL_PATH = XGBOOST_DIR / "xgb_priority_classifier.joblib"
CRITICALITY_MODEL_PATH = XGBOOST_DIR / "xgb_criticality_regressor.joblib"

_feature_builder = MLFeatureBuilder()
_normalizer = MLNormalizer()


@lru_cache
def load_models():
    """
    Load the XGBoost models once and reuse them for every request.
    """
    priority_model = joblib.load(PRIORITY_MODEL_PATH)
    criticality_model = joblib.load(CRITICALITY_MODEL_PATH)

    return priority_model, criticality_model


def predict(features: dict[str, float]) -> dict:
    """
    Extract, validate, normalize features and obtain predictions
    from both XGBoost models.
    """
    priority_model, criticality_model = load_models()

    # 1. Extract and normalize features through the ML preprocessing pipeline
    raw_ml_features = _feature_builder.build(features)
    normalized_features = _normalizer.transform(raw_ml_features)

    # 2. DataFrame preserves the canonical feature names and required order
    model_input = pd.DataFrame(
        [[normalized_features[feature_name] for feature_name in ML_MODEL_FEATURES]],
        columns=ML_MODEL_FEATURES,
    )

    priority_class = int(priority_model.predict(model_input)[0])
    priority_probabilities = priority_model.predict_proba(model_input)[0]
    criticality_score = float(criticality_model.predict(model_input)[0])

    return {
        "priority_class": priority_class,
        "priority_label": PRIORITY_CLASS_LABELS.get(priority_class, "UNKNOWN"),
        "priority_probabilities": {
            str(class_label): round(float(probability), 6)
            for class_label, probability in zip(
                priority_model.classes_,
                priority_probabilities,
            )
        },
        "criticality_score": round(criticality_score, 6),
    }