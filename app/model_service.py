from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from app.feature_schema import MODEL_FEATURES


PROJECT_DIR = Path(__file__).resolve().parent.parent
XGBOOST_DIR = PROJECT_DIR / "models" / "xgboost"

PRIORITY_MODEL_PATH = XGBOOST_DIR / "xgb_priority_classifier.joblib"
CRITICALITY_MODEL_PATH = XGBOOST_DIR / "xgb_criticality_regressor.joblib"


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
    Convert validated features to the exact saved model order
    and obtain predictions from both XGBoost models.
    """
    priority_model, criticality_model = load_models()

    # DataFrame preserves the feature names and their required order.
    model_input = pd.DataFrame(
        [[features[feature_name] for feature_name in MODEL_FEATURES]],
        columns=MODEL_FEATURES,
    )

    priority_class = priority_model.predict(model_input)[0]
    priority_probabilities = priority_model.predict_proba(model_input)[0]
    criticality_score = criticality_model.predict(model_input)[0]

    return {
        "priority_class": int(priority_class),
        "priority_probabilities": {
            str(class_label): round(float(probability), 6)
            for class_label, probability in zip(
                priority_model.classes_,
                priority_probabilities,
            )
        },
        "criticality_score": round(float(criticality_score), 6),
    }