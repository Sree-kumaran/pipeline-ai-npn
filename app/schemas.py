import math

from pydantic import BaseModel, Field, model_validator

from app.feature_schema import MODEL_FEATURES


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="All 61 XGBoost model features as numeric values."
    )

    @model_validator(mode="after")
    def validate_features(self):
        received_features = set(self.features.keys())
        required_features = set(MODEL_FEATURES)

        missing_features = sorted(required_features - received_features)
        unexpected_features = sorted(received_features - required_features)

        if missing_features:
            raise ValueError(
                f"Missing required features: {', '.join(missing_features)}"
            )

        if unexpected_features:
            raise ValueError(
                f"Unexpected features: {', '.join(unexpected_features)}"
            )

        for feature_name, value in self.features.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"Feature '{feature_name}' must be a finite number."
                )

        return self

class ExplanationRequest(PredictionRequest):
    clinical_summary: str

class CriticalityRuleResult(BaseModel):
    level: str
    score: float


class PriorityRuleResult(BaseModel):
    level: str
    score: float


class MedicalNecessityRuleResult(BaseModel):
    status: str
    score: float


class AuthorizationRuleResult(BaseModel):
    required: bool


class RuleEngineResult(BaseModel):
    criticality: CriticalityRuleResult
    priority: PriorityRuleResult
    medical_necessity: MedicalNecessityRuleResult
    authorization: AuthorizationRuleResult
    decision: str


class TriggeredRule(BaseModel):
    rule_id: str
    rule_name: str
    result: str
    impact: str


class AssessmentRequest(ExplanationRequest):
    patient_id: str
    rule_engine_result: RuleEngineResult
    triggered_rules: list[TriggeredRule]
    explanation: list[str]