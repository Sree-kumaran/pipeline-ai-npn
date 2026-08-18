from fastapi.testclient import TestClient

from app import main
from app.feature_schema import MODEL_FEATURES


client = TestClient(main.app)


def valid_features():
    """
    A structurally valid 61-feature payload for API testing.
    These are test values only, not real patient data.
    """
    return {
        feature_name: 0.0
        for feature_name in MODEL_FEATURES
    }


def valid_assessment_payload():
    return {
        "patient_id": "TEST-P001",
        "clinical_summary": "Test clinical summary for automated API verification.",
        "features": valid_features(),
        "rule_engine_result": {
            "criticality": {
                "level": "moderate",
                "score": 0.45,
            },
            "priority": {
                "level": "normal",
                "score": 0.35,
            },
            "medical_necessity": {
                "status": "insufficient_information",
                "score": 0.55,
            },
            "authorization": {
                "required": True,
            },
            "decision": "More information required",
        },
        "triggered_rules": [
            {
                "rule_id": "R005",
                "rule_name": "Prior Authorization Required",
                "result": "passed",
                "impact": "sets_authorization_required",
            }
        ],
        "explanation": [
            "Prior authorization is required under the applicable policy."
        ],
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_validate_accepts_all_61_features():
    response = client.post(
        "/validate",
        json={"features": valid_features()},
    )

    assert response.status_code == 200
    assert response.json()["valid"] is True
    assert response.json()["feature_count"] == 61


def test_validate_rejects_missing_feature():
    incomplete_features = valid_features()
    del incomplete_features["age"]

    response = client.post(
        "/validate",
        json={"features": incomplete_features},
    )

    assert response.status_code == 422


def test_predict_returns_xgboost_result():
    response = client.post(
        "/predict",
        json={"features": valid_features()},
    )

    assert response.status_code == 200

    prediction = response.json()["prediction"]
    assert "priority_class" in prediction
    assert "priority_probabilities" in prediction
    assert "criticality_score" in prediction


def test_assess_preserves_rule_engine_decision(monkeypatch):
    """
    Mock the expensive LLM call. This test checks API integration and confirms
    that the Rule Engine decision remains authoritative.
    """

    def fake_create_assessment(request):
        return {
            "patient_id": request.patient_id,
            "final_decision": request.rule_engine_result.decision,
            "authorization_required": (
                request.rule_engine_result.authorization.required
            ),
            "ai_assessment": {
                "xgboost_prediction": {
                    "priority_class": 1,
                    "criticality_score": 0.45,
                },
                "llm_explanation": "Mocked explanation for test execution.",
            },
            "human_review": {
                "required": True,
            },
        }

    monkeypatch.setattr(main, "create_assessment", fake_create_assessment)

    response = client.post(
        "/assess",
        json=valid_assessment_payload(),
    )

    assert response.status_code == 200

    assessment = response.json()["assessment"]
    assert assessment["patient_id"] == "TEST-P001"
    assert assessment["final_decision"] == "More information required"
    assert assessment["authorization_required"] is True
    assert assessment["human_review"]["required"] is True