from app.llm_service import generate_explanation
from app.schemas import AssessmentRequest


def create_assessment(request: AssessmentRequest) -> dict:
    """
    The Rule Engine remains the authority for authorization and decision.
    AI outputs enrich the result but never override business rules.
    """
    ai_result = generate_explanation(
        features=request.features,
        clinical_summary=request.clinical_summary,
    )

    rule_result = request.rule_engine_result
    needs_human_review = (
        rule_result.medical_necessity.status == "insufficient_information"
        or rule_result.decision.lower() == "more information required"
    )

    return {
        "patient_id": request.patient_id,

        # Authoritative result from the existing Rule Engine
        "final_decision": rule_result.decision,
        "authorization_required": rule_result.authorization.required,

        "rule_engine_assessment": {
            "criticality": rule_result.criticality.model_dump(),
            "priority": rule_result.priority.model_dump(),
            "medical_necessity": rule_result.medical_necessity.model_dump(),
            "triggered_rules": [
                rule.model_dump() for rule in request.triggered_rules
            ],
            "explanation": request.explanation,
        },

        # Assistive model outputs
        "ai_assessment": ai_result,

        "human_review": {
            "required": needs_human_review,
            "reason": (
                "Rule Engine reports insufficient information."
                if needs_human_review
                else "No Rule Engine information-gap flag."
            ),
        },

        "governance_note": (
            "The Rule Engine is authoritative. AI predictions and explanation "
            "support review and do not independently approve or deny a request."
        ),
    }