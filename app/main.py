from fastapi import FastAPI, HTTPException

from app.llm_service import generate_explanation
from app.model_service import predict
from app.schemas import ExplanationRequest, PredictionRequest

app = FastAPI(
    title="Inference Pipeline",
    version="0.3.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "phase": "llm explanation integration",
    }


@app.post("/validate")
def validate_input(request: PredictionRequest):
    return {
        "valid": True,
        "feature_count": len(request.features),
        "message": "Input is valid and ready for inference.",
    }


@app.post("/predict")
def make_prediction(request: PredictionRequest):
    try:
        return {
            "success": True,
            "prediction": predict(request.features),
            "note": "Priority classes 0/1/2 need confirmed business-label mapping.",
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference failed: {error}")


@app.post("/explain")
def create_explanation(request: ExplanationRequest):
    try:
        return {
            "success": True,
            "result": generate_explanation(
                features=request.features,
                clinical_summary=request.clinical_summary,
            ),
            "note": (
                "This is an assistive explanation, not a final clinical "
                "or authorization decision."
            ),
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation generation failed: {error}",
        )