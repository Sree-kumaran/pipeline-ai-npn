from fastapi import FastAPI, HTTPException

from app.model_service import predict
from app.schemas import PredictionRequest

app = FastAPI(
    title="Inference Pipeline",
    version="0.2.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "phase": "xgboost inference",
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
        prediction = predict(request.features)

        return {
            "success": True,
            "feature_count": len(request.features),
            "prediction": prediction,
            "note": (
                "priority_class is the original model label. "
                "Map 0/1/2 to business labels only after confirming "
                "the training label mapping."
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(error)}",
        )