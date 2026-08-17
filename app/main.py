from fastapi import FastAPI

from app.schemas import PredictionRequest

app = FastAPI(
    title="Inference Pipeline",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "phase": "input validation",
    }


@app.post("/validate")
def validate_input(request: PredictionRequest):
    return {
        "valid": True,
        "feature_count": len(request.features),
        "message": "Input is valid and ready for feature processing.",
    }