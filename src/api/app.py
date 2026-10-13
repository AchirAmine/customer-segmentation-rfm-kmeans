from fastapi import FastAPI, HTTPException
from src.api.schemas import (
    CustomerRFMPayload,
    HealthResponse,
    SegmentationResponse,
)
from src.models.predict import SegmentPredictor

app = FastAPI(
    title="Customer Segmentation RFM K-Means Service",
    description="Production REST microservice for customer segmentation using RFM metrics and K-Means clustering.",
    version="1.0.0",
)

predictor_instance = None


def get_predictor() -> SegmentPredictor:
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = SegmentPredictor()
    return predictor_instance


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        pred = get_predictor()
        return HealthResponse(
            status="healthy",
            model_version="1.0.0",
            algorithm="K-Means (K=5)",
            n_clusters=len(pred.cluster_persona_map),
            total_trained_customers=pred.metrics.get("total_customers", 5878),
        )
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Service unavailable: {str(e)}"
        )


@app.post("/segment", response_model=SegmentationResponse)
def segment(payload: CustomerRFMPayload):
    try:
        pred = get_predictor()
        res = pred.predict(
            recency=payload.recency,
            frequency=float(payload.frequency),
            monetary=payload.monetary,
        )
        return SegmentationResponse(
            cluster=res["cluster"],
            persona=res["persona"],
            description=res["description"],
            recommended_action=res["recommended_action"],
            inputs=res["inputs"],
            model_version="1.0.0",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=SegmentationResponse)
def predict(payload: CustomerRFMPayload):
    return segment(payload)
