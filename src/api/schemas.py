from typing import Any, Dict, Literal
from pydantic import BaseModel, Field


class CustomerRFMPayload(BaseModel):
    recency: float = Field(..., ge=0.0)
    frequency: int = Field(..., ge=1)
    monetary: float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    status: str
    model_version: str
    algorithm: str
    n_clusters: int
    total_trained_customers: int


class SegmentationResponse(BaseModel):
    cluster: int
    persona: Literal[
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "At Risk",
        "Hibernating",
        "Unknown",
    ]
    description: str
    recommended_action: str
    inputs: Dict[str, float]
    model_version: str
