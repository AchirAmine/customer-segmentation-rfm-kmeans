from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "1.0.0"
    assert data["n_clusters"] == 5
    assert data["total_trained_customers"] == 5878


def test_segment_endpoint_champions():
    payload = {"recency": 5.0, "frequency": 40, "monetary": 25000.0}
    response = client.post("/segment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["persona"] == "Champions"
    assert data["cluster"] == 3
    assert "VIP" in data["recommended_action"]


def test_segment_endpoint_hibernating():
    payload = {"recency": 450.0, "frequency": 1, "monetary": 120.0}
    response = client.post("/segment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["persona"] == "Hibernating"
    assert data["cluster"] == 0


def test_segment_endpoint_validation_error():
    invalid_payload = {"recency": -10.0, "frequency": 0, "monetary": -50.0}
    response = client.post("/segment", json=invalid_payload)
    assert response.status_code == 422


def test_predict_alias_endpoint():
    payload = {"recency": 15.0, "frequency": 10, "monetary": 3000.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["persona"] in [
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
    ]
