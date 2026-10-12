import pandas as pd
import pytest
from src.models.predict import SegmentPredictor


@pytest.fixture
def predictor():
    return SegmentPredictor("models/model_v1.joblib")


def test_predictor_initialization(predictor):
    assert predictor.model is not None
    assert predictor.preprocessor is not None
    assert len(predictor.cluster_persona_map) == 5


def test_predict_single_champions(predictor):
    result = predictor.predict(recency=5.0, frequency=40.0, monetary=25000.0)
    assert result["persona"] == "Champions"
    assert result["cluster"] == 3
    assert "VIP" in result["recommended_action"]


def test_predict_single_hibernating(predictor):
    result = predictor.predict(recency=450.0, frequency=1.0, monetary=150.0)
    assert result["persona"] == "Hibernating"
    assert result["cluster"] == 0


def test_predict_batch(predictor):
    batch_df = pd.DataFrame(
        {
            "Recency": [5.0, 450.0, 30.0],
            "Frequency": [40.0, 1.0, 3.0],
            "Monetary": [25000.0, 150.0, 700.0],
        }
    )
    result_df = predictor.predict_batch(batch_df)
    assert len(result_df) == 3
    assert "Cluster" in result_df.columns
    assert "Persona" in result_df.columns
    assert result_df["Persona"].iloc[0] == "Champions"
    assert result_df["Persona"].iloc[1] == "Hibernating"


def test_predict_invalid_inputs(predictor):
    with pytest.raises(ValueError):
        predictor.predict(recency=-1.0, frequency=1.0, monetary=10.0)
    with pytest.raises(ValueError):
        predictor.predict(recency=10.0, frequency=0.0, monetary=10.0)
    with pytest.raises(ValueError):
        predictor.predict(recency=10.0, frequency=1.0, monetary=-10.0)
