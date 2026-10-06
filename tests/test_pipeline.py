import numpy as np
import pandas as pd
import pytest
from src.models.pipeline import Log1pTransformer, build_rfm_preprocessor


def test_log1p_transformer_dataframe():
    df = pd.DataFrame(
        {"Recency": [0.0, 9.0], "Frequency": [1.0, 3.0], "Monetary": [10.0, 100.0]}
    )
    transformer = Log1pTransformer()
    res = transformer.transform(df)
    assert res.shape == (2, 3)
    assert np.isclose(res[0, 0], 0.0)
    assert np.isclose(res[0, 1], np.log1p(1.0))


def test_build_rfm_preprocessor_standard():
    df = pd.DataFrame(
        {
            "Recency": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Frequency": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Monetary": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )
    pipeline = build_rfm_preprocessor(scaler_type="standard")
    transformed = pipeline.fit_transform(df)
    assert transformed.shape == (5, 3)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-7)


def test_build_rfm_preprocessor_robust():
    df = pd.DataFrame(
        {
            "Recency": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Frequency": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Monetary": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )
    pipeline = build_rfm_preprocessor(scaler_type="robust")
    transformed = pipeline.fit_transform(df)
    assert transformed.shape == (5, 3)


def test_build_rfm_preprocessor_invalid_scaler():
    with pytest.raises(ValueError):
        build_rfm_preprocessor(scaler_type="invalid")
