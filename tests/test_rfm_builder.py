import numpy as np
import pandas as pd
import pytest
from src.data.rfm_builder import build_rfm_features, transform_log1p


def test_build_rfm_features_calculation():
    df = pd.DataFrame(
        {
            "Customer ID": [101, 101, 102],
            "Invoice": ["INV1", "INV2", "INV3"],
            "InvoiceDate": pd.to_datetime(
                ["2011-12-01", "2011-12-05", "2011-12-02"]
            ),
            "TotalAmount": [100.0, 150.0, 50.0],
        }
    )
    ref_date = pd.to_datetime("2011-12-06")
    rfm = build_rfm_features(df, reference_date=ref_date)

    assert rfm.loc[101, "Recency"] == 1
    assert rfm.loc[101, "Frequency"] == 2
    assert rfm.loc[101, "Monetary"] == 250.0

    assert rfm.loc[102, "Recency"] == 4
    assert rfm.loc[102, "Frequency"] == 1
    assert rfm.loc[102, "Monetary"] == 50.0


def test_build_rfm_missing_column():
    df = pd.DataFrame({"Customer ID": [101]})
    with pytest.raises(ValueError):
        build_rfm_features(df)


def test_transform_log1p():
    rfm = pd.DataFrame(
        {"Recency": [0.0, 99.0], "Frequency": [1.0, 10.0], "Monetary": [10.0, 1000.0]}
    )
    log_rfm = transform_log1p(rfm)
    assert "Recency_log" in log_rfm.columns
    assert "Frequency_log" in log_rfm.columns
    assert "Monetary_log" in log_rfm.columns
    assert np.isclose(log_rfm.loc[0, "Recency_log"], 0.0)
    assert np.isclose(log_rfm.loc[0, "Frequency_log"], np.log1p(1.0))
