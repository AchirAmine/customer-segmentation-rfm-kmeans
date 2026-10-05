import numpy as np
import pandas as pd


def build_rfm_features(
    clean_df: pd.DataFrame, reference_date: pd.Timestamp = None
) -> pd.DataFrame:
    required_cols = ["Customer ID", "InvoiceDate", "Invoice", "TotalAmount"]
    for col in required_cols:
        if col not in clean_df.columns:
            raise ValueError(f"Missing required column: {col}")

    if clean_df.empty:
        return pd.DataFrame(
            columns=["Recency", "Frequency", "Monetary"],
            index=pd.Index([], name="Customer ID"),
        )

    if reference_date is None:
        reference_date = clean_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = clean_df.groupby("Customer ID").agg(
        {
            "InvoiceDate": lambda x: (reference_date - x.max()).days,
            "Invoice": "nunique",
            "TotalAmount": "sum",
        }
    )
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    rfm["Monetary"] = rfm["Monetary"].clip(lower=0.0)
    return rfm


def transform_log1p(rfm_df: pd.DataFrame) -> pd.DataFrame:
    features = ["Recency", "Frequency", "Monetary"]
    for col in features:
        if col not in rfm_df.columns:
            raise ValueError(f"Missing required column for log1p: {col}")

    transformed = pd.DataFrame(index=rfm_df.index)
    for col in features:
        transformed[f"{col}_log"] = np.log1p(rfm_df[col].astype(float))
    return transformed
