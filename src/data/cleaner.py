import pandas as pd


def clean_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["Invoice", "Quantity", "Price", "Customer ID", "InvoiceDate"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    cleaned = df.dropna(subset=["Customer ID"]).copy()
    cleaned = cleaned[~cleaned["Invoice"].astype(str).str.upper().str.startswith("C")]
    cleaned = cleaned[(cleaned["Quantity"] > 0) & (cleaned["Price"] > 0)]
    cleaned["Customer ID"] = cleaned["Customer ID"].astype(int)
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"])
    cleaned["TotalAmount"] = cleaned["Quantity"] * cleaned["Price"]
    return cleaned.reset_index(drop=True)
