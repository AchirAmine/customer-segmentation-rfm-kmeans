import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


class Log1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = (
            columns
            if columns is not None
            else ["Recency", "Frequency", "Monetary"]
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            for col in self.columns:
                if col in X_df.columns:
                    X_df[col] = np.log1p(np.maximum(X_df[col].astype(float), 0.0))
            return X_df[self.columns].to_numpy()
        arr = np.asarray(X, dtype=float)
        return np.log1p(np.maximum(arr, 0.0))


def build_rfm_preprocessor(scaler_type: str = "standard") -> Pipeline:
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    return Pipeline(
        [
            ("log1p", Log1pTransformer()),
            ("scaler", scaler),
        ]
    )
