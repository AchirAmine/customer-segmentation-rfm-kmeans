import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def evaluate_clustering_k(
    X: np.ndarray, k_range=range(2, 9), random_state: int = 42
) -> pd.DataFrame:
    records = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        records.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(X, labels)),
                "davies_bouldin": float(davies_bouldin_score(X, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            }
        )
    return pd.DataFrame(records)
