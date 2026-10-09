import time
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def run_clustering_benchmark(X: np.ndarray) -> pd.DataFrame:
    models = {
        "K-Means (K=4)": KMeans(n_clusters=4, random_state=42, n_init=10),
        "Agglomerative (K=4)": AgglomerativeClustering(n_clusters=4),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=15),
    }

    records = []
    for name, model in models.items():
        t0 = time.time()
        labels = model.fit_predict(X)
        latency = (time.time() - t0) * 1000

        mask = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1 and mask.sum() > 0:
            sil = float(silhouette_score(X[mask], labels[mask]))
            db = float(davies_bouldin_score(X[mask], labels[mask]))
            ch = float(calinski_harabasz_score(X[mask], labels[mask]))
        else:
            sil, db, ch = float("nan"), float("nan"), float("nan")

        records.append(
            {
                "model": name,
                "n_clusters": n_clusters,
                "silhouette": round(sil, 4),
                "davies_bouldin": round(db, 4),
                "calinski_harabasz": round(ch, 1),
                "latency_ms": round(latency, 2),
            }
        )
    return pd.DataFrame(records)
