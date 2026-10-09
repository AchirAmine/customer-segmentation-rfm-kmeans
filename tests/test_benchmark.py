import numpy as np
from src.models.benchmark import run_clustering_benchmark


def test_run_clustering_benchmark():
    np.random.seed(42)
    c1 = np.random.normal(loc=[-2.0, -2.0, -2.0], scale=0.4, size=(25, 3))
    c2 = np.random.normal(loc=[2.0, 2.0, 2.0], scale=0.4, size=(25, 3))
    c3 = np.random.normal(loc=[0.0, 0.0, 0.0], scale=0.4, size=(25, 3))
    X = np.vstack([c1, c2, c3])

    df = run_clustering_benchmark(X)
    assert len(df) == 3
    assert "model" in df.columns
    assert "silhouette" in df.columns
    assert "davies_bouldin" in df.columns
    kmeans_row = df[df["model"] == "K-Means (K=4)"].iloc[0]
    assert not np.isnan(kmeans_row["silhouette"])
