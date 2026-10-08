import numpy as np
from src.models.clustering_eval import evaluate_clustering_k


def test_evaluate_clustering_k():
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[-2.0, -2.0, -2.0], scale=0.5, size=(30, 3))
    cluster2 = np.random.normal(loc=[2.0, 2.0, 2.0], scale=0.5, size=(30, 3))
    X = np.vstack([cluster1, cluster2])

    df = evaluate_clustering_k(X, k_range=range(2, 4), random_state=42)
    assert len(df) == 2
    assert "k" in df.columns
    assert "inertia" in df.columns
    assert "silhouette" in df.columns
    assert "davies_bouldin" in df.columns
    assert "calinski_harabasz" in df.columns
    assert df.loc[df["k"] == 2, "silhouette"].values[0] > 0.5
