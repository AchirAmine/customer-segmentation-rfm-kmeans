import pandas as pd
from src.models.profiler import (
    CLUSTER_PERSONA_MAP,
    PERSONA_DEFINITIONS,
    assign_personas,
    build_cluster_profiles,
)


def test_assign_personas():
    rfm = pd.DataFrame(
        {
            "Recency": [10, 50, 300, 20, 400],
            "Frequency": [30, 10, 4, 2, 1],
            "Monetary": [20000, 3500, 1500, 600, 250],
        }
    )
    clusters = [3, 1, 2, 4, 0]
    result = assign_personas(rfm, clusters)
    assert result.loc[0, "Persona"] == "Champions"
    assert result.loc[1, "Persona"] == "Loyal Customers"
    assert result.loc[2, "Persona"] == "At Risk"
    assert result.loc[3, "Persona"] == "Potential Loyalists"
    assert result.loc[4, "Persona"] == "Hibernating"


def test_build_cluster_profiles():
    rfm_with_persona = pd.DataFrame(
        {
            "Cluster": [3, 3, 0],
            "Persona": ["Champions", "Champions", "Hibernating"],
            "Recency": [10.0, 20.0, 400.0],
            "Frequency": [30.0, 40.0, 1.0],
            "Monetary": [1000.0, 2000.0, 100.0],
        }
    )
    profiles = build_cluster_profiles(rfm_with_persona)
    assert len(profiles) == 2
    assert "Customer_Share_%" in profiles.columns
    assert "Revenue_Share_%" in profiles.columns


def test_persona_definitions():
    for persona in CLUSTER_PERSONA_MAP.values():
        assert persona in PERSONA_DEFINITIONS
        assert "description" in PERSONA_DEFINITIONS[persona]
        assert "action" in PERSONA_DEFINITIONS[persona]
