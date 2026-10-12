from pathlib import Path
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd


class SegmentPredictor:
    def __init__(self, model_path: str = "models/model_v1.joblib"):
        path = Path(model_path)
        if not path.is_absolute():
            base_dir = Path(__file__).resolve().parent.parent.parent
            path = base_dir / model_path

        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path}")

        artifact = joblib.load(path)
        self.preprocessor = artifact["preprocessor"]
        self.model = artifact["model"]
        self.cluster_persona_map = artifact["cluster_persona_map"]
        self.persona_definitions = artifact["persona_definitions"]
        self.cluster_profiles = artifact.get("cluster_profiles", [])
        self.metrics = artifact.get("metrics", {})

    def predict(
        self, recency: float, frequency: float, monetary: float
    ) -> Dict[str, Any]:
        if recency < 0:
            raise ValueError("Recency cannot be negative")
        if frequency < 1:
            raise ValueError("Frequency must be at least 1")
        if monetary < 0:
            raise ValueError("Monetary cannot be negative")

        input_df = pd.DataFrame(
            [
                {
                    "Recency": float(recency),
                    "Frequency": float(frequency),
                    "Monetary": float(monetary),
                }
            ]
        )

        transformed = self.preprocessor.transform(input_df)
        cluster = int(self.model.predict(transformed)[0])
        persona = self.cluster_persona_map.get(cluster, "Unknown")
        meta = self.persona_definitions.get(
            persona, {"description": "N/A", "action": "N/A"}
        )

        return {
            "cluster": cluster,
            "persona": persona,
            "description": meta["description"],
            "recommended_action": meta["action"],
            "inputs": {
                "recency": float(recency),
                "frequency": float(frequency),
                "monetary": float(monetary),
            },
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["Recency", "Frequency", "Monetary"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        transformed = self.preprocessor.transform(df[required])
        clusters = self.model.predict(transformed)

        result = df.copy()
        result["Cluster"] = clusters
        result["Persona"] = result["Cluster"].map(self.cluster_persona_map)
        result["Description"] = result["Persona"].apply(
            lambda p: self.persona_definitions.get(p, {}).get("description", "")
        )
        result["Recommended_Action"] = result["Persona"].apply(
            lambda p: self.persona_definitions.get(p, {}).get("action", "")
        )
        return result
