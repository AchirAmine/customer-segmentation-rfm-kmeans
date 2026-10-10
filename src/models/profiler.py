import pandas as pd


PERSONA_DEFINITIONS = {
    "Champions": {
        "description": "High frequency, highest spend, recent purchases. Elite customer tier.",
        "action": "VIP access, premier loyalty rewards, early product launches.",
    },
    "Loyal Customers": {
        "description": "Consistent repeat buyers with high lifetime value.",
        "action": "Tiered loyalty incentives, cross-sell premium catalog.",
    },
    "Potential Loyalists": {
        "description": "Recent buyers with moderate spend and growing frequency.",
        "action": "Onboarding incentives, personalized category recommendations.",
    },
    "At Risk": {
        "description": "High past spend and repeat visits, but inactive for months.",
        "action": "Win-back email offers, reactivation discounts, re-engagement surveys.",
    },
    "Hibernating": {
        "description": "Long inactive, single or few low-value purchases.",
        "action": "Automated clearance promotions or budget suppression.",
    },
}

CLUSTER_PERSONA_MAP = {
    3: "Champions",
    1: "Loyal Customers",
    4: "Potential Loyalists",
    2: "At Risk",
    0: "Hibernating",
}


def assign_personas(
    rfm_df: pd.DataFrame, cluster_labels: pd.Series
) -> pd.DataFrame:
    df = rfm_df.copy()
    df["Cluster"] = cluster_labels
    df["Persona"] = df["Cluster"].map(CLUSTER_PERSONA_MAP).fillna("Unknown")
    return df


def build_cluster_profiles(rfm_with_persona: pd.DataFrame) -> pd.DataFrame:
    profiles = (
        rfm_with_persona.groupby(["Cluster", "Persona"])
        .agg(
            Customer_Count=("Recency", "count"),
            Recency_Mean=("Recency", "mean"),
            Recency_Median=("Recency", "median"),
            Frequency_Mean=("Frequency", "mean"),
            Frequency_Median=("Frequency", "median"),
            Monetary_Mean=("Monetary", "mean"),
            Monetary_Median=("Monetary", "median"),
            Total_Spend=("Monetary", "sum"),
        )
        .reset_index()
    )
    total_customers = profiles["Customer_Count"].sum()
    total_revenue = profiles["Total_Spend"].sum()
    profiles["Customer_Share_%"] = (
        profiles["Customer_Count"] / total_customers
    ) * 100
    profiles["Revenue_Share_%"] = (
        profiles["Total_Spend"] / total_revenue
    ) * 100
    return profiles.round(2)
