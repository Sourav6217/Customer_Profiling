import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data():
    """Load raw data from CSV in the repo."""
    df = pd.read_csv("Customer_Profiling.csv")
    return df


@st.cache_data
def preprocess_data(df: pd.DataFrame):
    """Clean and engineer all features needed for modelling & clustering."""
    df = df.copy()

    # --- 1. Handle missing numeric values with median ---
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    # --- 2. Remove extreme outliers (99.5 percentile) ---
    upper_bounds = df[numeric_cols].quantile(0.995)
    for col in numeric_cols:
        df = df[df[col] <= upper_bounds[col]]

    # --- 3. Date → tenure & enrollment year ---
    df["Dt_Customer_dt"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    today = datetime.now()
    df["Customer_Tenure_Days"] = (today - df["Dt_Customer_dt"]).dt.days
    df["Customer_Tenure_Months"] = df["Customer_Tenure_Days"] / 30.0
    df["Enrollment_Year"] = today.year - df["Dt_Customer_dt"].dt.year

    # --- 4. Core behaviour features ---
    df["Total_Expenditure"] = (
        df["MntWines"]
        + df["MntFruits"]
        + df["MntMeatProducts"]
        + df["MntFishProducts"]
        + df["MntSweetProducts"]
        + df["MntGoldProds"]
    )

    # Avoid division by zero
    df["Average_Monthly_Spend"] = df["Total_Expenditure"] / df["Customer_Tenure_Months"].replace(
        0, np.nan
    )
    df["Average_Monthly_Spend"].fillna(0, inplace=True)

    df["Dependents"] = df["Kidhome"] + df["Teenhome"]
    df["Engagement_Score"] = df["NumWebVisitsMonth"] * 0.4 + df["NumStorePurchases"] * 0.6

    # Binary campaign response label from past campaigns
    df["Campaign_Response"] = df[
        ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
    ].max(axis=1)

    # Age and family flags
    current_year = datetime.now().year
    df["Age"] = current_year - df["Year_Birth"]
    df["Has Kids"] = (df["Kidhome"] > 0).astype(int)
    df["Has Teens"] = (df["Teenhome"] > 0).astype(int)

    # Scale some continuous columns for stability
    value_scaler = MinMaxScaler()
    scale_cols = ["Income", "Total_Expenditure", "Customer_Tenure_Months"]
    df[scale_cols] = value_scaler.fit_transform(df[scale_cols])

    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    """
    Train Random Forest for campaign response & K-Means for segmentation.
    Returns the enriched df plus models, profiles & metrics.
    """
    # Features for campaign-response model (all numeric, no one-hot)
    model_features = [
        "Income",
        "Recency",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "Complain",
        "Enrollment_Year",
        "Total_Expenditure",
        "Customer_Tenure_Months",
        "Average_Monthly_Spend",
        "Dependents",
        "Engagement_Score",
        "Age",
        "Has Kids",
        "Has Teens",
    ]

    df_model = df.dropna(subset=["Response"]).copy()
    X = df_model[model_features]
    y = df_model["Response"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Add prediction probabilities back to full dataframe
    df["Prediction_Probability"] = rf.predict_proba(df[model_features])[:, 1]

    # --- Clustering ---
    cluster_features = [
        "Income",
        "Recency",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "Total_Expenditure",
        "Customer_Tenure_Months",
        "Average_Monthly_Spend",
        "Dependents",
        "Engagement_Score",
        "Age",
        "Has Kids",
        "Has Teens",
        "Prediction_Probability",
    ]

    cluster_scaler = StandardScaler()
    cluster_scaled = cluster_scaler.fit_transform(df[cluster_features])

    kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(cluster_scaled)

    cluster_profiles = df.groupby("Cluster")[cluster_features + ["Response", "Campaign_Response"]].mean()

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(cluster_scaled)
    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    pca_df["Cluster"] = df["Cluster"].values

    metrics = {"auc": auc, "precision": precision, "recall": recall}

    return (
        df,
        rf,
        model_features,
        kmeans,
        cluster_scaler,
        cluster_features,
        cluster_profiles,
        pca_df,
        metrics,
    )


def main():
    st.set_page_config(page_title="Customer Profiling & Targeted Marketing", layout="wide")

    st.title("Customer Profiling & Targeted Marketing Strategy")

    # Load + preprocess + train (all cached)
    df_raw = load_data()
    df_pre = preprocess_data(df_raw)
    (
        df,
        rf,
        model_features,
        kmeans,
        cluster_scaler,
        cluster_features,
        cluster_profiles,
        pca_df,
        metrics,
    ) = train_models(df_pre)

    menu = st.sidebar.radio(
        "Navigate",
        ["Overview", "Campaign Prediction Explorer", "Customer Segmentation Explorer"],
    )

    # -------- Overview --------
    if menu == "Overview":
        st.header("Data & Feature Engineering Overview")

        st.subheader("Sample of Raw Data")
        st.dataframe(df_raw.head())

        st.subheader("Sample of Engineered Data")
        st.dataframe(
            df[
                [
                    "Income",
                    "Total_Expenditure",
                    "Average_Monthly_Spend",
                    "Engagement_Score",
                    "Age",
                ]
            ].head()
        )

        st.subheader("Model Performance (Random Forest – Campaign Response)")
        col1, col2, col3 = st.columns(3)
        col1.metric("AUC", f"{metrics['auc']:.4f}")
        col2.metric("Precision", f"{metrics['precision']:.4f}")
        col3.metric("Recall", f"{metrics['recall']:.4f}")

        st.subheader("Income Distribution (after cleaning)")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["Income"], kde=True, ax=ax)
        ax.set_xlabel("Scaled Income")
        st.pyplot(fig)

    # -------- Prediction Explorer --------
    elif menu == "Campaign Prediction Explorer":
        st.header("Campaign Response Prediction Explorer")

        st.write(
            "Select an existing customer from the dataset to view the predicted campaign "
            "response probability and their key behavioural signals."
        )

        # Use customer index or ID if available
        id_col = "ID" if "ID" in df.columns else None
        if id_col:
            id_list = df[id_col].tolist()
            selected_id = st.selectbox("Select Customer ID", id_list)
            customer_row = df[df[id_col] == selected_id].iloc[0]
        else:
            idx = st.number_input(
                "Select row index", min_value=0, max_value=int(df.index.max()), value=0, step=1
            )
            customer_row = df.loc[idx]

        st.subheader("Customer Behaviour Snapshot")
        behaviour_cols = [
            "Income",
            "Recency",
            "NumDealsPurchases",
            "NumWebPurchases",
            "NumCatalogPurchases",
            "Total_Expenditure",
            "Average_Monthly_Spend",
            "Engagement_Score",
            "Dependents",
            "Age",
            "Has Kids",
            "Has Teens",
        ]
        st.table(customer_row[behaviour_cols].to_frame("Value"))

        # Prediction using the model_features subset
        X_single = customer_row[model_features].to_frame().T
        proba = rf.predict_proba(X_single)[0, 1]
        pred = rf.predict(X_single)[0]

        st.subheader("Predicted Campaign Response")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Probability", f"{proba:.3f}")
        col2.metric("Predicted Class (0 = No, 1 = Yes)", int(pred))

        st.info(
            "Prediction is based on behaviour variables such as recency, channel purchases, "
            "total & monthly spend, engagement score, age and family flags."
        )

    # -------- Segmentation Explorer --------
    elif menu == "Customer Segmentation Explorer":
        st.header("Customer Segmentation (K-Means, K = 3)")

        st.subheader("Cluster Size Distribution")
        cluster_counts = df["Cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(
            cluster_counts,
            labels=[f"Cluster {i}" for i in cluster_counts.index],
            autopct="%1.1f%%",
        )
        ax.axis("equal")
        st.pyplot(fig)

        st.subheader("Cluster Profiles (Mean of Key Features)")
        st.dataframe(cluster_profiles.style.format("{:.3f}"))

        st.subheader("Cluster Comparison on Key Attributes")
        key_compare = [
            "Income",
            "Total_Expenditure",
            "Average_Monthly_Spend",
            "Engagement_Score",
            "Prediction_Probability",
            "Age",
        ]
        compare_df = (
            cluster_profiles[key_compare]
            .reset_index()
            .melt(id_vars="Cluster", var_name="Feature", value_name="Value")
        )

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=compare_df, x="Feature", y="Value", hue="Cluster", ax=ax2)
        ax2.set_title("Cluster Comparison on Selected Attributes")
        ax2.tick_params(axis="x", rotation=45)
        st.pyplot(fig2)

        st.subheader("PCA Visualization of Clusters (2D)")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="viridis", ax=ax3, alpha=0.7
        )
        ax3.set_title("Customer Clusters (PCA Reduced Dimensions)")
        st.pyplot(fig3)

        st.markdown(
            """
**High-level interpretation (aligns with your cluster descriptions):**

- **Cluster 0 – Budget-Conscious Young Families**  
  Lower income & spend, higher dependents, more deal-driven, lowest campaign response probability.

- **Cluster 1 – High-Value, Campaign-Responsive Individuals/Couples**  
  Highest income & spend, strong web/catalog usage, highest predicted probability. Prime target segment.

- **Cluster 2 – Engaged Families with Teenagers**  
  Mid income, high engagement across channels, heavy web + deals, moderate response probability.
"""
        )


if __name__ == "__main__":
    main()
