import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


INDICATOR_COLUMNS = [
    "fiscal_intensity_ratio",
    "financial_depth_ratio",
    "fdi_gdp_ratio",
    "retail_per_capita",
    "service_openness_proxy",
]


def fill_missing_with_year_median(df: pd.DataFrame, indicator_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in indicator_columns:
        out[f"{col}_imputed_flag"] = 0
        year_median = out.groupby("year")[col].transform("median")
        missing_mask = out[col].isna()
        out.loc[missing_mask, col] = year_median[missing_mask]
        out.loc[missing_mask, f"{col}_imputed_flag"] = 1
    return out


def min_max_normalize(df: pd.DataFrame, indicator_columns: list[str]) -> pd.DataFrame:
    norm = pd.DataFrame(index=df.index)
    for col in indicator_columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if np.isclose(col_max, col_min):
            norm[col] = 1.0
        else:
            norm[col] = (df[col] - col_min) / (col_max - col_min)
        norm[col] = norm[col].clip(lower=0) + 1e-12
    return norm


def compute_entropy_weights(df: pd.DataFrame, indicator_columns: list[str]) -> pd.DataFrame:
    normalized = min_max_normalize(df, indicator_columns)
    p = normalized.div(normalized.sum(axis=0), axis=1)
    n = len(normalized)
    k = 1.0 / np.log(n)
    entropy = -k * (p * np.log(p)).sum(axis=0)
    redundancy = 1 - entropy
    weights = redundancy / redundancy.sum()
    return pd.DataFrame(
        {
            "indicator": indicator_columns,
            "entropy": entropy.values,
            "information_redundancy": redundancy.values,
            "entropy_weight": weights.values,
        }
    )


def compute_entropy_topsis_scores(df: pd.DataFrame, indicator_columns: list[str], weights: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    normalized = min_max_normalize(df, indicator_columns)
    weight_map = weights.set_index("indicator")["entropy_weight"]
    weighted = normalized.mul(weight_map, axis=1)
    positive_ideal = weighted.max(axis=0)
    negative_ideal = weighted.min(axis=0)
    d_pos = np.sqrt(((weighted - positive_ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - negative_ideal) ** 2).sum(axis=1))
    topsis_score = d_neg / (d_pos + d_neg)
    out["innovation_support_entropy_topsis_score"] = topsis_score
    out["innovation_support_entropy_rank_within_year"] = out.groupby("year")[
        "innovation_support_entropy_topsis_score"
    ].rank(method="dense", ascending=False)
    return out


def compute_pca_scores(df: pd.DataFrame, indicator_columns: list[str]) -> tuple[pd.DataFrame, float]:
    out = df.copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[indicator_columns])
    pca = PCA(n_components=1, random_state=42)
    scores = pca.fit_transform(X).reshape(-1)
    out["innovation_support_pca_score"] = scores
    out["innovation_support_pca_rank_within_year"] = out.groupby("year")["innovation_support_pca_score"].rank(
        method="dense", ascending=False
    )
    return out, float(pca.explained_variance_ratio_[0])


def build_city_profile_with_clusters(
    matched_df: pd.DataFrame,
    upgraded_innovation_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = matched_df.merge(
        upgraded_innovation_df[
            [
                "city_key",
                "year",
                "innovation_support_entropy_topsis_score",
                "innovation_support_pca_score",
            ]
        ],
        on=["city_key", "year"],
        how="left",
    )

    city_profile = (
        merged.groupby("city_name", as_index=False)
        .agg(
            ai_agglomeration_mean=("ai_agglomeration_composite", "mean"),
            innovation_support_entropy_mean=("innovation_support_entropy_topsis_score", "mean"),
            innovation_support_pca_mean=("innovation_support_pca_score", "mean"),
            coordination_capacity_mean=("coordination_capacity_composite", "mean"),
            sample_rows=("year", "count"),
        )
    )

    ai_median = city_profile["ai_agglomeration_mean"].median()
    support_median = city_profile["innovation_support_entropy_mean"].median()
    city_profile["quadrant_ai_level"] = np.where(city_profile["ai_agglomeration_mean"] >= ai_median, "high_ai", "low_ai")
    city_profile["quadrant_support_level"] = np.where(
        city_profile["innovation_support_entropy_mean"] >= support_median, "high_support", "low_support"
    )
    city_profile["quadrant_label"] = city_profile["quadrant_ai_level"] + "_" + city_profile["quadrant_support_level"]

    features = city_profile[["ai_agglomeration_mean", "innovation_support_entropy_mean"]].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    diagnostics = []
    for k in [2, 3, 4]:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels) if len(city_profile) > k else np.nan
        diagnostics.append({"k": k, "silhouette_score": score, "inertia": float(km.inertia_)})

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)
    city_profile["kmeans_cluster_id"] = labels

    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=["ai_agglomeration_mean_center", "innovation_support_entropy_mean_center"],
    )
    centers["kmeans_cluster_id"] = centers.index
    centers["cluster_ai_level"] = np.where(
        centers["ai_agglomeration_mean_center"] >= ai_median, "high_ai", "low_ai"
    )
    centers["cluster_support_level"] = np.where(
        centers["innovation_support_entropy_mean_center"] >= support_median, "high_support", "low_support"
    )
    centers["cluster_label"] = centers["cluster_ai_level"] + "_" + centers["cluster_support_level"]

    city_profile = city_profile.merge(
        centers[["kmeans_cluster_id", "cluster_label"]],
        on="kmeans_cluster_id",
        how="left",
    )

    label_map = {
        "high_ai_high_support": "高集聚-高支撑",
        "high_ai_low_support": "高集聚-低支撑",
        "low_ai_high_support": "低集聚-高支撑",
        "low_ai_low_support": "低集聚-低支撑",
    }
    city_profile["quadrant_label_cn"] = city_profile["quadrant_label"].map(label_map)
    city_profile["cluster_label_cn"] = city_profile["cluster_label"].map(label_map)
    city_profile["quadrant_cluster_consistent_flag"] = (
        city_profile["quadrant_label"] == city_profile["cluster_label"]
    ).astype(int)

    diagnostics_df = pd.DataFrame(diagnostics)
    return city_profile.sort_values(["quadrant_label", "city_name"]).reset_index(drop=True), diagnostics_df
