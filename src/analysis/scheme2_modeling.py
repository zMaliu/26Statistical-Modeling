from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


DESCRIPTIVE_COLUMNS = [
    "ai_agglomeration_composite",
    "innovation_support_substitute_index",
    "innovation_support_entropy_topsis_score",
    "innovation_support_pca_score",
    "coordination_capacity_composite",
    "gdp_per_capita",
    "population",
    "retail_sales",
    "fdi_actual_used",
    "financial_deposit_loan",
    "fiscal_expenditure",
    "secondary_industry_share",
    "tertiary_industry_share",
]

CORRELATION_COLUMNS = [
    "ai_agglomeration_composite",
    "innovation_support_substitute_index",
    "innovation_support_entropy_topsis_score",
    "coordination_capacity_composite",
    "gdp_per_capita",
    "population",
    "retail_sales",
    "fdi_actual_used",
    "financial_deposit_loan",
    "fiscal_expenditure",
    "secondary_industry_share",
    "tertiary_industry_share",
]

CORRELATION_FOCUS_PAIRS = [
    ("ai_agglomeration_composite", "innovation_support_substitute_index"),
    ("ai_agglomeration_composite", "innovation_support_entropy_topsis_score"),
    ("ai_agglomeration_composite", "coordination_capacity_composite"),
    ("innovation_support_substitute_index", "coordination_capacity_composite"),
    ("innovation_support_entropy_topsis_score", "coordination_capacity_composite"),
    ("service_openness_proxy", "coordination_capacity_composite"),
]


def _available_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def build_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    cols = _available_columns(df, DESCRIPTIVE_COLUMNS)
    records: list[dict] = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        non_null = s.dropna()
        records.append(
            {
                "variable": col,
                "n": int(non_null.shape[0]),
                "mean": float(non_null.mean()) if not non_null.empty else np.nan,
                "std": float(non_null.std(ddof=1)) if non_null.shape[0] > 1 else np.nan,
                "min": float(non_null.min()) if not non_null.empty else np.nan,
                "median": float(non_null.median()) if not non_null.empty else np.nan,
                "max": float(non_null.max()) if not non_null.empty else np.nan,
            }
        )
    return pd.DataFrame(records)


def build_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = _available_columns(df, CORRELATION_COLUMNS)
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    corr.index.name = "variable"
    return corr.reset_index()


def build_correlation_focus(df: pd.DataFrame) -> pd.DataFrame:
    corr = df.apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    records: list[dict] = []
    for x, y in CORRELATION_FOCUS_PAIRS:
        if x in corr.index and y in corr.columns:
            records.append(
                {
                    "variable_x": x,
                    "variable_y": y,
                    "correlation": float(corr.loc[x, y]),
                }
            )
    return pd.DataFrame(records)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _fit_ols(df: pd.DataFrame, y_col: str, x_cols: list[str], model_name: str) -> tuple[pd.DataFrame, dict]:
    work = df[[y_col] + x_cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
    n = len(work)
    k = len(x_cols) + 1  # intercept

    X = np.column_stack([np.ones(n), work[x_cols].to_numpy(dtype=float)])
    y = work[y_col].to_numpy(dtype=float)

    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    y_hat = X @ beta
    resid = y - y_hat

    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - sse / sst if sst > 0 else np.nan
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else np.nan

    dof = max(n - k, 1)
    sigma2 = sse / dof
    vcov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(vcov))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = beta / se
    p_vals = np.array([2 * (1 - _normal_cdf(abs(t))) if np.isfinite(t) else np.nan for t in t_stat])

    param_names = ["intercept"] + x_cols
    results = pd.DataFrame(
        {
            "model_name": model_name,
            "variable": param_names,
            "coefficient": beta,
            "std_error": se,
            "t_stat": t_stat,
            "p_value": p_vals,
            "n_obs": n,
            "r_squared": r2,
            "adj_r_squared": adj_r2,
        }
    )

    summary = {
        "model_name": model_name,
        "n_obs": n,
        "n_parameters": k,
        "r_squared": r2,
        "adj_r_squared": adj_r2,
        "ai_coefficient": float(results.loc[results["variable"] == "ai_agglomeration_composite", "coefficient"].iloc[0])
        if "ai_agglomeration_composite" in x_cols
        else np.nan,
        "ai_p_value": float(results.loc[results["variable"] == "ai_agglomeration_composite", "p_value"].iloc[0])
        if "ai_agglomeration_composite" in x_cols
        else np.nan,
    }
    return results, summary


def build_regression_outputs(matched_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = {
        "scheme2_baseline_model": [
            "ai_agglomeration_composite",
            "innovation_support_entropy_topsis_score",
            "gdp_per_capita",
            "population",
            "secondary_industry_share",
            "tertiary_industry_share",
        ],
        "scheme2_support_components_model": [
            "ai_agglomeration_composite",
            "fiscal_intensity_ratio",
            "financial_depth_ratio",
            "fdi_gdp_ratio",
            "retail_per_capita",
            "service_openness_proxy",
        ],
        "scheme2_ai_only_model": [
            "ai_agglomeration_composite",
        ],
    }

    result_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for model_name, x_cols in models.items():
        x_cols = [c for c in x_cols if c in matched_df.columns]
        res, summ = _fit_ols(
            matched_df,
            y_col="coordination_capacity_composite",
            x_cols=x_cols,
            model_name=model_name,
        )
        result_frames.append(res)
        summaries.append(summ)
    return pd.concat(result_frames, ignore_index=True), pd.DataFrame(summaries)


def build_regression_focus(regression_results: pd.DataFrame) -> pd.DataFrame:
    focus_variables = {
        "scheme2_ai_only_model": ["ai_agglomeration_composite"],
        "scheme2_baseline_model": [
            "ai_agglomeration_composite",
            "innovation_support_entropy_topsis_score",
            "gdp_per_capita",
            "population",
            "secondary_industry_share",
            "tertiary_industry_share",
        ],
        "scheme2_support_components_model": [
            "ai_agglomeration_composite",
            "fiscal_intensity_ratio",
            "financial_depth_ratio",
            "fdi_gdp_ratio",
            "retail_per_capita",
            "service_openness_proxy",
        ],
    }
    mask = regression_results.apply(
        lambda row: row["variable"] in focus_variables.get(row["model_name"], []), axis=1
    )
    out = regression_results.loc[mask].copy()
    return out.reset_index(drop=True)


def build_city_group_summary(city_profile: pd.DataFrame) -> pd.DataFrame:
    out = (
        city_profile.groupby("quadrant_label_cn", as_index=False)
        .agg(
            city_count=("city_name", "count"),
            ai_agglomeration_mean=("ai_agglomeration_mean", "mean"),
            innovation_support_entropy_mean=("innovation_support_entropy_mean", "mean"),
            innovation_support_pca_mean=("innovation_support_pca_mean", "mean"),
            coordination_capacity_mean=("coordination_capacity_mean", "mean"),
        )
        .sort_values("quadrant_label_cn")
        .reset_index(drop=True)
    )
    return out


def build_ai_city_summary(ai_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        ai_df.groupby("city_name", as_index=False)
        .agg(
            ai_observation_count=("year", "count"),
            ai_agglomeration_mean=("ai_agglomeration_composite", "mean"),
            ai_agglomeration_std=("ai_agglomeration_composite", "std"),
            ai_agglomeration_min=("ai_agglomeration_composite", "min"),
            ai_agglomeration_max=("ai_agglomeration_composite", "max"),
            ai_company_count_mean=("ai_company_count", "mean"),
            ai_hit_ratio_mean=("ai_hit_ratio", "mean"),
            ai_keyword_mentions_mean=("ai_keyword_mentions_company_sum", "mean"),
            ai_small_sample_observation_count=("ai_small_sample_flag", lambda s: int((s == "yes").sum())),
        )
        .sort_values("ai_agglomeration_mean", ascending=False)
        .reset_index(drop=True)
    )
    out["ai_rank"] = range(1, len(out) + 1)
    return out


def build_ai_year_summary(ai_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        ai_df.groupby("year", as_index=False)
        .agg(
            city_count=("city_name", "nunique"),
            observation_count=("city_name", "count"),
            ai_agglomeration_mean=("ai_agglomeration_composite", "mean"),
            ai_agglomeration_std=("ai_agglomeration_composite", "std"),
            ai_company_count_mean=("ai_company_count", "mean"),
            ai_hit_ratio_mean=("ai_hit_ratio", "mean"),
            ai_small_sample_observation_count=("ai_small_sample_flag", lambda s: int((s == "yes").sum())),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    return out


def build_innovation_city_summary(innovation_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        innovation_df.groupby("city_name", as_index=False)
        .agg(
            innovation_observation_count=("year", "count"),
            innovation_support_entropy_mean=("innovation_support_entropy_topsis_score", "mean"),
            innovation_support_entropy_std=("innovation_support_entropy_topsis_score", "std"),
            innovation_support_pca_mean=("innovation_support_pca_score", "mean"),
            fiscal_intensity_mean=("fiscal_intensity_ratio", "mean"),
            financial_depth_mean=("financial_depth_ratio", "mean"),
            openness_mean=("fdi_gdp_ratio", "mean"),
            retail_per_capita_mean=("retail_per_capita", "mean"),
            service_openness_mean=("service_openness_proxy", "mean"),
        )
        .sort_values("innovation_support_entropy_mean", ascending=False)
        .reset_index(drop=True)
    )
    out["innovation_rank"] = range(1, len(out) + 1)
    return out


def build_innovation_year_change(innovation_df: pd.DataFrame) -> pd.DataFrame:
    pivot = innovation_df.pivot_table(
        index="city_name",
        columns="year",
        values="innovation_support_entropy_topsis_score",
        aggfunc="mean",
    )
    years = sorted(pivot.columns.tolist())
    if len(years) >= 2:
        base_year = years[0]
        compare_year = years[-1]
        out = pivot.reset_index().rename(
            columns={
                base_year: f"innovation_support_score_{base_year}",
                compare_year: f"innovation_support_score_{compare_year}",
            }
        )
        out["innovation_support_change"] = (
            out[f"innovation_support_score_{compare_year}"] - out[f"innovation_support_score_{base_year}"]
        )
        out = out.sort_values("innovation_support_change", ascending=False).reset_index(drop=True)
        return out
    return pd.DataFrame()


def build_stratification_detailed(stratification_df: pd.DataFrame) -> pd.DataFrame:
    out = stratification_df.copy()
    quadrant_order = {
        "高集聚-高支撑": 1,
        "高集聚-低支撑": 2,
        "低集聚-高支撑": 3,
        "低集聚-低支撑": 4,
    }
    out["quadrant_order"] = out["city_quadrant"].map(quadrant_order)
    out = out.sort_values(
        ["quadrant_order", "ai_agglomeration_mean", "innovation_support_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return out.drop(columns=["quadrant_order"])
