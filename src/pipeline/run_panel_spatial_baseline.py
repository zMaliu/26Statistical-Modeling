"""Spatial weights, Moran's I, and baseline TWFE models for the completed panel.

This script is intentionally dependency-light.  It avoids PySAL/linearmodels so
the project can run in a plain scientific Python environment.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_ready" / "panel_21city_2018_2023_completed.csv"
OFFICIAL_CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned" / "analysis_city_panel_official_cleaned.csv"
SPATIAL_DIR = PROJECT_ROOT / "data" / "processed" / "spatial"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "analysis_ready"
PICTURE_DIR = PROJECT_ROOT / "picture"

COORDS_PATH = SPATIAL_DIR / "city_coordinates.csv"
WIDE_W_PATH = SPATIAL_DIR / "spatial_weights_inverse_distance.csv"
LONG_W_PATH = SPATIAL_DIR / "spatial_weights_inverse_distance_long.csv"
SPATIAL_PANEL_PATH = ANALYSIS_DIR / "panel_21city_2018_2023_spatial_ready.csv"
MORAN_RESULTS_PATH = ANALYSIS_DIR / "moran_global_results.csv"
LISA_RESULTS_PATH = ANALYSIS_DIR / "lisa_local_results.csv"
TWFE_RESULTS_PATH = ANALYSIS_DIR / "twfe_baseline_results.csv"
TWFE_SUMMARY_PATH = ANALYSIS_DIR / "twfe_model_summary.csv"
SLX_RESULTS_PATH = ANALYSIS_DIR / "slx_twfe_results.csv"
SLX_SUMMARY_PATH = ANALYSIS_DIR / "slx_twfe_model_summary.csv"
STATA_DIR = PROJECT_ROOT / "stata"
STATA_PANEL_CSV = STATA_DIR / "panel_sdm_stata.csv"
STATA_PANEL_DTA = STATA_DIR / "panel_sdm_stata.dta"
STATA_CITY_MAP_CSV = STATA_DIR / "city_id_map.csv"
STATA_W_INV_CSV = STATA_DIR / "w_inverse_distance_stata.csv"
STATA_W_KNN4_CSV = STATA_DIR / "w_knn4_distance_stata.csv"
STATA_W_GEO_ECON_CSV = STATA_DIR / "w_geo_economic_stata.csv"
STATA_W_CSV = STATA_W_INV_CSV
STATA_DO_PATH = STATA_DIR / "run_panel_sdm.do"
STATA_README_PATH = STATA_DIR / "SDM操作与解读说明.md"
MORAN_AI_FIG = PICTURE_DIR / "fig_moran_ai_trend.png"
MORAN_COORD_FIG = PICTURE_DIR / "fig_moran_coordination_trend.png"
LISA_SCATTER_FIG = PICTURE_DIR / "fig_lisa_ai_2023_scatter.png"


CITY_ORDER_COL = "city_name"
YEARS = list(range(2018, 2024))
MORAN_VARIABLES = ["ai_full_panel_index", "coordination_reference_index", "innovation_support_index"]


def read_completed_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_PATH, encoding="utf-8-sig")
    needed = ["city_name", "year", "ai_full_panel_index", "coordination_reference_index"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Completed panel missing required columns: {missing}")
    return df


def build_coordinates(panel: pd.DataFrame) -> pd.DataFrame:
    official = pd.read_csv(OFFICIAL_CLEANED_PATH, encoding="utf-8-sig")
    coords = (
        official[["city_name", "longitude", "latitude"]]
        .dropna(subset=["longitude", "latitude"])
        .drop_duplicates("city_name")
        .copy()
    )
    base = panel[["city_name", "city_en", "region_group"]].drop_duplicates("city_name")
    coords = base.merge(coords, on="city_name", how="left")
    if coords[["longitude", "latitude"]].isna().any().any():
        missing = coords.loc[coords[["longitude", "latitude"]].isna().any(axis=1), "city_name"].tolist()
        raise ValueError(f"Missing coordinates for cities: {missing}")
    coords = coords.sort_values("city_name").reset_index(drop=True)
    return coords


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0088
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def build_inverse_distance_weights(coords: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cities = coords["city_name"].tolist()
    n = len(cities)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist[i, j] = haversine_km(
                coords.loc[i, "longitude"],
                coords.loc[i, "latitude"],
                coords.loc[j, "longitude"],
                coords.loc[j, "latitude"],
            )
    weights = np.zeros_like(dist)
    nonzero = dist > 0
    weights[nonzero] = 1 / dist[nonzero]
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, row_sums, where=row_sums != 0)

    wide = pd.DataFrame(weights, index=cities, columns=cities)
    wide.index.name = "city_name"
    long_records = []
    for i, origin in enumerate(cities):
        for j, dest in enumerate(cities):
            long_records.append(
                {
                    "origin_city": origin,
                    "dest_city": dest,
                    "distance_km": dist[i, j],
                    "weight_inverse_distance_rowstd": weights[i, j],
                }
            )
    return wide, pd.DataFrame(long_records)


def row_standardize(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)


def build_distance_matrix(coords: pd.DataFrame) -> pd.DataFrame:
    cities = coords["city_name"].tolist()
    n = len(cities)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist[i, j] = haversine_km(
                coords.loc[i, "longitude"],
                coords.loc[i, "latitude"],
                coords.loc[j, "longitude"],
                coords.loc[j, "latitude"],
            )
    return pd.DataFrame(dist, index=cities, columns=cities)


def build_knn_weights(distance: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Build a row-standardized k-nearest-neighbor distance matrix."""
    dist = distance.to_numpy(dtype=float)
    n = dist.shape[0]
    weights = np.zeros((n, n), dtype=float)
    for i in range(n):
        order = np.argsort(np.where(np.arange(n) == i, np.inf, dist[i]))
        neighbors = order[:k]
        weights[i, neighbors] = 1.0
    weights = row_standardize(weights)
    return pd.DataFrame(weights, index=distance.index, columns=distance.columns)


def build_geo_economic_weights(panel: pd.DataFrame, distance: pd.DataFrame) -> pd.DataFrame:
    """Build a geography-economy nested matrix for robustness checks.

    The weight is inverse distance divided by GDP-per-capita gap.  It gives
    larger weights to geographically close cities with similar development
    levels, which is a common robustness alternative to a pure geography matrix.
    """
    city_order = distance.index.tolist()
    gdp_pc = panel.groupby("city_name")["gdp_per_capita"].mean().reindex(city_order).to_numpy(dtype=float)
    dist = distance.to_numpy(dtype=float)
    n = dist.shape[0]
    weights = np.zeros((n, n), dtype=float)
    scale = np.nanmean(gdp_pc)
    for i in range(n):
        for j in range(n):
            if i == j or dist[i, j] <= 0:
                continue
            econ_gap = abs(gdp_pc[i] - gdp_pc[j]) / scale
            weights[i, j] = 1.0 / (dist[i, j] * (1.0 + econ_gap))
    weights = row_standardize(weights)
    return pd.DataFrame(weights, index=city_order, columns=city_order)


def spatial_lag_by_year(panel: pd.DataFrame, weights: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    out = panel.copy()
    city_order = weights.index.tolist()
    W = weights.loc[city_order, city_order].to_numpy()
    for var in variables:
        out[f"w_{var}"] = np.nan
    for year, sub in out.groupby("year"):
        sub_indexed = sub.set_index("city_name").loc[city_order]
        for var in variables:
            lagged = W @ sub_indexed[var].astype(float).to_numpy()
            for city, value in zip(city_order, lagged):
                out.loc[(out["city_name"] == city) & (out["year"] == year), f"w_{var}"] = value
    return out


def moran_i(values: np.ndarray, W: np.ndarray) -> float:
    z = values - values.mean()
    denom = np.dot(z, z)
    if np.isclose(denom, 0):
        return np.nan
    n = len(values)
    s0 = W.sum()
    return float((n / s0) * (z @ W @ z) / denom)


def global_moran_with_permutation(values: np.ndarray, W: np.ndarray, permutations: int = 999, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    observed = moran_i(values, W)
    permuted = np.array([moran_i(rng.permutation(values), W) for _ in range(permutations)])
    finite = permuted[np.isfinite(permuted)]
    if not np.isfinite(observed) or len(finite) == 0:
        return {
            "moran_i": observed,
            "expected_i": -1 / (len(values) - 1),
            "permutation_mean": np.nan,
            "permutation_sd": np.nan,
            "z_sim": np.nan,
            "p_sim_two_sided": np.nan,
        }
    p = (np.sum(np.abs(finite - finite.mean()) >= abs(observed - finite.mean())) + 1) / (len(finite) + 1)
    z = (observed - finite.mean()) / finite.std(ddof=1) if finite.std(ddof=1) > 0 else np.nan
    return {
        "moran_i": observed,
        "expected_i": -1 / (len(values) - 1),
        "permutation_mean": float(finite.mean()),
        "permutation_sd": float(finite.std(ddof=1)),
        "z_sim": float(z),
        "p_sim_two_sided": float(p),
    }


def local_moran(values: np.ndarray, W: np.ndarray, permutations: int = 999, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = values - values.mean()
    z_std = z / z.std(ddof=0) if z.std(ddof=0) > 0 else z
    wz = W @ z_std
    local_i = z_std * wz
    p_values = []
    for i in range(len(values)):
        perm_i = []
        for _ in range(permutations):
            permuted = rng.permutation(z_std)
            perm_i.append(z_std[i] * (W[i, :] @ permuted))
        perm_i = np.array(perm_i)
        p = (np.sum(np.abs(perm_i - perm_i.mean()) >= abs(local_i[i] - perm_i.mean())) + 1) / (permutations + 1)
        p_values.append(p)

    clusters = []
    for zi, wzi, p in zip(z_std, wz, p_values):
        if p > 0.10:
            clusters.append("Not significant")
        elif zi >= 0 and wzi >= 0:
            clusters.append("High-High")
        elif zi < 0 and wzi < 0:
            clusters.append("Low-Low")
        elif zi >= 0 and wzi < 0:
            clusters.append("High-Low")
        else:
            clusters.append("Low-High")
    return pd.DataFrame(
        {
            "z_value": z_std,
            "spatial_lag_z": wz,
            "local_moran_i": local_i,
            "p_sim_two_sided": p_values,
            "lisa_cluster_p10": clusters,
        }
    )


def compute_moran_outputs(panel: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    city_order = weights.index.tolist()
    W = weights.loc[city_order, city_order].to_numpy()
    global_rows: list[dict] = []
    local_frames: list[pd.DataFrame] = []
    for var in MORAN_VARIABLES:
        if var not in panel.columns:
            continue
        for year in YEARS:
            sub = panel.loc[panel["year"] == year].set_index("city_name").loc[city_order]
            values = sub[var].astype(float).to_numpy()
            row = {"variable": var, "year": year}
            row.update(global_moran_with_permutation(values, W, permutations=999, seed=20260429 + year))
            global_rows.append(row)

            local = local_moran(values, W, permutations=499, seed=20260429 + year)
            local.insert(0, "city_name", city_order)
            local.insert(1, "variable", var)
            local.insert(2, "year", year)
            local_frames.append(local)
    return pd.DataFrame(global_rows), pd.concat(local_frames, ignore_index=True)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if np.isclose(std, 0):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def prepare_regression_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["ln_gdp"] = np.log(out["gdp"])
    out["ln_population"] = np.log(out["population"])
    z_cols = [
        "ai_full_panel_index",
        "fiscal_intensity_ratio",
        "financial_depth_ratio",
        "fdi_gdp_ratio",
        "ln_gdp",
        "ln_population",
        "secondary_industry_share",
    ]
    for col in z_cols:
        out[f"z_{col}"] = zscore(out[col].astype(float))
    return out


def normal_pvalue_from_t(t_value: float) -> float:
    if not np.isfinite(t_value):
        return np.nan
    return math.erfc(abs(float(t_value)) / math.sqrt(2))


def ols_with_city_cluster(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    model_name: str,
) -> tuple[pd.DataFrame, dict]:
    city_dummies = pd.get_dummies(df["city_name"], prefix="city", drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(df["year"].astype(str), prefix="year", drop_first=True, dtype=float)
    X_df = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="Intercept"),
            df[x_cols].astype(float),
            city_dummies,
            year_dummies,
        ],
        axis=1,
    )
    y = df[y_col].astype(float).to_numpy().reshape(-1, 1)
    X = X_df.to_numpy(dtype=float)
    n, k = X.shape

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residual = y - X @ beta
    y_centered = y - y.mean()
    sse = float((residual.T @ residual)[0, 0])
    tss = float((y_centered.T @ y_centered)[0, 0])
    r2 = 1 - sse / tss if tss > 0 else np.nan
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k) if n > k and np.isfinite(r2) else np.nan

    meat = np.zeros((k, k), dtype=float)
    groups = df["city_name"].to_numpy()
    unique_groups = np.unique(groups)
    for group in unique_groups:
        idx = np.where(groups == group)[0]
        Xg = X[idx, :]
        ug = residual[idx, :]
        meat += Xg.T @ ug @ ug.T @ Xg
    G = len(unique_groups)
    correction = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    cov = correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    params = beta.reshape(-1)

    coef_rows = []
    for term in x_cols:
        pos = X_df.columns.get_loc(term)
        coef = params[pos]
        std_err = se[pos]
        t_val = coef / std_err if std_err > 0 else np.nan
        coef_rows.append(
            {
                "model": model_name,
                "term": term,
                "coef": coef,
                "std_err": std_err,
                "t_value": t_val,
                "p_value": normal_pvalue_from_t(t_val),
                "nobs": n,
                "r2": r2,
                "adj_r2": adj_r2,
            }
        )
    summary = {
        "model": model_name,
        "nobs": n,
        "r2": r2,
        "adj_r2": adj_r2,
        "city_fe": 1,
        "year_fe": 1,
        "cov_type": "cluster_by_city_manual",
        "parameter_count": k,
        "cluster_count": G,
    }
    return pd.DataFrame(coef_rows), summary


def run_twfe_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prepare_regression_panel(panel)
    models = {
        "M1_ai_only_twfe": ["z_ai_full_panel_index"],
        "M2_economic_controls_twfe": [
            "z_ai_full_panel_index",
            "z_ln_gdp",
            "z_ln_population",
            "z_secondary_industry_share",
        ],
        "M3_support_controls_twfe": [
            "z_ai_full_panel_index",
            "z_fiscal_intensity_ratio",
            "z_financial_depth_ratio",
            "z_fdi_gdp_ratio",
        ],
        "M4_full_controls_twfe": [
            "z_ai_full_panel_index",
            "z_ln_gdp",
            "z_ln_population",
            "z_secondary_industry_share",
            "z_fiscal_intensity_ratio",
            "z_financial_depth_ratio",
            "z_fdi_gdp_ratio",
        ],
    }
    coef_rows: list[dict] = []
    summary_rows: list[dict] = []
    for name, x_cols in models.items():
        coef_df, summary = ols_with_city_cluster(df, "coordination_reference_index", x_cols, name)
        coef_rows.extend(coef_df.to_dict("records"))
        summary_rows.append(summary)
    return pd.DataFrame(coef_rows), pd.DataFrame(summary_rows)


def run_slx_twfe_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run SLX-TWFE models as a dependency-light spillover pretest.

    SLX adds spatial lags of explanatory variables but does not include WY.
    Therefore it is a useful pretest for spillover patterns, not a substitute
    for a maximum-likelihood/GMM spatial Durbin model.
    """
    df = prepare_regression_panel(panel)
    for col in [
        "w_ai_full_panel_index",
        "w_fiscal_intensity_ratio",
        "w_financial_depth_ratio",
        "w_fdi_gdp_ratio",
    ]:
        df[f"z_{col}"] = zscore(df[col].astype(float))

    models = {
        "S1_ai_wai_twfe": ["z_ai_full_panel_index", "z_w_ai_full_panel_index"],
        "S2_support_spillover_twfe": [
            "z_ai_full_panel_index",
            "z_w_ai_full_panel_index",
            "z_fiscal_intensity_ratio",
            "z_financial_depth_ratio",
            "z_fdi_gdp_ratio",
            "z_w_fiscal_intensity_ratio",
            "z_w_financial_depth_ratio",
            "z_w_fdi_gdp_ratio",
        ],
        "S3_full_slx_twfe": [
            "z_ai_full_panel_index",
            "z_w_ai_full_panel_index",
            "z_ln_gdp",
            "z_ln_population",
            "z_secondary_industry_share",
            "z_fiscal_intensity_ratio",
            "z_financial_depth_ratio",
            "z_fdi_gdp_ratio",
            "z_w_fiscal_intensity_ratio",
            "z_w_financial_depth_ratio",
            "z_w_fdi_gdp_ratio",
        ],
    }

    coef_rows: list[dict] = []
    summary_rows: list[dict] = []
    for name, x_cols in models.items():
        coef_df, summary = ols_with_city_cluster(df, "coordination_reference_index", x_cols, name)
        summary["model_note"] = "SLX-TWFE spillover pretest; not a full SDM because WY is excluded"
        coef_rows.extend(coef_df.to_dict("records"))
        summary_rows.append(summary)
    return pd.DataFrame(coef_rows), pd.DataFrame(summary_rows)


def export_stata_weight_matrix(weights: pd.DataFrame, city_id_map: dict[str, int], path: Path) -> None:
    city_order = weights.index.tolist()
    out = weights.loc[city_order, city_order].copy()
    out.columns = [f"w{city_id_map[city]}" for city in city_order]
    out.insert(0, "city_id", [city_id_map[city] for city in city_order])
    out.to_csv(path, index=False, encoding="utf-8-sig")


def export_stata_sdm_package(
    panel: pd.DataFrame,
    weights_inv: pd.DataFrame,
    weights_knn4: pd.DataFrame,
    weights_geo_econ: pd.DataFrame,
) -> None:
    """Export Stata-ready files and a do-file for Panel SDM estimation."""
    STATA_DIR.mkdir(parents=True, exist_ok=True)
    city_order = weights_inv.index.tolist()
    city_id_map = {city: idx + 1 for idx, city in enumerate(city_order)}
    export = panel.copy()
    export["city_id"] = export["city_name"].map(city_id_map).astype(int)

    rename_map = {
        "coordination_reference_index": "coord",
        "ai_full_panel_index": "ai",
        "innovation_support_index": "innov",
        "fiscal_intensity_ratio": "fiscal",
        "financial_depth_ratio": "finance",
        "fdi_gdp_ratio": "fdi",
        "retail_per_capita": "retail_pc",
        "service_openness_proxy": "service",
        "gdp_per_capita": "gdp_pc",
        "secondary_industry_share": "sec_share",
        "tertiary_industry_share": "ter_share",
    }
    keep = ["city_id", "city_name", "city_en", "region_group", "year"] + list(rename_map.keys())
    export = export[keep].rename(columns=rename_map)
    # Stata likes short ASCII variable names. Keep labels in a separate CSV.
    export.to_csv(STATA_PANEL_CSV, index=False, encoding="utf-8-sig")
    try:
        export.to_stata(STATA_PANEL_DTA, write_index=False, version=118)
    except Exception:
        # CSV remains the guaranteed exchange format if local pandas/stata
        # encoding support is limited.
        pass

    city_map = (
        export[["city_id", "city_name", "city_en", "region_group"]]
        .drop_duplicates("city_id")
        .sort_values("city_id")
    )
    city_map.to_csv(STATA_CITY_MAP_CSV, index=False, encoding="utf-8-sig")

    export_stata_weight_matrix(weights_inv, city_id_map, STATA_W_INV_CSV)
    export_stata_weight_matrix(weights_knn4, city_id_map, STATA_W_KNN4_CSV)
    export_stata_weight_matrix(weights_geo_econ, city_id_map, STATA_W_GEO_ECON_CSV)

    do_text = f"""********************************************************************************
* Panel SDM workflow for Guangdong 21-city panel
* Generated by src/pipeline/run_panel_spatial_baseline.py
* Recommended workflow: run in Stata 16+; xsmle is used for effect decomposition.
*
* Important:
* 1. Run this do-file from the repository stata/ directory.
* 2. Main model uses inverse-distance W.
* 3. Robustness models use KNN-4 and geography-economy nested W.
* 4. In SDM, prioritize direct/indirect/total effects over raw coefficients.
********************************************************************************

clear all
set more off
capture log close
log using "sdm_run_log.log", text replace

capture mkdir "results"

global y coord
global x_main ai
global controls fiscal finance fdi retail_pc service
global rhs $x_main $controls

* 1. Load panel data
import delimited using "panel_sdm_stata.csv", clear encoding("utf-8")
xtset city_id year
describe
summarize $y $rhs

tabulate year, generate(yr)
global year_dummies yr2 yr3 yr4 yr5 yr6

* 2. Helper program for loading a spatial matrix
capture program drop load_w_matrix
program define load_w_matrix
    syntax using/, NAME(name)
    preserve
    import delimited using "`using'", clear encoding("utf-8")
    sort city_id
    drop city_id
    mkmat w*, matrix(`name')
    restore
end

load_w_matrix using "w_inverse_distance_stata.csv", name(W_inv)
load_w_matrix using "w_knn4_distance_stata.csv", name(W_knn4)
load_w_matrix using "w_geo_economic_stata.csv", name(W_ge)

* 3. Baseline two-way FE model
capture which esttab
if _rc ssc install estout, replace

estimates clear
xtreg $y $rhs $year_dummies, fe vce(cluster city_id)
estimates store TWFE
esttab TWFE using "results/twfe_baseline.rtf", replace b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    stats(N r2_w r2_o, labels("N" "Within R2" "Overall R2")) title("Baseline TWFE")

* 4. Spatial Durbin model with xsmle
capture which xsmle
if _rc ssc install xsmle, replace

* Main SDM: inverse-distance matrix.
* type(both) requests direct, indirect, and total effects.
xsmle $y $rhs, wmat(W_inv) model(sdm) fe type(both) nolog effects
estimates store SDM_inv
estimates save "results/sdm_inverse_distance.ster", replace

* Robustness 1: K-nearest-neighbor matrix.
xsmle $y $rhs, wmat(W_knn4) model(sdm) fe type(both) nolog effects
estimates store SDM_knn4
estimates save "results/sdm_knn4.ster", replace

* Robustness 2: geography-economy nested matrix.
xsmle $y $rhs, wmat(W_ge) model(sdm) fe type(both) nolog effects
estimates store SDM_geo_econ
estimates save "results/sdm_geo_economic.ster", replace

* Export comparable raw coefficient table.
esttab TWFE SDM_inv SDM_knn4 SDM_geo_econ using "results/model_comparison_raw_coefficients.rtf", ///
    replace b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    title("TWFE and Panel SDM Raw Coefficients")

display as text "=============================================================="
display as text "READING GUIDE"
display as text "For SDM, do NOT over-interpret raw beta/theta coefficients."
display as text "Use xsmle output Effects section: Direct / Indirect / Total."
display as text "If Indirect(ai) > 0 and significant: positive spatial spillover."
display as text "If Indirect(ai) < 0 and significant: siphon / polarization effect."
display as text "If Indirect(ai) insignificant: point-island stage, weak spillover."
display as text "=============================================================="

log close
********************************************************************************
"""
    STATA_DO_PATH.write_text(do_text, encoding="utf-8")

    readme = """# Stata SDM操作与解读说明

## 运行顺序

1. 在 Stata 中把工作目录切换到本文件夹：`cd "E:/文档/文档/统计建模/26Statistical-Modeling/stata"`。
2. 运行：`do run_panel_sdm.do`。
3. 重点查看每次 `xsmle` 输出中 `Effects` 或 `Direct / Indirect / Total` 部分。
4. 运行日志会保存为 `sdm_run_log.smcl`，模型结果会保存到 `results/` 文件夹。

## 三类空间矩阵

- `w_inverse_distance_stata.csv`：主模型，地理反距离矩阵。
- `w_knn4_distance_stata.csv`：稳健性检验，4近邻矩阵。
- `w_geo_economic_stata.csv`：稳健性检验，地理距离与经济距离嵌套矩阵。

## 结果解读优先级

SDM 不建议直接解读原始回归表中的 `ai` 或空间滞后项系数。论文中应优先汇报效应分解：

- `Direct Effect`：本市 AI/数字产业对本市协调发展参照指标的影响。
- `Indirect Effect`：本市 AI/数字产业对邻近城市的空间溢出影响。
- `Total Effect`：直接效应与间接效应之和。

## 三种论文叙事预案

- 如果 `Indirect(ai)` 显著为正：说明存在正向空间溢出，可写“核心城市 AI/数字产业通过技术扩散、产业链协作带动周边城市”。
- 如果 `Indirect(ai)` 显著为负：说明存在虹吸或极化，可写“AI/数字产业集聚强化核心城市吸附能力，短期可能扩大区域差异”。
- 如果 `Indirect(ai)` 不显著：说明尚未形成稳定外溢，可写“广东 AI/数字产业仍呈点状集聚，跨市扩散机制不足”。

## 注意

当前 `coord` 是协调发展参照指标，`ai` 是全域 AI/数字产业代理指数。若后续获得更完整的年报文本 AI 指数，应替换 `ai` 后重跑全流程。
"""
    STATA_README_PATH.write_text(readme, encoding="utf-8")


def plot_moran_trends(global_moran: pd.DataFrame) -> None:
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    for var, path, title in [
        ("ai_full_panel_index", MORAN_AI_FIG, "Global Moran's I: AI/Digital Proxy"),
        ("coordination_reference_index", MORAN_COORD_FIG, "Global Moran's I: Coordination Reference"),
    ]:
        sub = global_moran.loc[global_moran["variable"] == var].sort_values("year")
        if sub.empty:
            continue
        plt.figure(figsize=(7, 4.2), dpi=160)
        plt.plot(sub["year"], sub["moran_i"], marker="o", linewidth=2)
        plt.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
        for _, row in sub.iterrows():
            label = "*" if row["p_sim_two_sided"] < 0.10 else ""
            plt.text(row["year"], row["moran_i"], label, ha="center", va="bottom", fontsize=11)
        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel("Moran's I")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


def plot_lisa_scatter_2023(local_moran: pd.DataFrame) -> None:
    sub = local_moran[
        (local_moran["variable"] == "ai_full_panel_index") & (local_moran["year"] == 2023)
    ].copy()
    if sub.empty:
        return
    color_map = {
        "High-High": "#d73027",
        "Low-Low": "#4575b4",
        "High-Low": "#fdae61",
        "Low-High": "#74add1",
        "Not significant": "#bdbdbd",
    }
    plt.figure(figsize=(6.2, 5), dpi=160)
    for cluster, group in sub.groupby("lisa_cluster_p10"):
        plt.scatter(
            group["z_value"],
            group["spatial_lag_z"],
            label=cluster,
            s=45,
            color=color_map.get(cluster, "#666666"),
            edgecolor="white",
            linewidth=0.6,
        )
    plt.axhline(0, color="#555555", linewidth=0.8)
    plt.axvline(0, color="#555555", linewidth=0.8)
    plt.xlabel("AI/Digital Proxy (standardized)")
    plt.ylabel("Spatial Lag")
    plt.title("LISA Scatter: AI/Digital Proxy, 2023")
    plt.legend(frameon=False, fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(LISA_SCATTER_FIG)
    plt.close()


def main() -> None:
    SPATIAL_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)

    panel = read_completed_panel()
    coords = build_coordinates(panel)
    weights_wide, weights_long = build_inverse_distance_weights(coords)
    distance = build_distance_matrix(coords)
    weights_knn4 = build_knn_weights(distance, k=4)
    weights_geo_econ = build_geo_economic_weights(panel, distance)
    spatial_panel = panel.merge(coords[["city_name", "longitude", "latitude"]], on="city_name", how="left")
    spatial_panel = spatial_lag_by_year(
        spatial_panel,
        weights_wide,
        [
            "ai_full_panel_index",
            "coordination_reference_index",
            "innovation_support_index",
            "fiscal_intensity_ratio",
            "financial_depth_ratio",
            "fdi_gdp_ratio",
        ],
    )
    global_moran, local_moran_df = compute_moran_outputs(spatial_panel, weights_wide)
    twfe_results, twfe_summary = run_twfe_models(spatial_panel)
    slx_results, slx_summary = run_slx_twfe_models(spatial_panel)

    coords.to_csv(COORDS_PATH, index=False, encoding="utf-8-sig")
    weights_wide.to_csv(WIDE_W_PATH, encoding="utf-8-sig")
    weights_long.to_csv(LONG_W_PATH, index=False, encoding="utf-8-sig")
    spatial_panel.to_csv(SPATIAL_PANEL_PATH, index=False, encoding="utf-8-sig")
    global_moran.to_csv(MORAN_RESULTS_PATH, index=False, encoding="utf-8-sig")
    local_moran_df.to_csv(LISA_RESULTS_PATH, index=False, encoding="utf-8-sig")
    twfe_results.to_csv(TWFE_RESULTS_PATH, index=False, encoding="utf-8-sig")
    twfe_summary.to_csv(TWFE_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    slx_results.to_csv(SLX_RESULTS_PATH, index=False, encoding="utf-8-sig")
    slx_summary.to_csv(SLX_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    export_stata_sdm_package(spatial_panel, weights_wide, weights_knn4, weights_geo_econ)
    plot_moran_trends(global_moran)
    plot_lisa_scatter_2023(local_moran_df)

    print(f"Wrote {COORDS_PATH}")
    print(f"Wrote {WIDE_W_PATH}")
    print(f"Wrote {LONG_W_PATH}")
    print(f"Wrote {SPATIAL_PANEL_PATH}")
    print(f"Wrote {MORAN_RESULTS_PATH}")
    print(f"Wrote {LISA_RESULTS_PATH}")
    print(f"Wrote {TWFE_RESULTS_PATH}")
    print(f"Wrote {TWFE_SUMMARY_PATH}")
    print(f"Wrote {SLX_RESULTS_PATH}")
    print(f"Wrote {SLX_SUMMARY_PATH}")
    print(f"Wrote {STATA_PANEL_CSV}")
    print(f"Wrote {STATA_PANEL_DTA}")
    print(f"Wrote {STATA_CITY_MAP_CSV}")
    print(f"Wrote {STATA_W_INV_CSV}")
    print(f"Wrote {STATA_W_KNN4_CSV}")
    print(f"Wrote {STATA_W_GEO_ECON_CSV}")
    print(f"Wrote {STATA_DO_PATH}")
    print(f"Wrote {STATA_README_PATH}")
    print("Global Moran's I:")
    print(global_moran.to_string(index=False))
    print("TWFE key coefficients:")
    print(twfe_results.to_string(index=False))
    print("SLX-TWFE key coefficients:")
    print(slx_results.to_string(index=False))


if __name__ == "__main__":
    main()
