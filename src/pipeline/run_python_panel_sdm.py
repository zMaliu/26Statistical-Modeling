"""Automatic Python implementation for Panel SDM estimation.

This script estimates a two-way fixed-effects spatial Durbin model by
concentrated maximum likelihood:

    y = rho W y + X beta + W X theta + city FE + year FE + e

and reports LeSage-Pace style direct, indirect, and total impacts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATA_DIR = PROJECT_ROOT / "stata"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "analysis_ready"

PANEL_PATH = STATA_DIR / "panel_sdm_stata.csv"
WEIGHT_FILES = {
    "inverse_distance": STATA_DIR / "w_inverse_distance_stata.csv",
    "knn4": STATA_DIR / "w_knn4_distance_stata.csv",
    "geo_economic": STATA_DIR / "w_geo_economic_stata.csv",
}

COEF_OUT = ANALYSIS_DIR / "python_panel_sdm_coefficients.csv"
IMPACT_OUT = ANALYSIS_DIR / "python_panel_sdm_impacts.csv"
SUMMARY_OUT = ANALYSIS_DIR / "python_panel_sdm_summary.csv"
INTERPRETATION_OUT = ANALYSIS_DIR / "python_panel_sdm_interpretation.md"

Y_COL = "coord"
X_COLS = ["ai", "fiscal", "finance", "fdi", "retail_pc", "service"]
MAIN_VARIABLE = "ai"
RHO_BOUNDS = (-0.95, 0.95)


@dataclass
class SDMResult:
    matrix_name: str
    rho: float
    beta: np.ndarray
    theta: np.ndarray
    sigma2: float
    loglik: float
    aic: float
    bic: float
    residual: np.ndarray
    z_matrix: np.ndarray
    cov_beta_theta: np.ndarray
    rho_se: float
    nobs: int
    k_params: int


def normal_pvalue(z_value: float) -> float:
    if not np.isfinite(z_value):
        return np.nan
    return math.erfc(abs(float(z_value)) / math.sqrt(2.0))


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if np.isclose(std, 0):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def two_way_demean(values: np.ndarray, city_ids: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Apply city and year fixed-effect within transformation."""
    out = values.astype(float).copy()
    if out.ndim == 1:
        out = out.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    grand = out.mean(axis=0, keepdims=True)
    city_mean = np.zeros_like(out)
    year_mean = np.zeros_like(out)
    for city in np.unique(city_ids):
        mask = city_ids == city
        city_mean[mask] = out[mask].mean(axis=0, keepdims=True)
    for year in np.unique(years):
        mask = years == year
        year_mean[mask] = out[mask].mean(axis=0, keepdims=True)
    demeaned = out - city_mean - year_mean + grand
    return demeaned.ravel() if squeeze else demeaned


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_PATH)
    required = ["city_id", "year", Y_COL, *X_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Panel missing columns: {missing}")
    df = df.sort_values(["year", "city_id"]).reset_index(drop=True)
    for col in [Y_COL, *X_COLS]:
        df[f"z_{col}"] = zscore(df[col].astype(float))
    return df


def load_weight(path: Path) -> np.ndarray:
    w = pd.read_csv(path).sort_values("city_id")
    mat = w.drop(columns=["city_id"]).to_numpy(dtype=float)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Weight matrix is not square: {path}")
    return mat


def spatial_lag_panel(df: pd.DataFrame, W: np.ndarray, cols: list[str]) -> np.ndarray:
    years = sorted(df["year"].unique())
    out = []
    for year in years:
        sub = df.loc[df["year"] == year].sort_values("city_id")
        X = sub[cols].to_numpy(dtype=float)
        out.append(W @ X)
    return np.vstack(out)


def logdet_i_minus_rho_w(W: np.ndarray, rho: float, periods: int) -> float:
    sign, logdet = np.linalg.slogdet(np.eye(W.shape[0]) - rho * W)
    if sign <= 0:
        return -np.inf
    return periods * float(logdet)


def ols_fit(y: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta = np.linalg.pinv(Z.T @ Z) @ Z.T @ y
    residual = y - Z @ beta
    ssr = float(residual.T @ residual)
    return beta, residual, ssr


def estimate_sdm(df: pd.DataFrame, W: np.ndarray, matrix_name: str) -> SDMResult:
    z_y_col = f"z_{Y_COL}"
    z_x_cols = [f"z_{col}" for col in X_COLS]
    city_ids = df["city_id"].to_numpy()
    years = df["year"].to_numpy()
    periods = df["year"].nunique()

    y = df[z_y_col].to_numpy(dtype=float)
    X = df[z_x_cols].to_numpy(dtype=float)
    Wy = spatial_lag_panel(df, W, [z_y_col]).ravel()
    WX = spatial_lag_panel(df, W, z_x_cols)

    y_dm = two_way_demean(y, city_ids, years)
    Wy_dm = two_way_demean(Wy, city_ids, years)
    X_dm = two_way_demean(X, city_ids, years)
    WX_dm = two_way_demean(WX, city_ids, years)
    Z = np.column_stack([X_dm, WX_dm])
    nobs = len(y_dm)
    p = Z.shape[1]

    def neg_profile_loglik(rho: float) -> float:
        y_rho = y_dm - rho * Wy_dm
        _, _, ssr = ols_fit(y_rho, Z)
        if ssr <= 0:
            return np.inf
        logdet = logdet_i_minus_rho_w(W, rho, periods)
        if not np.isfinite(logdet):
            return np.inf
        ll = logdet - (nobs / 2.0) * (math.log(ssr / nobs) + math.log(2 * math.pi) + 1.0)
        return -ll

    opt = minimize_scalar(neg_profile_loglik, bounds=RHO_BOUNDS, method="bounded", options={"xatol": 1e-8})
    rho = float(opt.x)
    y_rho = y_dm - rho * Wy_dm
    beta_theta, residual, ssr = ols_fit(y_rho, Z)
    sigma2 = ssr / nobs
    ll = -float(opt.fun)
    k_params = p + 2  # beta/theta + rho + sigma2
    aic = -2 * ll + 2 * k_params
    bic = -2 * ll + math.log(nobs) * k_params

    zz_inv = np.linalg.pinv(Z.T @ Z)
    cov_bt = sigma2 * zz_inv

    # Numeric profile-likelihood Hessian for rho.
    h = 1e-4
    f0 = neg_profile_loglik(rho)
    fp = neg_profile_loglik(min(RHO_BOUNDS[1] - 1e-6, rho + h))
    fm = neg_profile_loglik(max(RHO_BOUNDS[0] + 1e-6, rho - h))
    hessian = (fp - 2 * f0 + fm) / (h * h)
    rho_se = math.sqrt(1 / hessian) if hessian > 0 else np.nan

    k = len(X_COLS)
    return SDMResult(
        matrix_name=matrix_name,
        rho=rho,
        beta=beta_theta[:k],
        theta=beta_theta[k:],
        sigma2=sigma2,
        loglik=ll,
        aic=aic,
        bic=bic,
        residual=residual,
        z_matrix=Z,
        cov_beta_theta=cov_bt,
        rho_se=rho_se,
        nobs=nobs,
        k_params=k_params,
    )


def compute_impacts(W: np.ndarray, rho: float, beta: float, theta: float) -> tuple[float, float, float]:
    S = np.linalg.inv(np.eye(W.shape[0]) - rho * W) @ (beta * np.eye(W.shape[0]) + theta * W)
    direct = float(np.trace(S) / W.shape[0])
    total = float(S.sum(axis=1).mean())
    indirect = total - direct
    return direct, indirect, total


def impact_simulation(result: SDMResult, W: np.ndarray, draws: int = 1000, seed: int = 20260429) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    k = len(X_COLS)
    mean = np.concatenate([[result.rho], result.beta, result.theta])
    cov = np.zeros((1 + 2 * k, 1 + 2 * k), dtype=float)
    cov[0, 0] = result.rho_se**2 if np.isfinite(result.rho_se) else 0.0
    cov[1:, 1:] = result.cov_beta_theta

    # Numerical covariance matrices can be slightly non-PSD.  Stabilize gently.
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 1e-10:
        cov += np.eye(cov.shape[0]) * (abs(eigvals.min()) + 1e-8)

    samples = rng.multivariate_normal(mean, cov, size=draws)
    rows = []
    for var_idx, var in enumerate(X_COLS):
        simulated = {"direct": [], "indirect": [], "total": []}
        for sample in samples:
            rho = float(np.clip(sample[0], RHO_BOUNDS[0] + 1e-4, RHO_BOUNDS[1] - 1e-4))
            beta = float(sample[1 + var_idx])
            theta = float(sample[1 + k + var_idx])
            direct, indirect, total = compute_impacts(W, rho, beta, theta)
            simulated["direct"].append(direct)
            simulated["indirect"].append(indirect)
            simulated["total"].append(total)

        point = compute_impacts(W, result.rho, result.beta[var_idx], result.theta[var_idx])
        for effect_type, estimate in zip(["direct", "indirect", "total"], point):
            arr = np.asarray(simulated[effect_type], dtype=float)
            se = float(arr.std(ddof=1))
            z_val = estimate / se if se > 0 else np.nan
            rows.append(
                {
                    "matrix": result.matrix_name,
                    "variable": var,
                    "effect_type": effect_type,
                    "estimate": estimate,
                    "std_err_sim": se,
                    "z_value": z_val,
                    "p_value": normal_pvalue(z_val),
                    "ci95_low": float(np.quantile(arr, 0.025)),
                    "ci95_high": float(np.quantile(arr, 0.975)),
                    "draws": draws,
                }
            )
    return pd.DataFrame(rows)


def coefficients_table(result: SDMResult) -> pd.DataFrame:
    k = len(X_COLS)
    rows = []
    terms = ["rho"] + [f"beta_{col}" for col in X_COLS] + [f"theta_w_{col}" for col in X_COLS]
    estimates = np.concatenate([[result.rho], result.beta, result.theta])
    ses = np.concatenate([[result.rho_se], np.sqrt(np.maximum(np.diag(result.cov_beta_theta), 0))])
    for term, estimate, se in zip(terms, estimates, ses):
        z_val = estimate / se if se > 0 else np.nan
        rows.append(
            {
                "matrix": result.matrix_name,
                "term": term,
                "estimate": estimate,
                "std_err": se,
                "z_value": z_val,
                "p_value": normal_pvalue(z_val),
                "nobs": result.nobs,
                "loglik": result.loglik,
                "aic": result.aic,
                "bic": result.bic,
            }
        )
    return pd.DataFrame(rows)


def write_interpretation(impacts: pd.DataFrame, summaries: pd.DataFrame) -> None:
    main = impacts[(impacts["matrix"] == "inverse_distance") & (impacts["variable"] == MAIN_VARIABLE)]
    lines = [
        "# Panel SDM 估计结果解读",
        "",
        "说明：本结果由开源 Python 空间面板程序生成，用于估计双向固定效应 Panel SDM 并分解直接效应、间接效应与总效应。",
        "",
        "## 模型概况",
        "",
        summaries.to_markdown(index=False),
        "",
        "## 主变量 AI 的效应分解",
        "",
        main[["effect_type", "estimate", "std_err_sim", "z_value", "p_value", "ci95_low", "ci95_high"]].to_markdown(index=False),
        "",
        "## 叙事判断",
        "",
    ]
    indirect = main[main["effect_type"] == "indirect"].iloc[0]
    estimate = indirect["estimate"]
    p_value = indirect["p_value"]
    if p_value < 0.10 and estimate > 0:
        story = "AI 间接效应显著为正，可写作正向空间溢出：核心城市 AI/数字产业通过技术扩散、产业链协作带动周边城市。"
    elif p_value < 0.10 and estimate < 0:
        story = "AI 间接效应显著为负，可写作空间虹吸或极化：核心城市 AI/数字产业可能短期吸附周边资源，扩大区域差异。"
    else:
        story = "AI 间接效应未通过显著性检验，更稳妥的写法是：AI/数字产业存在空间集聚，但跨市外溢尚未形成稳定统计证据。"
    lines.extend([story, ""])
    INTERPRETATION_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_panel()
    coef_frames = []
    impact_frames = []
    summary_rows = []
    for matrix_name, path in WEIGHT_FILES.items():
        W = load_weight(path)
        result = estimate_sdm(df, W, matrix_name)
        coef_frames.append(coefficients_table(result))
        impact_frames.append(impact_simulation(result, W, draws=1000))
        summary_rows.append(
            {
                "matrix": matrix_name,
                "rho": result.rho,
                "rho_se": result.rho_se,
                "sigma2": result.sigma2,
                "loglik": result.loglik,
                "aic": result.aic,
                "bic": result.bic,
                "nobs": result.nobs,
                "note": "Two-way FE Panel SDM; variables standardized before estimation",
            }
        )

    coefs = pd.concat(coef_frames, ignore_index=True)
    impacts = pd.concat(impact_frames, ignore_index=True)
    summaries = pd.DataFrame(summary_rows)

    coefs.to_csv(COEF_OUT, index=False, encoding="utf-8-sig")
    impacts.to_csv(IMPACT_OUT, index=False, encoding="utf-8-sig")
    summaries.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
    write_interpretation(impacts, summaries)

    print(f"Wrote {COEF_OUT}")
    print(f"Wrote {IMPACT_OUT}")
    print(f"Wrote {SUMMARY_OUT}")
    print(f"Wrote {INTERPRETATION_OUT}")
    print("AI impacts:")
    print(
        impacts[impacts["variable"] == MAIN_VARIABLE][
            ["matrix", "effect_type", "estimate", "std_err_sim", "z_value", "p_value"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
