"""Generate last-mile competition checks for the paper.

The script adds two lightweight robustness/extension outputs:
1. regional heterogeneity summaries; and
2. one-period-lagged AI Panel SDM impacts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "analysis_ready"
SPATIAL_DIR = PROJECT_ROOT / "data" / "processed" / "spatial"
STATA_DIR = PROJECT_ROOT / "stata"
TABLE_DIR = PROJECT_ROOT / "paper_tables"

SPATIAL_PANEL_PATH = ANALYSIS_DIR / "panel_21city_2018_2023_spatial_ready.csv"
STATA_PANEL_PATH = STATA_DIR / "panel_sdm_stata.csv"
REGIONAL_SUMMARY_CSV = ANALYSIS_DIR / "regional_heterogeneity_summary.csv"
LAGGED_IMPACTS_CSV = ANALYSIS_DIR / "lagged_ai_sdm_impacts.csv"
PY_IMPACTS_CSV = ANALYSIS_DIR / "python_panel_sdm_impacts.csv"
REGIONAL_TABLE_TEX = TABLE_DIR / "table_regional_heterogeneity_summary.tex"
LAGGED_TABLE_TEX = TABLE_DIR / "table_lagged_ai_endogeneity.tex"
FIG_REGIONAL_HETEROGENEITY = PROJECT_ROOT / "picture" / "fig_regional_heterogeneity_summary.png"
FIG_SDM_DYNAMIC_COMPARISON = PROJECT_ROOT / "picture" / "fig_sdm_dynamic_comparison.png"

WEIGHT_FILES = {
    "inverse_distance": STATA_DIR / "w_inverse_distance_stata.csv",
    "knn4": STATA_DIR / "w_knn4_distance_stata.csv",
    "geo_economic": STATA_DIR / "w_geo_economic_stata.csv",
}

MATRIX_LABELS = {
    "inverse_distance": "地理反距离矩阵",
    "knn4": "4近邻矩阵",
    "geo_economic": "地理-经济嵌套矩阵",
}

REGION_ORDER = ["珠三角", "粤东", "粤西", "粤北"]
REGION_COLORS = ["#174A7C", "#2F855A", "#D9822B", "#7C3AED"]
Y_COL = "coord"
X_COLS = ["lag_ai", "fiscal", "finance", "fdi", "retail_pc", "service"]
RHO_BOUNDS = (-0.95, 0.95)


@dataclass
class SDMResult:
    matrix_name: str
    rho: float
    beta: np.ndarray
    theta: np.ndarray
    cov_beta_theta: np.ndarray
    rho_se: float
    nobs: int


def stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def fmt_coef(value: float, p_value: float | None = None) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.4f}{stars(p_value) if p_value is not None else ''}"


def configure_plot_style() -> None:
    font_files = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_file in font_files:
        if font_file.exists():
            try:
                fm.fontManager.addfont(str(font_file))
            except Exception:
                pass
    preferred = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]
    available = {font.name for font in fm.fontManager.ttflist}
    selected = next((font for font in preferred if font in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": selected,
            "font.sans-serif": [selected],
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 360,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def normal_pvalue(z_value: float) -> float:
    if not np.isfinite(z_value):
        return np.nan
    return math.erfc(abs(float(z_value)) / math.sqrt(2.0))


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = values.std(ddof=0)
    if np.isclose(std, 0):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def two_way_demean(values: np.ndarray, city_ids: np.ndarray, years: np.ndarray) -> np.ndarray:
    out = values.astype(float).copy()
    squeeze = False
    if out.ndim == 1:
        out = out.reshape(-1, 1)
        squeeze = True

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


def spatial_lag_panel(df: pd.DataFrame, weights: np.ndarray, columns: list[str]) -> np.ndarray:
    frames = []
    for year in sorted(df["year"].unique()):
        sub = df[df["year"] == year].sort_values("city_id")
        frames.append(weights @ sub[columns].to_numpy(dtype=float))
    return np.vstack(frames)


def logdet_i_minus_rho_w(weights: np.ndarray, rho: float, periods: int) -> float:
    sign, logdet = np.linalg.slogdet(np.eye(weights.shape[0]) - rho * weights)
    if sign <= 0:
        return -np.inf
    return periods * float(logdet)


def ols_fit(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    residual = y - x @ beta
    ssr = float(residual.T @ residual)
    return beta, residual, ssr


def load_lagged_panel() -> pd.DataFrame:
    df = pd.read_csv(STATA_PANEL_PATH, encoding="utf-8-sig").sort_values(["city_id", "year"])
    df["lag_ai"] = df.groupby("city_id")["ai"].shift(1)
    df = df.dropna(subset=["lag_ai"]).sort_values(["year", "city_id"]).reset_index(drop=True)
    for col in [Y_COL, *X_COLS]:
        df[f"z_{col}"] = zscore(df[col])
    return df


def load_weight(path: Path) -> np.ndarray:
    raw = pd.read_csv(path, encoding="utf-8-sig").sort_values("city_id")
    return raw.drop(columns=["city_id"]).to_numpy(dtype=float)


def estimate_lagged_sdm(df: pd.DataFrame, weights: np.ndarray, matrix_name: str) -> SDMResult:
    z_y_col = f"z_{Y_COL}"
    z_x_cols = [f"z_{col}" for col in X_COLS]
    city_ids = df["city_id"].to_numpy()
    years = df["year"].to_numpy()
    periods = df["year"].nunique()

    y = df[z_y_col].to_numpy(dtype=float)
    x = df[z_x_cols].to_numpy(dtype=float)
    wy = spatial_lag_panel(df, weights, [z_y_col]).ravel()
    wx = spatial_lag_panel(df, weights, z_x_cols)

    y_dm = two_way_demean(y, city_ids, years)
    wy_dm = two_way_demean(wy, city_ids, years)
    x_dm = two_way_demean(x, city_ids, years)
    wx_dm = two_way_demean(wx, city_ids, years)
    z = np.column_stack([x_dm, wx_dm])
    nobs = len(y_dm)

    def neg_profile_loglik(rho: float) -> float:
        y_rho = y_dm - rho * wy_dm
        _, _, ssr = ols_fit(y_rho, z)
        if ssr <= 0:
            return np.inf
        logdet = logdet_i_minus_rho_w(weights, rho, periods)
        if not np.isfinite(logdet):
            return np.inf
        ll = logdet - (nobs / 2.0) * (math.log(ssr / nobs) + math.log(2 * math.pi) + 1.0)
        return -ll

    opt = minimize_scalar(neg_profile_loglik, bounds=RHO_BOUNDS, method="bounded", options={"xatol": 1e-8})
    rho = float(opt.x)
    beta_theta, _, ssr = ols_fit(y_dm - rho * wy_dm, z)
    sigma2 = ssr / nobs
    cov_beta_theta = sigma2 * np.linalg.pinv(z.T @ z)

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
        cov_beta_theta=cov_beta_theta,
        rho_se=rho_se,
        nobs=nobs,
    )


def compute_impacts(weights: np.ndarray, rho: float, beta: float, theta: float) -> tuple[float, float, float]:
    s_matrix = np.linalg.inv(np.eye(weights.shape[0]) - rho * weights) @ (
        beta * np.eye(weights.shape[0]) + theta * weights
    )
    direct = float(np.trace(s_matrix) / weights.shape[0])
    total = float(s_matrix.sum(axis=1).mean())
    return direct, total - direct, total


def simulate_lagged_ai_impacts(
    result: SDMResult,
    weights: np.ndarray,
    draws: int = 1000,
    seed: int = 20260429,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    k = len(X_COLS)
    mean = np.concatenate([[result.rho], result.beta, result.theta])
    cov = np.zeros((1 + 2 * k, 1 + 2 * k), dtype=float)
    cov[0, 0] = result.rho_se**2 if np.isfinite(result.rho_se) else 0.0
    cov[1:, 1:] = result.cov_beta_theta

    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 1e-10:
        cov += np.eye(cov.shape[0]) * (abs(eigvals.min()) + 1e-8)

    samples = rng.multivariate_normal(mean, cov, size=draws)
    point = compute_impacts(weights, result.rho, result.beta[0], result.theta[0])
    rows = []
    for effect_type, estimate in zip(["direct", "indirect", "total"], point):
        simulated = []
        for sample in samples:
            rho = float(np.clip(sample[0], RHO_BOUNDS[0] + 1e-4, RHO_BOUNDS[1] - 1e-4))
            beta = float(sample[1])
            theta = float(sample[1 + k])
            simulated.append(compute_impacts(weights, rho, beta, theta)[["direct", "indirect", "total"].index(effect_type)])
        arr = np.asarray(simulated, dtype=float)
        se = float(arr.std(ddof=1))
        z_value = estimate / se if se > 0 else np.nan
        rows.append(
            {
                "matrix": result.matrix_name,
                "matrix_label": MATRIX_LABELS[result.matrix_name],
                "effect_type": effect_type,
                "estimate": estimate,
                "std_err_sim": se,
                "z_value": z_value,
                "p_value": normal_pvalue(z_value),
                "ci95_low": float(np.quantile(arr, 0.025)),
                "ci95_high": float(np.quantile(arr, 0.975)),
                "rho": result.rho,
                "nobs": result.nobs,
                "draws": draws,
            }
        )
    return pd.DataFrame(rows)


def build_regional_summary() -> pd.DataFrame:
    panel = pd.read_csv(SPATIAL_PANEL_PATH, encoding="utf-8-sig")
    summary = (
        panel.groupby("region_group", as_index=False)
        .agg(
            city_count=("city_name", "nunique"),
            ai_mean=("ai_full_panel_index", "mean"),
            coordination_mean=("coordination_reference_index", "mean"),
            neighboring_ai_exposure_mean=("w_ai_full_panel_index", "mean"),
            innovation_support_mean=("innovation_support_index", "mean"),
        )
    )
    summary["region_group"] = pd.Categorical(summary["region_group"], categories=REGION_ORDER, ordered=True)
    summary = summary.sort_values("region_group").reset_index(drop=True)
    summary.to_csv(REGIONAL_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    return summary


def build_lagged_impacts() -> pd.DataFrame:
    df = load_lagged_panel()
    frames = []
    for matrix_name, path in WEIGHT_FILES.items():
        weights = load_weight(path)
        result = estimate_lagged_sdm(df, weights, matrix_name)
        frames.append(simulate_lagged_ai_impacts(result, weights))
    impacts = pd.concat(frames, ignore_index=True)
    impacts.to_csv(LAGGED_IMPACTS_CSV, index=False, encoding="utf-8-sig")
    return impacts


def write_regional_table(summary: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{区域异质性分组描述结果}",
        r"\label{tab:regional-heterogeneity}",
        r"{\tablebodyformat",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"区域 & 城市数 & AI集聚均值 & 邻近AI暴露均值 & 创新支撑均值 & 协调发展均值 \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['region_group']} & {int(row['city_count'])} & "
            f"{row['ai_mean']:.4f} & {row['neighboring_ai_exposure_mean']:.4f} & "
            f"{row['innovation_support_mean']:.4f} & {row['coordination_mean']:.4f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    REGIONAL_TABLE_TEX.write_text("\n".join(lines), encoding="utf-8")


def write_lagged_table(impacts: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pivot = impacts.pivot(index=["matrix", "matrix_label"], columns="effect_type", values=["estimate", "p_value"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    pivot["matrix"] = pd.Categorical(pivot["matrix"], categories=list(WEIGHT_FILES), ordered=True)
    pivot = pivot.sort_values("matrix").reset_index(drop=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\caption{滞后一期AI产业集聚的SDM效应分解}",
        r"\label{tab:lagged-ai-sdm}",
        r"{\tablebodyformat",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"空间矩阵 & 直接效应 & 间接效应 & 总效应 \\",
        r"\midrule",
    ]
    for _, row in pivot.iterrows():
        lines.append(
            f"{row['matrix_label']} & "
            f"{fmt_coef(row['estimate_direct'], row['p_value_direct'])} & "
            f"{fmt_coef(row['estimate_indirect'], row['p_value_indirect'])} & "
            f"{fmt_coef(row['estimate_total'], row['p_value_total'])} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item 注：核心解释变量替换为滞后一期AI产业集聚代理指标，样本为2019—2023年105个城市—年份观测值；*、**、***分别表示在10\%、5\%、1\%水平显著。",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
            "",
        ]
    )
    LAGGED_TABLE_TEX.write_text("\n".join(lines), encoding="utf-8")


def plot_regional_heterogeneity(summary: pd.DataFrame) -> None:
    configure_plot_style()
    FIG_REGIONAL_HETEROGENEITY.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("ai_mean", "AI集聚"),
        ("neighboring_ai_exposure_mean", "邻近AI暴露"),
        ("innovation_support_mean", "创新支撑"),
        ("coordination_mean", "协调发展"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 4.8), sharey=True)
    y = np.arange(len(summary))
    for ax, (col, title) in zip(axes, metrics):
        ax.barh(
            y,
            summary[col],
            color=REGION_COLORS,
            edgecolor="white",
            linewidth=0.8,
            height=0.62,
        )
        ax.axvline(0, color="#55606E", linewidth=0.8, linestyle="--")
        for idx, value in enumerate(summary[col]):
            ax.text(
                value + (0.025 if value >= 0 else -0.025),
                idx,
                f"{value:.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8.5,
                color="#263238",
            )
        ax.set_title(title, color="#102A43", pad=8, fontweight="bold")
        ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.32)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(summary["region_group"])
    axes[0].invert_yaxis()
    fig.suptitle("广东四大区域AI集聚、邻近暴露与协调发展差异", fontsize=14, fontweight="bold", color="#102A43")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_REGIONAL_HETEROGENEITY, bbox_inches="tight")
    plt.close(fig)


def plot_sdm_dynamic_comparison(lagged_impacts: pd.DataFrame) -> None:
    configure_plot_style()
    FIG_SDM_DYNAMIC_COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    contemporaneous = pd.read_csv(PY_IMPACTS_CSV, encoding="utf-8-sig")
    contemporaneous = contemporaneous[contemporaneous["variable"] == "ai"].copy()
    contemporaneous["matrix_label"] = contemporaneous["matrix"].map(MATRIX_LABELS)
    contemporaneous["period_type"] = "同期AI"
    lagged = lagged_impacts.copy()
    lagged["period_type"] = "滞后一期AI"
    keep_cols = ["matrix", "matrix_label", "effect_type", "estimate", "p_value", "period_type"]
    combined = pd.concat([contemporaneous[keep_cols], lagged[keep_cols]], ignore_index=True)

    effect_labels = {"direct": "直接效应", "indirect": "间接效应", "total": "总效应"}
    matrix_order = list(WEIGHT_FILES)
    matrix_labels = [MATRIX_LABELS[m] for m in matrix_order]
    colors = {"同期AI": "#B83227", "滞后一期AI": "#174A7C"}

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharey=False)
    x = np.arange(len(matrix_order))
    width = 0.34
    for ax, effect in zip(axes, ["direct", "indirect", "total"]):
        sub = combined[combined["effect_type"] == effect]
        for offset, period in [(-width / 2, "同期AI"), (width / 2, "滞后一期AI")]:
            vals = []
            ps = []
            for matrix in matrix_order:
                row = sub[(sub["matrix"] == matrix) & (sub["period_type"] == period)].iloc[0]
                vals.append(row["estimate"])
                ps.append(row["p_value"])
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=colors[period],
                alpha=0.88,
                edgecolor="white",
                linewidth=0.8,
                label=period if effect == "direct" else None,
            )
            for bar, value, p_value in zip(bars, vals, ps):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (0.035 if value >= 0 else -0.055),
                    f"{value:.2f}{stars(p_value)}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                    color="#263238",
                )
        ax.axhline(0, color="#55606E", linewidth=0.8)
        ax.set_title(effect_labels[effect], color="#102A43", pad=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(matrix_labels, rotation=18, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("同期虹吸与滞后扩散：AI产业集聚SDM效应对比", fontsize=14, fontweight="bold", color="#102A43")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG_SDM_DYNAMIC_COMPARISON, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    regional_summary = build_regional_summary()
    lagged_impacts = build_lagged_impacts()
    write_regional_table(regional_summary)
    write_lagged_table(lagged_impacts)
    plot_regional_heterogeneity(regional_summary)
    plot_sdm_dynamic_comparison(lagged_impacts)

    print(f"Wrote {REGIONAL_SUMMARY_CSV}")
    print(f"Wrote {LAGGED_IMPACTS_CSV}")
    print(f"Wrote {REGIONAL_TABLE_TEX}")
    print(f"Wrote {LAGGED_TABLE_TEX}")
    print(f"Wrote {FIG_REGIONAL_HETEROGENEITY}")
    print(f"Wrote {FIG_SDM_DYNAMIC_COMPARISON}")
    print(regional_summary.to_string(index=False))
    print(lagged_impacts[lagged_impacts['effect_type'].isin(['direct', 'indirect', 'total'])].to_string(index=False))


if __name__ == "__main__":
    main()
