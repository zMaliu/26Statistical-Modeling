"""Refresh publication-quality figures used in the final paper.

The model outputs are not changed here.  This script only redraws figures with
CJK-safe fonts, consistent styling, and high-resolution export settings.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "analysis_ready"
PICTURE_DIR = PROJECT_ROOT / "picture"

PRIMARY = "#174A7C"
ACCENT = "#D9822B"
DEEP_RED = "#B83227"
DEEP_BLUE = "#1F6F9F"
MUTED = "#6B7280"
GRID = "#D8DEE9"


def _select_cjk_font() -> str:
    """Select a Chinese-capable font and register common Windows font files."""
    font_files = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_file in font_files:
        if font_file.exists():
            fm.fontManager.addfont(str(font_file))

    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    return next((font for font in preferred_fonts if font in available), "DejaVu Sans")


def configure_plot_style() -> None:
    selected = _select_cjk_font()
    plt.rcParams.update(
        {
            "font.family": selected,
            "font.sans-serif": [selected],
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 360,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.9,
            "savefig.facecolor": "white",
        }
    )


def polish_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#FBFCFE")
    ax.grid(axis="both", alpha=0.42, linestyle="--", linewidth=0.65, color=GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#263238", length=3.5, width=0.8)


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=360)
    plt.close(fig)


def significance_marker(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def plot_moran_trend(df: pd.DataFrame, variable: str, title: str, out_path: Path) -> None:
    sub = df[df["variable"] == variable].sort_values("year")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.55))
    ax.plot(
        sub["year"],
        sub["moran_i"],
        marker="o",
        linewidth=2.6,
        markersize=7.2,
        color=PRIMARY,
        markerfacecolor=ACCENT,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=3,
    )
    ax.fill_between(
        sub["year"],
        sub["moran_i"],
        0,
        color=PRIMARY,
        alpha=0.08,
        zorder=1,
    )
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--", zorder=2)
    for _, row in sub.iterrows():
        label = f"{row['moran_i']:.3f}{significance_marker(row['p_sim_two_sided'])}"
        ax.text(
            row["year"],
            row["moran_i"] + 0.006,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#7A3E00",
        )
    ax.set_title(title, pad=13, color="#102A43", fontweight="bold")
    ax.set_xlabel("年份")
    ax.set_ylabel("Moran's I")
    ax.set_xticks(sub["year"].astype(int).tolist())
    y_min = min(0, sub["moran_i"].min()) - 0.025
    y_max = sub["moran_i"].max() + 0.040
    ax.set_ylim(y_min, y_max)
    polish_axes(ax)
    save_figure(fig, out_path)


def plot_lisa_scatter(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[(df["variable"] == "ai_full_panel_index") & (df["year"] == 2023)].copy()
    if sub.empty:
        return
    color_map = {
        "High-High": DEEP_RED,
        "Low-Low": DEEP_BLUE,
        "High-Low": "#F59E0B",
        "Low-High": "#56A6D6",
        "Not significant": "#B8C0CC",
    }
    label_map = {
        "High-High": "高-高集聚",
        "Low-Low": "低-低集聚",
        "High-Low": "高-低离群",
        "Low-High": "低-高离群",
        "Not significant": "不显著",
    }
    order = ["High-High", "Low-Low", "High-Low", "Low-High", "Not significant"]

    fig, ax = plt.subplots(figsize=(6.9, 5.35))
    for cluster in order:
        group = sub[sub["lisa_cluster_p10"] == cluster]
        if group.empty:
            continue
        ax.scatter(
            group["z_value"],
            group["spatial_lag_z"],
            label=label_map[cluster],
            s=68,
            color=color_map[cluster],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            zorder=3,
        )
    highlight = sub[sub["lisa_cluster_p10"].isin(["High-High", "Low-Low", "Low-High"])]
    for _, row in highlight.iterrows():
        ax.annotate(
            row["city_name"].replace("市", ""),
            (row["z_value"], row["spatial_lag_z"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8.5,
            color="#102A43",
        )
    ax.axhline(0, color="#344054", linewidth=1.0, zorder=2)
    ax.axvline(0, color="#344054", linewidth=1.0, zorder=2)
    ax.text(0.98, 0.96, "高-高", transform=ax.transAxes, ha="right", va="top", color=DEEP_RED, fontweight="bold")
    ax.text(0.02, 0.04, "低-低", transform=ax.transAxes, ha="left", va="bottom", color=DEEP_BLUE, fontweight="bold")
    ax.text(0.98, 0.04, "高-低", transform=ax.transAxes, ha="right", va="bottom", color="#9A5B00", fontweight="bold")
    ax.text(0.02, 0.96, "低-高", transform=ax.transAxes, ha="left", va="top", color="#22749B", fontweight="bold")
    ax.set_xlabel("AI产业集聚标准化值")
    ax.set_ylabel("空间滞后项")
    ax.set_title("2023年AI产业集聚局部空间关联散点图", pad=13, color="#102A43", fontweight="bold")
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#D9E2EC",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        borderpad=0.7,
        labelspacing=0.45,
        columnspacing=1.3,
        handletextpad=0.5,
    )
    polish_axes(ax)
    save_figure(fig, out_path)


def plot_text_proxy_validation(panel: pd.DataFrame, out_path: Path) -> None:
    sub = panel.dropna(subset=["ai_text_index_original", "ai_full_panel_index"]).copy()
    if sub.empty:
        return
    x = sub["ai_text_index_original"].astype(float).to_numpy()
    y = sub["ai_full_panel_index"].astype(float).to_numpy()
    coef = np.polyfit(x, y, deg=1)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = coef[0] * xs + coef[1]
    r = np.corrcoef(x, y)[0, 1]
    n = len(sub)
    stats_path = ANALYSIS_DIR / "text_proxy_validation_stats.csv"
    if stats_path.exists():
        stats = pd.read_csv(stats_path, encoding="utf-8-sig").iloc[0]
        r = float(stats["pearson_r"])
        p = float(stats["p_value"])
        n = int(stats["n"])
    else:
        t = r * math.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else np.nan
        p = math.erfc(abs(t) / math.sqrt(2)) if np.isfinite(t) else np.nan

    fig, ax = plt.subplots(figsize=(7.0, 4.95))
    regions = sorted(sub["region_group"].dropna().unique())
    palette = ["#174A7C", "#D9822B", "#2F855A", "#7C3AED"]
    for region, color in zip(regions, palette):
        group = sub[sub["region_group"] == region]
        ax.scatter(
            group["ai_text_index_original"],
            group["ai_full_panel_index"],
            s=66,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.90,
            label=region,
            zorder=3,
        )
    ax.plot(xs, ys, color="#2D3748", linewidth=2.0, linestyle="--", label="线性拟合", zorder=2)
    ax.text(
        0.04,
        0.96,
        f"N={n}, Pearson r={r:.3f}\np={p:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "white", "edgecolor": "#D9E2EC", "alpha": 0.96},
    )
    ax.set_xlabel("年报文本AI指数")
    ax.set_ylabel("官方统计AI代理指标")
    ax.set_title("文本挖掘指标与宏观代理指标的匹配验证", pad=13, color="#102A43", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#D9E2EC", loc="best", borderpad=0.7)
    polish_axes(ax)
    save_figure(fig, out_path)


def plot_city_ai_support_bubble(panel: pd.DataFrame, out_path: Path) -> None:
    """Plot city-level AI agglomeration against innovation support."""
    city = (
        panel.groupby(["city_name", "region_group"], as_index=False)
        .agg(
            ai=("ai_full_panel_index", "mean"),
            innovation=("innovation_support_index", "mean"),
            coordination=("coordination_reference_index", "mean"),
            gdp=("gdp", "mean"),
        )
        .sort_values("ai", ascending=False)
    )
    region_colors = {
        "珠三角": PRIMARY,
        "粤东": ACCENT,
        "粤西": DEEP_RED,
        "粤北": DEEP_BLUE,
    }
    coord_min = city["coordination"].min()
    coord_span = city["coordination"].max() - coord_min
    city["bubble"] = 120 + 520 * (city["coordination"] - coord_min) / coord_span

    fig, ax = plt.subplots(figsize=(7.45, 5.35))
    for region, group in city.groupby("region_group"):
        ax.scatter(
            group["ai"],
            group["innovation"],
            s=group["bubble"],
            color=region_colors.get(region, MUTED),
            edgecolor="white",
            linewidth=0.9,
            alpha=0.86,
            label=region,
            zorder=3,
        )

    ai_median = city["ai"].median()
    innovation_median = city["innovation"].median()
    ax.axvline(ai_median, color="#64748B", linestyle="--", linewidth=1.0, zorder=2)
    ax.axhline(innovation_median, color="#64748B", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(ai_median, ax.get_ylim()[1], "AI中位数", ha="right", va="top", fontsize=8.8, color="#475569")
    ax.text(ax.get_xlim()[1], innovation_median, "创新支撑中位数", ha="right", va="bottom", fontsize=8.8, color="#475569")

    label_cities = {"深圳市", "广州市", "珠海市", "东莞市", "韶关市", "湛江市", "汕头市"}
    for _, row in city[city["city_name"].isin(label_cities)].iterrows():
        ax.annotate(
            row["city_name"].replace("市", ""),
            (row["ai"], row["innovation"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.8,
            color="#102A43",
        )

    ax.set_xlabel("AI产业集聚代理指标均值")
    ax.set_ylabel("创新支撑环境指数均值")
    ax.set_title("城市AI集聚与创新支撑二维分布", pad=13, color="#102A43", fontweight="bold")
    ax.legend(
        title="区域",
        frameon=True,
        facecolor="white",
        edgecolor="#D9E2EC",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
    )
    ax.text(
        0.02,
        0.03,
        "气泡大小表示协调发展参照指标均值",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "white", "edgecolor": "#E2E8F0", "alpha": 0.94},
    )
    polish_axes(ax)
    save_figure(fig, out_path)


def plot_region_multimetric_comparison(regional: pd.DataFrame, out_path: Path) -> None:
    """Draw grouped bars for regional heterogeneity."""
    order = ["珠三角", "粤东", "粤西", "粤北"]
    regional = regional.set_index("region_group").loc[order].reset_index()
    metrics = [
        ("ai_mean", "AI集聚"),
        ("neighboring_ai_exposure_mean", "邻近AI暴露"),
        ("innovation_support_mean", "创新支撑"),
        ("coordination_mean", "协调发展"),
    ]
    x = np.arange(len(order))
    width = 0.18
    colors = [PRIMARY, ACCENT, "#2F855A", DEEP_RED]

    fig, ax = plt.subplots(figsize=(8.45, 4.95))
    for idx, ((col, label), color) in enumerate(zip(metrics, colors)):
        offset = (idx - 1.5) * width
        bars = ax.bar(
            x + offset,
            regional[col],
            width=width,
            label=label,
            color=color,
            alpha=0.90,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        for bar in bars:
            value = bar.get_height()
            va = "bottom" if value >= 0 else "top"
            dy = 0.018 if value >= 0 else -0.018
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + dy,
                f"{value:.2f}",
                ha="center",
                va=va,
                fontsize=8.4,
                color="#263238",
            )

    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("指标均值")
    ax.set_title("广东四大区域多指标对比", pad=13, color="#102A43", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#D9E2EC", ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    polish_axes(ax)
    save_figure(fig, out_path)


def plot_innovation_weight_structure(report: pd.DataFrame, out_path: Path) -> None:
    """Visualize entropy weights of the innovation-support system."""
    name_map = {
        "entropy_weight_fiscal_intensity_ratio": "财政强度",
        "entropy_weight_financial_depth_ratio": "金融深度",
        "entropy_weight_fdi_gdp_ratio": "开放强度",
        "entropy_weight_retail_per_capita": "人均消费",
        "entropy_weight_service_openness_proxy": "服务开放",
    }
    sub = report[report["metric"].isin(name_map)].copy()
    sub["label"] = sub["metric"].map(name_map)
    sub["value"] = sub["value"].astype(float)
    sub = sub.sort_values("value", ascending=True)

    fig, ax = plt.subplots(figsize=(6.9, 4.6))
    bars = ax.barh(
        sub["label"],
        sub["value"],
        color=[PRIMARY, DEEP_BLUE, "#2F855A", ACCENT, DEEP_RED][: len(sub)],
        alpha=0.90,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for bar in bars:
        value = bar.get_width()
        ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, max(sub["value"]) + 0.08)
    ax.set_xlabel("熵值权重")
    ax.set_title("创新支撑环境指标权重结构", pad=13, color="#102A43", fontweight="bold")
    polish_axes(ax)
    save_figure(fig, out_path)


def plot_sdm_dynamic_comparison(
    sync_impacts: pd.DataFrame,
    lagged_impacts: pd.DataFrame,
    out_path: Path,
) -> None:
    """Compare synchronous and lagged SDM impact decomposition."""
    sync = sync_impacts[sync_impacts["matrix"] == "inverse_distance"].copy()
    sync_rows = pd.DataFrame(
        {
            "effect_type": ["direct", "indirect", "total"],
            "同期效应": [
                float(sync["python_estimate_direct"].iloc[0]),
                float(sync["python_estimate_indirect"].iloc[0]),
                float(sync["python_estimate_total"].iloc[0]),
            ],
        }
    )
    lag = lagged_impacts[lagged_impacts["matrix"] == "inverse_distance"][["effect_type", "estimate"]].copy()
    lag = lag.rename(columns={"estimate": "滞后一期效应"})
    plot_df = sync_rows.merge(lag, on="effect_type", how="left")
    label_map = {"direct": "直接效应", "indirect": "间接效应", "total": "总效应"}
    plot_df["label"] = plot_df["effect_type"].map(label_map)

    x = np.arange(len(plot_df))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7.5, 4.85))
    b1 = ax.bar(x - width / 2, plot_df["同期效应"], width, label="同期模型", color=DEEP_RED, alpha=0.88, edgecolor="white")
    b2 = ax.bar(x + width / 2, plot_df["滞后一期效应"], width, label="滞后一期模型", color=PRIMARY, alpha=0.88, edgecolor="white")
    for bars in (b1, b2):
        for bar in bars:
            value = bar.get_height()
            va = "bottom" if value >= 0 else "top"
            dy = 0.025 if value >= 0 else -0.025
            ax.text(bar.get_x() + bar.get_width() / 2, value + dy, f"{value:.2f}", ha="center", va=va, fontsize=9)
    ax.axhline(0, color="#334155", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"])
    ax.set_ylabel("效应估计值")
    ax.set_title("AI集聚空间效应的同期虹吸与滞后扩散", pad=13, color="#102A43", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#D9E2EC", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    polish_axes(ax)
    save_figure(fig, out_path)


def main() -> None:
    configure_plot_style()
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    moran = pd.read_csv(ANALYSIS_DIR / "moran_global_results.csv", encoding="utf-8-sig")
    lisa = pd.read_csv(ANALYSIS_DIR / "lisa_local_results.csv", encoding="utf-8-sig")
    panel = pd.read_csv(ANALYSIS_DIR / "panel_21city_2018_2023_completed.csv", encoding="utf-8-sig")
    regional = pd.read_csv(ANALYSIS_DIR / "regional_heterogeneity_summary.csv", encoding="utf-8-sig")
    report = pd.read_csv(ANALYSIS_DIR / "panel_21city_2018_2023_completion_report.csv", encoding="utf-8-sig")
    sdm = pd.read_csv(ANALYSIS_DIR / "sdm_ai_effects_summary.csv", encoding="utf-8-sig")
    lagged = pd.read_csv(ANALYSIS_DIR / "lagged_ai_sdm_impacts.csv", encoding="utf-8-sig")

    plot_moran_trend(moran, "ai_full_panel_index", "AI产业集聚全局Moran's I", PICTURE_DIR / "fig_moran_ai_trend.png")
    plot_moran_trend(
        moran,
        "coordination_reference_index",
        "协调发展参照指标全局Moran's I",
        PICTURE_DIR / "fig_moran_coordination_trend.png",
    )
    plot_lisa_scatter(lisa, PICTURE_DIR / "fig_lisa_ai_2023_scatter.png")
    plot_text_proxy_validation(panel, PICTURE_DIR / "fig_text_proxy_validation.png")
    plot_city_ai_support_bubble(panel, PICTURE_DIR / "fig_city_ai_support_bubble.png")
    plot_region_multimetric_comparison(regional, PICTURE_DIR / "fig_region_multimetric_comparison.png")
    plot_innovation_weight_structure(report, PICTURE_DIR / "fig_innovation_weight_structure.png")
    plot_sdm_dynamic_comparison(sdm, lagged, PICTURE_DIR / "fig_sdm_dynamic_comparison.png")


if __name__ == "__main__":
    main()
