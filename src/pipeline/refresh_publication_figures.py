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


def main() -> None:
    configure_plot_style()
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    moran = pd.read_csv(ANALYSIS_DIR / "moran_global_results.csv", encoding="utf-8-sig")
    lisa = pd.read_csv(ANALYSIS_DIR / "lisa_local_results.csv", encoding="utf-8-sig")
    panel = pd.read_csv(ANALYSIS_DIR / "panel_21city_2018_2023_completed.csv", encoding="utf-8-sig")

    plot_moran_trend(moran, "ai_full_panel_index", "AI产业集聚全局Moran's I", PICTURE_DIR / "fig_moran_ai_trend.png")
    plot_moran_trend(
        moran,
        "coordination_reference_index",
        "协调发展参照指标全局Moran's I",
        PICTURE_DIR / "fig_moran_coordination_trend.png",
    )
    plot_lisa_scatter(lisa, PICTURE_DIR / "fig_lisa_ai_2023_scatter.png")
    plot_text_proxy_validation(panel, PICTURE_DIR / "fig_text_proxy_validation.png")


if __name__ == "__main__":
    main()
