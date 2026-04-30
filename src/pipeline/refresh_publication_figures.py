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
SPATIAL_DIR = PROJECT_ROOT / "data" / "processed" / "spatial"
PICTURE_DIR = PROJECT_ROOT / "picture"

PRIMARY = "#174A7C"
ACCENT = "#D9822B"
DEEP_RED = "#B83227"
DEEP_BLUE = "#1F6F9F"
MUTED = "#6B7280"
GRID = "#D8DEE9"
NATURE_RED = "#E64B35"
NATURE_GREEN = "#00A087"
NATURE_BLUE = "#3C5488"


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def polish_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="both", alpha=0.42, linestyle="--", linewidth=0.65, color=GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#263238", length=3.5, width=0.8)


def save_figure(fig: plt.Figure, out_path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=360)
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
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
    save_figure(fig, out_path, tight=False)


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
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    xs = np.linspace(x.min(), x.max(), 100)
    Xs = np.column_stack([np.ones_like(xs), xs])
    ys = Xs @ beta
    resid = y - X @ beta
    dof = max(len(x) - 2, 1)
    mse = float((resid @ resid) / dof)
    xtx_inv = np.linalg.inv(X.T @ X)
    se_fit = np.sqrt(np.sum((Xs @ xtx_inv) * Xs, axis=1) * mse)
    ci_low = ys - 1.96 * se_fit
    ci_high = ys + 1.96 * se_fit
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

    fig, ax = plt.subplots(figsize=(7.1, 5.0))

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
    ax.fill_between(xs, ci_low, ci_high, color="#475569", alpha=0.13, linewidth=0, label="95%置信带", zorder=1)
    ax.plot(xs, ys, color="#2D3748", linewidth=2.1, linestyle="--", label="线性拟合", zorder=2)
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
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#D9E2EC",
        loc="lower right",
        ncol=1,
        borderpad=0.55,
        labelspacing=0.35,
        handlelength=1.4,
    )
    polish_axes(ax)
    save_figure(fig, out_path)


def plot_spatial_network_topology(
    panel: pd.DataFrame,
    coords: pd.DataFrame,
    weights_long: pd.DataFrame,
    out_path: Path,
) -> None:
    """Draw a compact geography-economy spatial weight network."""
    city_stats = (
        panel.groupby(["city_name", "region_group"], as_index=False)
        .agg(
            ai_mean=("ai_full_panel_index", "mean"),
            gdp_pc=("gdp_per_capita", "mean"),
        )
        .merge(coords, on=["city_name", "region_group"], how="left")
    )
    city_stats = city_stats.dropna(subset=["longitude", "latitude", "gdp_pc"]).copy()
    city_lookup = city_stats.set_index("city_name")

    edges = []
    eta = 1000.0
    for origin, group in weights_long[weights_long["origin_city"] != weights_long["dest_city"]].groupby("origin_city"):
        if origin not in city_lookup.index:
            continue
        origin_gdp = float(city_lookup.loc[origin, "gdp_pc"])
        local_edges = []
        for _, row in group.iterrows():
            dest = row["dest_city"]
            if dest not in city_lookup.index:
                continue
            dest_gdp = float(city_lookup.loc[dest, "gdp_pc"])
            distance = float(row["distance_km"])
            if distance <= 0:
                continue
            geo_econ_weight = (1.0 / distance) * (1.0 / (abs(origin_gdp - dest_gdp) + eta))
            local_edges.append((origin, dest, geo_econ_weight))
        local_edges.sort(key=lambda item: item[2], reverse=True)
        edges.extend(local_edges[:3])

    undirected = {}
    for origin, dest, weight in edges:
        key = tuple(sorted([origin, dest]))
        undirected[key] = max(undirected.get(key, 0.0), weight)
    edge_df = pd.DataFrame(
        [{"origin": key[0], "dest": key[1], "weight": weight} for key, weight in undirected.items()]
    ).sort_values("weight", ascending=False)
    edge_df = edge_df.head(34)

    region_colors = {
        "珠三角": NATURE_RED,
        "粤东": "#F39B7F",
        "粤西": "#4DBBD5",
        "粤北": NATURE_BLUE,
    }
    ai_min = city_stats["ai_mean"].min()
    ai_span = city_stats["ai_mean"].max() - ai_min
    city_stats["node_size"] = 85 + 360 * (city_stats["ai_mean"] - ai_min) / ai_span
    focus = {"广州市", "深圳市"}

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.set_facecolor("white")
    if not edge_df.empty:
        weights = edge_df["weight"].to_numpy()
        w_min = weights.min()
        w_span = weights.max() - w_min if weights.max() > w_min else 1.0
        for _, row in edge_df.iterrows():
            o = city_lookup.loc[row["origin"]]
            d = city_lookup.loc[row["dest"]]
            strength = (float(row["weight"]) - w_min) / w_span
            ax.plot(
                [o["longitude"], d["longitude"]],
                [o["latitude"], d["latitude"]],
                color="#64748B",
                linewidth=0.45 + 2.2 * strength,
                alpha=0.25 + 0.42 * strength,
                zorder=1,
            )

    for region, group in city_stats.groupby("region_group"):
        ax.scatter(
            group["longitude"],
            group["latitude"],
            s=group["node_size"],
            color=region_colors.get(region, MUTED),
            edgecolor="white",
            linewidth=1.1,
            alpha=0.92,
            label=region,
            zorder=3,
        )

    for city in focus:
        if city not in city_lookup.index:
            continue
        row = city_lookup.loc[city]
        ax.scatter(
            row["longitude"],
            row["latitude"],
            s=560,
            facecolor="none",
            edgecolor="#111827",
            linewidth=1.8,
            zorder=4,
        )

    label_cities = {"广州市", "深圳市", "珠海市", "东莞市", "佛山市", "惠州市", "韶关市", "汕头市", "湛江市", "梅州市"}
    offsets = {
        "广州市": (-0.24, 0.18),
        "深圳市": (0.18, -0.10),
        "东莞市": (0.18, 0.08),
        "佛山市": (-0.26, -0.10),
        "珠海市": (-0.18, -0.16),
        "惠州市": (0.18, 0.08),
        "韶关市": (0.10, 0.12),
        "汕头市": (0.16, -0.12),
        "湛江市": (-0.08, 0.12),
        "梅州市": (0.12, 0.10),
    }
    for _, row in city_stats.iterrows():
        city = row["city_name"]
        if city not in label_cities:
            continue
        dx, dy = offsets.get(city, (0.06, 0.06))
        ax.text(
            row["longitude"] + dx,
            row["latitude"] + dy,
            city.replace("市", ""),
            ha="center",
            va="center",
            fontsize=9.2 if city in focus else 8.2,
            color="#0F172A",
            fontweight="bold" if city in focus else "normal",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            zorder=5,
        )

    ax.set_title("广东AI产业地理--经济空间关联网络", pad=14, color="#102A43", fontweight="bold")
    ax.text(
        0.015,
        0.03,
        "注：节点大小表示AI集聚均值，连线粗细表示地理距离与经济相似性嵌套后的相对联系强度。",
        transform=ax.transAxes,
        fontsize=8.3,
        color="#475569",
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "white", "edgecolor": "#E2E8F0", "alpha": 0.90},
    )
    ax.legend(
        title="区域",
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#D9E2EC",
        borderpad=0.75,
        fontsize=8.7,
        title_fontsize=9.0,
    )
    ax.set_xlim(city_stats["longitude"].min() - 0.40, city_stats["longitude"].max() + 0.55)
    ax.set_ylim(city_stats["latitude"].min() - 0.28, city_stats["latitude"].max() + 0.28)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_aspect("equal", adjustable="box")
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
    label_offsets = {
        "深圳市": (7, -2),
        "广州市": (7, 8),
        "珠海市": (7, 8),
        "东莞市": (7, -10),
        "韶关市": (7, 8),
        "湛江市": (-20, -10),
        "汕头市": (8, -12),
    }
    for _, row in city[city["city_name"].isin(label_cities)].iterrows():
        dx, dy = label_offsets.get(row["city_name"], (5, 5))
        ax.annotate(
            row["city_name"].replace("市", ""),
            (row["ai"], row["innovation"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.8,
            color="#102A43",
            ha="right" if dx < 0 else "left",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
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


def plot_core_correlation_heatmap(panel: pd.DataFrame, out_path: Path) -> None:
    """Plot a compact Pearson-correlation heatmap for core variables."""
    cols = [
        ("ai_full_panel_index", "AI集聚"),
        ("innovation_support_index", "创新支撑"),
        ("coordination_reference_index", "协调发展"),
        ("gdp_per_capita", "人均GDP"),
        ("financial_depth_ratio", "金融深度"),
        ("tertiary_industry_share", "三产占比"),
    ]
    use_cols = [col for col, _ in cols]
    labels = [label for _, label in cols]
    corr = panel[use_cols].astype(float).corr().to_numpy()

    fig, ax = plt.subplots(figsize=(6.8, 5.55))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            color = "white" if abs(corr[i, j]) > 0.55 else "#0F172A"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8.8, color=color)
    ax.set_title("核心变量Pearson相关性热力图", pad=12, color="#102A43", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("相关系数")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
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


def _impact_row(frame: pd.DataFrame, effect_type: str) -> pd.Series:
    row = frame[frame["effect_type"] == effect_type]
    if row.empty:
        raise ValueError(f"Missing SDM impact row: {effect_type}")
    return row.iloc[0]


def plot_sdm_effect_forest(
    sync_impacts: pd.DataFrame,
    lagged_impacts: pd.DataFrame,
    out_path: Path,
) -> None:
    """Draw a forest plot for synchronous siphon and lagged empowerment effects."""
    sync = sync_impacts[
        (sync_impacts["matrix"] == "inverse_distance")
        & (sync_impacts.get("variable", "ai") == "ai")
    ].copy()
    lagged = lagged_impacts[lagged_impacts["matrix"] == "inverse_distance"].copy()

    rows = []
    specs = [
        ("同期直接效应", sync, "direct", NATURE_BLUE, "s"),
        ("同期间接效应（虹吸）", sync, "indirect", NATURE_RED, "o"),
        ("同期总效应", sync, "total", NATURE_RED, "D"),
        ("滞后一期直接效应", lagged, "direct", NATURE_GREEN, "s"),
        ("滞后一期间接效应（赋能）", lagged, "indirect", NATURE_GREEN, "o"),
        ("滞后一期总效应", lagged, "total", NATURE_GREEN, "D"),
    ]
    for label, frame, effect_type, color, marker in specs:
        row = _impact_row(frame, effect_type)
        estimate = float(row["estimate"])
        low = float(row["ci95_low"]) if "ci95_low" in row.index else estimate - 1.96 * float(row["std_err_sim"])
        high = float(row["ci95_high"]) if "ci95_high" in row.index else estimate + 1.96 * float(row["std_err_sim"])
        p_value = float(row["p_value"]) if "p_value" in row.index else np.nan
        rows.append(
            {
                "label": label,
                "estimate": estimate,
                "low": low,
                "high": high,
                "color": color,
                "marker": marker,
                "stars": significance_marker(p_value),
            }
        )

    plot_df = pd.DataFrame(rows)
    y = np.arange(len(plot_df))[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 5.25))
    ax.axvline(0, color="#111827", linestyle=(0, (4, 4)), linewidth=1.1, alpha=0.75, zorder=1)
    ax.axhspan(2.5, 5.5, color=NATURE_RED, alpha=0.055, zorder=0)
    ax.axhspan(-0.5, 2.5, color=NATURE_GREEN, alpha=0.055, zorder=0)

    for idx, row in plot_df.iterrows():
        yy = y[idx]
        ax.errorbar(
            row["estimate"],
            yy,
            xerr=[[row["estimate"] - row["low"]], [row["high"] - row["estimate"]]],
            fmt=row["marker"],
            color=row["color"],
            ecolor=row["color"],
            markersize=8.2,
            markeredgecolor="white",
            markeredgewidth=1.2,
            elinewidth=2.2,
            capsize=4.2,
            zorder=4,
        )
        text_x = row["high"] + 0.055 if row["estimate"] >= 0 else row["low"] - 0.055
        ha = "left" if row["estimate"] >= 0 else "right"
        ax.text(text_x, yy, f"{row['estimate']:.2f}{row['stars']}", va="center", ha=ha, fontsize=9.4, color="#1F2937")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    xmin = float(plot_df["low"].min())
    xmax = float(plot_df["high"].max())
    pad = max((xmax - xmin) * 0.16, 0.18)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_xlabel("偏微分效应估计值（点）及95%置信区间（线）")
    ax.set_title("AI集聚空间效应森林图：同期虹吸与滞后赋能", pad=14, color="#102A43", fontweight="bold")
    ax.text(
        0.02,
        0.96,
        "同期",
        transform=ax.transAxes,
        color=NATURE_RED,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.02,
        0.42,
        "滞后一期",
        transform=ax.transAxes,
        color=NATURE_GREEN,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    polish_axes(ax)
    ax.grid(axis="y", visible=False)
    save_figure(fig, out_path)


def main() -> None:
    configure_plot_style()
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    moran = pd.read_csv(ANALYSIS_DIR / "moran_global_results.csv", encoding="utf-8-sig")
    lisa = pd.read_csv(ANALYSIS_DIR / "lisa_local_results.csv", encoding="utf-8-sig")
    panel = pd.read_csv(ANALYSIS_DIR / "panel_21city_2018_2023_completed.csv", encoding="utf-8-sig")
    coords = pd.read_csv(SPATIAL_DIR / "city_coordinates.csv", encoding="utf-8-sig")
    weights_long = pd.read_csv(SPATIAL_DIR / "spatial_weights_inverse_distance_long.csv", encoding="utf-8-sig")
    regional = pd.read_csv(ANALYSIS_DIR / "regional_heterogeneity_summary.csv", encoding="utf-8-sig")
    report = pd.read_csv(ANALYSIS_DIR / "panel_21city_2018_2023_completion_report.csv", encoding="utf-8-sig")
    sdm = pd.read_csv(ANALYSIS_DIR / "sdm_ai_effects_summary.csv", encoding="utf-8-sig")
    sdm_impacts = pd.read_csv(ANALYSIS_DIR / "python_panel_sdm_impacts.csv", encoding="utf-8-sig")
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
    plot_spatial_network_topology(panel, coords, weights_long, PICTURE_DIR / "fig_spatial_network_topology.png")
    plot_city_ai_support_bubble(panel, PICTURE_DIR / "fig_city_ai_support_bubble.png")
    plot_region_multimetric_comparison(regional, PICTURE_DIR / "fig_region_multimetric_comparison.png")
    plot_innovation_weight_structure(report, PICTURE_DIR / "fig_innovation_weight_structure.png")
    plot_core_correlation_heatmap(panel, PICTURE_DIR / "fig_core_variable_correlation_heatmap.png")
    plot_sdm_dynamic_comparison(sdm, lagged, PICTURE_DIR / "fig_sdm_dynamic_comparison.png")
    plot_sdm_effect_forest(sdm_impacts, lagged, PICTURE_DIR / "fig_sdm_effect_forest.png")


if __name__ == "__main__":
    main()
