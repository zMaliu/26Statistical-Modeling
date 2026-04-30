"""Draw the 2023 Guangdong LISA cluster map used in the paper.

The map is kept as a standalone script so the final figure can be rebuilt
without losing Chinese labels or the grayscale-friendly marker overlay.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "analysis_ready"
SPATIAL_DIR = PROJECT_ROOT / "data" / "processed" / "spatial"
PICTURE_DIR = PROJECT_ROOT / "picture"

GEOJSON_PATH = SPATIAL_DIR / "guangdong_city_boundaries_datav_440000_full.json"
LISA_PATH = ANALYSIS_DIR / "lisa_local_results.csv"


def select_cjk_font() -> str:
    """Register and select a Chinese-capable font on Windows or Linux."""
    font_files = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_file in font_files:
        if font_file.exists():
            fm.fontManager.addfont(str(font_file))

    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    return next((font for font in preferred if font in available), "DejaVu Sans")


def lisa_display_type(row: pd.Series) -> str:
    cluster = row["lisa_cluster_p10"]
    if cluster == "High-High":
        return "高-高集聚"
    if cluster == "Low-High":
        return "低-高邻近"
    if cluster == "Low-Low":
        return "低-低集聚"
    if cluster == "High-Low":
        return "高-低离群"
    if float(row["z_value"]) > 0:
        return "高值不显著"
    return "不显著"


def draw_lisa_map() -> None:
    font = select_cjk_font()
    plt.rcParams.update(
        {
            "font.family": font,
            "font.sans-serif": [font],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
            "savefig.dpi": 360,
        }
    )

    boundary = gpd.read_file(GEOJSON_PATH)
    lisa = pd.read_csv(LISA_PATH, encoding="utf-8-sig")
    lisa_2023 = lisa[(lisa["variable"] == "ai_full_panel_index") & (lisa["year"] == 2023)].copy()
    lisa_2023["name"] = lisa_2023["city_name"]
    lisa_2023["display_type"] = lisa_2023.apply(lisa_display_type, axis=1)

    gdf = boundary.merge(lisa_2023, on="name", how="left")
    gdf["display_type"] = gdf["display_type"].fillna("不显著")

    fill_colors = {
        "高-高集聚": "#C8372D",
        "低-高邻近": "#F2A019",
        "低-低集聚": "#1F6F9F",
        "高-低离群": "#8B5CF6",
        "高值不显著": "#F3B2AE",
        "不显著": "#E8EDF3",
    }
    edge_color = "#93A4B8"

    fig, ax = plt.subplots(figsize=(9.0, 6.25))
    ax.set_facecolor("white")

    for category, color in fill_colors.items():
        sub = gdf[gdf["display_type"] == category]
        if sub.empty:
            continue
        sub.plot(
            ax=ax,
            color=color,
            edgecolor=edge_color,
            linewidth=0.75,
            alpha=0.98,
            zorder=2,
        )

    # Add a thin outer boundary to make the map legible in grayscale printing.
    gdf.boundary.plot(ax=ax, color="#7B8FA6", linewidth=0.55, zorder=3)

    label_offsets = {
        "广州市": (-0.10, 0.11),
        "佛山市": (-0.12, -0.08),
        "东莞市": (0.08, -0.09),
        "深圳市": (0.10, -0.08),
        "中山市": (-0.06, -0.09),
        "珠海市": (0.06, -0.10),
        "惠州市": (0.10, 0.08),
        "清远市": (0.00, 0.08),
        "河源市": (0.10, 0.06),
        "江门市": (-0.08, -0.09),
    }
    marker_styles = {
        "高-高集聚": ("^", "#0F172A", 58),
        "低-高邻近": ("o", "#0F172A", 46),
        "高值不显著": ("P", "#0F172A", 46),
    }

    for _, row in gdf.iterrows():
        centroid = row.geometry.representative_point()
        city = row["name"]
        category = row["display_type"]
        dx, dy = label_offsets.get(city, (0.0, 0.0))
        is_focus = category in marker_styles
        ax.text(
            centroid.x + dx,
            centroid.y + dy,
            city.replace("市", ""),
            ha="center",
            va="center",
            fontsize=8.8 if is_focus else 7.8,
            color="#111827" if is_focus else "#64748B",
            fontweight="bold" if is_focus else "normal",
            bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.76} if is_focus else None,
            zorder=5,
        )
        if is_focus:
            marker, marker_color, size = marker_styles[category]
            ax.scatter(
                centroid.x,
                centroid.y,
                marker=marker,
                s=size,
                c=marker_color,
                edgecolors="white",
                linewidths=1.35,
                zorder=6,
            )

    legend_handles = [
        Patch(facecolor=fill_colors["高-高集聚"], edgecolor=edge_color, label="高-高集聚"),
        Patch(facecolor=fill_colors["低-高邻近"], edgecolor=edge_color, label="低-高邻近"),
        Patch(facecolor=fill_colors["高值不显著"], edgecolor=edge_color, label="高值不显著"),
        Patch(facecolor=fill_colors["不显著"], edgecolor=edge_color, label="不显著"),
        Line2D([0], [0], marker="^", color="w", label="高-高符号", markerfacecolor="#0F172A", markeredgecolor="white", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="低-高符号", markerfacecolor="#0F172A", markeredgecolor="white", markersize=6),
        Line2D([0], [0], marker="P", color="w", label="高值不显著符号", markerfacecolor="#0F172A", markeredgecolor="white", markersize=6),
    ]
    legend = ax.legend(
        handles=legend_handles,
        title="LISA类型（10%）",
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=9.0,
        title_fontsize=9.5,
        borderpad=0.75,
        labelspacing=0.55,
        handlelength=1.55,
    )
    legend.get_frame().set_linewidth(0.9)

    ax.set_title("2023年AI产业集聚LISA空间分布", fontsize=15, fontweight="bold", pad=12, color="#0F172A")
    ax.text(
        0.01,
        0.025,
        "注：底色表示局部空间关联类型，深色符号用于黑白打印识别；显著性口径为10%。",
        transform=ax.transAxes,
        fontsize=8.3,
        color="#475569",
        ha="left",
        va="bottom",
    )
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.35)

    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(PICTURE_DIR / f"fig_lisa_ai_2023_map.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw_lisa_map()
