from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _pick_cjk_font() -> str:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\Deng.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                fm.fontManager.addfont(str(path))
                return fm.FontProperties(fname=str(path)).get_name()
            except Exception:
                continue
    return "DejaVu Sans"


CHINESE_FONT = _pick_cjk_font()

sns.set_theme(style="whitegrid")

# seaborn resets font-related rcParams, so apply the CJK font after theme setup.
plt.rcParams["font.family"] = CHINESE_FONT
plt.rcParams["font.sans-serif"] = [CHINESE_FONT]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 170
plt.rcParams["savefig.dpi"] = 340
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10


MAIN_BLUE = "#2F5C85"
ACCENT_RED = "#C65D57"
ACCENT_GOLD = "#D9A441"
ACCENT_GREEN = "#5B8E55"
ACCENT_PURPLE = "#7468A6"
DARK = "#2B2B2B"
GRID = "#E7EBF0"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _base_ax(ax, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(colors="#444444")
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-", alpha=0.9)
    ax.set_axisbelow(True)


def _clean_label(text: str) -> str:
    mapping = {
        "ai_agglomeration_composite": "AI集聚",
        "innovation_support_substitute_index": "创新支撑替代",
        "innovation_support_entropy_topsis_score": "创新支撑综合",
        "coordination_capacity_composite": "协调发展",
        "service_openness_proxy": "服务开放度",
        "fiscal_intensity_ratio": "财政强度",
        "financial_depth_ratio": "金融深度",
        "fdi_gdp_ratio": "开放强度",
        "retail_per_capita": "人均消费",
        "gdp_per_capita": "人均GDP",
        "population": "人口规模",
        "retail_sales": "社零总额",
        "fdi_actual_used": "实际外资",
        "financial_deposit_loan": "金融存贷款",
        "fiscal_expenditure": "财政支出",
        "secondary_industry_share": "第二产业占比",
        "tertiary_industry_share": "第三产业占比",
    }
    return mapping.get(text, text)


def plot_ai_city_ranking(df: pd.DataFrame, output_path: Path) -> None:
    d = df.sort_values("ai_agglomeration_mean", ascending=True).copy()
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    colors = [ACCENT_RED if x <= 1 else MAIN_BLUE for x in d["ai_small_sample_observation_count"]]
    ax.barh(d["city_name"], d["ai_agglomeration_mean"], color=colors, edgecolor="white", linewidth=1.0)
    ax.axvline(0, color="#666666", linewidth=1)
    _base_ax(ax)
    ax.set_title("样本城市人工智能产业集聚均值排序", color=DARK, pad=12)
    ax.set_xlabel("AI产业集聚均值")
    ax.set_ylabel("")
    for i, v in enumerate(d["ai_agglomeration_mean"]):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=10, color=DARK)
    ax.text(
        0.98,
        0.03,
        "红色柱表示存在小样本观测",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ai_year_trend(df: pd.DataFrame, output_path: Path) -> None:
    _ensure_parent(output_path)
    fig, ax1 = plt.subplots(figsize=(8.6, 4.9))
    ax1.plot(df["year"], df["ai_agglomeration_mean"], marker="o", color=MAIN_BLUE, linewidth=2.3)
    ax1.fill_between(df["year"], df["ai_agglomeration_mean"], color=MAIN_BLUE, alpha=0.10)
    ax1.set_xlabel("年份")
    ax1.set_ylabel("AI产业集聚均值", color=MAIN_BLUE)
    ax1.tick_params(axis="y", labelcolor=MAIN_BLUE)
    ax1.spines["top"].set_visible(False)
    ax1.grid(axis="y", color=GRID, linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.bar(df["year"], df["ai_small_sample_observation_count"], color=ACCENT_GOLD, alpha=0.35, width=0.45)
    ax2.set_ylabel("小样本观测数", color=ACCENT_GOLD)
    ax2.tick_params(axis="y", labelcolor=ACCENT_GOLD)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color("#777777")

    ax1.set_title("人工智能产业集聚年度变化", color=DARK, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_innovation_city_ranking(df: pd.DataFrame, output_path: Path, top_n: int = 12) -> None:
    d = df.sort_values("innovation_support_entropy_mean", ascending=False).head(top_n).iloc[::-1]
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(9.4, 6.3))
    colors = sns.color_palette("Blues_r", len(d))
    ax.barh(d["city_name"], d["innovation_support_entropy_mean"], color=colors, edgecolor="white")
    _base_ax(ax)
    ax.set_title("创新支撑环境综合得分排名（前12位）", color=DARK, pad=12)
    ax.set_xlabel("熵值-TOPSIS综合得分")
    ax.set_ylabel("")
    for i, v in enumerate(d["innovation_support_entropy_mean"]):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", ha="left", fontsize=10, color=DARK)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_innovation_year_change(df: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    d = df.sort_values("innovation_support_change", ascending=False).head(top_n).copy()
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    colors = [ACCENT_GREEN if v >= 0 else ACCENT_RED for v in d["innovation_support_change"]]
    ax.barh(d["city_name"], d["innovation_support_change"], color=colors, edgecolor="white")
    ax.axvline(0, color="#666666", linewidth=1)
    _base_ax(ax)
    ax.set_title("创新支撑环境年度变化（增幅前10位）", color=DARK, pad=12)
    ax.set_xlabel("2023年相对2022年的得分变化")
    ax.set_ylabel("")
    for i, v in enumerate(d["innovation_support_change"]):
        ax.text(v + (0.004 if v >= 0 else -0.004), i, f"{v:.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=10, color=DARK)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_weights(df: pd.DataFrame, output_path: Path) -> None:
    d = df.sort_values("entropy_weight", ascending=True).copy()
    name_map = {
        "fiscal_intensity_ratio": "财政强度",
        "financial_depth_ratio": "金融深度",
        "fdi_gdp_ratio": "开放强度",
        "retail_per_capita": "人均消费",
        "service_openness_proxy": "服务开放度",
    }
    labels = [name_map.get(x, x) for x in d["indicator"]]
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    colors = sns.color_palette("crest", len(d))
    ax.barh(labels, d["entropy_weight"], color=colors, edgecolor="white")
    _base_ax(ax)
    ax.set_title("创新支撑环境评价指标权重", color=DARK, pad=12)
    ax.set_xlabel("熵值权重")
    ax.set_ylabel("")
    for i, v in enumerate(d["entropy_weight"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", ha="left", fontsize=10, color=DARK)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_city_quadrant(df: pd.DataFrame, output_path: Path) -> None:
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(8.3, 6.3))
    x_med = df["innovation_support_mean"].median()
    y_med = df["ai_agglomeration_mean"].median()
    color_map = {
        "高集聚-高支撑": MAIN_BLUE,
        "高集聚-低支撑": ACCENT_RED,
        "低集聚-高支撑": ACCENT_GREEN,
        "低集聚-低支撑": ACCENT_GOLD,
    }
    for _, row in df.iterrows():
        ax.scatter(
            row["innovation_support_mean"],
            row["ai_agglomeration_mean"],
            s=150,
            color=color_map.get(row["city_quadrant"], MAIN_BLUE),
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            row["innovation_support_mean"] + 0.03,
            row["ai_agglomeration_mean"] + 0.03,
            row["city_name"],
            fontsize=10,
            color=DARK,
        )
    ax.axvline(x_med, color="#777777", linestyle="--", linewidth=1)
    ax.axhline(y_med, color="#777777", linestyle="--", linewidth=1)
    ax.grid(color="#ECECEC", linewidth=0.8)
    ax.set_title("AI集聚—创新支撑四象限分层", color=DARK, pad=12)
    ax.set_xlabel("创新支撑环境均值")
    ax.set_ylabel("AI产业集聚均值")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.text(x_med + 0.03, y1 - 0.08 * (y1 - y0), "高支撑", fontsize=10, color="#666666")
    ax.text(x0 + 0.03, y1 - 0.08 * (y1 - y0), "低支撑", fontsize=10, color="#666666")
    ax.text(x1 - 0.18 * (x1 - x0), y_med + 0.05 * (y1 - y0), "高集聚", fontsize=10, color="#666666")
    ax.text(x1 - 0.18 * (x1 - x0), y0 + 0.05 * (y1 - y0), "低集聚", fontsize=10, color="#666666")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_group_comparison(df: pd.DataFrame, output_path: Path) -> None:
    d = df.copy()
    _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), sharey=True)
    metrics = [
        ("ai_agglomeration_mean", "AI集聚均值", MAIN_BLUE),
        ("innovation_support_entropy_mean", "创新支撑均值", ACCENT_GREEN),
        ("coordination_capacity_mean", "协调发展均值", ACCENT_RED),
    ]
    for ax, (col, title, color) in zip(axes, metrics):
        ax.barh(d["quadrant_label_cn"], d[col], color=color, edgecolor="white")
        ax.axvline(0, color="#666666", linewidth=0.8)
        _base_ax(ax)
        ax.set_title(title, fontsize=12, color=DARK, pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for i, v in enumerate(d[col]):
            ax.text(v + (0.015 if v >= 0 else -0.015), i, f"{v:.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9, color=DARK)
    fig.suptitle("不同分层城市组的关键均值比较", fontsize=14, color=DARK, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    _ensure_parent(output_path)
    corr = df.set_index("variable").copy()
    corr.index = [_clean_label(x) for x in corr.index]
    corr.columns = [_clean_label(x) for x in corr.columns]
    fig, ax = plt.subplots(figsize=(9.6, 7.6))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.85, "label": "相关系数"},
    )
    ax.set_title("核心变量相关性热力图", fontsize=14, color=DARK, pad=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_regression_coefficients(df: pd.DataFrame, output_path: Path) -> None:
    keep = df[df["variable"] != "gdp_per_capita"].copy()
    keep["coef_label"] = keep["variable"].map(_clean_label)
    _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.9), sharex=False)
    model_order = [
        ("scheme2_baseline_model", "基准模型"),
        ("scheme2_support_components_model", "支撑组成项模型"),
    ]
    for ax, (model_name, title) in zip(axes, model_order):
        sub = keep[keep["model_name"] == model_name].copy()
        sub = sub.sort_values("coefficient", ascending=True)
        ci = 1.96 * sub["std_error"]
        color = MAIN_BLUE if model_name == "scheme2_baseline_model" else ACCENT_GREEN
        ax.barh(sub["coef_label"], sub["coefficient"], color=color, alpha=0.88)
        ax.errorbar(sub["coefficient"], np.arange(len(sub)), xerr=ci, fmt="none", ecolor=DARK, elinewidth=1, capsize=3)
        ax.axvline(0, color="#666666", linewidth=1)
        _base_ax(ax)
        ax.set_title(title, fontsize=12, color=DARK, pad=8)
        ax.set_xlabel("系数估计值")
        ax.set_ylabel("")
    fig.suptitle("辅助回归重点系数比较", fontsize=14, color=DARK, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_city_profile_dual(df: pd.DataFrame, output_path: Path) -> None:
    d = df.sort_values(["innovation_support_entropy_mean", "ai_agglomeration_mean"], ascending=[False, False]).copy()
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    y = np.arange(len(d))
    h = 0.35
    ax.barh(y + h / 2, d["innovation_support_entropy_mean"], height=h, color=ACCENT_GREEN, label="创新支撑环境")
    ax.barh(y - h / 2, d["ai_agglomeration_mean"], height=h, color=MAIN_BLUE, label="AI产业集聚")
    ax.axvline(0, color="#666666", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(d["city_name"])
    _base_ax(ax)
    ax.set_title("样本城市画像：AI集聚与创新支撑双维比较", color=DARK, pad=12)
    ax.set_xlabel("标准化均值")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_city_profile_radar(df: pd.DataFrame, output_path: Path) -> None:
    # 选择代表性城市做画像展示，避免图面过载
    candidates = ["广州市", "深圳市", "佛山市", "中山市", "珠海市", "东莞市"]
    d = df[df["city_name"].isin(candidates)].copy()
    if d.empty:
        return

    metrics = [
        ("ai_agglomeration_mean", "AI集聚"),
        ("innovation_support_entropy_mean", "创新支撑"),
        ("coordination_capacity_mean", "协调发展"),
    ]
    plot_df = d[["city_name"] + [m[0] for m in metrics]].copy()
    for col, _ in metrics:
        col_min = plot_df[col].min()
        col_max = plot_df[col].max()
        if col_max == col_min:
            plot_df[col] = 0.5
        else:
            plot_df[col] = (plot_df[col] - col_min) / (col_max - col_min)

    labels = [m[1] for m in metrics]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(8.0, 6.8), subplot_kw=dict(polar=True))
    palette = [MAIN_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_GOLD, ACCENT_PURPLE, "#4C9A92"]

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        values = [row[m[0]] for m in metrics]
        values += values[:1]
        color = palette[idx % len(palette)]
        ax.plot(angles, values, color=color, linewidth=2, label=row["city_name"])
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="#666666")
    ax.set_ylim(0, 1.0)
    ax.set_title("代表性城市画像雷达图", color=DARK, pad=18)
    ax.grid(color=GRID)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
