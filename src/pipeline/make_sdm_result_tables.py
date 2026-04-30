"""Create paper-ready SDM result tables from automated Python outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "analysis_ready"
TABLE_DIR = PROJECT_ROOT / "paper_tables"

PY_IMPACTS = ANALYSIS_DIR / "python_panel_sdm_impacts.csv"
SPREG_COEFS = ANALYSIS_DIR / "spreg_panel_sdm_coefficients.csv"
AI_SUMMARY_CSV = ANALYSIS_DIR / "sdm_ai_effects_summary.csv"
MODEL_COMPARISON_CSV = ANALYSIS_DIR / "sdm_model_comparison_summary.csv"
AI_SUMMARY_TEX = TABLE_DIR / "table_sdm_ai_effects.tex"
MODEL_COMPARISON_TEX = TABLE_DIR / "table_sdm_model_comparison.tex"
NARRATIVE_MD = ANALYSIS_DIR / "sdm_final_narrative.md"


MATRIX_LABELS = {
    "inverse_distance": "地理反距离矩阵",
    "knn4": "4近邻矩阵",
    "geo_economic": "地理-经济嵌套矩阵",
}
MATRIX_ORDER = ["inverse_distance", "knn4", "geo_economic"]


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt_coef(v: float, p: float | None = None) -> str:
    if pd.isna(v):
        return ""
    suffix = stars(p) if p is not None else ""
    return f"{v:.4f}{suffix}"


def make_ai_summary() -> pd.DataFrame:
    impacts = pd.read_csv(PY_IMPACTS)
    spreg = pd.read_csv(SPREG_COEFS)

    py_ai = impacts[impacts["variable"] == "ai"].copy()
    py_pivot = py_ai.pivot(index="matrix", columns="effect_type", values=["estimate", "p_value"])
    py_pivot.columns = [f"python_{a}_{b}" for a, b in py_pivot.columns]
    py_pivot = py_pivot.reset_index()

    rows = []
    for matrix, sub in spreg.groupby("matrix"):
        row = {"matrix": matrix}
        for term in ["ai", "W_ai", "W_coord"]:
            one = sub[sub["term"] == term]
            if one.empty:
                row[f"spreg_{term}_coef"] = pd.NA
                row[f"spreg_{term}_p"] = pd.NA
            else:
                row[f"spreg_{term}_coef"] = one.iloc[0]["estimate"]
                row[f"spreg_{term}_p"] = one.iloc[0]["p_value"]
        rows.append(row)
    spreg_ai = pd.DataFrame(rows)

    out = py_pivot.merge(spreg_ai, on="matrix", how="left")
    out.insert(1, "matrix_label", out["matrix"].map(MATRIX_LABELS))
    out.to_csv(AI_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    return out


def make_model_comparison() -> pd.DataFrame:
    spreg = pd.read_csv(SPREG_COEFS)
    rows = []
    for matrix, sub in spreg.groupby("matrix"):
        rho = sub[sub["term"] == "W_coord"].iloc[0]
        wai = sub[sub["term"] == "W_ai"].iloc[0]
        ai = sub[sub["term"] == "ai"].iloc[0]
        rows.append(
            {
                "matrix": matrix,
                "matrix_label": MATRIX_LABELS.get(matrix, matrix),
                "rho": rho["estimate"],
                "rho_p": rho["p_value"],
                "ai_local": ai["estimate"],
                "ai_local_p": ai["p_value"],
                "w_ai": wai["estimate"],
                "w_ai_p": wai["p_value"],
                "loglik": rho["loglik"],
                "aic": rho["aic"],
                "bic": rho["bic"],
            }
        )
    out = pd.DataFrame(rows)
    out["matrix"] = pd.Categorical(out["matrix"], categories=MATRIX_ORDER, ordered=True)
    out = out.sort_values("matrix").reset_index(drop=True)
    out["matrix"] = out["matrix"].astype(str)
    out.to_csv(MODEL_COMPARISON_CSV, index=False, encoding="utf-8-sig")
    return out


def write_latex_table(df: pd.DataFrame, path: Path, table_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if table_type == "ai_effects":
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\begin{threeparttable}",
            r"\caption{Panel SDM中AI产业代理指标的效应分解结果}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"空间矩阵 & 直接效应 & 间接效应 & 总效应 \\",
            r"\midrule",
        ]
        for _, row in df.iterrows():
            lines.append(
                f"{row['matrix_label']} & "
                f"{fmt_coef(row['python_estimate_direct'], row['python_p_value_direct'])} & "
                f"{fmt_coef(row['python_estimate_indirect'], row['python_p_value_indirect'])} & "
                f"{fmt_coef(row['python_estimate_total'], row['python_p_value_total'])} " + r"\\"
            )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\begin{tablenotes}",
                r"\footnotesize",
                r"\item 注：*、**、***分别表示在10\%、5\%、1\%水平显著。直接效应、间接效应和总效应分别反映本地影响、邻近城市空间溢出影响及二者合计影响。",
                r"\end{tablenotes}",
                r"\end{threeparttable}",
                r"\end{table}",
                "",
            ]
        )
    else:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\begin{threeparttable}",
            r"\caption{空间模型设定与空间矩阵信息准则比较}",
            r"\label{tab:model-comparison}",
            r"{\compacttableformat",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"空间矩阵 & LogLik & AIC & BIC & $WAI$ & $WCoord$ \\",
            r"\midrule",
        ]
        for _, row in df.iterrows():
            lines.append(
                f"{row['matrix_label']} & "
                f"{row['loglik']:.4f} & "
                f"{row['aic']:.4f} & "
                f"{row['bic']:.4f} & "
                f"{fmt_coef(row['w_ai'], row['w_ai_p'])} & "
                f"{fmt_coef(row['rho'], row['rho_p'])} " + r"\\"
            )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"}",
                r"\begin{tablenotes}",
                r"\footnotesize",
                r"\item 注：*、**、***分别表示在10\%、5\%、1\%水平显著。LogLik越大、AIC和BIC越小，表示模型信息准则表现越优；$WAI$和$WCoord$分别表示AI产业集聚与协调发展参照指标的空间滞后项系数。",
                r"\end{tablenotes}",
                r"\end{threeparttable}",
                r"\end{table}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_narrative(ai_summary: pd.DataFrame, model_summary: pd.DataFrame) -> None:
    inv = ai_summary[ai_summary["matrix"] == "inverse_distance"].iloc[0]
    lines = [
        "# SDM估计结果可写入论文的解释",
        "",
        "本文采用两套空间面板估计路线进行交叉校验：一是双向固定效应 Panel SDM 的效应分解，二是空间滞后极大似然模型的核心系数估计。",
        "",
        "## 核心发现",
        "",
        f"主模型（地理反距离矩阵）下，AI 的直接效应为 {inv['python_estimate_direct']:.4f}，p={inv['python_p_value_direct']:.4f}，未通过显著性检验；间接效应为 {inv['python_estimate_indirect']:.4f}，p={inv['python_p_value_indirect']:.4f}，呈显著负向；总效应为 {inv['python_estimate_total']:.4f}，p={inv['python_p_value_total']:.4f}。",
        "",
        "空间滞后模型结果也显示，主矩阵下 W_ai 系数为负且显著，地理-经济嵌套矩阵下 W_ai 也在10%水平附近为负，说明“空间虹吸/极化”方向具有一定一致性；但4近邻矩阵下不显著，因此论文应采用谨慎表述。",
        "",
        "## 建议论文表述",
        "",
        "在广东省样本中，AI/数字产业代理指标呈现显著空间集聚，但这种集聚尚未稳定转化为正向跨市溢出。主模型结果提示核心城市AI产业可能存在一定空间虹吸效应，即高集聚地区在短期内吸附周边城市的人才、资本和数字服务资源，导致邻近城市协调发展参照指标承压。稳健性检验显示该负向溢出方向存在但显著性依赖空间矩阵设定，因此本文更适合将其解释为“阶段性极化风险”，而非绝对因果结论。",
        "",
        "## 写作底线",
        "",
        "不要写“AI显著赋能区域协调发展”。更稳的标题和结论应改为“AI产业集聚与创新支撑环境匹配研究”或“AI产业空间集聚、创新支撑与区域差异”。",
        "",
    ]
    NARRATIVE_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ai_summary = make_ai_summary()
    model_summary = make_model_comparison()
    write_latex_table(ai_summary, AI_SUMMARY_TEX, "ai_effects")
    write_latex_table(model_summary, MODEL_COMPARISON_TEX, "model_comparison")
    write_narrative(ai_summary, model_summary)
    print(f"Wrote {AI_SUMMARY_CSV}")
    print(f"Wrote {MODEL_COMPARISON_CSV}")
    print(f"Wrote {AI_SUMMARY_TEX}")
    print(f"Wrote {MODEL_COMPARISON_TEX}")
    print(f"Wrote {NARRATIVE_MD}")


if __name__ == "__main__":
    main()
