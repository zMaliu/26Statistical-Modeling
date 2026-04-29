"""Estimate Panel SDM with PySAL/spreg as an open-source Stata alternative."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from libpysal.weights import W
from spreg import ML_LagFE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATA_DIR = PROJECT_ROOT / "stata"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "analysis_ready"

PANEL_PATH = STATA_DIR / "panel_sdm_stata.csv"
WEIGHT_FILES = {
    "inverse_distance": STATA_DIR / "w_inverse_distance_stata.csv",
    "knn4": STATA_DIR / "w_knn4_distance_stata.csv",
    "geo_economic": STATA_DIR / "w_geo_economic_stata.csv",
}

SUMMARY_OUT = ANALYSIS_DIR / "spreg_panel_sdm_summary.txt"
COEF_OUT = ANALYSIS_DIR / "spreg_panel_sdm_coefficients.csv"

Y_COL = "coord"
X_COLS = ["ai", "fiscal", "finance", "fdi", "retail_pc", "service"]


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if np.isclose(std, 0):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def load_panel_long() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(PANEL_PATH).sort_values(["year", "city_id"])
    for col in [Y_COL, *X_COLS]:
        df[f"z_{col}"] = zscore(df[col].astype(float))

    y_long = pd.DataFrame({Y_COL: df[f"z_{Y_COL}"].to_numpy(dtype=float)})
    x_long = pd.DataFrame({col: df[f"z_{col}"].to_numpy(dtype=float) for col in X_COLS})
    return y_long, x_long, X_COLS


def load_libpysal_weight(path: Path) -> W:
    raw = pd.read_csv(path).sort_values("city_id")
    ids = raw["city_id"].astype(int).tolist()
    mat = raw.drop(columns=["city_id"]).to_numpy(dtype=float)
    neighbors: dict[int, list[int]] = {}
    weights: dict[int, list[float]] = {}
    for row_idx, city_id in enumerate(ids):
        js = np.where(mat[row_idx] > 0)[0]
        neighbors[city_id] = [ids[j] for j in js]
        weights[city_id] = [float(mat[row_idx, j]) for j in js]
    w_obj = W(neighbors, weights, ids=ids)
    w_obj.transform = "r"
    return w_obj


def model_to_rows(matrix_name: str, model: ML_LagFE) -> list[dict]:
    rows = []
    names = list(model.name_x)
    betas = model.betas.ravel()
    std_err = np.asarray(model.std_err).ravel()
    z_stats = list(model.z_stat)
    for i, (name, beta, se) in enumerate(zip(names, betas, std_err)):
        z, p = z_stats[i]
        rows.append(
            {
                "matrix": matrix_name,
                "term": name,
                "estimate": beta,
                "std_err": se,
                "z_value": z,
                "p_value": p,
                "rho": getattr(model, "rho", np.nan),
                "loglik": getattr(model, "logll", np.nan),
                "aic": getattr(model, "aic", np.nan),
                "bic": getattr(model, "schwarz", np.nan),
            }
        )
    return rows


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    y, x, x_names = load_panel_long()
    summaries = []
    coef_rows = []
    for matrix_name, weight_path in WEIGHT_FILES.items():
        w = load_libpysal_weight(weight_path)
        model = ML_LagFE(
            y,
            x,
            w,
            slx_lags=1,
            slx_vars="all",
            spat_impacts="all",
            vm=True,
            name_y=Y_COL,
            name_x=x_names,
            name_w=matrix_name,
            name_ds="Guangdong_21city_2018_2023",
        )
        summaries.append(f"\n\n{'=' * 90}\nMATRIX: {matrix_name}\n{'=' * 90}\n{model.summary}")
        coef_rows.extend(model_to_rows(matrix_name, model))

    SUMMARY_OUT.write_text("\n".join(summaries), encoding="utf-8")
    pd.DataFrame(coef_rows).to_csv(COEF_OUT, index=False, encoding="utf-8-sig")
    print(f"Wrote {SUMMARY_OUT}")
    print(f"Wrote {COEF_OUT}")


if __name__ == "__main__":
    main()
