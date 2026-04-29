"""Build a 21-city x 2018-2023 Guangdong panel from official yearbooks.

The goal of this script is to create a reproducible large panel for later
model upgrading.  It prioritizes official Guangdong Statistical Yearbook
tables, keeps source/imputation flags, and separates the full-panel AI/digital
proxy from the smaller annual-report text-mining AI index.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YEARBOOK_DIR = PROJECT_ROOT / "data" / "raw" / "guangdong_statistical_yearbook"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "analysis_ready"
OUTPUT_PANEL = OUTPUT_DIR / "panel_21city_2018_2023_completed.csv"
OUTPUT_REPORT = OUTPUT_DIR / "panel_21city_2018_2023_completion_report.csv"

YEARS = list(range(2018, 2024))

CITY_EN_TO_CN = {
    "Guangzhou": "广州市",
    "Shenzhen": "深圳市",
    "Zhuhai": "珠海市",
    "Shantou": "汕头市",
    "Foshan": "佛山市",
    "Shaoguan": "韶关市",
    "Heyuan": "河源市",
    "Meizhou": "梅州市",
    "Huizhou": "惠州市",
    "Shanwei": "汕尾市",
    "Dongguan": "东莞市",
    "Zhongshan": "中山市",
    "Jiangmen": "江门市",
    "Yangjiang": "阳江市",
    "Zhanjiang": "湛江市",
    "Maoming": "茂名市",
    "Zhaoqing": "肇庆市",
    "Qingyuan": "清远市",
    "Chaozhou": "潮州市",
    "Jieyang": "揭阳市",
    "Yunfu": "云浮市",
}

CITY_CN_TO_REGION = {
    "广州市": "珠三角",
    "深圳市": "珠三角",
    "珠海市": "珠三角",
    "佛山市": "珠三角",
    "惠州市": "珠三角",
    "东莞市": "珠三角",
    "中山市": "珠三角",
    "江门市": "珠三角",
    "肇庆市": "珠三角",
    "汕头市": "粤东",
    "汕尾市": "粤东",
    "潮州市": "粤东",
    "揭阳市": "粤东",
    "湛江市": "粤西",
    "茂名市": "粤西",
    "阳江市": "粤西",
    "韶关市": "粤北",
    "河源市": "粤北",
    "梅州市": "粤北",
    "清远市": "粤北",
    "云浮市": "粤北",
}


@dataclass(frozen=True)
class TableSpec:
    variable: str
    suffix: str
    yearbook_min: int = 2019
    yearbook_max: int = 2024


TIME_SERIES_SPECS = [
    TableSpec("gdp", "directory/02/excel/02-14-1.xls", 2023, 2024),
    TableSpec("population", "directory/03/excel/03-07.xls", 2020, 2024),
    TableSpec("retail_sales", "directory/16/excel/16-04.xls", 2020, 2024),
    TableSpec("fiscal_expenditure", "directory/08/excel/08-03-1.xls", 2020, 2024),
    TableSpec("financial_deposits", "directory/08/excel/08-10-0.xls", 2020, 2024),
    TableSpec("financial_loans", "directory/08/excel/08-10-1.xls", 2020, 2024),
]


def yearbook_zip(yearbook_year: int) -> Path:
    return YEARBOOK_DIR / f"guangdong_statistical_yearbook_{yearbook_year}.zip"


def valid_yearbook_years() -> list[int]:
    years: list[int] = []
    for path in sorted(YEARBOOK_DIR.glob("guangdong_statistical_yearbook_*.zip")):
        match = re.search(r"_(\d{4})\.zip$", path.name)
        if not match:
            continue
        year = int(match.group(1))
        try:
            with zipfile.ZipFile(path):
                years.append(year)
        except zipfile.BadZipFile:
            # The 2018 download is an HTML error page in the current cache.
            continue
    return years


def find_member(zf: zipfile.ZipFile, suffix: str) -> str | None:
    suffix = suffix.replace("\\", "/")
    for name in zf.namelist():
        if name.replace("\\", "/").endswith(suffix):
            return name
    return None


def read_yearbook_table(yearbook_year: int, suffix: str) -> pd.DataFrame | None:
    path = yearbook_zip(yearbook_year)
    if not path.exists():
        return None
    try:
        with zipfile.ZipFile(path) as zf:
            member = find_member(zf, suffix)
            if member is None:
                return None
            return pd.read_excel(io.BytesIO(zf.read(member)), header=None, engine="xlrd")
    except zipfile.BadZipFile:
        return None


def numeric_year(cell: object) -> int | None:
    if pd.isna(cell):
        return None
    if isinstance(cell, (int, np.integer)):
        value = int(cell)
        return value if 1900 <= value <= 2100 else None
    if isinstance(cell, (float, np.floating)) and np.isfinite(cell):
        value = int(cell)
        return value if abs(cell - value) < 1e-6 and 1900 <= value <= 2100 else None
    text = str(cell).strip()
    match = re.fullmatch(r"(\d{4})(?:\.0)?", text)
    if match:
        return int(match.group(1))
    return None


def to_float(cell: object) -> float:
    if pd.isna(cell):
        return np.nan
    if isinstance(cell, (int, float, np.integer, np.floating)):
        return float(cell)
    text = str(cell).strip().replace(",", "")
    if text in {"", "--", "—", "nan"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def find_year_columns(df: pd.DataFrame, target_years: Iterable[int] = YEARS) -> dict[int, int]:
    year_cols: dict[int, int] = {}
    target = set(target_years)
    for col in range(df.shape[1]):
        values = [numeric_year(df.iat[row, col]) for row in range(min(12, len(df)))]
        hits = [v for v in values if v in target]
        if hits:
            year_cols[hits[-1]] = col
    return year_cols


def extract_city_year_series(
    df: pd.DataFrame,
    variable: str,
    source_yearbook: int,
    target_years: Iterable[int] = YEARS,
) -> pd.DataFrame:
    year_cols = find_year_columns(df, target_years)
    records: list[dict[str, object]] = []

    for row_idx in range(len(df)):
        row = df.iloc[row_idx]
        city_en = None
        for cell in row:
            text = "" if pd.isna(cell) else str(cell).strip()
            if text in CITY_EN_TO_CN:
                city_en = text
                break
        if city_en is None:
            continue
        for year, col_idx in year_cols.items():
            value = to_float(df.iat[row_idx, col_idx])
            if np.isfinite(value):
                records.append(
                    {
                        "city_en": city_en,
                        "city_name": CITY_EN_TO_CN[city_en],
                        "year": year,
                        variable: value,
                        f"{variable}_source_yearbook": source_yearbook,
                    }
                )

    return pd.DataFrame(records)


def extract_series_from_specs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    valid_years = valid_yearbook_years()
    for spec in TIME_SERIES_SPECS:
        pieces: list[pd.DataFrame] = []
        for yearbook_year in valid_years:
            if not (spec.yearbook_min <= yearbook_year <= spec.yearbook_max):
                continue
            df = read_yearbook_table(yearbook_year, spec.suffix)
            if df is None:
                continue
            piece = extract_city_year_series(df, spec.variable, yearbook_year)
            if not piece.empty:
                pieces.append(piece)
        if pieces:
            combined = pd.concat(pieces, ignore_index=True)
            combined = (
                combined.sort_values(["city_name", "year", f"{spec.variable}_source_yearbook"])
                .drop_duplicates(["city_name", "year"], keep="last")
                .reset_index(drop=True)
            )
            frames.append(combined)

    panel = base_panel()
    for frame in frames:
        panel = panel.merge(frame, on=["city_en", "city_name", "year"], how="left")
    return panel


def base_panel() -> pd.DataFrame:
    records = []
    for city_en, city_name in CITY_EN_TO_CN.items():
        for year in YEARS:
            records.append(
                {
                    "province_name": "广东省",
                    "city_en": city_en,
                    "city_name": city_name,
                    "city_key": f"广东省_{city_name}",
                    "region_group": CITY_CN_TO_REGION.get(city_name, "其他"),
                    "year": year,
                }
            )
    return pd.DataFrame(records)


def extract_value_added_by_title(keyword: str, variable: str) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for yearbook_year in valid_yearbook_years():
        path = yearbook_zip(yearbook_year)
        if not path.exists():
            continue
        try:
            with zipfile.ZipFile(path) as zf:
                for member in zf.namelist():
                    normalized = member.replace("\\", "/")
                    if "/directory/02/excel/" not in normalized or not normalized.endswith(".xls"):
                        continue
                    df_head = pd.read_excel(io.BytesIO(zf.read(member)), header=None, engine="xlrd", nrows=3)
                    title = " ".join(str(x) for x in df_head.fillna("").to_numpy().ravel())
                    if keyword not in title:
                        continue
                    df = pd.read_excel(io.BytesIO(zf.read(member)), header=None, engine="xlrd")
                    piece = extract_city_year_series(df, variable, yearbook_year)
                    if not piece.empty:
                        pieces.append(piece)
        except zipfile.BadZipFile:
            continue
    if not pieces:
        return pd.DataFrame(columns=["city_en", "city_name", "year", variable, f"{variable}_source_yearbook"])
    out = pd.concat(pieces, ignore_index=True)
    return (
        out.sort_values(["city_name", "year", f"{variable}_source_yearbook"])
        .drop_duplicates(["city_name", "year"], keep="last")
        .reset_index(drop=True)
    )


def extract_fdi() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for yearbook_year in valid_yearbook_years():
        df = read_yearbook_table(yearbook_year, "directory/06/excel/06-22.xls")
        if df is None:
            continue
        year_cols: dict[int, int] = {}
        for col in range(df.shape[1]):
            for row in range(min(6, len(df))):
                year = numeric_year(df.iat[row, col])
                if year in YEARS:
                    # In this table each year has three columns: newly
                    # established firms, contracted amount, actually utilized
                    # foreign capital.  The actual amount is the rightmost one.
                    year_cols[year] = min(col + 1, df.shape[1] - 1)
        records = []
        for row_idx in range(len(df)):
            city_en = None
            for cell in df.iloc[row_idx]:
                text = "" if pd.isna(cell) else str(cell).strip()
                if text in CITY_EN_TO_CN:
                    city_en = text
                    break
            if city_en is None:
                continue
            for year, col_idx in year_cols.items():
                value = to_float(df.iat[row_idx, col_idx])
                if np.isfinite(value):
                    records.append(
                        {
                            "city_en": city_en,
                            "city_name": CITY_EN_TO_CN[city_en],
                            "year": year,
                            "fdi_actual_used": value / 10000.0,
                            "fdi_actual_used_unit": "亿元人民币",
                            "fdi_actual_used_source_yearbook": yearbook_year,
                        }
                    )
                else:
                    pass
        if records:
            pieces.append(pd.DataFrame(records))

    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    return (
        out.sort_values(["city_name", "year", "fdi_actual_used_source_yearbook"])
        .drop_duplicates(["city_name", "year"], keep="last")
        .reset_index(drop=True)
    )


def extract_information_software_entities() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for yearbook_year in valid_yearbook_years():
        path = yearbook_zip(yearbook_year)
        if not path.exists():
            continue
        try:
            with zipfile.ZipFile(path) as zf:
                candidate_members = []
                for member in zf.namelist():
                    normalized = member.replace("\\", "/")
                    if "/directory/01/excel/" not in normalized or not normalized.endswith(".xls"):
                        continue
                    df_head = pd.read_excel(io.BytesIO(zf.read(member)), header=None, engine="xlrd", nrows=14)
                    text = " ".join(str(x) for x in df_head.fillna("").to_numpy().ravel())
                    if "Information" in text and "Software" in text:
                        candidate_members.append(member)
                if not candidate_members:
                    continue
                # Prefer table 01-11-1 when available; otherwise use the first
                # matching continuation table.
                member = sorted(candidate_members, key=lambda x: ("01-11-1" not in x, x))[0]
                df = pd.read_excel(io.BytesIO(zf.read(member)), header=None, engine="xlrd")
        except zipfile.BadZipFile:
            continue

        info_col = None
        for col in range(df.shape[1]):
            header = " ".join(str(df.iat[row, col]) for row in range(min(12, len(df))) if pd.notna(df.iat[row, col]))
            if "Information" in header and ("Software" in header or "Computer" in header):
                info_col = col
                break
        if info_col is None:
            continue
        # Legal-entity tables in this chapter lag the yearbook by about two
        # years in recent Guangdong yearbooks.  We keep the explicit source
        # yearbook and flag this as an official proxy year.
        data_year = yearbook_year - 2
        if data_year not in YEARS:
            continue
        records = []
        for row_idx in range(len(df)):
            city_en = None
            for cell in df.iloc[row_idx]:
                text = "" if pd.isna(cell) else str(cell).strip()
                if text in CITY_EN_TO_CN:
                    city_en = text
                    break
            if city_en is None:
                continue
            value = to_float(df.iat[row_idx, info_col])
            if np.isfinite(value):
                records.append(
                    {
                        "city_en": city_en,
                        "city_name": CITY_EN_TO_CN[city_en],
                        "year": data_year,
                        "information_software_entity_count": value,
                        "information_software_entity_source_yearbook": yearbook_year,
                        "information_software_entity_imputed_flag": 0,
                    }
                )
        if records:
            pieces.append(pd.DataFrame(records))
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    return (
        out.sort_values(["city_name", "year", "information_software_entity_source_yearbook"])
        .drop_duplicates(["city_name", "year"], keep="last")
        .reset_index(drop=True)
    )


def extract_all_society_rd() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for yearbook_year in valid_yearbook_years():
        path = yearbook_zip(yearbook_year)
        if not path.exists():
            continue
        try:
            with zipfile.ZipFile(path) as zf:
                for member in zf.namelist():
                    normalized = member.replace("\\", "/")
                    if "/directory/19/excel/" not in normalized or not normalized.endswith(".xls"):
                        continue
                    df_head = pd.read_excel(io.BytesIO(zf.read(member)), header=None, engine="xlrd", nrows=10)
                    text = " ".join(str(x) for x in df_head.fillna("").to_numpy().ravel())

                    is_expenditure_only = (
                        "分市全社会研究与试验发展经费" in text
                        or "Research and Development Expenditure by City" in text
                    )
                    is_personnel_expenditure = (
                        "分市全社会研究与试验发展人员与经费" in text
                        or "Personnel and Intramural Expenditure on R&D by City" in text
                    )
                    if not (is_expenditure_only or is_personnel_expenditure):
                        continue

                    df = pd.read_excel(io.BytesIO(zf.read(member)), header=None, engine="xlrd")
                    match = re.search(r"20\d{2}", text)
                    if match:
                        data_year = int(match.group(0))
                    else:
                        years_in_table = sorted(
                            {
                                numeric_year(cell)
                                for cell in df_head.to_numpy().ravel()
                                if numeric_year(cell) in YEARS
                            }
                        )
                        data_year = years_in_table[-1] if years_in_table else yearbook_year - 1
                    if data_year not in YEARS:
                        continue

                    records = []
                    for row_idx in range(len(df)):
                        city_en = None
                        for cell in df.iloc[row_idx]:
                            val = "" if pd.isna(cell) else str(cell).strip()
                            if val in CITY_EN_TO_CN:
                                city_en = val
                                break
                        if city_en is None:
                            continue
                        if is_expenditure_only:
                            personnel = np.nan
                            expenditure = to_float(df.iat[row_idx, 2]) / 10000.0 if df.shape[1] > 2 else np.nan
                        else:
                            personnel = to_float(df.iat[row_idx, 2]) if df.shape[1] > 2 else np.nan
                            expenditure = to_float(df.iat[row_idx, 4]) if df.shape[1] > 4 else np.nan
                        if not np.isfinite(expenditure):
                            continue
                        records.append(
                            {
                                "city_en": city_en,
                                "city_name": CITY_EN_TO_CN[city_en],
                                "year": data_year,
                                "rd_personnel_all_society": personnel,
                                "rd_expenditure_all_society": expenditure,
                                "rd_all_society_source_yearbook": yearbook_year,
                                "rd_all_society_imputed_flag": 0,
                            }
                        )
                    if records:
                        pieces.append(pd.DataFrame(records))
        except zipfile.BadZipFile:
            continue
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    return (
        out.sort_values(["city_name", "year", "rd_all_society_source_yearbook"])
        .drop_duplicates(["city_name", "year"], keep="last")
        .reset_index(drop=True)
    )


def extract_industrial_rd() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for yearbook_year in valid_yearbook_years():
        path = yearbook_zip(yearbook_year)
        if not path.exists():
            continue
        try:
            with zipfile.ZipFile(path) as zf:
                for member in zf.namelist():
                    normalized = member.replace("\\", "/")
                    if "/excel/" not in normalized or not normalized.endswith(".xls"):
                        continue
                    try:
                        df_head = pd.read_excel(io.BytesIO(zf.read(member)), header=None, engine="xlrd", nrows=8)
                    except Exception:
                        continue
                    text = " ".join(str(x) for x in df_head.fillna("").to_numpy().ravel())
                    has_rd = "R&D" in text or "R＆D" in text or "R D" in text
                    has_industrial_rd = "规模以上工业企业" in text or "Industrial Enterprises by City" in text
                    has_personnel_expenditure = "活动人员和经费" in text or "Personnel and Expenditure" in text
                    if not (has_rd and has_industrial_rd and has_personnel_expenditure):
                        continue
                    try:
                        df = pd.read_excel(io.BytesIO(zf.read(member)), header=None, engine="xlrd")
                    except Exception:
                        continue
                    # The table has two year blocks: personnel and expenditure.
                    # For expenditure, use the right-side occurrence of each year.
                    exp_cols: dict[int, int] = {}
                    for year in YEARS:
                        cols = []
                        for col in range(df.shape[1]):
                            for row in range(min(8, len(df))):
                                if numeric_year(df.iat[row, col]) == year:
                                    cols.append(col)
                        if cols:
                            exp_cols[year] = max(cols)
                    records = []
                    for row_idx in range(len(df)):
                        city_en = None
                        for cell in df.iloc[row_idx]:
                            val = "" if pd.isna(cell) else str(cell).strip()
                            if val in CITY_EN_TO_CN:
                                city_en = val
                                break
                        if city_en is None:
                            continue
                        for year, col_idx in exp_cols.items():
                            value = to_float(df.iat[row_idx, col_idx])
                            if np.isfinite(value):
                                records.append(
                                    {
                                        "city_en": city_en,
                                        "city_name": CITY_EN_TO_CN[city_en],
                                        "year": year,
                                        "rd_expenditure_industrial": value,
                                        "rd_industrial_source_yearbook": yearbook_year,
                                    }
                                )
                    if records:
                        pieces.append(pd.DataFrame(records))
        except zipfile.BadZipFile:
            continue
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    return (
        out.sort_values(["city_name", "year", "rd_industrial_source_yearbook"])
        .drop_duplicates(["city_name", "year"], keep="last")
        .reset_index(drop=True)
    )


def add_existing_text_ai(panel: pd.DataFrame) -> pd.DataFrame:
    ai_path = OUTPUT_DIR / "scheme2_ai_measurement_panel.csv"
    if not ai_path.exists():
        panel["ai_text_index_original"] = np.nan
        panel["ai_text_coverage_flag"] = 0
        return panel
    ai = pd.read_csv(ai_path, encoding="utf-8-sig")
    keep = [
        "city_name",
        "year",
        "ai_agglomeration_composite",
        "ai_company_count",
        "ai_hit_ratio",
        "ai_keyword_mentions_company_sum",
        "ai_small_sample_flag",
    ]
    keep = [col for col in keep if col in ai.columns]
    ai = ai[keep].rename(columns={"ai_agglomeration_composite": "ai_text_index_original"})
    panel = panel.merge(ai, on=["city_name", "year"], how="left")
    panel["ai_text_coverage_flag"] = panel["ai_text_index_original"].notna().astype(int)
    return panel


def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def entropy_topsis(df: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    X = df[cols].astype(float).copy()
    normalized = X.apply(minmax, axis=0).clip(lower=0) + 1e-12
    p = normalized.div(normalized.sum(axis=0), axis=1)
    entropy = -(p * np.log(p)).sum(axis=0) / np.log(len(normalized))
    redundancy = 1 - entropy
    if np.isclose(redundancy.sum(), 0):
        weights = pd.Series(np.full(len(cols), 1 / len(cols)), index=cols)
    else:
        weights = redundancy / redundancy.sum()
    weighted = normalized.mul(weights, axis=1)
    d_pos = np.sqrt(((weighted - weighted.max(axis=0)) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - weighted.min(axis=0)) ** 2).sum(axis=1))
    score = d_neg / (d_pos + d_neg)
    weights_df = pd.DataFrame({"indicator": cols, "entropy_weight": weights.values})
    return score, weights_df


def add_indices(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    out["financial_deposit_loan"] = out["financial_deposits"] + out["financial_loans"]
    out["gdp_per_capita"] = out["gdp"] * 10000 / out["population"]
    out["retail_per_capita"] = out["retail_sales"] / out["population"]
    out["secondary_industry_share"] = out["secondary_value_added"] / out["gdp"] * 100
    out["tertiary_industry_share"] = out["tertiary_value_added"] / out["gdp"] * 100
    out["fiscal_intensity_ratio"] = out["fiscal_expenditure"] / out["gdp"]
    out["financial_depth_ratio"] = out["financial_deposit_loan"] / out["gdp"]
    out["fdi_gdp_ratio"] = out["fdi_actual_used"] / out["gdp"]
    out["service_openness_proxy"] = out["tertiary_industry_share"]
    out["rd_expenditure_proxy"] = out["rd_expenditure_all_society"].fillna(out["rd_expenditure_industrial"])
    out["rd_expenditure_proxy_source"] = np.where(
        out["rd_expenditure_all_society"].notna(), "all_society_rd", "industrial_enterprise_rd"
    )
    out["rd_expenditure_proxy_imputed_flag"] = out["rd_expenditure_proxy"].isna().astype(int)

    # Fill remaining gaps conservatively by within-city interpolation, then
    # within-year median.  Flags make these rows auditable.
    impute_cols = [
        "information_software_entity_count",
        "rd_expenditure_proxy",
        "fiscal_intensity_ratio",
        "financial_depth_ratio",
        "fdi_gdp_ratio",
        "retail_per_capita",
        "service_openness_proxy",
    ]
    for col in impute_cols:
        flag_col = f"{col}_imputed_flag"
        if flag_col not in out.columns:
            out[flag_col] = 0
        missing_before = out[col].isna()
        out[col] = out.sort_values(["city_name", "year"]).groupby("city_name")[col].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
        med = out.groupby("year")[col].transform("median")
        out[col] = out[col].fillna(med)
        out.loc[missing_before & out[col].notna(), flag_col] = 1

    out["information_software_entities_per_10k_persons"] = (
        out["information_software_entity_count"] / out["population"]
    )
    out["rd_expenditure_intensity"] = out["rd_expenditure_proxy"] / out["gdp"]

    ai_components = ["information_software_entities_per_10k_persons", "rd_expenditure_intensity"]
    # Year-wise z-score keeps the proxy comparable as a relative city position.
    for col in ai_components:
        out[f"{col}_z"] = out.groupby("year")[col].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if not np.isclose(s.std(ddof=0), 0) else 0
        )
    out["ai_digital_proxy_index"] = out[[f"{col}_z" for col in ai_components]].mean(axis=1)
    out["ai_digital_proxy_flag"] = 1
    # The full 126-row panel must use one consistent measurement口径.
    # Annual-report text-mining values are kept separately for robustness or
    # case discussion; mixing them with the official proxy would contaminate
    # panel regressions and spatial models.
    out["ai_full_panel_index"] = out["ai_digital_proxy_index"]
    out["ai_full_panel_index_source"] = "official_information_software_and_rd_proxy"

    support_cols = [
        "fiscal_intensity_ratio",
        "financial_depth_ratio",
        "fdi_gdp_ratio",
        "retail_per_capita",
        "service_openness_proxy",
    ]
    out["innovation_support_index"], weights_df = entropy_topsis(out, support_cols)
    out["innovation_support_rank_within_year"] = out.groupby("year")["innovation_support_index"].rank(
        method="dense", ascending=False
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(out[support_cols])
    pca = PCA(n_components=1, random_state=42)
    out["innovation_support_pca_score"] = pca.fit_transform(X).reshape(-1)
    out["innovation_support_pca_explained_ratio"] = pca.explained_variance_ratio_[0]

    coord_cols = ["gdp_per_capita", "retail_per_capita", "tertiary_industry_share", "innovation_support_index"]
    for col in coord_cols:
        out[f"{col}_z"] = out.groupby("year")[col].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if not np.isclose(s.std(ddof=0), 0) else 0
        )
    out["coordination_reference_index"] = out[[f"{col}_z" for col in coord_cols]].mean(axis=1)
    return out, weights_df


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = extract_series_from_specs()

    secondary = extract_value_added_by_title("Secondary Industry by City", "secondary_value_added")
    tertiary = extract_value_added_by_title("Tertiary Industry by City", "tertiary_value_added")
    fdi = extract_fdi()
    info_entities = extract_information_software_entities()
    rd_all = extract_all_society_rd()
    rd_ind = extract_industrial_rd()

    for frame in [secondary, tertiary, fdi, info_entities, rd_all, rd_ind]:
        if frame is not None and not frame.empty:
            panel = panel.merge(frame, on=["city_en", "city_name", "year"], how="left")

    panel = add_existing_text_ai(panel)
    panel, weights_df = add_indices(panel)

    ordered_first = [
        "province_name",
        "city_key",
        "city_name",
        "city_en",
        "region_group",
        "year",
        "ai_full_panel_index",
        "ai_full_panel_index_source",
        "ai_text_index_original",
        "ai_text_coverage_flag",
        "ai_digital_proxy_index",
        "innovation_support_index",
        "innovation_support_pca_score",
        "coordination_reference_index",
    ]
    rest = [col for col in panel.columns if col not in ordered_first]
    panel = panel[ordered_first + rest].sort_values(["city_name", "year"]).reset_index(drop=True)

    report_rows = [
        {"metric": "target_rows_21_cities_x_6_years", "value": 21 * 6},
        {"metric": "actual_rows", "value": len(panel)},
        {"metric": "city_count", "value": panel["city_name"].nunique()},
        {"metric": "year_count", "value": panel["year"].nunique()},
        {"metric": "ai_text_original_rows", "value": int(panel["ai_text_coverage_flag"].sum())},
        {
            "metric": "ai_full_panel_non_missing_rows",
            "value": int(panel["ai_full_panel_index"].notna().sum()),
        },
        {
            "metric": "innovation_support_non_missing_rows",
            "value": int(panel["innovation_support_index"].notna().sum()),
        },
        {
            "metric": "coordination_reference_non_missing_rows",
            "value": int(panel["coordination_reference_index"].notna().sum()),
        },
    ]
    key_cols = [
        "gdp",
        "population",
        "retail_sales",
        "fiscal_expenditure",
        "financial_deposit_loan",
        "fdi_actual_used",
        "secondary_value_added",
        "tertiary_value_added",
        "information_software_entity_count",
        "rd_expenditure_proxy",
    ]
    for col in key_cols:
        if col in panel.columns:
            report_rows.append({"metric": f"{col}_non_missing_rows", "value": int(panel[col].notna().sum())})
    for _, row in weights_df.iterrows():
        report_rows.append({"metric": f"entropy_weight_{row['indicator']}", "value": float(row["entropy_weight"])})
    report = pd.DataFrame(report_rows)
    return panel, report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel, report = build_panel()
    panel.to_csv(OUTPUT_PANEL, index=False, encoding="utf-8-sig")
    report.to_csv(OUTPUT_REPORT, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUTPUT_PANEL}")
    print(f"Wrote {OUTPUT_REPORT}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
