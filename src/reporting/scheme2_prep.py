import numpy as np
import pandas as pd

from src.config import scheme2_config as cfg


def load_official_cleaned_panel() -> pd.DataFrame:
    return pd.read_csv(cfg.OFFICIAL_CLEANED_PATH)


def build_scheme2_main_panel(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [col for col in cfg.ALL_COLUMNS if col in df.columns]
    main_panel = df[keep_cols].copy()
    main_panel["scheme2_data_role"] = np.where(
        main_panel["ai_agglomeration_composite"].notna(),
        "ai_city_sample",
        "official_reference_panel",
    )
    return main_panel.sort_values(["city_name", "year"]).reset_index(drop=True)


def build_innovation_support_panel(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in cfg.ID_COLUMNS + cfg.INNOVATION_SUPPORT_COLUMNS + cfg.ECONOMIC_SUPPORT_COLUMNS if col in df.columns]
    panel = df.loc[df["innovation_support_substitute_index"].notna(), cols].copy()
    panel = panel.sort_values(["year", "innovation_support_substitute_index"], ascending=[True, False]).reset_index(drop=True)
    panel["innovation_support_rank_within_year"] = panel.groupby("year")["innovation_support_substitute_index"].rank(
        method="dense", ascending=False
    )
    panel["innovation_support_percentile_within_year"] = panel.groupby("year")["innovation_support_substitute_index"].rank(
        pct=True
    )
    return panel


def build_ai_measurement_panel(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in cfg.ID_COLUMNS + cfg.AI_COLUMNS + cfg.INNOVATION_SUPPORT_COLUMNS + cfg.COORDINATION_COLUMNS if col in df.columns]
    panel = df.loc[df["ai_agglomeration_composite"].notna(), cols].copy()
    return panel.sort_values(["city_name", "year"]).reset_index(drop=True)


def build_matched_panel(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        col
        for col in cfg.ID_COLUMNS
        + cfg.AI_COLUMNS
        + cfg.INNOVATION_SUPPORT_COLUMNS
        + cfg.ECONOMIC_SUPPORT_COLUMNS
        + ["coordination_capacity_composite", "formal_main_sample_flag", "sample_tier"]
        if col in df.columns
    ]
    mask = df["ai_agglomeration_composite"].notna() & df["innovation_support_substitute_index"].notna()
    panel = df.loc[mask, cols].copy()
    return panel.sort_values(["city_name", "year"]).reset_index(drop=True)


def build_city_stratification(df: pd.DataFrame) -> pd.DataFrame:
    matched = df.loc[
        df["ai_agglomeration_composite"].notna() & df["innovation_support_substitute_index"].notna(),
        ["city_name", "year", "ai_agglomeration_composite", "innovation_support_substitute_index", "coordination_capacity_composite"],
    ].copy()
    city_avg = (
        matched.groupby("city_name", as_index=False)
        .agg(
            ai_agglomeration_mean=("ai_agglomeration_composite", "mean"),
            ai_observation_count=("ai_agglomeration_composite", "count"),
            innovation_support_mean=("innovation_support_substitute_index", "mean"),
            innovation_observation_count=("innovation_support_substitute_index", "count"),
            coordination_capacity_mean=("coordination_capacity_composite", "mean"),
        )
    )

    ai_median = city_avg["ai_agglomeration_mean"].median()
    innovation_median = city_avg["innovation_support_mean"].median()

    city_avg["ai_level"] = np.where(city_avg["ai_agglomeration_mean"] >= ai_median, "高集聚", "低集聚")
    city_avg["innovation_support_level"] = np.where(
        city_avg["innovation_support_mean"] >= innovation_median, "高支撑", "低支撑"
    )
    city_avg["city_quadrant"] = city_avg["ai_level"] + "-" + city_avg["innovation_support_level"]
    city_avg["scheme2_interpretation"] = city_avg["city_quadrant"].map(
        {
            "高集聚-高支撑": "人工智能集聚与创新支撑协同较强，可作为样本城市中的优势层级。",
            "高集聚-低支撑": "人工智能集聚较强，但创新支撑条件相对偏弱，适合重点讨论支撑短板。",
            "低集聚-高支撑": "创新支撑条件较好，但人工智能集聚表现偏弱，适合讨论潜力释放问题。",
            "低集聚-低支撑": "人工智能集聚与创新支撑均偏弱，更适合作为基础薄弱型样本。",
        }
    )
    return city_avg.sort_values(["ai_level", "innovation_support_level", "city_name"], ascending=[False, False, True]).reset_index(drop=True)


def build_scheme2_summary(
    main_panel: pd.DataFrame,
    innovation_panel: pd.DataFrame,
    ai_panel: pd.DataFrame,
    matched_panel: pd.DataFrame,
    stratification: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = [
        {"metric": "scheme2_main_panel_rows", "value": int(len(main_panel))},
        {"metric": "scheme2_main_panel_cities", "value": int(main_panel["city_name"].nunique())},
        {
            "metric": "scheme2_main_panel_years",
            "value": f"{int(main_panel['year'].min())}-{int(main_panel['year'].max())}",
        },
        {"metric": "innovation_support_panel_rows", "value": int(len(innovation_panel))},
        {"metric": "innovation_support_panel_cities", "value": int(innovation_panel["city_name"].nunique())},
        {"metric": "innovation_support_panel_years", "value": ",".join(map(str, sorted(innovation_panel["year"].unique().tolist())))},
        {"metric": "ai_measurement_panel_rows", "value": int(len(ai_panel))},
        {"metric": "ai_measurement_panel_cities", "value": int(ai_panel["city_name"].nunique())},
        {"metric": "ai_measurement_panel_years", "value": ",".join(map(str, sorted(ai_panel["year"].unique().tolist())))},
        {"metric": "ai_innovation_matched_rows", "value": int(len(matched_panel))},
        {"metric": "ai_innovation_matched_cities", "value": int(matched_panel["city_name"].nunique())},
        {"metric": "city_stratification_rows", "value": int(len(stratification))},
    ]
    return pd.DataFrame(summary_rows)


def write_scheme2_outputs(
    main_panel: pd.DataFrame,
    innovation_panel: pd.DataFrame,
    ai_panel: pd.DataFrame,
    matched_panel: pd.DataFrame,
    stratification: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    cfg.ANALYSIS_READY_DIR.mkdir(parents=True, exist_ok=True)
    main_panel.to_csv(cfg.SCHEME2_MAIN_PANEL_PATH, index=False, encoding="utf-8-sig")
    innovation_panel.to_csv(cfg.SCHEME2_INNOVATION_PANEL_PATH, index=False, encoding="utf-8-sig")
    ai_panel.to_csv(cfg.SCHEME2_AI_PANEL_PATH, index=False, encoding="utf-8-sig")
    matched_panel.to_csv(cfg.SCHEME2_MATCHED_PANEL_PATH, index=False, encoding="utf-8-sig")
    stratification.to_csv(cfg.SCHEME2_STRATIFICATION_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(cfg.SCHEME2_SUMMARY_PATH, index=False, encoding="utf-8-sig")
