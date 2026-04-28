import pandas as pd

from src.analysis.scheme2_modeling import (
    build_ai_city_summary,
    build_ai_year_summary,
    build_city_group_summary,
    build_correlation_matrix,
    build_correlation_focus,
    build_descriptive_statistics,
    build_innovation_city_summary,
    build_innovation_year_change,
    build_regression_focus,
    build_regression_outputs,
    build_stratification_detailed,
)
from src.config import scheme2_config as cfg


def main() -> None:
    ai_panel = pd.read_csv(cfg.SCHEME2_AI_PANEL_PATH)
    matched = pd.read_csv(cfg.SCHEME2_MATCHED_PANEL_PATH)
    innovation_upgraded = pd.read_csv(cfg.SCHEME2_INNOVATION_UPGRADED_PATH)
    city_profile = pd.read_csv(cfg.SCHEME2_CITY_PROFILE_PATH)
    stratification = pd.read_csv(cfg.SCHEME2_STRATIFICATION_PATH)

    matched_upgraded = matched.merge(
        innovation_upgraded[
            [
                "city_key",
                "year",
                "innovation_support_entropy_topsis_score",
                "innovation_support_pca_score",
            ]
        ],
        on=["city_key", "year"],
        how="left",
    )

    ai_city_summary = build_ai_city_summary(ai_panel)
    ai_year_summary = build_ai_year_summary(ai_panel)
    innovation_city_summary = build_innovation_city_summary(innovation_upgraded)
    innovation_year_change = build_innovation_year_change(innovation_upgraded)
    stratification_detailed = build_stratification_detailed(stratification)
    descriptive = build_descriptive_statistics(matched_upgraded)
    corr = build_correlation_matrix(matched_upgraded)
    corr_focus = build_correlation_focus(matched_upgraded)
    regression_results, regression_summary = build_regression_outputs(matched_upgraded)
    regression_focus = build_regression_focus(regression_results)
    city_group_summary = build_city_group_summary(city_profile)

    cfg.ANALYSIS_READY_DIR.mkdir(parents=True, exist_ok=True)
    ai_city_summary.to_csv(cfg.SCHEME2_AI_CITY_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    ai_year_summary.to_csv(cfg.SCHEME2_AI_YEAR_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    innovation_city_summary.to_csv(cfg.SCHEME2_INNOVATION_CITY_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    innovation_year_change.to_csv(cfg.SCHEME2_INNOVATION_YEAR_CHANGE_PATH, index=False, encoding="utf-8-sig")
    stratification_detailed.to_csv(cfg.SCHEME2_STRATIFICATION_DETAILED_PATH, index=False, encoding="utf-8-sig")
    descriptive.to_csv(cfg.SCHEME2_DESCRIPTIVE_STATS_PATH, index=False, encoding="utf-8-sig")
    corr.to_csv(cfg.SCHEME2_CORRELATION_MATRIX_PATH, index=False, encoding="utf-8-sig")
    corr_focus.to_csv(cfg.SCHEME2_CORRELATION_FOCUS_PATH, index=False, encoding="utf-8-sig")
    regression_results.to_csv(cfg.SCHEME2_REGRESSION_RESULTS_PATH, index=False, encoding="utf-8-sig")
    regression_summary.to_csv(cfg.SCHEME2_REGRESSION_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    regression_focus.to_csv(cfg.SCHEME2_REGRESSION_FOCUS_PATH, index=False, encoding="utf-8-sig")
    city_group_summary.to_csv(cfg.SCHEME2_CITY_GROUP_SUMMARY_PATH, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
