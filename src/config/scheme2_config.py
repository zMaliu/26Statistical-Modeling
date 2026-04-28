from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OFFICIAL_CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned" / "analysis_city_panel_official_cleaned.csv"

ANALYSIS_READY_DIR = PROJECT_ROOT / "data" / "processed" / "analysis_ready"
PICTURE_DIR = PROJECT_ROOT / "picture"

SCHEME2_MAIN_PANEL_PATH = ANALYSIS_READY_DIR / "scheme2_main_city_panel.csv"
SCHEME2_INNOVATION_PANEL_PATH = ANALYSIS_READY_DIR / "scheme2_innovation_support_panel.csv"
SCHEME2_AI_PANEL_PATH = ANALYSIS_READY_DIR / "scheme2_ai_measurement_panel.csv"
SCHEME2_MATCHED_PANEL_PATH = ANALYSIS_READY_DIR / "scheme2_ai_innovation_matched_panel.csv"
SCHEME2_STRATIFICATION_PATH = ANALYSIS_READY_DIR / "scheme2_city_stratification.csv"
SCHEME2_SUMMARY_PATH = ANALYSIS_READY_DIR / "scheme2_data_summary.csv"
SCHEME2_ENTROPY_WEIGHTS_PATH = ANALYSIS_READY_DIR / "scheme2_entropy_weights.csv"
SCHEME2_INNOVATION_UPGRADED_PATH = ANALYSIS_READY_DIR / "scheme2_innovation_support_panel_upgraded.csv"
SCHEME2_CITY_PROFILE_PATH = ANALYSIS_READY_DIR / "scheme2_city_profile_upgraded.csv"
SCHEME2_CLUSTER_DIAGNOSTICS_PATH = ANALYSIS_READY_DIR / "scheme2_cluster_diagnostics.csv"
SCHEME2_DESCRIPTIVE_STATS_PATH = ANALYSIS_READY_DIR / "scheme2_descriptive_statistics.csv"
SCHEME2_CORRELATION_MATRIX_PATH = ANALYSIS_READY_DIR / "scheme2_correlation_matrix.csv"
SCHEME2_REGRESSION_RESULTS_PATH = ANALYSIS_READY_DIR / "scheme2_regression_results.csv"
SCHEME2_REGRESSION_SUMMARY_PATH = ANALYSIS_READY_DIR / "scheme2_regression_summary.csv"
SCHEME2_CITY_GROUP_SUMMARY_PATH = ANALYSIS_READY_DIR / "scheme2_city_group_summary.csv"
SCHEME2_AI_CITY_SUMMARY_PATH = ANALYSIS_READY_DIR / "scheme2_ai_city_summary.csv"
SCHEME2_AI_YEAR_SUMMARY_PATH = ANALYSIS_READY_DIR / "scheme2_ai_year_summary.csv"
SCHEME2_INNOVATION_CITY_SUMMARY_PATH = ANALYSIS_READY_DIR / "scheme2_innovation_city_summary.csv"
SCHEME2_INNOVATION_YEAR_CHANGE_PATH = ANALYSIS_READY_DIR / "scheme2_innovation_year_change.csv"
SCHEME2_STRATIFICATION_DETAILED_PATH = ANALYSIS_READY_DIR / "scheme2_stratification_detailed.csv"
SCHEME2_CORRELATION_FOCUS_PATH = ANALYSIS_READY_DIR / "scheme2_correlation_focus.csv"
SCHEME2_REGRESSION_FOCUS_PATH = ANALYSIS_READY_DIR / "scheme2_regression_focus.csv"

FIG_AI_CITY_RANKING_PATH = PICTURE_DIR / "fig_ai_city_ranking.png"
FIG_AI_YEAR_TREND_PATH = PICTURE_DIR / "fig_ai_year_trend.png"
FIG_INNOVATION_CITY_RANKING_PATH = PICTURE_DIR / "fig_innovation_city_ranking.png"
FIG_INNOVATION_YEAR_CHANGE_PATH = PICTURE_DIR / "fig_innovation_year_change.png"
FIG_ENTROPY_WEIGHTS_PATH = PICTURE_DIR / "fig_entropy_weights.png"
FIG_CITY_QUADRANT_PATH = PICTURE_DIR / "fig_city_quadrant.png"
FIG_GROUP_COMPARISON_PATH = PICTURE_DIR / "fig_group_comparison.png"
FIG_CORRELATION_HEATMAP_PATH = PICTURE_DIR / "fig_correlation_heatmap.png"
FIG_REGRESSION_COEFFICIENTS_PATH = PICTURE_DIR / "fig_regression_coefficients.png"
FIG_CITY_PROFILE_DUAL_PATH = PICTURE_DIR / "fig_city_profile_dual.png"
FIG_CITY_PROFILE_RADAR_PATH = PICTURE_DIR / "fig_city_profile_radar.png"

ID_COLUMNS = [
    "city_key",
    "city_name",
    "province_name",
    "region_group",
    "longitude",
    "latitude",
    "year",
]

AI_COLUMNS = [
    "ai_agglomeration_composite",
    "ai_company_count",
    "ai_hit_company_count",
    "ai_hit_ratio",
    "ai_keyword_mentions_company_sum",
    "ai_lq_company_hit",
    "ai_lq_keyword_mass",
    "ai_lq_ai_char_mass",
    "ai_small_sample_flag",
    "ai_small_sample_note",
]

INNOVATION_SUPPORT_COLUMNS = [
    "innovation_support_substitute_index",
    "fiscal_intensity_ratio",
    "financial_depth_ratio",
    "fdi_gdp_ratio",
    "retail_per_capita",
    "service_openness_proxy",
    "innovation_support_substitute_components",
    "innovation_support_substitute_flag",
    "science_or_substitute_available",
]

ECONOMIC_SUPPORT_COLUMNS = [
    "gdp",
    "gdp_per_capita",
    "population",
    "retail_sales",
    "fdi_actual_used",
    "financial_deposit_loan",
    "fiscal_expenditure",
    "secondary_industry_share",
    "tertiary_industry_share",
    "science_tech_expenditure",
]

COORDINATION_COLUMNS = [
    "coordination_capacity_composite",
    "official_ai_sample_flag",
    "formal_main_sample_flag",
    "strict_science_sample_flag",
    "science_or_substitute_sample_flag",
    "sample_tier",
]

ALL_COLUMNS = ID_COLUMNS + AI_COLUMNS + INNOVATION_SUPPORT_COLUMNS + ECONOMIC_SUPPORT_COLUMNS + COORDINATION_COLUMNS
