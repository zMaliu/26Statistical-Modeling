import pandas as pd

from src.analysis.scheme2_plotting import (
    plot_ai_city_ranking,
    plot_ai_year_trend,
    plot_city_profile_dual,
    plot_city_profile_radar,
    plot_city_quadrant,
    plot_correlation_heatmap,
    plot_entropy_weights,
    plot_group_comparison,
    plot_innovation_city_ranking,
    plot_innovation_year_change,
    plot_regression_coefficients,
)
from src.config import scheme2_config as cfg


def main() -> None:
    cfg.PICTURE_DIR.mkdir(parents=True, exist_ok=True)

    ai_city = pd.read_csv(cfg.SCHEME2_AI_CITY_SUMMARY_PATH)
    ai_year = pd.read_csv(cfg.SCHEME2_AI_YEAR_SUMMARY_PATH)
    innovation_city = pd.read_csv(cfg.SCHEME2_INNOVATION_CITY_SUMMARY_PATH)
    innovation_change = pd.read_csv(cfg.SCHEME2_INNOVATION_YEAR_CHANGE_PATH)
    weights = pd.read_csv(cfg.SCHEME2_ENTROPY_WEIGHTS_PATH)
    stratification = pd.read_csv(cfg.SCHEME2_STRATIFICATION_DETAILED_PATH)
    city_group = pd.read_csv(cfg.SCHEME2_CITY_GROUP_SUMMARY_PATH)
    city_profile = pd.read_csv(cfg.SCHEME2_CITY_PROFILE_PATH)
    corr = pd.read_csv(cfg.SCHEME2_CORRELATION_MATRIX_PATH)
    regression_focus = pd.read_csv(cfg.SCHEME2_REGRESSION_FOCUS_PATH)

    plot_ai_city_ranking(ai_city, cfg.FIG_AI_CITY_RANKING_PATH)
    plot_ai_year_trend(ai_year, cfg.FIG_AI_YEAR_TREND_PATH)
    plot_innovation_city_ranking(innovation_city, cfg.FIG_INNOVATION_CITY_RANKING_PATH)
    plot_innovation_year_change(innovation_change, cfg.FIG_INNOVATION_YEAR_CHANGE_PATH)
    plot_entropy_weights(weights, cfg.FIG_ENTROPY_WEIGHTS_PATH)
    plot_city_quadrant(stratification, cfg.FIG_CITY_QUADRANT_PATH)
    plot_group_comparison(city_group, cfg.FIG_GROUP_COMPARISON_PATH)
    plot_city_profile_dual(city_profile, cfg.FIG_CITY_PROFILE_DUAL_PATH)
    plot_city_profile_radar(city_profile, cfg.FIG_CITY_PROFILE_RADAR_PATH)
    plot_correlation_heatmap(corr, cfg.FIG_CORRELATION_HEATMAP_PATH)
    plot_regression_coefficients(regression_focus, cfg.FIG_REGRESSION_COEFFICIENTS_PATH)


if __name__ == "__main__":
    main()
