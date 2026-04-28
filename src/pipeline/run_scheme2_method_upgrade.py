import pandas as pd

from src.analysis.scheme2_methods import (
    INDICATOR_COLUMNS,
    build_city_profile_with_clusters,
    compute_entropy_topsis_scores,
    compute_entropy_weights,
    compute_pca_scores,
    fill_missing_with_year_median,
)
from src.config import scheme2_config as cfg


def main() -> None:
    innovation_panel = pd.read_csv(cfg.SCHEME2_INNOVATION_PANEL_PATH)
    matched_panel = pd.read_csv(cfg.SCHEME2_MATCHED_PANEL_PATH)

    innovation_filled = fill_missing_with_year_median(innovation_panel, INDICATOR_COLUMNS)
    weights = compute_entropy_weights(innovation_filled, INDICATOR_COLUMNS)
    innovation_upgraded = compute_entropy_topsis_scores(innovation_filled, INDICATOR_COLUMNS, weights)
    innovation_upgraded, pca_explained = compute_pca_scores(innovation_upgraded, INDICATOR_COLUMNS)
    innovation_upgraded["pca_explained_variance_ratio"] = pca_explained

    city_profile, diagnostics = build_city_profile_with_clusters(matched_panel, innovation_upgraded)

    cfg.ANALYSIS_READY_DIR.mkdir(parents=True, exist_ok=True)
    weights.to_csv(cfg.SCHEME2_ENTROPY_WEIGHTS_PATH, index=False, encoding="utf-8-sig")
    innovation_upgraded.to_csv(cfg.SCHEME2_INNOVATION_UPGRADED_PATH, index=False, encoding="utf-8-sig")
    city_profile.to_csv(cfg.SCHEME2_CITY_PROFILE_PATH, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(cfg.SCHEME2_CLUSTER_DIAGNOSTICS_PATH, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
