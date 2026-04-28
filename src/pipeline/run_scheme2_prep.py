from src.reporting.scheme2_prep import (
    build_ai_measurement_panel,
    build_city_stratification,
    build_innovation_support_panel,
    build_matched_panel,
    build_scheme2_main_panel,
    build_scheme2_summary,
    load_official_cleaned_panel,
    write_scheme2_outputs,
)


def main() -> None:
    df = load_official_cleaned_panel()
    main_panel = build_scheme2_main_panel(df)
    innovation_panel = build_innovation_support_panel(df)
    ai_panel = build_ai_measurement_panel(df)
    matched_panel = build_matched_panel(df)
    stratification = build_city_stratification(df)
    summary = build_scheme2_summary(main_panel, innovation_panel, ai_panel, matched_panel, stratification)
    write_scheme2_outputs(main_panel, innovation_panel, ai_panel, matched_panel, stratification, summary)


if __name__ == "__main__":
    main()
