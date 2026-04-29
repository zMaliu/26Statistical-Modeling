$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "Step 1/6: rebuild spatial panel and Stata package"
python src/pipeline/run_panel_spatial_baseline.py

Write-Host "Step 2/6: check Stata package inputs"
python stata/check_stata_package.py

Write-Host "Step 3/6: run dependency-light Python Panel SDM"
python src/pipeline/run_python_panel_sdm.py

Write-Host "Step 4/6: run PySAL/spreg Panel SDM validation"
python src/pipeline/run_spreg_panel_sdm.py

Write-Host "Step 5/6: generate paper-ready SDM tables"
python src/pipeline/make_sdm_result_tables.py

Write-Host "Step 6/6: generate competition upgrade checks"
python src/pipeline/run_competition_upgrade_checks.py

Write-Host "All SDM automation steps completed."
Write-Host "Key narrative: data/processed/analysis_ready/sdm_final_narrative.md"
Write-Host "Key table: paper_tables/table_sdm_ai_effects.tex"
Write-Host "Upgrade tables: paper_tables/table_lagged_ai_endogeneity.tex, paper_tables/table_regional_heterogeneity_summary.tex"
