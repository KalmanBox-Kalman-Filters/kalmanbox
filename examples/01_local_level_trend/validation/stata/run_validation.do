* ===========================================================================
* Runner: Executa todos os scripts de validacao Stata
* ===========================================================================
* Este script executa sequencialmente os 3 scripts de validacao:
*   1. Local Level (Nile)
*   2. Local Linear Trend (Airline Passengers)
*   3. Basic Structural Model (UK Drivers)
*
* Uso: Em Stata, navegue ate o diretorio deste script e execute:
*   do run_validation.do
*
* Os resultados serao salvos como CSVs no diretorio corrente:
*   - results_local_level_stata.csv
*   - results_llt_stata.csv
*   - results_bsm_stata.csv
* ===========================================================================

di _newline
di "=========================================="
di "Iniciando validacao Stata (sspace)"
di "=========================================="
di _newline

* 1. Local Level Model - Nile
di ">>> Executando: validate_local_level.do"
do validate_local_level.do
di _newline

* 2. Local Linear Trend - Airline Passengers
di ">>> Executando: validate_local_linear_trend.do"
do validate_local_linear_trend.do
di _newline

* 3. Basic Structural Model - UK Drivers
di ">>> Executando: validate_bsm.do"
do validate_bsm.do
di _newline

di "=========================================="
di "All Stata validations complete."
di "=========================================="
