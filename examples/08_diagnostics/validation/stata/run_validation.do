* ===========================================================================
* Runner: FASE 8 - Validacao Stata (Diagnostics)
* ===========================================================================
* Executa os 2 scripts de validacao em sequencia:
*   1. Residual diagnostics (Local Level no Nile)
*   2. Missing data (BSM no log(airline))
*
* Uso: no Stata, navegue ate este diretorio e rode:
*   do run_validation.do
*
* Arquivos CSV gerados (no diretorio corrente):
*   - results_residual_diagnostics_stata.csv
*   - results_residual_diagnostics_tests_stata.csv
*   - results_residual_diagnostics_model_stata.csv
*   - results_missing_data_stata.csv
*   - results_missing_data_params_stata.csv
*   - results_missing_data_summary_stata.csv
*
* Limitacoes conhecidas do Stata para diagnosticos avancados de SSM:
*   - Sem auxiliary residuals padronizados (e_eps_std, e_eta_std).
*   - Sem Jarque-Bera nativo (usar sktest como proxy).
*   - Sem CUSUM/CUSUMSQ para state-space models (estat sbcusum cobre
*     regressoes, nao sspace).
*   - Sem simulation smoother (Durbin-Koopman 2002) nativo; para
*     bootstrap/draws de estado use Mata ou o backend R (KFAS).
* ===========================================================================

clear all
set more off

di _newline
di "=========================================="
di "FASE 8 - Stata Validation Suite (Diagnostics)"
di "Backend: sspace + predict + sktest/wntestq"
di "=========================================="
di _newline

* --------------------------------------------------------------------------
* 1. Residual diagnostics
* --------------------------------------------------------------------------
di ">>> Executando: validate_residual_diagnostics.do"
do validate_residual_diagnostics.do
di _newline

* --------------------------------------------------------------------------
* 2. Missing data
* --------------------------------------------------------------------------
di ">>> Executando: validate_missing_data.do"
do validate_missing_data.do
di _newline

di "=========================================="
di "All Stata validations complete."
di "=========================================="
