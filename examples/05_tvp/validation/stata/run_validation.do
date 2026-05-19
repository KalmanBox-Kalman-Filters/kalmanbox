* =============================================================================
* Runner: Execute todos os scripts de validacao Stata para TVP (FASE5.5)
*
* Scripts executados sequencialmente:
*   1. validate_tvp_regression.do  - TVP regression (inflation ~ 1 + gdp_gap)
*   2. validate_phillips_curve.do  - Phillips curve TVP (SSW + NAIRU fixo)
*
* Nota: Stata `sspace` so suporta modelos state-space lineares-Gaussianos, entao
* a Phillips curve com produto beta_t * NAIRU_t NAO pode ser especificada
* diretamente. O script validate_phillips_curve.do usa a reparametrizacao SSW
* (alpha_t = -beta_t * NAIRU_t) e um modelo slope-only TV para comparacao.
*
* Uso (a partir deste diretorio):
*   cd $PROJECT_ROOT/examples/05_tvp/validation/stata
*   do run_validation.do
*
* Output: CSVs com parametros estimados e trajetorias dos coeficientes para
*         cross-validacao com kalmanbox (Python), dlm (R) e bvarsv (R).
* =============================================================================

clear all
set more off

display "============================================================"
display "  Kalmanbox Validation: TVP (Stata sspace) - FASE5.5"
display "============================================================"
display ""

local start_time = c(current_time)
display "Start time: `start_time'"
display ""

* --- 1. TVP Regression ---
display "============================================================"
display "  [1/2] TVP Regression Validation (inflation ~ 1 + gdp_gap)"
display "============================================================"
do validate_tvp_regression.do
display ""

* --- 2. Phillips Curve ---
display "============================================================"
display "  [2/2] Phillips Curve TVP Validation (Delta pi ~ 1 + u)"
display "============================================================"
do validate_phillips_curve.do
display ""

* --- Resumo ---
display "============================================================"
display "  All TVP sspace validations complete!"
display "============================================================"
display ""
display "Generated CSV files:"
display "  - results_tvp_regression_stata_coefs.csv"
display "  - results_tvp_regression_stata_summary.csv"
display "  - results_phillips_curve_stata.csv"
display "  - results_phillips_curve_stata_summary.csv"
display ""

local end_time = c(current_time)
display "End time: `end_time'"
display ""
display "Use estes CSVs para comparar com kalmanbox (Python), dlm e bvarsv (R)."
