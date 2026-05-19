* =============================================================================
* Runner: Execute all Stata validation scripts for structural models
*
* This script runs all three validation .do files sequentially:
*   1. validate_ucm.do         - UCM with level, trend, cycle (Brazil GDP)
*   2. validate_stochastic_cycle.do - Stochastic cycle (Brazil IPCA)
*   3. validate_regression_ssm.do   - Regression SSM / TVP (UK Gas)
*
* Usage:
*   cd $PROJECT_ROOT/examples/02_structural_models/validation/stata
*   do run_validation.do
*
* Output: CSV files with estimated parameters and decompositions for
*         cross-validation against kalmanbox (Python), KFAS (R), and others.
* =============================================================================

clear all
set more off

display "============================================================"
display "  Kalmanbox Validation: Structural Models (Stata)"
display "============================================================"
display ""

* Record start time
local start_time = c(current_time)
display "Start time: `start_time'"
display ""

* --- 1. UCM: Brazil GDP ---
display "============================================================"
display "  [1/3] UCM Validation: Brazil GDP"
display "============================================================"
do validate_ucm.do
display ""

* --- 2. Stochastic Cycle: Brazil IPCA ---
display "============================================================"
display "  [2/3] Stochastic Cycle Validation: Brazil IPCA"
display "============================================================"
do validate_stochastic_cycle.do
display ""

* --- 3. Regression SSM: UK Gas ---
display "============================================================"
display "  [3/3] Regression SSM Validation: UK Gas"
display "============================================================"
do validate_regression_ssm.do
display ""

* --- Summary ---
display "============================================================"
display "  All validations complete!"
display "============================================================"
display ""
display "Generated CSV files:"
display "  - results_ucm_params.csv"
display "  - results_ucm_decomposition.csv"
display "  - results_stochastic_cycle_params.csv"
display "  - results_stochastic_cycle_decomposition.csv"
display "  - results_regression_ssm_params.csv"
display "  - results_regression_ssm_fixed_vs_ols.csv"
display "  - results_regression_ssm_tvp_coefs.csv"
display ""

local end_time = c(current_time)
display "End time: `end_time'"
display ""
display "Use these CSV files to compare with kalmanbox (Python) and KFAS (R) results."
