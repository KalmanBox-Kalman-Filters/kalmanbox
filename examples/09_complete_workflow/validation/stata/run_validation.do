*! run_validation.do
*! ============================================================================
*!  Runner: executa toda a suite de validacao Stata para FASE9 (complete
*!  workflow examples).
*!
*!    [1/2] Univariate (sspace + ucm) on uk_drivers
*!    [2/2] Multivariate (dfactor)    on us_macro_panel
*!
*!  Uso (a partir de examples/09_complete_workflow/validation/stata/):
*!    stata -b do run_validation.do
*!
*!  Saida: arquivos stata_univariate_*.csv / stata_multivariate_*.csv neste
*!  mesmo diretorio + um log run_validation.log.
*! ============================================================================

version 14
clear all
set more off
capture log close
log using "run_validation.log", replace text

display as text "============================================================="
display as text "  FASE9 - Stata Validation Suite (complete workflow)"
display as text "    [1/2] Univariate (sspace + ucm)  -> uk_drivers"
display as text "    [2/2] Multivariate (dfactor)    -> us_macro_panel"
display as text "============================================================="

* ---------------------------------------------------------------------------
* Verificar comandos disponiveis
* ---------------------------------------------------------------------------

local missing ""
foreach cmd in sspace ucm dfactor {
    capture which `cmd'
    if _rc local missing "`missing' `cmd'"
}
if "`missing'" != "" {
    display as error "Comandos Stata ausentes:`missing'"
    display as error "Estes comandos sao parte do Stata 12+ (oficial)."
    log close
    exit 198
}

* ---------------------------------------------------------------------------
* [1/2] Univariate
* ---------------------------------------------------------------------------

display _newline as text "============================================================="
display as text "  [1/2] Univariate workflow (validate_univariate.do)"
display as text "============================================================="

local uni_ok = 1
capture noisily do "validate_univariate.do"
if _rc {
    local uni_ok = 0
    display as error "[ERROR] validate_univariate.do failed (rc=" _rc ")"
}

* ---------------------------------------------------------------------------
* [2/2] Multivariate
* ---------------------------------------------------------------------------

display _newline as text "============================================================="
display as text "  [2/2] Multivariate workflow (validate_multivariate.do)"
display as text "============================================================="

local mul_ok = 1
capture noisily do "validate_multivariate.do"
if _rc {
    local mul_ok = 0
    display as error "[ERROR] validate_multivariate.do failed (rc=" _rc ")"
}

* ---------------------------------------------------------------------------
* Resumo dos arquivos gerados
* ---------------------------------------------------------------------------

display _newline as text "============================================================="
display as text "  Summary of generated CSV / TXT artefacts"
display as text "============================================================="

local expected ""
local expected "`expected' stata_univariate_model_comparison.csv"
local expected "`expected' stata_univariate_best_params.csv"
local expected "`expected' stata_univariate_best_summary.txt"
local expected "`expected' stata_univariate_decomposition.csv"
local expected "`expected' stata_univariate_diagnostics.csv"
local expected "`expected' stata_univariate_residuals.csv"
local expected "`expected' stata_univariate_forecast.csv"
local expected "`expected' stata_univariate_vs_kalmanbox.csv"
local expected "`expected' stata_multivariate_K_selection.csv"
local expected "`expected' stata_multivariate_best_params.csv"
local expected "`expected' stata_multivariate_best_summary.txt"
local expected "`expected' stata_multivariate_factors.csv"
local expected "`expected' stata_multivariate_loadings.csv"
local expected "`expected' stata_multivariate_variance_decomp.csv"
local expected "`expected' stata_multivariate_vs_kalmanbox.csv"

local n_ok = 0
local n_total = 0
foreach f of local expected {
    local ++n_total
    capture confirm file "`f'"
    if _rc == 0 {
        display as text "  [OK]      `f'"
        local ++n_ok
    }
    else {
        display as error "  [MISSING] `f'"
    }
}

display _newline as text "`n_ok' / `n_total' expected files present."

if `uni_ok' & `mul_ok' {
    display _newline as text "All Stata validations completed successfully."
}
else {
    display _newline as error "One or more validation scripts failed - see log."
    log close
    exit 1
}

log close
