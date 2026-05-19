/*******************************************************************************
* run_validation.do
*
* Runner: executa todos os scripts de validacao Stata para Dynamic Factor
* Models (FASE4 subfase F4.5).
*
* Uso:
*   stata -b do run_validation.do
*   ou interativamente: do run_validation.do
*
* Scripts executados:
*   1. validate_dfm.do              - DFM K=2 com dfactor
*   2. validate_mixed_freq.do        - DFM K=1 com missing data (nowcast GDP)
*   3. validate_variance_decomp.do   - Decomposicao de variancia / R^2
*
* Saidas esperadas (CSV) no mesmo diretorio:
*   - results_dfm_factors.csv
*   - results_dfm_loadings.csv
*   - results_dfm_fit_summary.csv
*   - results_mixed_freq_nowcast.csv
*   - results_mixed_freq_loadings.csv
*   - results_mixed_freq_summary.csv
*   - results_variance_decomp.csv
*   - results_variance_decomp_loadings.csv
*   - results_variance_decomp_summary.csv
*
* Logs:
*   - validate_dfm.log
*   - validate_mixed_freq.log
*   - validate_variance_decomp.log
*   - run_validation.log
*
* Autor: kalmanbox validation suite
* Fase:  FASE4 subfase F4.5
*******************************************************************************/

clear all
set more off
capture log close
log using "run_validation.log", replace

display "============================================================"
display " KALMANBOX - Validacao Stata Dynamic Factor Models (FASE4)"
display " Runner iniciado: $S_DATE $S_TIME"
display "============================================================"
display ""

* --- Configurar diretorio ---
* Garantir que estamos no diretorio dos scripts (necessario para paths
* relativos aos CSVs de dados em ../../data/).
local script_dir = c(pwd)
display "Diretorio de trabalho: `script_dir'"
display ""

local n_total = 3
local n_ok    = 0
local n_err   = 0
local failed_scripts ""

* -----------------------------------------------------------------------------
* Script 1: DFM K=2
* -----------------------------------------------------------------------------
display "============================================================"
display " [1/`n_total'] Executando validate_dfm.do"
display "============================================================"
display ""

capture noisily do "`script_dir'/validate_dfm.do"
if _rc != 0 {
    display as error "ERRO: validate_dfm.do falhou com _rc = " _rc
    local n_err = `n_err' + 1
    local failed_scripts "`failed_scripts' validate_dfm.do"
}
else {
    display ""
    display as result "[1/`n_total'] validate_dfm.do concluido com sucesso"
    local n_ok = `n_ok' + 1
}
display ""

* -----------------------------------------------------------------------------
* Script 2: Mixed Frequency Nowcast
* -----------------------------------------------------------------------------
display "============================================================"
display " [2/`n_total'] Executando validate_mixed_freq.do"
display "============================================================"
display ""

capture noisily do "`script_dir'/validate_mixed_freq.do"
if _rc != 0 {
    display as error "ERRO: validate_mixed_freq.do falhou com _rc = " _rc
    local n_err = `n_err' + 1
    local failed_scripts "`failed_scripts' validate_mixed_freq.do"
}
else {
    display ""
    display as result "[2/`n_total'] validate_mixed_freq.do concluido com sucesso"
    local n_ok = `n_ok' + 1
}
display ""

* -----------------------------------------------------------------------------
* Script 3: Variance Decomposition
* -----------------------------------------------------------------------------
display "============================================================"
display " [3/`n_total'] Executando validate_variance_decomp.do"
display "============================================================"
display ""

capture noisily do "`script_dir'/validate_variance_decomp.do"
if _rc != 0 {
    display as error "ERRO: validate_variance_decomp.do falhou com _rc = " _rc
    local n_err = `n_err' + 1
    local failed_scripts "`failed_scripts' validate_variance_decomp.do"
}
else {
    display ""
    display as result "[3/`n_total'] validate_variance_decomp.do concluido com sucesso"
    local n_ok = `n_ok' + 1
}
display ""

* -----------------------------------------------------------------------------
* Resumo final e verificacao de arquivos gerados
* -----------------------------------------------------------------------------
display "============================================================"
display " RESUMO DA VALIDACAO"
display "============================================================"
display ""
display " Scripts executados: `n_total'"
display " Concluidos com sucesso: `n_ok'"
display " Com erro: `n_err'"
if "`failed_scripts'" != "" {
    display as error " Scripts com falha:`failed_scripts'"
}
display ""

display " Verificando arquivos CSV gerados..."
local outputs ///
    results_dfm_factors.csv ///
    results_dfm_loadings.csv ///
    results_dfm_fit_summary.csv ///
    results_mixed_freq_nowcast.csv ///
    results_mixed_freq_loadings.csv ///
    results_mixed_freq_summary.csv ///
    results_variance_decomp.csv ///
    results_variance_decomp_loadings.csv ///
    results_variance_decomp_summary.csv

local n_missing = 0
foreach f of local outputs {
    capture confirm file "`f'"
    if _rc == 0 {
        display "   [OK]      `f'"
    }
    else {
        display as error "   [MISSING] `f'"
        local n_missing = `n_missing' + 1
    }
}

display ""
if `n_missing' == 0 {
    display as result " Todos os arquivos CSV esperados foram gerados."
}
else {
    display as error " `n_missing' arquivo(s) CSV nao foram gerados."
}

display ""
display " Logs disponiveis:"
display "   - validate_dfm.log"
display "   - validate_mixed_freq.log"
display "   - validate_variance_decomp.log"
display "   - run_validation.log"

display ""
display " Runner concluido: $S_DATE $S_TIME"
display "============================================================"

log close
