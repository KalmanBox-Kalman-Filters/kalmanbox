/*******************************************************************************
* run_validation.do
*
* Runner: executa todos os scripts de validacao Stata para ARIMA SSM
*
* Uso:
*   stata -b do run_validation.do
*   ou interativamente: do run_validation.do
*
* Saidas:
*   - results_arima_ssm_nile.csv       (arima vs sspace, Nile)
*   - results_arima_ssm_airline.csv     (airline model)
*   - results_arima_ssm_comparison.csv  (comparacao resumo)
*   - results_comparison_mle.csv        (MLE vs CSS vs sspace)
*   - results_comparison_mle_diffs.csv  (diferencas percentuais)
*   - validate_arima_ssm.log
*   - validate_comparison_mle.log
*
* Autor: kalmanbox validation suite
* Data: 2026-04-09
*******************************************************************************/

clear all
set more off

display "============================================================"
display " KALMANBOX - Validacao Stata ARIMA SSM"
display " Runner iniciado: $S_DATE $S_TIME"
display "============================================================"
display ""

* --- Configurar diretorio ---
* Garantir que estamos no diretorio correto dos scripts
* (ajustar se necessario para o ambiente local)
local script_dir = c(pwd)
display "Diretorio de trabalho: `script_dir'"
display ""

* --- Script 1: Validacao arima vs sspace ---
display "============================================================"
display " [1/2] Executando validate_arima_ssm.do"
display "============================================================"
display ""

capture noisily do "`script_dir'/validate_arima_ssm.do"

if _rc != 0 {
    display as error "ERRO: validate_arima_ssm.do falhou com codigo _rc = " _rc
    display as error "Continuando com proximo script..."
}
else {
    display ""
    display as result "[1/2] validate_arima_ssm.do concluido com sucesso"
}

display ""

* --- Script 2: Validacao MLE vs CSS ---
display "============================================================"
display " [2/2] Executando validate_comparison_mle.do"
display "============================================================"
display ""

capture noisily do "`script_dir'/validate_comparison_mle.do"

if _rc != 0 {
    display as error "ERRO: validate_comparison_mle.do falhou com codigo _rc = " _rc
}
else {
    display ""
    display as result "[2/2] validate_comparison_mle.do concluido com sucesso"
}

* --- Resumo ---
display ""
display "============================================================"
display " RESUMO DA VALIDACAO"
display "============================================================"
display ""
display " Scripts executados:"
display "   [1] validate_arima_ssm.do"
display "   [2] validate_comparison_mle.do"
display ""
display " Arquivos CSV gerados:"
display "   - results_arima_ssm_nile.csv"
display "   - results_arima_ssm_airline.csv"
display "   - results_arima_ssm_comparison.csv"
display "   - results_comparison_mle.csv"
display "   - results_comparison_mle_diffs.csv"
display ""
display " Logs:"
display "   - validate_arima_ssm.log"
display "   - validate_comparison_mle.log"
display ""
display " Runner concluido: $S_DATE $S_TIME"
display "============================================================"
