* =============================================================================
* Runner: Execute todos os scripts de validacao Stata para Advanced Filters
* (FASE6.5)
*
* ATENCAO: Stata `sspace` NAO SUPORTA filtros nao-lineares ou avancados:
*   - Sem EKF (Extended Kalman Filter)
*   - Sem UKF (Unscented Kalman Filter)
*   - Sem EnKF (Ensemble Kalman Filter)
*   - Sem forma square-root (Cholesky) explicita
*   - Sem forma de informacao (precisao)
*
* Estes scripts servem como REFERENCIAS PARCIAIS:
*   1. validate_reference.do   - pendulo LINEARIZADO (aproximacao grosseira;
*                                documenta por que precisamos de EKF/UKF)
*   2. validate_sqrt_linear.do - caso LINEAR puro (benchmark para SQRT/Info
*                                filter do kalmanbox; Stata so expoe forma
*                                de momento)
*
* Para validacao COMPLETA dos filtros nao-lineares e avancados, use os
* scripts R em `../R/` (implementacoes manuais em R) e o notebook Python
* do kalmanbox.
*
* Uso (a partir deste diretorio):
*   cd $PROJECT_ROOT/examples/06_advanced_filters/validation/stata
*   do run_validation.do
*
* Referencia: Stata Time-Series Reference Manual, [TS] sspace
* =============================================================================

clear all
set more off

display "============================================================"
display "  Kalmanbox Validation: Advanced Filters (Stata) - FASE6.5"
display "============================================================"
display ""

local start_time = c(current_time)
display "Start time: `start_time'"
display ""

display "AVISO IMPORTANTE:"
display "  Stata sspace cobre apenas filtros de Kalman LINEARES e"
display "  Gaussianos. Validacao completa de EKF/UKF/EnKF, SQRT e"
display "  Information filter requer outras ferramentas."
display ""

* --- 1. Referencia linearizada (pendulo) ---
display "============================================================"
display "  [1/2] Pendulo LINEARIZADO (limite do Stata para NL)"
display "============================================================"
capture noisily do validate_reference.do
if _rc != 0 {
    display as error "validate_reference.do retornou codigo " _rc
    display as error "(continuando para o proximo script)"
}
display ""

* --- 2. Caso linear (SQRT / Information) ---
display "============================================================"
display "  [2/2] Caso linear puro (benchmark SQRT / Info)"
display "============================================================"
capture noisily do validate_sqrt_linear.do
if _rc != 0 {
    display as error "validate_sqrt_linear.do retornou codigo " _rc
    display as error "(continuando)"
}
display ""

* --- Resumo ---
display "============================================================"
display "  Scripts Stata de FASE6.5 concluidos"
display "============================================================"
display ""
display "CSVs gerados:"
display "  - results_reference_stata_states.csv"
display "  - results_reference_stata_summary.csv"
display "  - results_sqrt_linear_stata_states.csv"
display "  - results_sqrt_linear_stata_summary.csv"
display ""
display "USO: compare os resultados linearizados com kalmanbox EKF/UKF"
display "     (espera-se pior desempenho no Stata linearizado) e o caso"
display "     linear com SQRT/Information filter do kalmanbox (espera-se"
display "     logLik proximo)."
display ""
display "Para filtros NAO-LINEARES ou avancados (SQRT/Info explicito),"
display "use kalmanbox (Python) ou os scripts R em ../R/."
display ""

local end_time = c(current_time)
display "End time: `end_time'"
