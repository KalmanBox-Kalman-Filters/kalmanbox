* ===========================================================================
* Runner: Executa todos os scripts de validacao Stata - FASE 7.5 (Bayesian)
* ===========================================================================
*
* Ordem de execucao:
*   1. validate_bayesian_ssm.do    -> bayes: sspace (Stata 17+)
*   2. validate_mcmc_reference.do  -> sspace MLE (referencia frequentista)
*
* Uso:
*   No Stata, navegue para este diretorio e execute:
*     do run_validation.do
*
* Arquivos gerados (neste diretorio):
*   - results_bayesian_ssm_stata.csv    (posterior summary, 2 variancias)
*   - results_mle_reference_states.csv  (mu_filtered/mu_smoothed do MLE)
*   - results_mle_reference_params.csv  (MLE + IC 95% das variancias)
*
* ---------------------------------------------------------------------------
* IMPORTANTE
* ---------------------------------------------------------------------------
* - Requer Stata >= 17 para o script Bayesian (bayes: sspace).
* - Se estiver em Stata <17, o script 1 falhara: voce ainda pode rodar
*   apenas validate_mcmc_reference.do (MLE).
* - Em cenarios com Stata 17+ mas sem licenca Bayesian, os comandos
*   `bayes:`, `bayesstats`, `bayespredict` retornarao erro.
*
* ===========================================================================

clear all
set more off

di _newline
di "=========================================="
di "FASE 7.5 - Stata validacao Bayesian SSM"
di "=========================================="
di _newline

* --------------------------------------------------------------------------
* 0. Verificar versao minima
* --------------------------------------------------------------------------
local stata_version = c(stata_version)
di "Stata version detectada: `stata_version'"

if `stata_version' < 17 {
    di as error "ATENCAO: bayes: sspace requer Stata >= 17."
    di as error "        validate_bayesian_ssm.do ira falhar."
    di as error "        Apenas validate_mcmc_reference.do (MLE) rodara."
    local skip_bayes 1
}
else {
    di "[OK] Stata >= 17 - bayes: sspace disponivel."
    local skip_bayes 0
}

* --------------------------------------------------------------------------
* 1. Bayesian Local-Level
* --------------------------------------------------------------------------
if `skip_bayes' == 0 {
    di _newline
    di ">>> Executando: validate_bayesian_ssm.do"
    capture noisily do validate_bayesian_ssm.do
    if _rc != 0 {
        di as error "[ERRO] validate_bayesian_ssm.do falhou (rc=" _rc ")."
        di as error "       Continuando com MLE reference..."
    }
    else {
        di "[OK] validate_bayesian_ssm.do concluido."
    }
    di _newline
}

* --------------------------------------------------------------------------
* 2. MLE Reference
* --------------------------------------------------------------------------
di ">>> Executando: validate_mcmc_reference.do"
capture noisily do validate_mcmc_reference.do
if _rc != 0 {
    di as error "[ERRO] validate_mcmc_reference.do falhou (rc=" _rc ")."
    exit _rc
}
else {
    di "[OK] validate_mcmc_reference.do concluido."
}

* --------------------------------------------------------------------------
* 3. Resumo dos arquivos gerados
* --------------------------------------------------------------------------
di _newline
di "=========================================="
di "Arquivos gerados:"
di "=========================================="

local expected `" "results_bayesian_ssm_stata.csv"   "results_mle_reference_states.csv"   "results_mle_reference_params.csv" "'

foreach f of local expected {
    capture confirm file "`f'"
    if _rc == 0 {
        di "  [OK]       `f'"
    }
    else {
        di "  [MISSING]  `f'"
    }
}

di _newline
di "=========================================="
di "FASE 7.5 Stata validation complete."
di "=========================================="
di _newline
di "Nota: Os CSVs serao comparados com os resultados do R (dlm) e"
di "      do kalmanbox.estimation.bayesian.BayesianSSM nos notebooks."
