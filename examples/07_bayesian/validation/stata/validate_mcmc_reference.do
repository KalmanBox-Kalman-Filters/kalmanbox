* ===========================================================================
* Validacao: MLE reference para comparacao com posterior Bayesian (Nile)
* ===========================================================================
*
* Propósito: gerar o MLE frequentista via `sspace` como ponto de comparacao
* com o posterior gerado por `bayes: sspace` (validate_bayesian_ssm.do) e
* com o Gibbs+FFBS em R (validate_gibbs_ffbs.R).
*
* Justificativa teorica (Bernstein-von Mises):
*   Para T grande e priors regulares, a posterior marginal de theta deve
*   colapsar em uma Normal centrada no MLE, com covariancia (-H)^-1 / T.
*   Logo: posterior_mean(sigma2) ~= MLE(sigma2), se a amostra for grande.
*
* ---------------------------------------------------------------------------
* REQUISITOS
* ---------------------------------------------------------------------------
* - Stata >= 12 (sspace existe desde Stata 12).
* - Versao minima testada: 17 (para consistencia com validate_bayesian_ssm).
*
* ---------------------------------------------------------------------------
* COMPARACAO: `bayes: sspace` vs `sspace` (MLE)
* ---------------------------------------------------------------------------
* sspace (MLE):
*   + Convergencia rapida (BFGS).
*   + Inferencia frequentista via hessiano.
*   + Padronizado, bem testado, reproduzivel.
*   - Nao fornece distribuicao completa dos parametros.
*   - IC assintotico pode ser ruim para variancias proximas de zero.
*
* bayes: sspace:
*   + Posterior completo (incerteza nas variancias).
*   + Incorpora info prior (util para amostras pequenas).
*   + Intervalos de credibilidade diretos.
*   - MH adaptativo, nao FFBS: mixing pior.
*   - Priors default (IG(0.01, 0.01)) podem ser inadequadas.
*   - Mais lento.
*
* Na pratica: se T >= 100 e nao ha info prior relevante, MLE via `sspace`
* eh a opcao mais pragmatica em Stata. Bayesian faz sentido quando:
*   (1) Quer incerteza em variancias (nao apenas IC de Wald),
*   (2) Tem priors substantivos,
*   (3) Amostra pequena onde MLE e instavel.
*
* ---------------------------------------------------------------------------
* LIMITACOES (revisao)
* ---------------------------------------------------------------------------
* `bayes: sspace` herda todas as limitacoes do `sspace`:
*   - Apenas modelos lineares Gaussianos.
*   - Sem suporte direto a missing data complexo.
*   - State equations com restricoes nao-triviais exigem parameterizacao
*     manual.
* Alem disso, `bayes: sspace` em Stata 17+:
*   - Nao expõe FFBS; draws de estados requerem `bayespredict`.
*   - Priors IG com shape < 1 (Jeffrey-like) nao suportados oficialmente.
*   - Gelman-Rubin R-hat so disponivel com nchains >= 2 via `bayesstats grubin`.
*
* Para quem precisa de FFBS real em Stata, a unica alternativa seh
* codar `bayesmh` com `llevaluator()` customizado que roda Kalman filter
* (vide Etapa 6 em validate_bayesian_ssm.do).
*
* ===========================================================================

clear all
set more off
version 17

* --------------------------------------------------------------------------
* 1. Carregar dados (Nile)
* --------------------------------------------------------------------------
import delimited "../../data/nile.csv", clear
tsset year

di _newline
di "=========================================="
di "MLE Reference (Nile) - sspace"
di "=========================================="
summarize flow

* --------------------------------------------------------------------------
* 2. Estimar Local-Level via MLE
* --------------------------------------------------------------------------
* Mesma especificacao de validate_bayesian_ssm.do (local-level):
*   obs:    flow = mu + eps   (mle estima var(eps))
*   state:  mu   = L.mu + eta (var(eta) estimada)
sspace (flow = {mu}, mle) ///
       (mu = L.mu, state noconstant)

* --------------------------------------------------------------------------
* 3. Extrair estimativas MLE
* --------------------------------------------------------------------------
matrix params = e(b)
matrix V      = e(V)

* Variancias (Stata parametriza em log-var internamente - expoe diretamente)
scalar sigma2_eps_mle = params[1, "/:var(flow.flow)"]
scalar sigma2_eta_mle = params[1, "/:var(mu.mu)"]

* Erros-padrao (delta method na log-scale ja aplicado pelo Stata)
scalar se_eps = sqrt(V["/:var(flow.flow)", "/:var(flow.flow)"])
scalar se_eta = sqrt(V["/:var(mu.mu)",     "/:var(mu.mu)"])

scalar loglik   = e(ll)
scalar aic_val  = -2 * loglik + 2 * 2
scalar bic_val  = -2 * loglik + 2 * log(e(N))

di _newline
di "=========================================="
di "MLE Local-Level (Nile) - resultados"
di "=========================================="
di "sigma2_eps (MLE) = " sigma2_eps_mle "  (SE = " se_eps ")"
di "sigma2_eta (MLE) = " sigma2_eta_mle "  (SE = " se_eta ")"
di "q = sigma2_eta / sigma2_eps = " sigma2_eta_mle / sigma2_eps_mle
di "log-likelihood   = " loglik
di "AIC              = " aic_val
di "BIC              = " bic_val
di "=========================================="

* --------------------------------------------------------------------------
* 4. Estados filtrados e suavizados
* --------------------------------------------------------------------------
predict mu_filtered, state equation(mu) smethod(filter)
predict mu_smoothed, state equation(mu) smethod(smooth)
predict mu_filtered_se, state equation(mu) smethod(filter) rmse
predict mu_smoothed_se, state equation(mu) smethod(smooth) rmse

* --------------------------------------------------------------------------
* 5. Exportar resultados MLE (referencia frequentista)
* --------------------------------------------------------------------------
export delimited year flow mu_filtered mu_smoothed ///
       mu_filtered_se mu_smoothed_se ///
       using "results_mle_reference_states.csv", replace

* Summary dos parametros MLE
preserve
    clear
    set obs 2
    gen str20 parameter = ""
    gen double mle      = .
    gen double se       = .
    gen double ci_lo95  = .
    gen double ci_hi95  = .

    replace parameter = "sigma2_eps" in 1
    replace mle       = sigma2_eps_mle in 1
    replace se        = se_eps         in 1
    replace ci_lo95   = sigma2_eps_mle - 1.96 * se_eps in 1
    replace ci_hi95   = sigma2_eps_mle + 1.96 * se_eps in 1

    replace parameter = "sigma2_eta" in 2
    replace mle       = sigma2_eta_mle in 2
    replace se        = se_eta         in 2
    replace ci_lo95   = sigma2_eta_mle - 1.96 * se_eta in 2
    replace ci_hi95   = sigma2_eta_mle + 1.96 * se_eta in 2

    export delimited using "results_mle_reference_params.csv", replace
restore

di _newline
di "Saved: results_mle_reference_states.csv"
di "Saved: results_mle_reference_params.csv"

* --------------------------------------------------------------------------
* 6. Nota final - uso combinado com resultados Bayesian
* --------------------------------------------------------------------------
* Para a comparacao Bernstein-von Mises, execute:
*   1. validate_bayesian_ssm.do  ->  results_bayesian_ssm_stata.csv
*   2. validate_mcmc_reference.do -> results_mle_reference_params.csv
*
* Em Python/R espera-se que:
*   |posterior_mean(sigma2_eps) - MLE(sigma2_eps)| < 2 * posterior_sd
*   |posterior_mean(sigma2_eta) - MLE(sigma2_eta)| < 2 * posterior_sd
*
* Se essa diferenca for grande, verifique:
*   - Priors default do Stata muito informativos para o tamanho amostral,
*   - Burn-in insuficiente,
*   - Mixing ruim (ESS baixo) em bayes: sspace.

di _newline
di "=========================================="
di "MLE reference validation complete."
di "=========================================="
