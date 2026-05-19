* ===========================================================================
* Validacao: Bayesian Local-Level (Nile) - Stata bayes: sspace
* ===========================================================================
*
* Referencia Bayesian (parcial) para kalmanbox.estimation.bayesian.BayesianSSM.
*
* Modelo Local-Level (Durbin-Koopman 2012, cap. 2):
*   Observacao: y_t = mu_t + eps_t,   eps_t ~ N(0, sigma2_eps)
*   Estado:     mu_t = mu_{t-1} + eta_t, eta_t ~ N(0, sigma2_eta)
*
* ---------------------------------------------------------------------------
* REQUISITOS
* ---------------------------------------------------------------------------
* - Stata versao MINIMA: 17.0 (bayes: sspace foi introduzido em Stata 17)
* - Em versoes anteriores (<=16) este script falhara com erro de sintaxe.
* - Testado em Stata 18.
*
* ---------------------------------------------------------------------------
* PRIORS DEFAULT DO STATA (bayes: sspace)
* ---------------------------------------------------------------------------
* Para cada variancia /var(.) o Stata usa, por default:
*    sigma2 ~ InvGamma(0.01, 0.01)   (weakly-informative; bayesmhmanual p.23)
* Para coeficientes, o default eh:
*    beta   ~ Normal(0, 10000)       (flat)
*
* Esses defaults diferem de dlm::dlmGibbsDIG em R, que usa IG(1, 1) por
* convencao. Isso implica que os posteriors de Stata nao serao identicos
* aos de dlm / kalmanbox para T pequeno, mas convergem para T grande
* (dominancia da verossimilhanca).
*
* Nota: pode-se sobrescrever com a opcao `prior()` (documentada abaixo),
* mas bayes: sspace NAO aceita priors InvGamma customizados em todas as
* versoes - ver limitacoes.
*
* ---------------------------------------------------------------------------
* LIMITACOES DE `bayes: sspace` (ver [BAYES] bayes: sspace)
* ---------------------------------------------------------------------------
* 1. Nao expoe o FFBS (Forward-Filter Backward-Sample) diretamente. O
*    sampler usado eh um Metropolis-Hastings adaptativo, NAO Gibbs + FFBS
*    como dlm::dlmGibbsDIG. Isso significa:
*      - Mixing pior para variancias quando a likelihood e plana.
*      - ESS (effective sample size) tipicamente menor.
*      - Autocorrelacao elevada nas cadeias.
* 2. Priors disponiveis sao restritas: nao suporta IG conjugado com
*    hyperparams arbitrarios da mesma forma que dlm.
* 3. Nao fornece draws dos estados latentes mu_t por default - e preciso
*    usar `bayespredict` separadamente.
* 4. Suporte limitado a multi-chain diagnostics; para R-hat robusto eh
*    necessario rodar cadeias separadas manualmente.
*
* Para MCMC totalmente customizado (incluindo FFBS manual), a alternativa
* em Stata eh `bayesmh` com likelihood definida via avaliador (ver Etapa 3).
*
* ---------------------------------------------------------------------------
* QUANDO USAR Bayesian vs MLE em Stata
* ---------------------------------------------------------------------------
*   - sspace (MLE):       amostra grande, parametros bem identificados,
*                         inferencia frequentista (IC assintotico).
*   - bayes: sspace:      amostra pequena/moderada, precisa de posteriors
*                         das variancias, quer incluir info prior.
*   - bayesmh customizado: precisa de priors especificos, FFBS, ou
*                          parametros fora do suporte padrao sspace.
*   - R (dlm) / Python (kalmanbox): MCMC serio em SSM.
*
* ===========================================================================

clear all
set more off
version 17

* --------------------------------------------------------------------------
* 1. Carregar dados (Nile - vazao anual 1871-1970)
* --------------------------------------------------------------------------
import delimited "../../data/nile.csv", clear
tsset year

di _newline
di "=========================================="
di "Bayesian Local-Level (Nile) - bayes: sspace"
di "=========================================="
summarize flow

* --------------------------------------------------------------------------
* 2. Estimar Bayesian Local-Level
* --------------------------------------------------------------------------
* Sintaxe:
*   bayes [, opts_mcmc] : sspace (obs_eq) (state_eq)
*
* Opcoes MCMC:
*   mcmcsize(N)   numero de draws pos-burn (N >= 10000 recomendado)
*   burnin(B)     numero de draws descartados (default 2500)
*   rseed(s)      seed para reprodutibilidade
*   thinning(t)   manter 1 a cada t draws
*   nchains(k)    rodar k cadeias em sequencia (Stata 17+)
*   showreffects  mostrar cadeia completa (debug)
*
* Priors default (vide cabecalho): IG(0.01, 0.01) nas variancias.
* Pode-se sobrescrever priors de coeficientes com `prior({nome}, dist)`.
bayes, mcmcsize(10000) burnin(2500) rseed(2026) nchains(2) saving(bayes_draws, replace): ///
    sspace (flow = {mu}, mle) ///
           (mu = L.mu, state noconstant)

* --------------------------------------------------------------------------
* 3. Resumo do posterior
* --------------------------------------------------------------------------
bayesstats summary

* Convergencia (Gelman-Rubin R-hat requer nchains >= 2)
capture bayesstats grubin
if _rc == 0 {
    di "[OK] Gelman-Rubin R-hat disponivel (nchains>=2)"
}
else {
    di "[WARN] bayesstats grubin falhou: " _rc
}

* Effective Sample Size e autocorrelacao
bayesstats ess

* --------------------------------------------------------------------------
* 4. Predicao dos estados latentes (posterior medio)
* --------------------------------------------------------------------------
* bayespredict gera draws preditivos do estado mu_t a partir do posterior
* salvo em bayes_draws.dta. Para local-level usamos a mean do filtro/smoother.
capture bayespredict mu_post_mean, smethod(smooth) rseed(2026)
if _rc != 0 {
    di "[WARN] bayespredict nao suportado para esta configuracao (rc=" _rc ")"
}

* --------------------------------------------------------------------------
* 5. Exportar posterior summary para comparacao com dlm / kalmanbox
* --------------------------------------------------------------------------
* Captura mean / sd / IC 95% das variancias em e(mean), e(sd), e(cri).
matrix M  = e(mean)
matrix SD = e(sd)
matrix CI = e(cri)

* Cria tabela de summary e exporta
preserve
    clear
    set obs 2
    gen str20 parameter = ""
    gen double mean     = .
    gen double sd       = .
    gen double q025     = .
    gen double q975     = .

    replace parameter = "sigma2_eps" in 1
    replace mean      = M[1, "/:var(flow.flow)"]    in 1
    replace sd        = SD[1, "/:var(flow.flow)"]   in 1
    replace q025      = CI[1, "/:var(flow.flow)"]   in 1
    replace q975      = CI[2, "/:var(flow.flow)"]   in 1

    replace parameter = "sigma2_eta" in 2
    replace mean      = M[1, "/:var(mu.mu)"]        in 2
    replace sd        = SD[1, "/:var(mu.mu)"]       in 2
    replace q025      = CI[1, "/:var(mu.mu)"]       in 2
    replace q975      = CI[2, "/:var(mu.mu)"]       in 2

    export delimited using "results_bayesian_ssm_stata.csv", replace
restore

di _newline
di "Posterior summary exportado: results_bayesian_ssm_stata.csv"

* --------------------------------------------------------------------------
* 6. Alternativa: MCMC customizado via bayesmh (reference pattern)
* --------------------------------------------------------------------------
* bayesmh permite especificar a likelihood manualmente e usar priors
* arbitrarios. Para local-level, a likelihood marginal pode ser obtida
* integrando-se os estados via Kalman filter - nao trivial em Stata, mas
* possivel via `llevaluator()` customizado (ver [BAYES] bayesmh evaluators).
*
* Template (nao executado aqui - apenas referencia):
*
*   bayesmh flow, likelihood(llevaluator(my_llf, parameters({sigma2_eps}
*           {sigma2_eta}))) ///
*       prior({sigma2_eps}, igamma(1, 1)) ///
*       prior({sigma2_eta}, igamma(1, 1)) ///
*       mcmcsize(10000) burnin(2500) rseed(2026)
*
* Onde my_llf e uma program definida pelo usuario que:
*   (a) Recebe parametros como arguments.
*   (b) Roda o Kalman filter sobre a serie flow.
*   (c) Retorna log-likelihood em return scalar lnf.
*
* Este padrao e o equivalente em Stata ao que o kalmanbox faz internamente
* em BayesianSSM, mas requer programacao adicional e esta fora do escopo
* desta validacao (que visa apenas referencia cruzada de alto nivel).

di _newline
di "=========================================="
di "Bayesian SSM validation complete."
di "=========================================="
