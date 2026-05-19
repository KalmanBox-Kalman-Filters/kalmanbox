* ===========================================================================
* Validacao: Residual Diagnostics - Local Level no Nile
* Comando: sspace + predict residuals + sktest + wntestq
* ===========================================================================
*
* Objetivo: Ajustar um Local Level a serie do Nilo via sspace e extrair
* residuos de uma-etapa-a-frente (innovations) para aplicar testes de
* diagnostico classicos:
*   - Normalidade: sktest (skewness/kurtosis), swilk (Shapiro-Wilk)
*   - Autocorrelacao: wntestq (Portmanteau Q / Ljung-Box)
*
* Modelo Local Level:
*   Observacao: y_t = mu_t + eps_t,       eps_t ~ N(0, sigma2_eps)
*   Estado   : mu_t = mu_{t-1} + eta_t,   eta_t ~ N(0, sigma2_eta)
*
* Limitacoes do Stata para diagnosticos avancados:
*   - Nao ha auxiliary residuals nativos (e_eps, e_eta standardized)
*     como os retornados por KFAS::residuals(type="response"/"state").
*     Eles devem ser aproximados via residuos observacionais padronizados.
*   - Nao ha funcao pronta para CUSUM / CUSUMSQ sobre residuos de sspace.
*     O teste estat sbcusum cobre regressoes, nao state-space.
*   - Nao ha Jarque-Bera nativo para qualquer serie arbitraria (em
*     contraste com tseries::jarque.bera.test); usamos sktest como
*     substituto padrao do Stata.
*
* Referencias:
*   Durbin & Koopman (2012), Time Series Analysis by State Space Methods
*   Harvey (1989), Forecasting, Structural Time Series Models
* ===========================================================================

clear all
set more off

* --------------------------------------------------------------------------
* 1. Carregar dados Nile
* --------------------------------------------------------------------------
import delimited "../../data/nile.csv", clear
tsset year

di _newline
di "=========================================="
di "Residual Diagnostics - Local Level (Nile)"
di "=========================================="
quietly count
di "Observacoes: " r(N)

* --------------------------------------------------------------------------
* 2. Estimar Local Level via sspace
* --------------------------------------------------------------------------
sspace (volume = {mu}, mle) ///
       (mu = L.mu, state noconstant)

matrix params = e(b)
scalar sigma2_eps = exp(params[1, "/:var(volume.volume)"])
scalar sigma2_eta = exp(params[1, "/:var(mu.mu)"])
scalar loglik     = e(ll)
scalar n_obs      = e(N)

di _newline
di "=========================================="
di "Parametros estimados"
di "=========================================="
di "sigma2_eps (obs)   = " sigma2_eps
di "sigma2_eta (state) = " sigma2_eta
di "q = eta/eps        = " sigma2_eta/sigma2_eps
di "log-likelihood     = " loglik
di "n_obs              = " n_obs
di "=========================================="

* --------------------------------------------------------------------------
* 3. Extrair residuos via predict
* --------------------------------------------------------------------------
* residuos de uma-etapa-a-frente (innovations, y_t - E[y_t|Y_{t-1}]):
predict resid_raw, residuals

* predicao do Kalman filter (one-step-ahead):
predict yhat_filter, xb smethod(onestep)

* media suavizada (usa toda a amostra):
predict yhat_smooth, xb smethod(smooth)

* estado suavizado (nivel):
predict mu_smooth, state equation(mu) smethod(smooth)

* --------------------------------------------------------------------------
* 4. Padronizar residuos
* --------------------------------------------------------------------------
* Padronizacao robusta: subtrai media e divide pelo desvio-padrao amostral.
* Em Kalman filter a variancia de cada innovation e F_t (diferente a cada
* passo), mas sspace + predict nao expoem F_t diretamente, entao usamos
* a padronizacao empirica como aproximacao. Apos burn-in difuso, a
* diferenca e pequena em regime estacionario.
quietly summarize resid_raw
scalar resid_mean = r(mean)
scalar resid_sd   = r(sd)
scalar resid_n    = r(N)

gen double resid_std = (resid_raw - resid_mean) / resid_sd

di _newline
di "--- Residuos (innovations) ---"
di "n (nao missing)    = " resid_n
di "mean               = " resid_mean
di "sd                 = " resid_sd

* --------------------------------------------------------------------------
* 5. Testes de diagnostico
* --------------------------------------------------------------------------

* 5a. sktest: skewness/kurtosis (D'Agostino, Belanger & D'Agostino 1990)
di _newline
di "--- 5a. sktest (skewness/kurtosis, normalidade) ---"
sktest resid_std
scalar sktest_pr_skew   = r(P_skew)
scalar sktest_pr_kurt   = r(P_kurt)
scalar sktest_chi2_all  = r(chi2)
scalar sktest_pr_all    = r(P_chi2)

* 5b. swilk: Shapiro-Wilk normality test
di _newline
di "--- 5b. swilk (Shapiro-Wilk) ---"
swilk resid_std
scalar swilk_W   = r(W)
scalar swilk_p   = r(p)

* 5c. wntestq: Portmanteau Q (Ljung-Box) no lag 10 e 20
di _newline
di "--- 5c. wntestq (Ljung-Box) ---"
di "Lag 10:"
wntestq resid_std, lags(10)
scalar lb10_stat = r(stat)
scalar lb10_df   = r(df)
scalar lb10_p    = r(p)

di "Lag 20:"
wntestq resid_std, lags(20)
scalar lb20_stat = r(stat)
scalar lb20_df   = r(df)
scalar lb20_p    = r(p)

di _newline
di "=========================================="
di "Resumo dos diagnosticos"
di "=========================================="
di "sktest chi2        = " sktest_chi2_all    "   p = " sktest_pr_all
di "swilk W            = " swilk_W            "   p = " swilk_p
di "Ljung-Box(10)      = " lb10_stat          "   p = " lb10_p
di "Ljung-Box(20)      = " lb20_stat          "   p = " lb20_p
di "=========================================="

* --------------------------------------------------------------------------
* 6. Exportar residuos
* --------------------------------------------------------------------------
export delimited year volume yhat_filter yhat_smooth mu_smooth ///
    resid_raw resid_std ///
    using "results_residual_diagnostics_stata.csv", replace

* --------------------------------------------------------------------------
* 7. Exportar testes (CSV tabular)
* --------------------------------------------------------------------------
preserve
    clear
    set obs 5
    gen str30 test      = ""
    gen double statistic = .
    gen double df       = .
    gen double p_value  = .

    replace test = "sktest_skew"    in 1
    replace p_value = sktest_pr_skew in 1

    replace test = "sktest_kurt"    in 2
    replace p_value = sktest_pr_kurt in 2

    replace test = "sktest_joint"   in 3
    replace statistic = sktest_chi2_all in 3
    replace p_value   = sktest_pr_all   in 3

    replace test = "shapiro_wilk"   in 4
    replace statistic = swilk_W     in 4
    replace p_value   = swilk_p     in 4

    set obs 6
    replace test = "ljung_box_10"   in 5
    replace statistic = lb10_stat   in 5
    replace df        = lb10_df     in 5
    replace p_value   = lb10_p      in 5

    replace test = "ljung_box_20"   in 6
    replace statistic = lb20_stat   in 6
    replace df        = lb20_df     in 6
    replace p_value   = lb20_p      in 6

    export delimited using "results_residual_diagnostics_tests_stata.csv", replace
restore

* --------------------------------------------------------------------------
* 8. Exportar resumo do modelo
* --------------------------------------------------------------------------
preserve
    clear
    set obs 1
    gen str20 source      = "Stata sspace"
    gen double sigma2_eps = .
    gen double sigma2_eta = .
    gen double loglik     = .
    gen double n_obs      = .
    gen double q_ratio    = .

    replace sigma2_eps = scalar(sigma2_eps) in 1
    replace sigma2_eta = scalar(sigma2_eta) in 1
    replace loglik     = scalar(loglik)     in 1
    replace n_obs      = scalar(n_obs)      in 1
    replace q_ratio    = scalar(sigma2_eta)/scalar(sigma2_eps) in 1

    export delimited using "results_residual_diagnostics_model_stata.csv", replace
restore

di _newline
di "Residual diagnostics (Stata) completo."
di "  results_residual_diagnostics_stata.csv"
di "  results_residual_diagnostics_tests_stata.csv"
di "  results_residual_diagnostics_model_stata.csv"
