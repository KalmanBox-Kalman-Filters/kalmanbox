* =============================================================================
* Validacao: TVP Regression via sspace (Stata)
*
* Modelo TVP (mesma especificacao usada em kalmanbox e dlm):
*   inflation_t = alpha_t + beta_t * gdp_gap_t + eps_t,  eps_t ~ N(0, sigma2_eps)
*   alpha_t     = alpha_{t-1} + eta_a_t,                 eta_a ~ N(0, w_alpha)
*   beta_t      = beta_{t-1}  + eta_b_t,                 eta_b ~ N(0, w_beta)
*
* Stata sspace TVP:
*   sspace permite medir `y` como soma de estados (latentes) com loading fixo
*   (coef=1) ou time-varying (loading dado por um regressor observado). Para
*   TVP com intercepto e slope aleatorios:
*
*     constraint 1 [y]alpha = 1
*     sspace (y = {alpha} {beta}*x, ...)                         ///
*            (alpha = L.alpha, state noconstant)                 ///
*            (beta  = L.beta,  state noconstant),                ///
*            constraints(1) covstate(diagonal)
*
*   - `{alpha}` sem multiplicador -> loading fixo 1 (pin via constraint)
*   - `{beta}*x` -> loading observado time-varying = gdp_gap
*   - Cada state equation define random walk com `state noconstant`
*   - `covstate(diagonal)` forca W diagonal (coerente com kalmanbox/dlm)
*
* Dataset: us_inflation_unemployment.csv (quarterly 1960Q1-2023Q4)
*   Regressao: inflation ~ 1 + gdp_gap (mesma spec dos notebooks kalmanbox).
* =============================================================================

clear all
set more off

display "=== TVP Regression Validation: inflation ~ 1 + gdp_gap (Stata sspace) ==="
display ""

* ---------------------------------------------------------------------------
* Dados
* ---------------------------------------------------------------------------
import delimited using "../../data/us_inflation_unemployment.csv", clear

generate date_stata = date(date, "YMD")
format date_stata %td
generate qdate = qofd(date_stata)
format qdate %tq
tsset qdate

local n_obs = _N
display "Observations: `n_obs' quarters"
display ""

* Loading observado constante (= 1) para o intercepto time-varying
generate ones = 1

* ---------------------------------------------------------------------------
* Benchmark OLS (pool, full-sample)
* ---------------------------------------------------------------------------
display "--- OLS benchmark (pool, full-sample) ---"
regress inflation gdp_gap

local ols_intercept = _b[_cons]
local ols_slope     = _b[gdp_gap]
local sigma2_ols    = e(rmse)^2
local loglik_ols    = e(ll)

display ""
display "OLS alpha (intercept) = " %10.4f `ols_intercept'
display "OLS beta  (gdp_gap)   = " %10.4f `ols_slope'
display "OLS sigma2            = " %10.4f `sigma2_ols'
display "OLS logLik            = " %10.4f `loglik_ols'
display ""

* ---------------------------------------------------------------------------
* sspace TVP: coeficientes como estados em random walk
* ---------------------------------------------------------------------------
display "--- sspace TVP: alpha_t, beta_t random walk ---"
display ""

* Fixar loadings dos estados em 1 (equivalente a G=I, F_t=[1, gdp_gap])
constraint 1 [inflation]alpha = 1
constraint 2 [inflation]beta  = 1

* Especificacao:
*   - State equations: random walk puro (sem drift) -> `state noconstant`
*   - Measurement: inflation = 1*alpha + 1*(gdp_gap * beta) + eps
*     expresso como {alpha}*ones + {beta}*gdp_gap, com loadings pinados = 1.
*   - covstate(diagonal): W diagonal
*   - noconstant no measurement evita constante extra (ja absorvido em alpha)
sspace (alpha = L.alpha, state noconstant) ///
       (beta  = L.beta,  state noconstant) ///
       (inflation = {alpha}*ones + {beta}*gdp_gap, noconstant), ///
       constraints(1 2) covstate(diagonal) ///
       difficult iterate(500)

estimates store tvp_model

* ---------------------------------------------------------------------------
* Extrai variancias estimadas (parametros livres)
* ---------------------------------------------------------------------------
local loglik_tvp     = e(ll)
local sigma2_obs     = exp(2 * _b[/ln_var(inflation)])
local sigma2_alpha   = exp(2 * _b[/ln_var(alpha)])
local sigma2_beta    = exp(2 * _b[/ln_var(beta)])

display ""
display "--- TVP Variance Parameters ---"
display "sigma2_obs     = " %12.6f `sigma2_obs'
display "sigma2_alpha   = " %12.6f `sigma2_alpha'
display "sigma2_beta    = " %12.6f `sigma2_beta'
display "logLik         = " %12.4f `loglik_tvp'
display ""

* ---------------------------------------------------------------------------
* Suavizacao dos coeficientes (trajetoria via predict ..., state smethod(smooth))
* ---------------------------------------------------------------------------
predict alpha_sm, state equation(alpha) smethod(smooth)
predict beta_sm,  state equation(beta)  smethod(smooth)

* Trajetoria filtrada (recursive) para comparacao
predict alpha_f, state equation(alpha) smethod(onestep)
predict beta_f,  state equation(beta)  smethod(onestep)

local last = _N
display "Smoothed TVP coefs at final t:"
display "  alpha (intercept) = " %10.4f alpha_sm[`last']
display "  beta  (gdp_gap)   = " %10.4f beta_sm[`last']
display ""
display "OLS reference:"
display "  alpha             = " %10.4f `ols_intercept'
display "  beta              = " %10.4f `ols_slope'
display ""

* ---------------------------------------------------------------------------
* Exporta trajetoria dos coeficientes (CSV)
* ---------------------------------------------------------------------------
preserve
keep date alpha_sm beta_sm alpha_f beta_f
rename alpha_sm beta_intercept
rename beta_sm  beta_slope
rename alpha_f  filt_intercept
rename beta_f   filt_slope
order date beta_intercept beta_slope filt_intercept filt_slope
export delimited using "results_tvp_regression_stata_coefs.csv", replace
display "Saved: results_tvp_regression_stata_coefs.csv"
restore

* ---------------------------------------------------------------------------
* Exporta sumario de parametros (CSV)
* ---------------------------------------------------------------------------
preserve
clear
set obs 10
generate str30 metric = ""
generate double value = .

replace metric = "logLik"             in 1
replace value  = `loglik_tvp'         in 1
replace metric = "sigma2_obs"         in 2
replace value  = `sigma2_obs'         in 2
replace metric = "sigma2_alpha"       in 3
replace value  = `sigma2_alpha'       in 3
replace metric = "sigma2_beta"        in 4
replace value  = `sigma2_beta'        in 4
replace metric = "ols_intercept"      in 5
replace value  = `ols_intercept'      in 5
replace metric = "ols_slope"          in 6
replace value  = `ols_slope'          in 6
replace metric = "ols_sigma2"         in 7
replace value  = `sigma2_ols'         in 7
replace metric = "ols_loglik"         in 8
replace value  = `loglik_ols'         in 8
replace metric = "n_obs"              in 9
replace value  = `n_obs'              in 9

drop if missing(metric) | metric == ""
export delimited using "results_tvp_regression_stata_summary.csv", replace
display "Saved: results_tvp_regression_stata_summary.csv"
restore

display ""
display "TVP regression sspace validation complete."
