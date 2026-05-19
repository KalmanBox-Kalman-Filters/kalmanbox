* =============================================================================
* Validacao: Regression SSM (UK Gas Consumption)
* Comandos: sspace (state-space) + regress (OLS comparison)
*
* Modelo 1 - Fixed coefficients (via OLS/regress):
*   y_t = X_t * beta + eps_t
*   beta is constant
*
* Modelo 2 - Time-varying coefficients (TVP via sspace):
*   y_t = X_t * beta_t + eps_t
*   beta_t = beta_{t-1} + eta_t   (random walk coefficients)
*
* Stata sspace syntax:
*   sspace defines the measurement and state equations explicitly.
*   State equations: beta_t = beta_{t-1} + eta_t for each regressor
*   Measurement equation: y_t = sum(x_k_t * beta_k_t) + eps_t
*
* Dados: UK quarterly gas consumption with trend and seasonal regressors
* =============================================================================

clear all
set more off

display "=== Regression SSM Validation: UK Gas (Stata sspace + regress) ==="
display ""

* --- Load data ---
import delimited using "../../data/uk_gas.csv", clear

* Parse date and set quarterly time series
generate date_stata = date(date, "YMD")
format date_stata %td
generate qdate = qofd(date_stata)
format qdate %tq
tsset qdate

local n_obs = _N
display "Observations: `n_obs'"
display "Frequency: quarterly"
display ""

* =============================================================================
* Construct regressors
* =============================================================================
* Intercept, linear trend, quarterly dummies (Q2, Q3, Q4 vs Q1 baseline)

generate intercept = 1
generate trend = _n
generate quarter_num = quarter(dofq(qdate))
generate q2 = (quarter_num == 2)
generate q3 = (quarter_num == 3)
generate q4 = (quarter_num == 4)

display "Regressors: intercept, trend, Q2, Q3, Q4"
display ""

* =============================================================================
* Model 1: Fixed Coefficients (OLS via regress)
* =============================================================================
display "--- Model 1: Fixed Coefficients (OLS) ---"
display ""

regress gas_consumption intercept trend q2 q3 q4, noconstant

* Store OLS results
estimates store ols_model

* Extract OLS coefficients
local ols_intercept = _b[intercept]
local ols_trend     = _b[trend]
local ols_q2        = _b[q2]
local ols_q3        = _b[q3]
local ols_q4        = _b[q4]
local sigma2_ols    = e(rmse)^2
local loglik_ols    = e(ll)

display ""
display "OLS Coefficients:"
display "  intercept  = " %10.4f `ols_intercept'
display "  trend      = " %10.4f `ols_trend'
display "  Q2         = " %10.4f `ols_q2'
display "  Q3         = " %10.4f `ols_q3'
display "  Q4         = " %10.4f `ols_q4'
display "  sigma2_ols = " %10.4f `sigma2_ols'
display "  logLik_ols = " %10.4f `loglik_ols'
display ""

* Predict OLS fitted values
predict yhat_ols, xb

* =============================================================================
* Model 2: Time-Varying Parameters (TVP via sspace)
* =============================================================================
display "--- Model 2: Time-Varying Parameters (sspace) ---"
display ""

* sspace specification:
*   - 5 unobserved states: beta_intercept, beta_trend, beta_q2, beta_q3, beta_q4
*   - Each state follows a random walk: beta_k,t = beta_k,t-1 + eta_k,t
*   - Measurement equation: y_t = sum(x_k,t * beta_k,t) + eps_t
*
* The measurement equation uses the OBS() option to specify time-varying
* loadings from observed regressors multiplied by latent states.

* Note: In Stata sspace, state equations are specified with the 'state' option
* and measurement equations link observed variables to latent states.

sspace (gas_consumption = intercept*{beta_intercept} + trend*{beta_trend} ///
    + q2*{beta_q2} + q3*{beta_q3} + q4*{beta_q4}, ///
    state noconstant), ///
    state(lagged(beta_intercept 1, beta_intercept) ///
          lagged(beta_trend 1, beta_trend) ///
          lagged(beta_q2 1, beta_q2) ///
          lagged(beta_q3 1, beta_q3) ///
          lagged(beta_q4 1, beta_q4)) ///
    covstate(diagonal) ///
    difficult iterate(500)

* Store TVP results
estimates store tvp_model

* --- Extract TVP parameters ---
display ""
display "--- TVP Estimated Variance Parameters ---"

* Observation noise variance
local sigma2_obs_tvp = exp(2 * _b[/ln_var(gas_consumption)])
display "sigma2_obs       = " %12.6f `sigma2_obs_tvp'

* State noise variances (coefficient evolution)
local sigma2_b_intercept = exp(2 * _b[/ln_var(beta_intercept)])
local sigma2_b_trend     = exp(2 * _b[/ln_var(beta_trend)])
local sigma2_b_q2        = exp(2 * _b[/ln_var(beta_q2)])
local sigma2_b_q3        = exp(2 * _b[/ln_var(beta_q3)])
local sigma2_b_q4        = exp(2 * _b[/ln_var(beta_q4)])

display "sigma2_intercept = " %12.6f `sigma2_b_intercept'
display "sigma2_trend     = " %12.6f `sigma2_b_trend'
display "sigma2_Q2        = " %12.6f `sigma2_b_q2'
display "sigma2_Q3        = " %12.6f `sigma2_b_q3'
display "sigma2_Q4        = " %12.6f `sigma2_b_q4'

local loglik_tvp = e(ll)
display "logLik           = " %12.4f `loglik_tvp'

* --- Predict smoothed states (time-varying coefficients) ---
predict beta_intercept_sm, state equation(beta_intercept) smethod(smooth)
predict beta_trend_sm, state equation(beta_trend) smethod(smooth)
predict beta_q2_sm, state equation(beta_q2) smethod(smooth)
predict beta_q3_sm, state equation(beta_q3) smethod(smooth)
predict beta_q4_sm, state equation(beta_q4) smethod(smooth)

* TVP coefficients at final time point
local last = _N
display ""
display "TVP coefficients at final time point:"
display "  intercept  = " %10.4f beta_intercept_sm[`last']
display "  trend      = " %10.4f beta_trend_sm[`last']
display "  Q2         = " %10.4f beta_q2_sm[`last']
display "  Q3         = " %10.4f beta_q3_sm[`last']
display "  Q4         = " %10.4f beta_q4_sm[`last']

* =============================================================================
* Model Comparison
* =============================================================================
display ""
display "--- Model Comparison ---"

local aic_ols = -2 * `loglik_ols' + 2 * 6
local bic_ols = -2 * `loglik_ols' + 6 * ln(`n_obs')
local aic_tvp = -2 * `loglik_tvp' + 2 * 6
local bic_tvp = -2 * `loglik_tvp' + 6 * ln(`n_obs')

display ""
display "Metric               OLS          TVP"
display "Log-likelihood  " %12.4f `loglik_ols' "  " %12.4f `loglik_tvp'
display "AIC             " %12.4f `aic_ols' "  " %12.4f `aic_tvp'
display "BIC             " %12.4f `bic_ols' "  " %12.4f `bic_tvp'

* Signal-to-noise ratios
display ""
display "Signal-to-noise ratios (sigma2_beta / sigma2_obs):"
display "  intercept: SNR = " %12.6f `sigma2_b_intercept' / `sigma2_obs_tvp'
display "  trend    : SNR = " %12.6f `sigma2_b_trend' / `sigma2_obs_tvp'
display "  Q2       : SNR = " %12.6f `sigma2_b_q2' / `sigma2_obs_tvp'
display "  Q3       : SNR = " %12.6f `sigma2_b_q3' / `sigma2_obs_tvp'
display "  Q4       : SNR = " %12.6f `sigma2_b_q4' / `sigma2_obs_tvp'

* =============================================================================
* Export: Fixed vs OLS comparison
* =============================================================================
display ""
display "--- Exporting Results ---"

preserve
clear
set obs 5
generate str15 regressor = ""
generate double ols_coef = .

replace regressor = "intercept" in 1
replace ols_coef  = `ols_intercept' in 1

replace regressor = "trend" in 2
replace ols_coef  = `ols_trend' in 2

replace regressor = "Q2" in 3
replace ols_coef  = `ols_q2' in 3

replace regressor = "Q3" in 4
replace ols_coef  = `ols_q3' in 4

replace regressor = "Q4" in 5
replace ols_coef  = `ols_q4' in 5

export delimited using "results_regression_ssm_fixed_vs_ols.csv", replace
display "Saved: results_regression_ssm_fixed_vs_ols.csv"
restore

* =============================================================================
* Export: All parameters (OLS + TVP)
* =============================================================================
preserve
clear
set obs 20
generate str30 parameter = ""
generate double value = .

* OLS parameters
replace parameter = "ols_coef_intercept" in 1
replace value = `ols_intercept' in 1
replace parameter = "ols_coef_trend" in 2
replace value = `ols_trend' in 2
replace parameter = "ols_coef_Q2" in 3
replace value = `ols_q2' in 3
replace parameter = "ols_coef_Q3" in 4
replace value = `ols_q3' in 4
replace parameter = "ols_coef_Q4" in 5
replace value = `ols_q4' in 5
replace parameter = "ols_sigma2" in 6
replace value = `sigma2_ols' in 6
replace parameter = "ols_loglik" in 7
replace value = `loglik_ols' in 7
replace parameter = "ols_aic" in 8
replace value = `aic_ols' in 8
replace parameter = "ols_bic" in 9
replace value = `bic_ols' in 9

* TVP parameters
replace parameter = "tvp_sigma2_obs" in 10
replace value = `sigma2_obs_tvp' in 10
replace parameter = "tvp_sigma2_intercept" in 11
replace value = `sigma2_b_intercept' in 11
replace parameter = "tvp_sigma2_trend" in 12
replace value = `sigma2_b_trend' in 12
replace parameter = "tvp_sigma2_Q2" in 13
replace value = `sigma2_b_q2' in 13
replace parameter = "tvp_sigma2_Q3" in 14
replace value = `sigma2_b_q3' in 14
replace parameter = "tvp_sigma2_Q4" in 15
replace value = `sigma2_b_q4' in 15
replace parameter = "tvp_loglik" in 16
replace value = `loglik_tvp' in 16
replace parameter = "tvp_aic" in 17
replace value = `aic_tvp' in 17
replace parameter = "tvp_bic" in 18
replace value = `bic_tvp' in 18

* Drop unused rows
drop if missing(parameter) | parameter == ""

export delimited using "results_regression_ssm_params.csv", replace
display "Saved: results_regression_ssm_params.csv"
restore

* =============================================================================
* Export: TVP coefficient evolution
* =============================================================================
preserve
keep qdate beta_intercept_sm beta_trend_sm beta_q2_sm beta_q3_sm beta_q4_sm
generate date_str = string(dofq(qdate), "%tdCCYY-NN-DD")
order date_str beta_intercept_sm beta_trend_sm beta_q2_sm beta_q3_sm beta_q4_sm
rename date_str date

export delimited using "results_regression_ssm_tvp_coefs.csv", replace
display "Saved: results_regression_ssm_tvp_coefs.csv"
restore

display ""
display "Regression SSM validation complete."
