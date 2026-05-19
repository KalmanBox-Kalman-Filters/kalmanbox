* =============================================================================
* Validacao: UCM - Unobserved Components Model (Brazil GDP)
* Comando: ucm
*
* Modelo: local linear trend + stochastic cycle
*   y_t = mu_t + psi_t + eps_t
*   mu_t = mu_{t-1} + nu_{t-1} + xi_t      (level, xi_t ~ N(0, sigma2_level))
*   nu_t = nu_{t-1} + zeta_t                (slope, zeta_t ~ N(0, sigma2_slope))
*   psi_t = rho*cos(lambda)*psi_{t-1} + rho*sin(lambda)*psi*_{t-1} + kappa_t
*
* Stata ucm syntax:
*   ucm depvar, model(rwalk) specifies random-walk level
*   trend(rwalk) adds a stochastic trend (slope)
*   cycle(1, period(P)) adds a stochastic cycle with period P
*
* Dados: Brazil quarterly GDP index (log), 2000-2023
* =============================================================================

clear all
set more off

display "=== UCM Validation: Brazil GDP (Stata ucm) ==="
display ""

* --- Load data ---
* Navigate to data directory relative to script location
import delimited using "../../data/brazil_gdp.csv", clear

* Parse date and set time series
generate date_stata = date(date, "YMD")
format date_stata %td
generate qdate = qofd(date_stata)
format qdate %tq
tsset qdate

* Log transform (multiplicative decomposition)
generate log_gdp = ln(gdp_index)

summarize log_gdp
local n_obs = r(N)
display "Observations: `n_obs'"
display "Frequency: quarterly"
display ""

* =============================================================================
* Model: UCM with level, trend, and cycle
* =============================================================================
* ucm estimates:
*   - Stochastic level (random walk)
*   - Stochastic trend/slope (random walk drift)
*   - Stochastic cycle with estimated period
*
* model(rwalk): random-walk level component
* trend(rwalk): random-walk trend/slope component
* cycle(1): one stochastic cycle (Stata estimates the period)
* =============================================================================

display "--- Fitting UCM: level + trend + cycle ---"
display ""

ucm log_gdp, model(rwalk) trend(rwalk) cycle(1)

* Store estimation results
estimates store ucm_full

* --- Extract estimated parameters ---
display ""
display "--- Estimated Parameters ---"

* Variance of observation disturbance (irregular)
local sigma2_obs = exp(2 * _b[/ln_var(epsilon)])
display "sigma2_obs   = " %9.6f `sigma2_obs'

* Variance of level disturbance
local sigma2_level = exp(2 * _b[/ln_var(level)])
display "sigma2_level = " %9.6f `sigma2_level'

* Variance of slope disturbance
local sigma2_slope = exp(2 * _b[/ln_var(slope)])
display "sigma2_slope = " %9.6f `sigma2_slope'

* Variance of cycle disturbance
local sigma2_cycle = exp(2 * _b[/ln_var(cycle)])
display "sigma2_cycle = " %9.6f `sigma2_cycle'

* Cycle period (estimated by ucm)
* ucm reports the cycle frequency parameter; period = 2*pi / frequency
local lambda_c = _b[/ln_p1.cycle]
local period_quarters = exp(`lambda_c')
local period_years = `period_quarters' / 4
display "period       = " %6.2f `period_quarters' " quarters (" %5.2f `period_years' " years)"

* Damping factor (rho) for the cycle
local rho = tanh(_b[/atanh_rho1.cycle])
display "rho (damping)= " %9.6f `rho'

* Log-likelihood
local loglik = e(ll)
display "logLik       = " %12.4f `loglik'

* =============================================================================
* Extract smoothed components (predict after ucm)
* =============================================================================
display ""
display "--- Extracting Smoothed Components ---"

* Predict smoothed level
predict level_smooth, level equation(level)

* Predict smoothed slope/trend
predict slope_smooth, slope equation(slope)

* Predict smoothed cycle
predict cycle_smooth, cycle equation(cycle)

* Predict full smoothed signal
predict yhat, xb

* Irregular component = observed - level - cycle
generate irregular = log_gdp - level_smooth - cycle_smooth

* Summary statistics of components
summarize level_smooth slope_smooth cycle_smooth irregular

* =============================================================================
* Export parameters to CSV
* =============================================================================
display ""
display "--- Exporting Results ---"

preserve
clear
set obs 8
generate str30 parameter = ""
generate double value = .

replace parameter = "sigma2_obs"       in 1
replace value = `sigma2_obs'           in 1

replace parameter = "sigma2_level"     in 2
replace value = `sigma2_level'         in 2

replace parameter = "sigma2_slope"     in 3
replace value = `sigma2_slope'         in 3

replace parameter = "sigma2_cycle"     in 4
replace value = `sigma2_cycle'         in 4

replace parameter = "rho"              in 5
replace value = `rho'                  in 5

replace parameter = "period_quarters"  in 6
replace value = `period_quarters'      in 6

replace parameter = "period_years"     in 7
replace value = `period_years'         in 7

replace parameter = "loglik"           in 8
replace value = `loglik'               in 8

export delimited using "results_ucm_params.csv", replace
display "Saved: results_ucm_params.csv"
restore

* =============================================================================
* Export decomposition to CSV
* =============================================================================
preserve
keep qdate log_gdp level_smooth slope_smooth cycle_smooth irregular
generate date_str = string(dofq(qdate), "%tdCCYY-NN-DD")
order date_str log_gdp level_smooth slope_smooth cycle_smooth irregular
rename log_gdp y
rename date_str date

export delimited using "results_ucm_decomposition.csv", replace
display "Saved: results_ucm_decomposition.csv"
restore

display ""
display "UCM validation complete."
