* =============================================================================
* Validacao: Stochastic Cycle Model (Brazil IPCA 12-month)
* Comando: ucm
*
* Modelo: local level + stochastic damped cycle
*   y_t = mu_t + psi_t + eps_t
*   mu_t = mu_{t-1} + eta_t                 (random walk level)
*   psi_t = rho*cos(lambda)*psi_{t-1} + rho*sin(lambda)*psi*_{t-1} + kappa_t
*   psi*_t = -rho*sin(lambda)*psi_{t-1} + rho*cos(lambda)*psi*_{t-1} + kappa*_t
*
* Stata ucm syntax:
*   model(rwalk): random-walk level component
*   cycle(1): stochastic cycle (period and damping estimated)
*
* Key outputs: rho, lambda_c, period, extracted cycle component
*
* Dados: Brazil IPCA 12-month inflation, 2000-2023 (monthly)
* =============================================================================

clear all
set more off

display "=== Stochastic Cycle Validation: Brazil IPCA (Stata ucm) ==="
display ""

* --- Load data ---
import delimited using "../../data/brazil_ipca.csv", clear

* Parse date and set monthly time series
generate date_stata = date(date, "YMD")
format date_stata %td
generate mdate = mofd(date_stata)
format mdate %tm
tsset mdate

summarize ipca_12m
local n_obs = r(N)
display "Observations: `n_obs'"
display "Frequency: monthly"
display ""

* =============================================================================
* Model: UCM with level and stochastic cycle
* =============================================================================
* model(rwalk): random-walk level (no slope)
* cycle(1): one stochastic cycle
*   - Stata estimates the cycle period and damping factor (rho) internally
*   - rho < 1 gives damped (stationary) cycle; rho = 1 is undamped
* =============================================================================

display "--- Fitting UCM: level + stochastic cycle ---"
display ""

ucm ipca_12m, model(rwalk) cycle(1)

* Store estimation results
estimates store cycle_model

* --- Extract estimated parameters ---
display ""
display "--- Estimated Parameters ---"

* Variance of observation disturbance (irregular)
local sigma2_obs = exp(2 * _b[/ln_var(epsilon)])
display "sigma2_obs   = " %9.6f `sigma2_obs'

* Variance of level disturbance
local sigma2_level = exp(2 * _b[/ln_var(level)])
display "sigma2_level = " %9.6f `sigma2_level'

* Variance of cycle disturbance
local sigma2_cycle = exp(2 * _b[/ln_var(cycle)])
display "sigma2_cycle = " %9.6f `sigma2_cycle'

* Cycle period (estimated)
* ucm parameterizes period via ln_p; period = exp(ln_p)
local lambda_c = _b[/ln_p1.cycle]
local period_months = exp(`lambda_c')
local period_years = `period_months' / 12
display "period       = " %6.2f `period_months' " months (" %5.2f `period_years' " years)"

* Cycle frequency in radians
local freq_rad = 2 * _pi / `period_months'
display "lambda_c     = " %9.6f `freq_rad' " rad"

* Damping factor (rho)
local rho = tanh(_b[/atanh_rho1.cycle])
display "rho (damping)= " %9.6f `rho'

* Log-likelihood
local loglik = e(ll)
display "logLik       = " %12.4f `loglik'

* =============================================================================
* Extract smoothed components
* =============================================================================
display ""
display "--- Extracting Smoothed Components ---"

* Predict smoothed level
predict level_smooth, level equation(level)

* Predict smoothed cycle
predict cycle_smooth, cycle equation(cycle)

* Predict full smoothed signal
predict yhat, xb

* Irregular component
generate irregular = ipca_12m - level_smooth - cycle_smooth

* Cycle amplitude (approximate from predicted cycle values)
* Note: Stata ucm does not directly output psi* (conjugate cycle state),
* so we compute amplitude from the smoothed cycle directly.
generate amplitude = abs(cycle_smooth)

* Summary statistics
summarize level_smooth cycle_smooth amplitude irregular

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

replace parameter = "sigma2_cycle"     in 3
replace value = `sigma2_cycle'         in 3

replace parameter = "rho"              in 4
replace value = `rho'                  in 4

replace parameter = "lambda_c"         in 5
replace value = `freq_rad'             in 5

replace parameter = "period_months"    in 6
replace value = `period_months'        in 6

replace parameter = "period_years"     in 7
replace value = `period_years'         in 7

replace parameter = "loglik"           in 8
replace value = `loglik'               in 8

export delimited using "results_stochastic_cycle_params.csv", replace
display "Saved: results_stochastic_cycle_params.csv"
restore

* =============================================================================
* Export decomposition to CSV
* =============================================================================
preserve
keep mdate ipca_12m level_smooth cycle_smooth irregular amplitude
generate date_str = string(dofm(mdate), "%tdCCYY-NN-DD")
order date_str ipca_12m level_smooth cycle_smooth amplitude irregular
rename ipca_12m y
rename date_str date

export delimited using "results_stochastic_cycle_decomposition.csv", replace
display "Saved: results_stochastic_cycle_decomposition.csv"
restore

display ""
display "Stochastic Cycle validation complete."
