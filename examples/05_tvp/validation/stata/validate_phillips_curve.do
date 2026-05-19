* =============================================================================
* Validacao: Phillips Curve TVP via sspace (Stata)
*
* Modelo accelerationist (Staiger-Stock-Watson):
*   Delta pi_t = alpha_t + beta_t * u_t + eps_t,  eps_t ~ N(0, sigma2_eps)
*   alpha_t    = alpha_{t-1} + eta_a_t,           eta_a ~ N(0, w_alpha)
*   beta_t     = beta_{t-1}  + eta_b_t,           eta_b ~ N(0, w_beta)
*
*   NAIRU_t = -alpha_t / beta_t
*
* =============================================================================
* LIMITACAO IMPORTANTE DO Stata sspace PARA PHILLIPS CURVE + NAIRU TV
* =============================================================================
* Na especificacao original da Phillips curve com NAIRU time-varying:
*   Delta pi_t = -beta_t * NAIRU_t + beta_t * u_t + eps_t
*              =  beta_t * (u_t - NAIRU_t) + eps_t
*
* Essa formulacao envolve o produto de dois estados (beta_t * NAIRU_t), o que
* a torna NAO-LINEAR no estado. O comando `sspace` do Stata so suporta modelos
* lineares-Gaussianos, entao NAO e possivel especifica-la diretamente.
*
* Alternativas validas em Stata:
*   (a) Reparametrizar com alpha_t = -beta_t * NAIRU_t como um estado linear
*       time-varying independente (abordagem SSW adotada aqui). NAIRU e recuperado
*       post-hoc via delta method: NAIRU_t = -alpha_t / beta_t.
*   (b) Fixar NAIRU = constante e estimar so a slope como TV (mais simples).
*   (c) Usar sspace em matrix-form (sspace using ... ) e implementar o produto via
*       linearizacao (EKF/UKF) - nao trivial, requer codigo Mata customizado.
*
* Este script usa (a) + (b) para comparacao:
*   Model A: ambos alpha_t e beta_t random walk (SSW puro)
*   Model B: apenas beta_t random walk, alpha fixo (NAIRU fixo)
*
* Dataset: us_inflation_unemployment.csv (quarterly 1960Q1-2023Q4)
* =============================================================================

clear all
set more off

display "=== Phillips Curve TVP Validation: Delta pi ~ 1 + u (Stata sspace) ==="
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

* Primeira diferenca da inflacao (acceleracionista): Delta pi_t = pi_t - pi_{t-1}
generate dpi = D.inflation
drop if missing(dpi)

* Loading unitario para alpha time-varying
generate ones = 1

local n_obs = _N
display "Observations after differencing: `n_obs' quarters"
display ""

* ---------------------------------------------------------------------------
* Benchmark OLS: Delta pi = alpha + beta * u
* ---------------------------------------------------------------------------
display "--- OLS benchmark (constant NAIRU) ---"
regress dpi unemployment

local ols_alpha  = _b[_cons]
local ols_beta   = _b[unemployment]
local sigma2_ols = e(rmse)^2
local loglik_ols = e(ll)

local ols_nairu = .
if abs(`ols_beta') > 1e-8 {
    local ols_nairu = -`ols_alpha' / `ols_beta'
}

display ""
display "OLS alpha           = " %10.4f `ols_alpha'
display "OLS beta (slope u)  = " %10.4f `ols_beta'
display "OLS NAIRU           = " %10.4f `ols_nairu' "  (% unemployment)"
display "OLS logLik          = " %10.4f `loglik_ols'
display ""

* ===========================================================================
* Model A: SSW puro -- alpha_t e beta_t ambos random walk
* ===========================================================================
display "--- Model A: sspace SSW (alpha_t, beta_t random walk) ---"
display ""

constraint 1 [dpi]alpha = 1
constraint 2 [dpi]beta  = 1

sspace (alpha = L.alpha, state noconstant) ///
       (beta  = L.beta,  state noconstant) ///
       (dpi = {alpha}*ones + {beta}*unemployment, noconstant), ///
       constraints(1 2) covstate(diagonal) ///
       difficult iterate(500)

estimates store pc_ssw

local loglik_A      = e(ll)
local sigma2_obs_A  = exp(2 * _b[/ln_var(dpi)])
local sigma2_alphaA = exp(2 * _b[/ln_var(alpha)])
local sigma2_betaA  = exp(2 * _b[/ln_var(beta)])

display ""
display "--- Model A Variance Parameters ---"
display "sigma2_obs     = " %12.6f `sigma2_obs_A'
display "sigma2_alpha   = " %12.6f `sigma2_alphaA'
display "sigma2_beta    = " %12.6f `sigma2_betaA'
display "logLik (SSW)   = " %12.4f `loglik_A'
display ""

predict alpha_sm, state equation(alpha) smethod(smooth)
predict beta_sm,  state equation(beta)  smethod(smooth)

* NAIRU_t = -alpha_t / beta_t (tratamento numerico para beta ~ 0)
generate nairu_t = .
replace nairu_t = -alpha_sm / beta_sm if abs(beta_sm) > 1e-3

summarize nairu_t
local nairu_mean = r(mean)
local nairu_min  = r(min)
local nairu_max  = r(max)

* Fracao em banda razoavel (3% a 8%) e fracao com slope negativo (esperado)
generate in_band   = (nairu_t >= 3 & nairu_t <= 8) if !missing(nairu_t)
generate neg_slope = (beta_sm < 0)
summarize in_band
local frac_in_band = r(mean)
summarize neg_slope
local frac_neg_beta = r(mean)

display "NAIRU (Model A): mean=" %6.3f `nairu_mean' ///
        "  min=" %6.3f `nairu_min' "  max=" %6.3f `nairu_max'
display "Fracao em [3, 8]: " %6.3f `frac_in_band'
display "Fracao com beta_t < 0: " %6.3f `frac_neg_beta'
display ""

* Comparacao pre vs pos 1990 (flattening literatura)
summarize beta_sm if qdate < yq(1990, 1)
local beta_pre_1990 = r(mean)
summarize beta_sm if qdate >= yq(1990, 1)
local beta_post_1990 = r(mean)

display "Slope pre-1990  mean: " %8.4f `beta_pre_1990'
display "Slope post-1990 mean: " %8.4f `beta_post_1990'
display ""

* ---------------------------------------------------------------------------
* Exporta resultados Model A
* ---------------------------------------------------------------------------
preserve
keep date alpha_sm beta_sm nairu_t
rename alpha_sm alpha
rename beta_sm  beta
rename nairu_t  nairu
order date alpha beta nairu
export delimited using "results_phillips_curve_stata.csv", replace
display "Saved: results_phillips_curve_stata.csv"
restore

* Limpa variaveis para reutilizar nomes no Model B
drop alpha_sm beta_sm nairu_t in_band neg_slope

* ===========================================================================
* Model B: NAIRU fixo -- so beta_t time-varying
* ===========================================================================
display ""
display "--- Model B: sspace Phillips (NAIRU fixo, so slope_t random walk) ---"
display "  Nota: sspace NAO lida com produto de estados (beta_t * NAIRU_t),"
display "  entao NAIRU e fixo aqui. Para NAIRU time-varying, ver Model A"
display "  (reparametrizacao linear alpha_t = -beta_t * NAIRU_t)."
display ""

constraint 3 [dpi]beta = 1

sspace (beta = L.beta, state noconstant) ///
       (dpi = {beta}*unemployment), ///
       constraints(3) covstate(diagonal) ///
       difficult iterate(500)

estimates store pc_slope_tv

local loglik_B      = e(ll)
local sigma2_obs_B  = exp(2 * _b[/ln_var(dpi)])
local sigma2_betaB  = exp(2 * _b[/ln_var(beta)])
local alpha_fix_B   = _b[dpi:_cons]

local nairu_fix_B = .
if abs(`alpha_fix_B') > 1e-12 {
    * usar slope medio como aproximacao; NAIRU = -alpha / mean(beta)
    predict beta_sm, state equation(beta) smethod(smooth)
    summarize beta_sm
    local beta_mean_B = r(mean)
    if abs(`beta_mean_B') > 1e-8 {
        local nairu_fix_B = -`alpha_fix_B' / `beta_mean_B'
    }
}

display ""
display "--- Model B Variance Parameters ---"
display "sigma2_obs              = " %12.6f `sigma2_obs_B'
display "sigma2_beta             = " %12.6f `sigma2_betaB'
display "alpha (fixo)            = " %12.6f `alpha_fix_B'
display "NAIRU (fixo, aprox)     = " %12.6f `nairu_fix_B'
display "logLik (slope-only TV)  = " %12.4f `loglik_B'
display ""

* ===========================================================================
* Comparacao Model A vs Model B
* ===========================================================================
display "--- Model Comparison ---"
display "Metric              Model A (SSW)    Model B (NAIRU fixo)"
display "logLik        " %12.4f `loglik_A' "   " %12.4f `loglik_B'
display "sigma2_obs    " %12.6f `sigma2_obs_A' "   " %12.6f `sigma2_obs_B'
display ""

* ---------------------------------------------------------------------------
* Exporta sumario
* ---------------------------------------------------------------------------
preserve
clear
set obs 20
generate str40 metric = ""
generate double value = .

replace metric = "modelA_logLik"           in 1
replace value  = `loglik_A'                in 1
replace metric = "modelA_sigma2_obs"       in 2
replace value  = `sigma2_obs_A'            in 2
replace metric = "modelA_sigma2_alpha"     in 3
replace value  = `sigma2_alphaA'           in 3
replace metric = "modelA_sigma2_beta"      in 4
replace value  = `sigma2_betaA'            in 4
replace metric = "modelA_mean_nairu"       in 5
replace value  = `nairu_mean'              in 5
replace metric = "modelA_min_nairu"        in 6
replace value  = `nairu_min'               in 6
replace metric = "modelA_max_nairu"        in 7
replace value  = `nairu_max'               in 7
replace metric = "modelA_frac_nairu_3_8"   in 8
replace value  = `frac_in_band'            in 8
replace metric = "modelA_frac_neg_beta"    in 9
replace value  = `frac_neg_beta'           in 9
replace metric = "modelA_beta_pre_1990"    in 10
replace value  = `beta_pre_1990'           in 10
replace metric = "modelA_beta_post_1990"   in 11
replace value  = `beta_post_1990'          in 11
replace metric = "modelB_logLik"           in 12
replace value  = `loglik_B'                in 12
replace metric = "modelB_sigma2_obs"       in 13
replace value  = `sigma2_obs_B'            in 13
replace metric = "modelB_sigma2_beta"      in 14
replace value  = `sigma2_betaB'            in 14
replace metric = "modelB_alpha_fixed"      in 15
replace value  = `alpha_fix_B'             in 15
replace metric = "modelB_nairu_fixed"      in 16
replace value  = `nairu_fix_B'             in 16
replace metric = "ols_alpha"               in 17
replace value  = `ols_alpha'               in 17
replace metric = "ols_beta"                in 18
replace value  = `ols_beta'                in 18
replace metric = "ols_nairu"               in 19
replace value  = `ols_nairu'               in 19
replace metric = "n_obs"                   in 20
replace value  = `n_obs'                   in 20

drop if missing(metric) | metric == ""
export delimited using "results_phillips_curve_stata_summary.csv", replace
display "Saved: results_phillips_curve_stata_summary.csv"
restore

display ""
display "Phillips curve sspace validation complete."
