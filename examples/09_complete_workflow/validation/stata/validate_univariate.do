*! validate_univariate.do
*! ============================================================================
*!  Validacao: complete univariate workflow - UK driver deaths (log scale)
*!  Stata commands: sspace, ucm
*!
*!  Workflow replicado do notebook 01_univariate_workflow.ipynb e do script
*!  R validate_univariate.R:
*!    1. Carregar uk_drivers.csv, aplicar log-transform e construir dummy
*!       belt_law (Feb 1983 onwards).
*!    2. Ajustar 4 candidatos (sspace + ucm):
*!         - Model 1: Local Level                (sspace)
*!         - Model 2: Local Linear Trend (LLT)   (sspace)
*!         - Model 3: BSM (LLT + seasonal 12)    (ucm)
*!         - Model 4: BSM + intervention         (ucm)
*!    3. Tabela AIC / BIC / logLik via -estat ic-.
*!    4. Decomposicao do melhor modelo via -predict-.
*!    5. Diagnosticos: Ljung-Box (wntestq) e Jarque-Bera (sktest).
*!    6. Forecast 24 meses via -predict, dynamic-.
*!    7. Exportar tudo em CSV para comparacao com kalmanbox / R.
*!
*!  Uso (a partir de examples/09_complete_workflow/validation/stata/):
*!    stata -b do validate_univariate.do
*! ============================================================================

version 14
clear all
set more off
capture log close

local OUT_PREFIX "stata_univariate"

* ---------------------------------------------------------------------------
* 1. Dados
* ---------------------------------------------------------------------------

import delimited using "../../data/uk_drivers.csv", varnames(1) clear stringcols(1)

gen date_d = date(date, "YMD")
format date_d %td
gen mdate = mofd(date_d)
format mdate %tm
tsset mdate

gen y = log(deaths)
gen belt_law = (date_d >= mdy(2,1,1983))

count
local n = r(N)
quietly summarize mdate
local first_m = r(min)
local last_m  = r(max)

display as text "============================================================="
display as text "  Univariate workflow validation (Stata sspace + ucm)"
display as text "  UK driver deaths (log scale)"
display as text "============================================================="
display as text "Observations: " `n'
display as text "Intervention obs (belt_law): " as result `=sum(belt_law)'

* ---------------------------------------------------------------------------
* 2. Ajuste dos 4 modelos candidatos
* ---------------------------------------------------------------------------

display _newline as text "--- Fitting 4 candidate models (sspace + ucm) ---"

* ---- Model 1: Local Level (sspace) ----
*   alpha_t = alpha_{t-1} + eta_t      (state, var_eta)
*   y_t     = alpha_t     + eps_t      (obs,   var_eps)
display _newline as text "[Model 1] Local Level (sspace)"
capture noisily sspace                                                    ///
    (alpha L.alpha, state noerror)                                        ///
    (y alpha, noconstant noerror),                                        ///
    covstate(diagonal) covobserved(diagonal)                              ///
    technique(bfgs) iterate(200)
local rc1 = _rc
if `rc1' == 0 {
    estimates store M1
    estat ic
    matrix ic1 = r(S)
    scalar ll1   = ic1[1, 4]
    scalar k1    = ic1[1, 5]
    scalar aic1  = ic1[1, 6]
    scalar bic1  = ic1[1, 7]
}
else {
    scalar ll1 = .
    scalar k1  = .
    scalar aic1 = .
    scalar bic1 = .
    display as error "  -> sspace failed for Model 1 (rc=`rc1')"
}

* ---- Model 2: Local Linear Trend (sspace) ----
*   level_t = level_{t-1} + slope_{t-1} + eta1_t
*   slope_t = slope_{t-1}                + eta2_t
*   y_t     = level_t                    + eps_t
display _newline as text "[Model 2] Local Linear Trend (sspace)"
capture noisily sspace                                                    ///
    (lvl L.lvl L.slp, state noerror)                                      ///
    (slp L.slp, state noerror)                                            ///
    (y lvl, noconstant noerror),                                          ///
    covstate(diagonal) covobserved(diagonal)                              ///
    technique(bfgs) iterate(300)
local rc2 = _rc
if `rc2' == 0 {
    estimates store M2
    estat ic
    matrix ic2 = r(S)
    scalar ll2  = ic2[1, 4]
    scalar k2   = ic2[1, 5]
    scalar aic2 = ic2[1, 6]
    scalar bic2 = ic2[1, 7]
}
else {
    scalar ll2 = .
    scalar k2  = .
    scalar aic2 = .
    scalar bic2 = .
    display as error "  -> sspace failed for Model 2 (rc=`rc2')"
}

* ---- Model 3: BSM (ucm) - LLT + seasonal(12) ----
display _newline as text "[Model 3] BSM = LLT + seasonal(12) (ucm)"
capture noisily ucm y, model(ltrend) seasonal(12) iterate(300)
local rc3 = _rc
if `rc3' == 0 {
    estimates store M3
    estat ic
    matrix ic3 = r(S)
    scalar ll3  = ic3[1, 4]
    scalar k3   = ic3[1, 5]
    scalar aic3 = ic3[1, 6]
    scalar bic3 = ic3[1, 7]
}
else {
    scalar ll3 = .
    scalar k3  = .
    scalar aic3 = .
    scalar bic3 = .
    display as error "  -> ucm failed for Model 3 (rc=`rc3')"
}

* ---- Model 4: BSM + intervention (ucm) ----
display _newline as text "[Model 4] BSM + intervention belt_law (ucm)"
capture noisily ucm y belt_law, model(ltrend) seasonal(12) iterate(300)
local rc4 = _rc
if `rc4' == 0 {
    estimates store M4
    estat ic
    matrix ic4 = r(S)
    scalar ll4  = ic4[1, 4]
    scalar k4   = ic4[1, 5]
    scalar aic4 = ic4[1, 6]
    scalar bic4 = ic4[1, 7]
}
else {
    scalar ll4 = .
    scalar k4  = .
    scalar aic4 = .
    scalar bic4 = .
    display as error "  -> ucm failed for Model 4 (rc=`rc4')"
}

* ---------------------------------------------------------------------------
* 3. Tabela de comparacao de modelos (AIC/BIC/logLik)
* ---------------------------------------------------------------------------

preserve
    clear
    set obs 4
    gen str20 model    = ""
    gen      k_params  = .
    gen      loglik    = .
    gen      aic       = .
    gen      bic       = .

    replace model = "LocalLevel"        in 1
    replace model = "LocalLinearTrend"  in 2
    replace model = "BSM"               in 3
    replace model = "BSM+intervention"  in 4

    replace k_params = k1   in 1
    replace k_params = k2   in 2
    replace k_params = k3   in 3
    replace k_params = k4   in 4

    replace loglik = ll1  in 1
    replace loglik = ll2  in 2
    replace loglik = ll3  in 3
    replace loglik = ll4  in 4

    replace aic = aic1  in 1
    replace aic = aic2  in 2
    replace aic = aic3  in 3
    replace aic = aic4  in 4

    replace bic = bic1  in 1
    replace bic = bic2  in 2
    replace bic = bic3  in 3
    replace bic = bic4  in 4

    quietly summarize aic
    gen delta_aic = aic - r(min)

    gsort aic
    list, clean noobs
    export delimited "`OUT_PREFIX'_model_comparison.csv", replace
restore
display as text "Saved: `OUT_PREFIX'_model_comparison.csv"

* ---------------------------------------------------------------------------
* 4. Selecao do melhor modelo (menor AIC)
* ---------------------------------------------------------------------------

local best_name ""
local best_aic = .
foreach pair in "LocalLevel:M1:`=aic1'" "LocalLinearTrend:M2:`=aic2'" ///
                "BSM:M3:`=aic3'" "BSM+intervention:M4:`=aic4'" {
    local name : word 1 of `=subinstr("`pair'", ":", " ", .)'
    local est  : word 2 of `=subinstr("`pair'", ":", " ", .)'
    local val  : word 3 of `=subinstr("`pair'", ":", " ", .)'
    if "`val'" != "." & ("`best_aic'" == "." | real("`val'") < `best_aic') {
        local best_aic  = real("`val'")
        local best_name "`name'"
        local best_est  "`est'"
    }
}

display _newline as text "Selected best model (min AIC): " as result "`best_name'"

estimates restore `best_est'

* ---------------------------------------------------------------------------
* 5. Decomposicao via -predict-
*    Para ucm: components(level smooth seasonal); para sspace: states.
* ---------------------------------------------------------------------------

capture drop level_hat slope_hat seasonal_hat intervention_hat irregular_hat signal_hat

if "`best_est'" == "M3" | "`best_est'" == "M4" {
    capture predict double level_hat,    smethod(smooth) component(trend)
    if _rc capture predict double level_hat, smethod(smooth) component(level)
    capture predict double slope_hat,    smethod(smooth) component(slope)
    capture predict double seasonal_hat, smethod(smooth) component(seasonal)
    capture predict double signal_hat,   smethod(smooth) xb
}
else {
    * sspace: melhor modelo (raro neste workflow); usar -predict, states-
    capture predict double level_hat, states equation(#1) smethod(smooth)
    capture predict double slope_hat, states equation(#2) smethod(smooth)
    gen double seasonal_hat = 0
    capture predict double signal_hat, smethod(smooth) xb
}

if "`best_est'" == "M4" {
    * Coeficiente da intervencao (regressor exogeno em ucm)
    matrix b_int = e(b)
    capture confirm matrix b_int
    scalar coef_belt = .
    capture {
        local cn : colnames b_int
        local i = 0
        foreach cname of local cn {
            local ++i
            if "`cname'" == "y:belt_law" | "`cname'" == "belt_law" {
                scalar coef_belt = b_int[1, `i']
            }
        }
    }
    if missing(coef_belt) scalar coef_belt = 0
    gen double intervention_hat = belt_law * coef_belt
}
else {
    gen double intervention_hat = 0
}

* slope_hat pode nao existir se modelo for so local level
capture confirm variable slope_hat
if _rc gen double slope_hat = 0
capture confirm variable seasonal_hat
if _rc gen double seasonal_hat = 0
capture confirm variable signal_hat
if _rc gen double signal_hat = level_hat + slope_hat + seasonal_hat + intervention_hat

gen double irregular_hat = y - signal_hat

preserve
    keep date y level_hat slope_hat seasonal_hat intervention_hat irregular_hat
    rename (y level_hat slope_hat seasonal_hat intervention_hat irregular_hat) ///
           (observed level slope seasonal intervention irregular)
    export delimited "`OUT_PREFIX'_decomposition.csv", replace
restore
display as text "Saved: `OUT_PREFIX'_decomposition.csv"

* ---------------------------------------------------------------------------
* 6. Diagnosticos: residuos padronizados, Ljung-Box, Jarque-Bera
* ---------------------------------------------------------------------------

capture drop resid_std resid_raw
if "`best_est'" == "M3" | "`best_est'" == "M4" {
    capture predict double resid_raw, residuals
    capture predict double resid_std, rstandard
    if _rc {
        * fallback: padronizar manualmente
        quietly summarize resid_raw
        gen double resid_std = (resid_raw - r(mean)) / r(sd)
    }
}
else {
    * sspace: usar innovations
    capture predict double resid_raw, innovations equation(#1)
    quietly summarize resid_raw
    gen double resid_std = (resid_raw - r(mean)) / r(sd)
}

local lags 10 12 20 24
preserve
    clear
    set obs 6
    gen str24 test       = ""
    gen double statistic = .
    gen double p_value   = .
    gen byte   reject_5pct = .
restore

tempname memhold
tempfile diag_file
postfile `memhold' str24 test double(statistic p_value) byte reject_5pct ///
    using "`diag_file'", replace

foreach L in `lags' {
    quietly wntestq resid_std, lags(`L')
    post `memhold' ("Ljung-Box(`L')") (r(stat)) (r(p)) (r(p) < 0.05)
}

* Jarque-Bera via sktest (combined p-value, chi2(2))
quietly sktest resid_std
* sktest retorna combined chi2 e p em r(chi2) / r(P_chi2) na ultima linha
return list
local jb_chi2 = r(chi2)
local jb_p    = r(P_chi2)
post `memhold' ("Jarque-Bera (sktest)") (`jb_chi2') (`jb_p') (`jb_p' < 0.05)

* Shapiro-Wilk
quietly swilk resid_std
local sw_stat = r(W)
local sw_p    = r(p)
post `memhold' ("Shapiro-Wilk") (`sw_stat') (`sw_p') (`sw_p' < 0.05)

postclose `memhold'

preserve
    use "`diag_file'", clear
    list, clean noobs
    export delimited "`OUT_PREFIX'_diagnostics.csv", replace
restore
display as text "Saved: `OUT_PREFIX'_diagnostics.csv"

preserve
    keep date resid_std
    rename resid_std standardized_residual
    export delimited "`OUT_PREFIX'_residuals.csv", replace
restore
display as text "Saved: `OUT_PREFIX'_residuals.csv"

* ---------------------------------------------------------------------------
* 7. Forecast 24 passos via -predict, dynamic-
* ---------------------------------------------------------------------------

local h = 24

* Estender o dataset com 24 observacoes futuras
preserve
    quietly summarize mdate
    local last_m = r(max)
    local new_n = `n' + `h'
    set obs `new_n'

    * Preencher mdate sequencialmente
    forvalues i = `=`n' + 1'/`new_n' {
        local k = `i' - `n'
        replace mdate = `last_m' + `k' in `i'
    }
    format mdate %tm
    tsset mdate

    * belt_law: assumir 1 apos 1983 (sempre 1 no futuro)
    capture confirm variable belt_law
    if _rc == 0 {
        replace belt_law = 1 if mdate > `last_m'
    }

    * Forecast: predict y_hat com dynamic
    capture drop y_fc y_fc_se y_fc_lo y_fc_up
    if "`best_est'" == "M3" | "`best_est'" == "M4" {
        capture predict double y_fc, dynamic(`=`last_m' + 1') xb
        capture predict double y_fc_se, dynamic(`=`last_m' + 1') rmse
    }
    else {
        capture predict double y_fc, dynamic(`=`last_m' + 1') xb
        capture predict double y_fc_se, dynamic(`=`last_m' + 1') rmse
    }
    if _rc {
        * fallback: forecast estatico ate o final
        gen double y_fc = level_hat + slope_hat + seasonal_hat + intervention_hat
        gen double y_fc_se = .
    }

    * Intervalo de 95%
    gen double y_fc_lo = y_fc - 1.959964 * y_fc_se
    gen double y_fc_up = y_fc + 1.959964 * y_fc_se

    keep if mdate > `last_m'
    keep mdate y_fc y_fc_lo y_fc_up
    rename (mdate y_fc y_fc_lo y_fc_up) (mdate mean lower upper)
    gen str10 date = string(dofm(mdate), "%tdCCYY-NN-DD")
    order date mean lower upper
    drop mdate
    export delimited "`OUT_PREFIX'_forecast.csv", replace
restore
display as text "Saved: `OUT_PREFIX'_forecast.csv"

* ---------------------------------------------------------------------------
* 8. Parametros do melhor modelo
* ---------------------------------------------------------------------------

estimates restore `best_est'
matrix b_best  = e(b)
matrix V_best  = e(V)
local cn : colnames b_best
local nc = colsof(b_best)

preserve
    clear
    set obs `nc'
    gen str40 parameter = ""
    gen double estimate  = .
    gen double std_error = .
    forvalues i = 1/`nc' {
        local nm : word `i' of `cn'
        replace parameter = "`nm'"             in `i'
        replace estimate  = b_best[1, `i']     in `i'
        replace std_error = sqrt(V_best[`i', `i']) in `i'
    }
    export delimited "`OUT_PREFIX'_best_params.csv", replace
restore
display as text "Saved: `OUT_PREFIX'_best_params.csv"

* ---------------------------------------------------------------------------
* 9. Sumario textual
* ---------------------------------------------------------------------------

tempname sumh
file open `sumh' using "`OUT_PREFIX'_best_summary.txt", write replace
file write `sumh' "======================================================================" _n
file write `sumh' "  Best model (Stata): `best_name' [`best_est']" _n
file write `sumh' "======================================================================" _n
file write `sumh' "Log-Likelihood:    " %12.4f (`best_aic') _n
* (Re-extract via estat ic for clean numbers)
quietly estat ic
matrix icb = r(S)
file write `sumh' "Log-Likelihood:    " %12.4f (icb[1,4]) _n
file write `sumh' "AIC:               " %12.4f (icb[1,6]) _n
file write `sumh' "BIC:               " %12.4f (icb[1,7]) _n
file write `sumh' "k_params:          " %12.0f (icb[1,5]) _n
file write `sumh' "n_obs:             " %12.0f (`n')      _n
file close `sumh'
display as text "Saved: `OUT_PREFIX'_best_summary.txt"

* ---------------------------------------------------------------------------
* 10. Tabela de comparacao kalmanbox vs Stata (e R, se disponivel)
* ---------------------------------------------------------------------------

local kbox_path "../../output/univariate_model_comparison.csv"
local r_path    "../R/r_univariate_model_comparison.csv"

capture confirm file "`kbox_path'"
local has_kbox = (_rc == 0)
capture confirm file "`r_path'"
local has_r = (_rc == 0)

if `has_kbox' {
    preserve
        import delimited using "`kbox_path'", varnames(1) clear
        rename (loglik aic bic) (loglik_kalmanbox aic_kalmanbox bic_kalmanbox)
        keep model loglik_kalmanbox aic_kalmanbox bic_kalmanbox
        tempfile kbox_tmp
        save "`kbox_tmp'", replace
    restore

    preserve
        import delimited using "`OUT_PREFIX'_model_comparison.csv", varnames(1) clear
        rename (loglik aic bic) (loglik_stata aic_stata bic_stata)
        keep model loglik_stata aic_stata bic_stata
        merge 1:1 model using "`kbox_tmp'", nogen

        if `has_r' {
            preserve
                import delimited using "`r_path'", varnames(1) clear
                rename (loglik aic bic) (loglik_R aic_R bic_R)
                keep model loglik_R aic_R bic_R
                tempfile r_tmp
                save "`r_tmp'", replace
            restore
            merge 1:1 model using "`r_tmp'", nogen
        }

        gen d_loglik_stata_vs_kbox = loglik_stata - loglik_kalmanbox
        gen d_aic_stata_vs_kbox    = aic_stata    - aic_kalmanbox
        gen d_bic_stata_vs_kbox    = bic_stata    - bic_kalmanbox

        list, clean noobs
        export delimited "`OUT_PREFIX'_vs_kalmanbox.csv", replace
    restore
    display as text "Saved: `OUT_PREFIX'_vs_kalmanbox.csv"
}

display _newline as text "Univariate Stata validation complete."
