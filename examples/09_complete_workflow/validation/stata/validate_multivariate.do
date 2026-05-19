*! validate_multivariate.do
*! ============================================================================
*!  Validacao: complete multivariate workflow - US macro panel
*!  Stata command: dfactor
*!
*!  Workflow replicado do notebook 02_multivariate_workflow.ipynb e do script
*!  R validate_multivariate.R:
*!    1. Carregar us_macro_panel.csv (T x 15 series mensais).
*!    2. Padronizar (mean=0, sd=1) cada serie (StandardScaler).
*!    3. Ajustar Dynamic Factor Model com K = 1, 2, 3 fatores via -dfactor-.
*!    4. Selecionar K via BIC (menor BIC) usando -estat ic-.
*!    5. Extrair fatores suavizados via -predict-, cargas (Lambda),
*!       AR(1) coefs e variancias idiossincraticas.
*!    6. Variance decomposition (R^2 por serie e por fator).
*!    7. Exportar tudo em CSV para comparacao com kalmanbox / R (MARSS).
*!
*!  Especificacao DFM (mesma do notebook):
*!     y_t = Z * f_t + v_t,   v_t ~ N(0, R)   (R diagonal)
*!     f_t = Phi * f_{t-1} + w_t, w_t ~ N(0, I_K) (Phi diagonal AR(1))
*!  Identificacao: triangular superior zero em Z nas primeiras K-1 linhas
*!  (Stata `dfactor` impoe sua propria normalizacao internamente).
*!
*!  Uso (a partir de examples/09_complete_workflow/validation/stata/):
*!    stata -b do validate_multivariate.do
*! ============================================================================

version 14
clear all
set more off
capture log close

local OUT_PREFIX "stata_multivariate"

* ---------------------------------------------------------------------------
* 1. Dados
* ---------------------------------------------------------------------------

import delimited using "../../data/us_macro_panel.csv", varnames(1) clear stringcols(1)

gen date_d = date(date, "YMD")
format date_d %td
gen mdate = mofd(date_d)
format mdate %tm
tsset mdate

* Lista de series (todas exceto date / date_d / mdate)
ds, has(type numeric)
local allnum `r(varlist)'
local series ""
foreach v of local allnum {
    if "`v'" != "mdate" {
        local series "`series' `v'"
    }
}
local series : list clean series
local N : word count `series'

count
local T = r(N)

display as text "============================================================="
display as text "  Multivariate workflow validation (Stata dfactor)"
display as text "  US macro panel"
display as text "============================================================="
display as text "Panel dimensions: T=" `T' " x N=" `N'
display as text "Series: `series'"

* ---------------------------------------------------------------------------
* 2. Padronizar (mean=0, sd=1) cada serie
* ---------------------------------------------------------------------------

local std_vars ""
foreach v of local series {
    quietly summarize `v'
    gen double z_`v' = (`v' - r(mean)) / r(sd)
    local std_vars "`std_vars' z_`v'"
}
local std_vars : list clean std_vars

* ---------------------------------------------------------------------------
* 3. Ajustar DFM com K = 1, 2, 3
* ---------------------------------------------------------------------------

local K_values 1 2 3

foreach K of local K_values {
    display _newline as text "--- Fitting DFM with K = `K' (dfactor) ---"

    * Construir lista de fatores: f1, f2, ..., fK
    local factors ""
    forvalues k = 1/`K' {
        local factors "`factors' f`k'"
    }
    local factors : list clean factors

    capture noisily dfactor                                                ///
        (`std_vars' = , noconstant)                                        ///
        (`factors' = , ar(1)),                                             ///
        iterate(500)

    if _rc == 0 {
        estimates store DFM`K'
        quietly estat ic
        matrix ic`K' = r(S)
        scalar ll`K'  = ic`K'[1, 4]
        scalar k`K'   = ic`K'[1, 5]
        scalar aic`K' = ic`K'[1, 6]
        scalar bic`K' = ic`K'[1, 7]
        display as text "  logLik = " %12.4f (ll`K') ", AIC = " %12.4f (aic`K') ", BIC = " %12.4f (bic`K') ", k = " %4.0f (k`K')
    }
    else {
        scalar ll`K'  = .
        scalar k`K'   = .
        scalar aic`K' = .
        scalar bic`K' = .
        display as error "  -> dfactor failed for K=`K' (rc=" _rc ")"
    }
}

* ---------------------------------------------------------------------------
* 4. Tabela de selecao por BIC
* ---------------------------------------------------------------------------

preserve
    clear
    set obs 3
    gen K        = .
    gen k_params = .
    gen loglik   = .
    gen aic      = .
    gen bic      = .

    replace K = 1 in 1
    replace K = 2 in 2
    replace K = 3 in 3

    replace k_params = k1 in 1
    replace k_params = k2 in 2
    replace k_params = k3 in 3

    replace loglik = ll1 in 1
    replace loglik = ll2 in 2
    replace loglik = ll3 in 3

    replace aic = aic1 in 1
    replace aic = aic2 in 2
    replace aic = aic3 in 3

    replace bic = bic1 in 1
    replace bic = bic2 in 2
    replace bic = bic3 in 3

    list, clean noobs
    export delimited "`OUT_PREFIX'_K_selection.csv", replace
restore
display as text "Saved: `OUT_PREFIX'_K_selection.csv"

* ---------------------------------------------------------------------------
* 5. Selecionar melhor K (menor BIC)
* ---------------------------------------------------------------------------

local best_K = 1
local best_bic = bic1
foreach K of local K_values {
    if !missing(bic`K') & (missing(`best_bic') | bic`K' < `best_bic') {
        local best_K = `K'
        local best_bic = bic`K'
    }
}

display _newline as text "Selected best K (min BIC): K = " as result `best_K'

estimates restore DFM`best_K'

* ---------------------------------------------------------------------------
* 6. Fatores suavizados via -predict-
* ---------------------------------------------------------------------------

* Limpar variaveis previas
forvalues k = 1/3 {
    capture drop f_hat_`k'
    capture drop f_se_`k'
}

forvalues k = 1/`best_K' {
    capture predict double f_hat_`k', factors smethod(smooth) equation(f`k')
    if _rc capture predict double f_hat_`k', latent smethod(smooth) equation(f`k')
    capture predict double f_se_`k', factors smethod(smooth) equation(f`k') stdp
}

preserve
    keep date f_hat_*
    forvalues k = 1/`best_K' {
        rename f_hat_`k' factor_`k'
    }
    capture confirm variable f_se_1
    if _rc == 0 {
        forvalues k = 1/`best_K' {
            capture rename f_se_`k' factor_`k'_se
        }
    }
    export delimited "`OUT_PREFIX'_factors.csv", replace
restore
display as text "Saved: `OUT_PREFIX'_factors.csv"

* ---------------------------------------------------------------------------
* 7. Cargas (Lambda), coefs AR(1), variancias idiossincraticas
* ---------------------------------------------------------------------------

matrix b = e(b)
local cn : colnames b
local nc = colsof(b)

* Construir tabela de loadings (N linhas, K colunas)
preserve
    clear
    set obs `N'
    gen str40 series = ""
    forvalues k = 1/`best_K' {
        gen double f`k' = .
    }

    local i = 0
    foreach s of local series {
        local ++i
        replace series = "`s'" in `i'

        forvalues k = 1/`best_K' {
            * Procurar coluna correspondente a carga (formatos possiveis):
            *   z_<s>:f`k'    ou    z_<s>:L.f`k'   ou    f`k':z_<s>
            local found = 0
            local j = 0
            foreach cname of local cn {
                local ++j
                if "`cname'" == "z_`s':f`k'" | "`cname'" == "z_`s':L.f`k'" | ///
                   "`cname'" == "f`k':z_`s'" {
                    replace f`k' = b[1, `j'] in `i'
                    local found = 1
                }
            }
        }
    }

    list, clean noobs
    export delimited "`OUT_PREFIX'_loadings.csv", replace
restore
display as text "Saved: `OUT_PREFIX'_loadings.csv"

* ---------------------------------------------------------------------------
* 8. Tabela de parametros (estilo kalmanbox: parameter, estimate)
* ---------------------------------------------------------------------------

preserve
    clear
    set obs `nc'
    gen str60 parameter = ""
    gen double estimate = .
    forvalues j = 1/`nc' {
        local nm : word `j' of `cn'
        replace parameter = "`nm'"          in `j'
        replace estimate  = b[1, `j']       in `j'
    }
    export delimited "`OUT_PREFIX'_best_params.csv", replace
restore
display as text "Saved: `OUT_PREFIX'_best_params.csv"

* ---------------------------------------------------------------------------
* 9. Variance decomposition
*    Series sao padronizadas (Var=1).
*    Var(y_i) ~ sum_k Lambda_{ik}^2 * Var(f_k) + R_ii
* ---------------------------------------------------------------------------

* Var amostral dos fatores suavizados
matrix factor_var = J(`best_K', 1, 0)
forvalues k = 1/`best_K' {
    quietly summarize f_hat_`k'
    matrix factor_var[`k', 1] = r(Var)
}

* Loadings: rebuild matrix (N x K)
matrix Lambda = J(`N', `best_K', 0)
local i = 0
foreach s of local series {
    local ++i
    forvalues k = 1/`best_K' {
        local j = 0
        foreach cname of local cn {
            local ++j
            if "`cname'" == "z_`s':f`k'" | "`cname'" == "z_`s':L.f`k'" | ///
               "`cname'" == "f`k':z_`s'" {
                matrix Lambda[`i', `k'] = b[1, `j']
            }
        }
    }
}

* Variancias idiossincraticas (R diagonal)
matrix Rdiag = J(`N', 1, 0)
local i = 0
foreach s of local series {
    local ++i
    local found = 0
    local j = 0
    foreach cname of local cn {
        local ++j
        if regexm("`cname'", "var.*z_`s'") | "`cname'" == "var(z_`s')" {
            matrix Rdiag[`i', 1] = b[1, `j']
            local found = 1
        }
    }
    if !`found' matrix Rdiag[`i', 1] = 1
}

preserve
    clear
    set obs `N'
    gen str40 series = ""
    forvalues k = 1/`best_K' {
        gen double factor_`k' = .
    }
    gen double idiosyncratic = .
    gen double r2            = .

    local i = 0
    foreach s of local series {
        local ++i
        replace series = "`s'" in `i'

        scalar total = 0
        forvalues k = 1/`best_K' {
            scalar share_k_`k' = (Lambda[`i', `k'])^2 * factor_var[`k', 1]
            scalar total = total + share_k_`k'
        }
        scalar idio = Rdiag[`i', 1]
        scalar total = total + idio
        if total > 0 {
            forvalues k = 1/`best_K' {
                replace factor_`k' = share_k_`k' / total in `i'
            }
            replace idiosyncratic = idio / total in `i'
        }
        else {
            forvalues k = 1/`best_K' {
                replace factor_`k' = 0 in `i'
            }
            replace idiosyncratic = 1 in `i'
        }

        scalar r2_i = 1 - idio / total
        replace r2 = r2_i in `i'
    }

    list, clean noobs
    export delimited "`OUT_PREFIX'_variance_decomp.csv", replace
restore
display as text "Saved: `OUT_PREFIX'_variance_decomp.csv"

* ---------------------------------------------------------------------------
* 10. Sumario textual
* ---------------------------------------------------------------------------

tempname sumh
file open `sumh' using "`OUT_PREFIX'_best_summary.txt", write replace
file write `sumh' "======================================================================" _n
file write `sumh' "  Best DFM (Stata): K = `best_K'" _n
file write `sumh' "======================================================================" _n
file write `sumh' "Log-Likelihood:    " %12.4f (ll`best_K')  _n
file write `sumh' "AIC:               " %12.4f (aic`best_K') _n
file write `sumh' "BIC:               " %12.4f (bic`best_K') _n
file write `sumh' "k_params:          " %12.0f (k`best_K')   _n
file write `sumh' "T (obs):           " %12.0f (`T')         _n
file write `sumh' "N (series):        " %12.0f (`N')         _n
file close `sumh'
display as text "Saved: `OUT_PREFIX'_best_summary.txt"

* ---------------------------------------------------------------------------
* 11. Tabela de comparacao kalmanbox vs Stata vs R
* ---------------------------------------------------------------------------

local kbox_path "../../output/multivariate_K_selection.csv"
local r_path    "../R/r_multivariate_K_selection.csv"

capture confirm file "`kbox_path'"
local has_kbox = (_rc == 0)
capture confirm file "`r_path'"
local has_r = (_rc == 0)

if `has_kbox' {
    preserve
        import delimited using "`kbox_path'", varnames(1) clear
        rename (loglik aic bic) (loglik_kalmanbox aic_kalmanbox bic_kalmanbox)
        keep K loglik_kalmanbox aic_kalmanbox bic_kalmanbox
        tempfile kbox_tmp
        save "`kbox_tmp'", replace
    restore

    preserve
        import delimited using "`OUT_PREFIX'_K_selection.csv", varnames(1) clear
        rename (loglik aic bic) (loglik_stata aic_stata bic_stata)
        keep K loglik_stata aic_stata bic_stata
        merge 1:1 K using "`kbox_tmp'", nogen

        if `has_r' {
            preserve
                import delimited using "`r_path'", varnames(1) clear
                rename (loglik aic bic) (loglik_R aic_R bic_R)
                keep K loglik_R aic_R bic_R
                tempfile r_tmp
                save "`r_tmp'", replace
            restore
            merge 1:1 K using "`r_tmp'", nogen
        }

        gen d_loglik_stata_vs_kbox = loglik_stata - loglik_kalmanbox
        gen d_aic_stata_vs_kbox    = aic_stata    - aic_kalmanbox
        gen d_bic_stata_vs_kbox    = bic_stata    - bic_kalmanbox

        list, clean noobs
        export delimited "`OUT_PREFIX'_vs_kalmanbox.csv", replace
    restore
    display as text "Saved: `OUT_PREFIX'_vs_kalmanbox.csv"
}

display _newline as text "Multivariate Stata validation complete."
