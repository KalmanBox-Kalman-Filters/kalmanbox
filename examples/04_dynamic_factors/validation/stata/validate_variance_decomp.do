/*******************************************************************************
* validate_variance_decomp.do
*
* Validacao: Decomposicao de Variancia (R^2 da componente comum) via dfactor
*
* Objetivo:
*   Para cada serie y_i do painel, decompor a variancia amostral em
*   contribuicao dos fatores comuns (f1, f2) vs idiossincratica:
*
*       Var(y_i) = sum_{j} lambda_ij^2 * Var(f_j)                    (comum)
*                 + 2 * sum_{j<l} lambda_ij * lambda_il * Cov(f_j,f_l)
*                 + sigma2_eps_i                                     (idio)
*
*   Com Q = I (identificacao padrao dfactor) os fatores tem variancia
*   teorica Var(f_j) = 1 / (1 - phi_j^2); usamos a variancia amostral dos
*   estados suavizados, que e aproximadamente igual e serve de sanity check.
*
*   R^2_i (comum) = 1 - sigma2_eps_i / Var_implied(y_i)
*
* Especificacao dfactor:
*   Mesma do validate_dfm.do (K=2, AR(1) independentes). Repetimos a
*   estimacao aqui para manter independencia entre scripts.
*
* Nota:
*   Stata dfactor expoe sigma2_eps via _b[/var(z_var)] quando covobserved
*   = diagonal (default). Extraimos via e(b) / e(Sigma) conforme disponivel.
*
* Saidas:
*   - results_variance_decomp.csv          (shares por serie)
*   - results_variance_decomp_loadings.csv (cargas Lambda)
*   - results_variance_decomp_summary.csv  (R^2 medio, Var(f))
*   - validate_variance_decomp.log
*
* Autor: kalmanbox validation suite
* Fase:  FASE4 subfase F4.5
*******************************************************************************/

clear all
set more off
capture log close
log using "validate_variance_decomp.log", replace

display "============================================================"
display " VALIDACAO: Variance Decomposition (R^2) via dfactor"
display " Data: $S_DATE $S_TIME"
display "============================================================"

* --- 1. Carregar painel ---
import delimited using "../../data/us_macro_panel.csv", clear varnames(1)

generate date_stata = date(date, "YMD")
format date_stata %td
generate mdate = mofd(date_stata)
format mdate %tm
tsset mdate

local series gdp_growth industrial_production unemployment payrolls ///
             retail_sales housing_starts consumer_confidence ///
             pmi_manufacturing cpi_inflation pce_inflation ///
             fed_funds_rate sp500_returns term_spread credit_spread ///
             oil_price_change

local N : word count `series'

* --- 2. Padronizar (Var(y_i) = 1 por construcao) ---
foreach v of local series {
    quietly summarize `v'
    quietly generate z_`v' = (`v' - r(mean)) / r(sd)
}

local zseries
foreach v of local series {
    local zseries `zseries' z_`v'
}

* =============================================================================
* 3. Estimar DFM K=2 (mesma especificacao do validate_dfm.do)
* =============================================================================

display ""
display "--- Estimando dfactor K=2 para decomposicao ---"
display ""

dfactor (`zseries' = , noconstant) (f1 = , ar(1)) (f2 = , ar(1))

scalar vd_ll    = e(ll)
scalar vd_phi1  = _b[f1:L.f1]
scalar vd_phi2  = _b[f2:L.f2]

display ""
display "Ajuste DFM K=2:"
display "  logLik = " vd_ll
display "  phi_1  = " vd_phi1
display "  phi_2  = " vd_phi2

* =============================================================================
* 4. Obter fatores suavizados para computar Var(f_j) e Cov(f_1, f_2)
* =============================================================================

predict fh1, states smethod(smooth) equation(f1)
predict fh2, states smethod(smooth) equation(f2)

quietly summarize fh1
scalar var_f1 = r(Var)
scalar mean_f1 = r(mean)

quietly summarize fh2
scalar var_f2 = r(Var)
scalar mean_f2 = r(mean)

quietly correlate fh1 fh2, covariance
scalar cov_f12 = r(cov_12)

display ""
display "Estatisticas dos fatores suavizados:"
display "  Var(f1)    = " var_f1
display "  Var(f2)    = " var_f2
display "  Cov(f1,f2) = " cov_f12

* Variancia teorica dos fatores (AR(1) estacionario com Var(eps_f)=1):
*   Var(f) = 1 / (1 - phi^2)
scalar var_f1_theo = 1 / (1 - vd_phi1^2)
scalar var_f2_theo = 1 / (1 - vd_phi2^2)
display "  Var(f1) teorica = " var_f1_theo
display "  Var(f2) teorica = " var_f2_theo

* =============================================================================
* 5. Montar tabela de cargas e variancias idiossincraticas
* =============================================================================
* Extrair sigma2_eps_i: dfactor guarda variancias da equacao de observacao
* em e(b) como /var(z_var) quando covobserved = diagonal (padrao).
* -----------------------------------------------------------------------------

tempname LAM SIG SHAREF1 SHAREF2 SHAREX SHAREE R2
matrix `LAM' = J(`N', 2, .)
matrix `SIG' = J(`N', 1, .)
matrix `SHAREF1' = J(`N', 1, .)
matrix `SHAREF2' = J(`N', 1, .)
matrix `SHAREX'  = J(`N', 1, .)
matrix `SHAREE'  = J(`N', 1, .)
matrix `R2'      = J(`N', 1, .)

local i = 1
foreach v of local series {
    * Cargas
    capture scalar lam1_i = _b[z_`v':f1]
    if _rc == 0 matrix `LAM'[`i', 1] = lam1_i

    capture scalar lam2_i = _b[z_`v':f2]
    if _rc == 0 matrix `LAM'[`i', 2] = lam2_i

    * Variancia idiossincratica (parametro /var(z_var))
    scalar sig_i = .
    capture scalar sig_i = _b[/var(z_`v')]
    * Em algumas versoes do Stata o parametro aparece como ln(var): tentar outros nomes
    if missing(sig_i) {
        capture scalar sig_i = exp(_b[/lnvar(z_`v')])
    }
    if missing(sig_i) {
        * Fallback: reconstroi via residuos
        tempvar res_`i'
        capture predict `res_`i'', residuals equation(z_`v')
        if _rc == 0 {
            quietly summarize `res_`i''
            scalar sig_i = r(Var)
        }
    }
    if !missing(sig_i) matrix `SIG'[`i', 1] = sig_i

    local ++i
}

* Contribuicoes e shares (loop em Mata-esque via scalar)
local i = 1
foreach v of local series {
    scalar l1 = `LAM'[`i', 1]
    scalar l2 = `LAM'[`i', 2]
    scalar s2 = `SIG'[`i', 1]

    * Contribuicoes individuais
    scalar c1    = (l1^2) * var_f1
    scalar c2    = (l2^2) * var_f2
    scalar ccrs  = 2 * l1 * l2 * cov_f12
    scalar ctot  = c1 + c2 + ccrs + s2        /* total implicada pelo modelo */

    if !missing(ctot) & ctot > 0 {
        matrix `SHAREF1'[`i', 1] = c1   / ctot
        matrix `SHAREF2'[`i', 1] = c2   / ctot
        matrix `SHAREX'[`i', 1]  = ccrs / ctot
        matrix `SHAREE'[`i', 1]  = s2   / ctot
        matrix `R2'[`i', 1]      = 1 - s2 / ctot
    }

    local ++i
}

* =============================================================================
* 6. Exportar CSV da decomposicao
* =============================================================================

display ""
display "--- Decomposicao de variancia (shares) ---"

preserve
    clear
    set obs `N'
    generate str30 series    = ""
    generate double share_f1 = .
    generate double share_f2 = .
    generate double share_cross = .
    generate double share_idio  = .
    generate double r2_common   = .
    generate double row_sum     = .

    local i = 1
    foreach v of local series {
        replace series      = "`v'"             in `i'
        replace share_f1    = `SHAREF1'[`i', 1] in `i'
        replace share_f2    = `SHAREF2'[`i', 1] in `i'
        replace share_cross = `SHAREX'[`i', 1]  in `i'
        replace share_idio  = `SHAREE'[`i', 1]  in `i'
        replace r2_common   = `R2'[`i', 1]      in `i'
        replace row_sum     = `SHAREF1'[`i', 1] + `SHAREF2'[`i', 1] ///
                              + `SHAREX'[`i', 1] + `SHAREE'[`i', 1] in `i'
        local ++i
    }

    list, noobs sep(0)
    export delimited using "results_variance_decomp.csv", replace
    display "Exportado: results_variance_decomp.csv"

    * R^2 medio
    quietly summarize r2_common
    scalar mean_r2 = r(mean)
    display ""
    display "  R^2 medio (componente comum): " mean_r2
restore

* --- Cargas ---
preserve
    clear
    set obs `N'
    generate str30 series    = ""
    generate double lambda_f1 = .
    generate double lambda_f2 = .

    local i = 1
    foreach v of local series {
        replace series    = "`v'"         in `i'
        replace lambda_f1 = `LAM'[`i', 1] in `i'
        replace lambda_f2 = `LAM'[`i', 2] in `i'
        local ++i
    }

    export delimited using "results_variance_decomp_loadings.csv", replace
    display "Exportado: results_variance_decomp_loadings.csv"
restore

* --- Sumario ---
preserve
    clear
    set obs 9
    generate str25 metric = ""
    generate double value = .

    replace metric = "logLik"          in 1
    replace value  = vd_ll             in 1
    replace metric = "mean_r2_common"  in 2
    replace value  = mean_r2           in 2
    replace metric = "var_f1_sample"   in 3
    replace value  = var_f1            in 3
    replace metric = "var_f2_sample"   in 4
    replace value  = var_f2            in 4
    replace metric = "cov_f1_f2"       in 5
    replace value  = cov_f12           in 5
    replace metric = "phi_1"           in 6
    replace value  = vd_phi1           in 6
    replace metric = "phi_2"           in 7
    replace value  = vd_phi2           in 7
    replace metric = "K"               in 8
    replace value  = 2                 in 8
    replace metric = "N"               in 9
    replace value  = `N'               in 9

    export delimited using "results_variance_decomp_summary.csv", replace
    display "Exportado: results_variance_decomp_summary.csv"
restore

display ""
display "============================================================"
display " VALIDACAO VARIANCE DECOMPOSITION CONCLUIDA"
display "============================================================"
display ""
display " Arquivos gerados:"
display "   - results_variance_decomp.csv"
display "   - results_variance_decomp_loadings.csv"
display "   - results_variance_decomp_summary.csv"
display "   - validate_variance_decomp.log"

log close
