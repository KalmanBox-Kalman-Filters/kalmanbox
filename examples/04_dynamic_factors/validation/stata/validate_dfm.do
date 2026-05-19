/*******************************************************************************
* validate_dfm.do
*
* Validacao: Dynamic Factor Model (K=2) via comando `dfactor`
*
* Objetivo:
*   Estimar um DFM com 2 fatores latentes independentes, cada um seguindo
*   AR(1), sobre o painel macro americano (15 series, T=288 meses).
*   Exportar fatores suavizados, cargas e parametros para comparacao com
*   kalmanbox (Python) e MARSS (R).
*
* Especificacao dfactor:
*   Sintaxe geral:
*     dfactor (depvars = [indepvars], obs_opts) [(depvars = ...)] ///
*             (factors = , state_opts) [(factors = , state_opts)]
*
*   - A(s) equacao(oes) de observacao define(m) quais series sao modeladas.
*     Com `noconstant` nao se estima intercepto (o dfactor centra Y por padrao
*     se a constante for incluida; ja padronizamos fora do dfactor).
*   - Cada bloco de fator define um vetor latente com opcoes state_opts, sendo
*     as mais importantes:
*       ar(numlist)       : ordem AR do fator (ex. ar(1))
*       noconstant        : sem constante na equacao do fator
*       identity          : covariancia do choque do fator = I (fixa escala;
*                           so aplicavel quando o fator e multivariado)
*   - Para K=2 fatores independentes com AR(1) cada, usamos dois blocos de
*     fator separados, um para cada fator. Isso equivale a B diagonal.
*   - dfactor estima cargas Z (Lambda) como coeficientes das series sobre os
*     fatores. Por padrao todas livres -> identificacao via normalizacao
*     interna do Stata (Lambda triangular nas K primeiras linhas). Os
*     resultados sao comparaveis a MARSS/kalmanbox a menos de rotacao.
*   - Variancias idiossincraticas sao diagonais por padrao (covstate/covobserved
*     controlam a estrutura; manter default = diagonal).
*
* Identificacao e comparacao cross-software:
*   DFM e identificado so a menos de (a) sinal, (b) rotacao ortogonal quando
*   K > 1. Dfactor aplica normalizacoes internas diferentes de MARSS; portanto
*   a comparacao direta requer:
*     - correlacao |abs| entre fatores estimados (apos sign-flip)
*     - ou rotacao Procrustes entre as matrizes de cargas.
*
* Saidas:
*   - results_dfm_factors.csv     (fatores suavizados f1, f2 com datas)
*   - results_dfm_loadings.csv    (cargas Lambda N x K)
*   - results_dfm_fit_summary.csv (logLik, AIC, BIC, parametros AR)
*   - validate_dfm.log
*
* Dataset: examples/04_dynamic_factors/data/us_macro_panel.csv
*          T=288 meses (2000-01 a 2023-12), N=15 series macro ja padronizadas.
*
* Autor: kalmanbox validation suite
* Fase:  FASE4 subfase F4.5
*******************************************************************************/

clear all
set more off
capture log close
log using "validate_dfm.log", replace

display "============================================================"
display " VALIDACAO: Dynamic Factor Model (K=2) via dfactor"
display " Data: $S_DATE $S_TIME"
display "============================================================"

* --- 1. Carregar painel ---
import delimited using "../../data/us_macro_panel.csv", clear varnames(1)

* Parse date e criar indice mensal
generate date_stata = date(date, "YMD")
format date_stata %td
generate mdate = mofd(date_stata)
format mdate %tm
tsset mdate

display ""
display "Painel carregado: `=_N' meses"

* Lista de series (todas as variaveis numericas exceto date)
local series gdp_growth industrial_production unemployment payrolls ///
             retail_sales housing_starts consumer_confidence ///
             pmi_manufacturing cpi_inflation pce_inflation ///
             fed_funds_rate sp500_returns term_spread credit_spread ///
             oil_price_change

local N : word count `series'
display "N series: `N'"
display "Series: `series'"

* --- 2. Padronizar series (media 0, var 1) para estabilidade numerica ---
* Os dados do painel ja estao padronizados conceitualmente, mas aplicamos
* novamente por robustez (evita problemas de escala entre series).
foreach v of local series {
    quietly summarize `v'
    quietly generate z_`v' = (`v' - r(mean)) / r(sd)
}

local zseries
foreach v of local series {
    local zseries `zseries' z_`v'
}

display ""
display "Series padronizadas (prefixo z_): `zseries'"

* =============================================================================
* 3. Estimacao DFM K=2 via dfactor
* =============================================================================
* Especificacao:
*   - Observacao: todas as series padronizadas, noconstant (ja centradas)
*   - Fator f1: AR(1), sem constante
*   - Fator f2: AR(1), sem constante
*
* Nota sobre identificacao:
*   dfactor normaliza internamente para resolver indeterminacao de escala/
*   rotacao. Por padrao, a variancia do choque do fator e igual a 1 e as
*   cargas sao livres. A ordem das series importa: a primeira serie da
*   equacao de observacao nao carrega no fator 2 (restricao implicita em
*   algumas implementacoes). Ajustamos a ordem de forma consistente com
*   MARSS (gdp_growth primeiro => Lambda[1,2] = 0 equivalente).

display ""
display "--- Estimando dfactor K=2 (AR(1) + AR(1)) ---"
display ""

dfactor (`zseries' = , noconstant) (f1 = , ar(1)) (f2 = , ar(1))

* --- 4. Armazenar estatisticas de ajuste ---
scalar dfm_ll     = e(ll)
scalar dfm_aic    = -2*e(ll) + 2*e(rank)
scalar dfm_bic    = -2*e(ll) + ln(e(N))*e(rank)
scalar dfm_N      = e(N)
scalar dfm_k      = e(rank)

* AR(1) dos fatores (coeficiente de L.f em cada equacao de estado)
scalar phi_1 = _b[f1:L.f1]
scalar phi_2 = _b[f2:L.f2]

display ""
display "DFM K=2 - Resumo:"
display "  log-lik     = " dfm_ll
display "  AIC         = " dfm_aic
display "  BIC         = " dfm_bic
display "  # params    = " dfm_k
display "  N obs       = " dfm_N
display "  phi_1 (AR1) = " phi_1
display "  phi_2 (AR1) = " phi_2

* =============================================================================
* 5. Extracao dos fatores suavizados
* =============================================================================
* predict ..., states smethod(smooth) equation(eqname)
*   - states: pede estados latentes (fatores)
*   - smethod(smooth): usa Kalman smoother (default: onestep)
*   - equation: nome do bloco de estado
* -----------------------------------------------------------------------------

display ""
display "--- Extraindo fatores suavizados ---"

predict f1_hat, states smethod(smooth) equation(f1)
predict f2_hat, states smethod(smooth) equation(f2)

* Erros-padrao dos estados suavizados (se disponiveis)
capture predict f1_se, states smethod(smooth) equation(f1) rstd
capture predict f2_se, states smethod(smooth) equation(f2) rstd

* =============================================================================
* 6. Exportar fatores com datas
* =============================================================================

display ""
display "--- Exportando fatores para CSV ---"

preserve
    local keepvars date mdate f1_hat f2_hat
    capture confirm variable f1_se
    if _rc == 0 local keepvars `keepvars' f1_se f2_se
    keep `keepvars'
    rename f1_hat f1
    rename f2_hat f2
    order date f1 f2
    export delimited using "results_dfm_factors.csv", replace
    display "Exportado: results_dfm_factors.csv"
restore

* =============================================================================
* 7. Exportar cargas fatoriais (Lambda)
* =============================================================================
* As cargas sao os coeficientes de cada serie (z_var) sobre f1 e f2 na equacao
* de observacao. Sao acessiveis via _b[z_var:f1] e _b[z_var:f2].
* -----------------------------------------------------------------------------

display ""
display "--- Exportando cargas Lambda N x K ---"

preserve
    clear
    set obs `N'
    generate str30 series = ""
    generate double lambda_f1 = .
    generate double lambda_f2 = .

    local i = 1
    foreach v of local series {
        replace series = "`v'" in `i'
        * coef da serie z_`v' sobre fator f1 (equacao de observacao z_`v')
        capture scalar lam1 = _b[z_`v':f1]
        if _rc == 0 replace lambda_f1 = lam1 in `i'
        capture scalar lam2 = _b[z_`v':f2]
        if _rc == 0 replace lambda_f2 = lam2 in `i'
        local ++i
    }

    export delimited using "results_dfm_loadings.csv", replace
    display "Exportado: results_dfm_loadings.csv"
restore

* =============================================================================
* 8. Exportar sumario do ajuste
* =============================================================================

preserve
    clear
    set obs 7
    generate str20 metric = ""
    generate double value = .

    replace metric = "logLik"      in 1
    replace value  = dfm_ll        in 1
    replace metric = "AIC"         in 2
    replace value  = dfm_aic       in 2
    replace metric = "BIC"         in 3
    replace value  = dfm_bic       in 3
    replace metric = "n_params"    in 4
    replace value  = dfm_k         in 4
    replace metric = "N"           in 5
    replace value  = dfm_N         in 5
    replace metric = "phi_1"       in 6
    replace value  = phi_1         in 6
    replace metric = "phi_2"       in 7
    replace value  = phi_2         in 7

    export delimited using "results_dfm_fit_summary.csv", replace
    display "Exportado: results_dfm_fit_summary.csv"
restore

display ""
display "============================================================"
display " VALIDACAO DFM K=2 CONCLUIDA"
display "============================================================"
display ""
display " Arquivos gerados:"
display "   - results_dfm_factors.csv"
display "   - results_dfm_loadings.csv"
display "   - results_dfm_fit_summary.csv"
display "   - validate_dfm.log"

log close
