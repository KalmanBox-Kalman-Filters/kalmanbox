/*******************************************************************************
* validate_mixed_freq.do
*
* Validacao: DFM com dados mixed-frequency (nowcasting de GDP)
*
* Objetivo:
*   Estimar um DFM com 1 fator (atividade economica comum) sobre um painel
*   em que a serie alvo (gdp_growth) esta disponivel apenas em fim de
*   trimestre (marco/junho/setembro/dezembro) e as demais 4 series sao
*   mensais. O fator suavizado reconstroi gdp_growth nos meses faltantes
*   -> nowcast.
*
* Especificacao dfactor:
*   - O dfactor trata valores missing (. no Stata) via Kalman filter: em
*     cada periodo, as observacoes faltantes sao simplesmente omitidas da
*     atualizacao do filtro, mas o estado continua evoluindo via transicao.
*   - Para mixed-frequency tipo Mariano-Murasawa nao usamos aqui (requer
*     agregacao temporal via sspace customizado). Usamos apenas o mecanismo
*     default: NA -> skip update.
*
* Dataset: mixed_freq_macro.csv
*   - gdp_growth: observado apenas em Mar/Jun/Set/Dez (~96 de 288)
*   - industrial_production, unemployment, cpi, pmi: mensais completos
*
* Nowcast:
*   Com K=1 fator comum F_t (AR(1), Var(eps_f)=1), o nowcast padronizado
*   de GDP em todos os meses e:
*       gdp_nowcast_std_t = lambda_gdp * F_t
*   Revertendo a padronizacao (mu, sd calculados somente em t observados):
*       gdp_nowcast_t = gdp_nowcast_std_t * sd_gdp + mu_gdp
*
* Saidas:
*   - results_mixed_freq_nowcast.csv  (fator, nowcast, GDP observado)
*   - results_mixed_freq_loadings.csv (cargas lambda de cada serie)
*   - results_mixed_freq_summary.csv  (logLik, RMSE, correlacao)
*   - validate_mixed_freq.log
*
* Autor: kalmanbox validation suite
* Fase:  FASE4 subfase F4.5
*******************************************************************************/

clear all
set more off
capture log close
log using "validate_mixed_freq.log", replace

display "============================================================"
display " VALIDACAO: Mixed-Frequency DFM Nowcasting via dfactor"
display " Data: $S_DATE $S_TIME"
display "============================================================"

* --- 1. Carregar dados mixed-frequency ---
* import delimited trata campos vazios como missing (.) por padrao.
import delimited using "../../data/mixed_freq_macro.csv", clear varnames(1)

generate date_stata = date(date, "YMD")
format date_stata %td
generate mdate = mofd(date_stata)
format mdate %tm
tsset mdate

display ""
display "Painel mixed-frequency carregado: `=_N' meses"

local series gdp_growth industrial_production unemployment cpi pmi
local N : word count `series'
display "N series: `N'"

* Contar missing por serie (diagnostico)
display ""
display "Missing por serie:"
foreach v of local series {
    quietly count if missing(`v')
    display "  `v' missing = " r(N) " / `=_N'"
}

* Verificar GDP: deve ter ~96 observacoes (1 por trimestre x 24 anos x 4)
quietly count if !missing(gdp_growth)
local n_gdp = r(N)
display ""
display "GDP observado: `n_gdp' trimestres (esperado ~96)"

* --- 2. Padronizacao por serie ---
* Calcula mu e sd ignorando missing (summarize ja ignora .)
* Guardamos mu_gdp e sd_gdp para reverter a padronizacao do nowcast.
foreach v of local series {
    quietly summarize `v'
    scalar mu_`v' = r(mean)
    scalar sd_`v' = r(sd)
    quietly generate z_`v' = (`v' - r(mean)) / r(sd)
}

scalar mu_gdp = mu_gdp_growth
scalar sd_gdp = sd_gdp_growth

display ""
display "Padronizacao GDP: mu = " mu_gdp " , sd = " sd_gdp

local zseries
foreach v of local series {
    local zseries `zseries' z_`v'
}

* =============================================================================
* 3. Estimar DFM K=1 com missing data
* =============================================================================
* Sintaxe:
*   dfactor (obs_eq) (factor_eq)
*   - obs_eq:    z_* = , noconstant
*   - factor_eq: f = , ar(1)
*
* Com 1 fator, nao ha ambiguidade de rotacao (apenas de sinal). A variancia
* do choque do fator e fixada em 1 (padrao dfactor).
* -----------------------------------------------------------------------------

display ""
display "--- Estimando dfactor K=1 com missing data ---"
display ""

dfactor (`zseries' = , noconstant) (f = , ar(1))

* Armazenar estatisticas
scalar mf_ll   = e(ll)
scalar mf_aic  = -2*e(ll) + 2*e(rank)
scalar mf_bic  = -2*e(ll) + ln(e(N))*e(rank)
scalar mf_phi  = _b[f:L.f]

display ""
display "Mixed-Freq DFM K=1:"
display "  log-lik = " mf_ll
display "  AIC     = " mf_aic
display "  BIC     = " mf_bic
display "  phi     = " mf_phi

* =============================================================================
* 4. Extrair fator suavizado
* =============================================================================

predict f_hat, states smethod(smooth) equation(f)

* =============================================================================
* 5. Construir nowcast de GDP
* =============================================================================
* Carga de gdp sobre o fator: _b[z_gdp_growth:f]
* Nowcast padronizado: f_hat * lambda_gdp
* Nowcast em escala original: nowcast_std * sd_gdp + mu_gdp
* -----------------------------------------------------------------------------

scalar lambda_gdp = _b[z_gdp_growth:f]
display ""
display "Carga gdp sobre fator: lambda_gdp = " lambda_gdp

generate double gdp_nowcast_std = lambda_gdp * f_hat
generate double gdp_nowcast = gdp_nowcast_std * sd_gdp + mu_gdp

* =============================================================================
* 6. Metricas de acuracia do nowcast
* =============================================================================
* Comparar gdp_nowcast com gdp_growth nos t em que GDP foi observado.
* RMSE DFM vs RMSE naive (media incondicional).
* -----------------------------------------------------------------------------

generate double err_dfm = gdp_growth - gdp_nowcast if !missing(gdp_growth)
quietly summarize err_dfm
scalar rmse_dfm = sqrt(r(Var) * (r(N) - 1)/r(N) + r(mean)^2)
scalar n_gdp_obs = r(N)

quietly summarize gdp_growth
scalar gdp_mean = r(mean)
generate double err_naive = gdp_growth - gdp_mean if !missing(gdp_growth)
quietly summarize err_naive
scalar rmse_naive = sqrt(r(Var) * (r(N) - 1)/r(N) + r(mean)^2)

scalar rel_improvement = 100 * (rmse_naive - rmse_dfm) / rmse_naive

* Correlacao nowcast vs GDP observado (em |.|)
quietly correlate gdp_nowcast gdp_growth if !missing(gdp_growth)
scalar corr_nowcast = abs(r(rho))

display ""
display "--- Metricas de acuracia (nos quarters observados) ---"
display "  N GDP obs    = " n_gdp_obs
display "  RMSE DFM     = " rmse_dfm
display "  RMSE naive   = " rmse_naive
display "  Improvement  = " rel_improvement " %"
display "  |corr|       = " corr_nowcast

* =============================================================================
* 7. Exportar CSVs
* =============================================================================

display ""
display "--- Exportando resultados ---"

* --- Nowcast completo com datas ---
preserve
    generate double gdp_observed = gdp_growth
    generate byte is_quarter_end = !missing(gdp_growth)
    keep date f_hat gdp_nowcast gdp_observed is_quarter_end
    rename f_hat factor
    order date factor gdp_nowcast gdp_observed is_quarter_end
    export delimited using "results_mixed_freq_nowcast.csv", replace
    display "Exportado: results_mixed_freq_nowcast.csv"
restore

* --- Cargas e variancias idiossincraticas ---
preserve
    clear
    set obs `N'
    generate str30 series = ""
    generate double lambda = .

    local i = 1
    foreach v of local series {
        replace series = "`v'" in `i'
        capture scalar lam = _b[z_`v':f]
        if _rc == 0 replace lambda = lam in `i'
        local ++i
    }

    export delimited using "results_mixed_freq_loadings.csv", replace
    display "Exportado: results_mixed_freq_loadings.csv"
restore

* --- Sumario com metricas ---
preserve
    clear
    set obs 9
    generate str25 metric = ""
    generate double value = .

    replace metric = "logLik"             in 1
    replace value  = mf_ll                in 1
    replace metric = "AIC"                in 2
    replace value  = mf_aic               in 2
    replace metric = "BIC"                in 3
    replace value  = mf_bic               in 3
    replace metric = "phi"                in 4
    replace value  = mf_phi               in 4
    replace metric = "lambda_gdp"         in 5
    replace value  = lambda_gdp           in 5
    replace metric = "rmse_dfm"           in 6
    replace value  = rmse_dfm             in 6
    replace metric = "rmse_naive"         in 7
    replace value  = rmse_naive           in 7
    replace metric = "rel_improvement_pct" in 8
    replace value  = rel_improvement      in 8
    replace metric = "corr_nowcast_gdp"   in 9
    replace value  = corr_nowcast         in 9

    export delimited using "results_mixed_freq_summary.csv", replace
    display "Exportado: results_mixed_freq_summary.csv"
restore

display ""
display "============================================================"
display " VALIDACAO MIXED-FREQUENCY CONCLUIDA"
display "============================================================"
display ""
display " Arquivos gerados:"
display "   - results_mixed_freq_nowcast.csv"
display "   - results_mixed_freq_loadings.csv"
display "   - results_mixed_freq_summary.csv"
display "   - validate_mixed_freq.log"

log close
