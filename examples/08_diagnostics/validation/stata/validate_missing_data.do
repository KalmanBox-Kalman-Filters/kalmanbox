* ===========================================================================
* Validacao: Missing Data - BSM no log(airline) (completo vs. com missings)
* Comando: sspace + predict
* ===========================================================================
*
* Objetivo: Demonstrar que o Stata sspace lida nativamente com valores
* faltantes (representados como "." ou string vazia no CSV convertida a
* missing pelo import delimited).
*
*   - Ajustamos um BSM (level+trend+seasonal) ao log(airline) completo.
*   - Repetimos a estimacao no log(airline) com NAs artificiais.
*   - Comparamos hiperparametros e log-likelihood.
*   - Usamos o smoother para imputar E[y|Y^n] nos pontos faltantes.
*   - Comparamos as imputacoes contra a serie original (truth).
*
* No Stata, quando a variavel dependente tem missing em um periodo t, o
* sspace monta a matriz de ganho Kalman para pular a atualizacao daquele
* t (K_t = 0) e propaga apenas a equacao de transicao. predict com
* smethod(smooth) devolve entao a media condicional E[y_t|Y^n] como
* imputacao dos pontos faltantes.
*
* Limitacoes:
*   - Bootstrap/simulation smoother nao esta integrado ao sspace; se
*     precisar de draws de estado, e necessario rodar em Mata (omitido
*     aqui; coberto no backend R via KFAS::simulateSSM).
*   - predict nao devolve a variancia da imputacao diretamente para
*     observacoes (somente RMSE do estado via option rmse). Usamos o
*     RMSE do nivel suavizado como proxy do se da imputacao.
*
* Referencias:
*   Durbin & Koopman (2012), cap. 4.10 (missing observations)
*   Harvey (1989), cap. 3.4
* ===========================================================================

clear all
set more off

* --------------------------------------------------------------------------
* 1. Carregar serie completa log(airline)
* --------------------------------------------------------------------------
import delimited "../../data/airline.csv", clear

* month no formato YYYY-MM -> mdate mensal
gen mdate = monthly(month, "YM")
format mdate %tm
tsset mdate

gen double ly_full = ln(passengers)

tempfile full
save `full'

quietly count
local n_full = r(N)
di _newline
di "=========================================="
di "Missing Data - BSM log(airline)"
di "=========================================="
di "Observacoes completas: " `n_full'

* --------------------------------------------------------------------------
* 2. Carregar serie com missings (passengers em branco -> missing)
* --------------------------------------------------------------------------
import delimited "../../data/airline_missing.csv", clear

* import delimited converte campos vazios para missing (.) em variaveis
* numericas. Caso o parser leia como string, forcamos conversao:
capture confirm numeric variable passengers
if _rc != 0 {
    destring passengers, replace force
}

gen mdate = monthly(month, "YM")
format mdate %tm
tsset mdate

gen double ly_miss = ln(passengers)

quietly count if missing(ly_miss)
local n_miss = r(N)
quietly count
local n_total = r(N)

di "Observacoes com missings: " `n_total'
di "  missings = " `n_miss' " (" 100*`n_miss'/`n_total' "%)"

* --------------------------------------------------------------------------
* 3. Juntar as duas series no mesmo dataset para comparacao posterior
* --------------------------------------------------------------------------
merge 1:1 mdate using `full', keepusing(ly_full)
drop _merge

tsset mdate

* --------------------------------------------------------------------------
* 4. Estimar BSM (level+trend+seasonal12) na serie completa
* --------------------------------------------------------------------------
* BSM via sspace (dummy seasonal representation, 11 lags + shift register):
*
*   ly = mu + g1 + eps
*   mu = L.mu + L.nu + xi
*   nu = L.nu + zeta
*   g1 = -sum_{j=1..11} L.gj + omega
*   g2..g11 = shift registers de g1
di _newline
di "--- 4. BSM na serie completa ---"

sspace (ly_full = {mu} + {g1}, mle)                           ///
       (mu = L.mu + L.nu, state noconstant)                   ///
       (nu = L.nu, state noconstant)                          ///
       (g1 = -L.g1 -L.g2 -L.g3 -L.g4 -L.g5 -L.g6 -L.g7        ///
             -L.g8 -L.g9 -L.g10 -L.g11, state noconstant)     ///
       (g2 = L.g1, state noconstant noisily)                  ///
       (g3 = L.g2, state noconstant noisily)                  ///
       (g4 = L.g3, state noconstant noisily)                  ///
       (g5 = L.g4, state noconstant noisily)                  ///
       (g6 = L.g5, state noconstant noisily)                  ///
       (g7 = L.g6, state noconstant noisily)                  ///
       (g8 = L.g7, state noconstant noisily)                  ///
       (g9 = L.g8, state noconstant noisily)                  ///
       (g10 = L.g9, state noconstant noisily)                 ///
       (g11 = L.g10, state noconstant noisily)

matrix pf = e(b)
scalar H_full    = exp(pf[1, "/:var(ly_full.ly_full)"])
scalar Qlvl_full = exp(pf[1, "/:var(mu.mu)"])
scalar Qtrd_full = exp(pf[1, "/:var(nu.nu)"])
scalar Qsea_full = exp(pf[1, "/:var(g1.g1)"])
scalar ll_full   = e(ll)
scalar nf_full   = e(N)

di "sigma2_obs   = " H_full
di "sigma2_level = " Qlvl_full
di "sigma2_trend = " Qtrd_full
di "sigma2_seas  = " Qsea_full
di "log-lik      = " ll_full

* Estado suavizado (nivel) e media suavizada (signal)
predict mu_full, state equation(mu) smethod(smooth)
predict ly_hat_full, xb smethod(smooth)

* --------------------------------------------------------------------------
* 5. Estimar o MESMO BSM na serie com missings
* --------------------------------------------------------------------------
di _newline
di "--- 5. BSM na serie com missings ---"

sspace (ly_miss = {mu2} + {g1m}, mle)                           ///
       (mu2 = L.mu2 + L.nu2, state noconstant)                  ///
       (nu2 = L.nu2, state noconstant)                          ///
       (g1m = -L.g1m -L.g2m -L.g3m -L.g4m -L.g5m -L.g6m -L.g7m  ///
              -L.g8m -L.g9m -L.g10m -L.g11m, state noconstant)  ///
       (g2m = L.g1m, state noconstant noisily)                  ///
       (g3m = L.g2m, state noconstant noisily)                  ///
       (g4m = L.g3m, state noconstant noisily)                  ///
       (g5m = L.g4m, state noconstant noisily)                  ///
       (g6m = L.g5m, state noconstant noisily)                  ///
       (g7m = L.g6m, state noconstant noisily)                  ///
       (g8m = L.g7m, state noconstant noisily)                  ///
       (g9m = L.g8m, state noconstant noisily)                  ///
       (g10m = L.g9m, state noconstant noisily)                 ///
       (g11m = L.g10m, state noconstant noisily)

matrix pm = e(b)
scalar H_miss    = exp(pm[1, "/:var(ly_miss.ly_miss)"])
scalar Qlvl_miss = exp(pm[1, "/:var(mu2.mu2)"])
scalar Qtrd_miss = exp(pm[1, "/:var(nu2.nu2)"])
scalar Qsea_miss = exp(pm[1, "/:var(g1m.g1m)"])
scalar ll_miss   = e(ll)
scalar nf_miss   = e(N)

di "sigma2_obs   = " H_miss
di "sigma2_level = " Qlvl_miss
di "sigma2_trend = " Qtrd_miss
di "sigma2_seas  = " Qsea_miss
di "log-lik      = " ll_miss

* Estado suavizado e media suavizada (imputacao nos missings)
predict mu_miss, state equation(mu2) smethod(smooth)
predict ly_hat_miss, xb smethod(smooth)
predict mu_miss_se, state equation(mu2) smethod(smooth) rmse

* --------------------------------------------------------------------------
* 6. Comparar parametros (complete vs. missing)
* --------------------------------------------------------------------------
di _newline
di "=========================================="
di "Comparacao de parametros"
di "=========================================="
di "                  complete       missing      abs_diff"
di "sigma2_obs    = " H_full       "   " H_miss    "   " abs(H_full-H_miss)
di "sigma2_level  = " Qlvl_full    "   " Qlvl_miss "   " abs(Qlvl_full-Qlvl_miss)
di "sigma2_trend  = " Qtrd_full    "   " Qtrd_miss "   " abs(Qtrd_full-Qtrd_miss)
di "sigma2_seas   = " Qsea_full    "   " Qsea_miss "   " abs(Qsea_full-Qsea_miss)
di "loglik        = " ll_full      "   " ll_miss
di "=========================================="

* --------------------------------------------------------------------------
* 7. Metricas de imputacao (apenas nos pontos missing)
* --------------------------------------------------------------------------
gen byte is_missing = missing(ly_miss)
gen double imp_err  = ly_hat_miss - ly_full if is_missing == 1

quietly summarize imp_err
scalar rmse_imp = sqrt(r(Var) + r(mean)^2)

gen double abs_err = abs(imp_err)
quietly summarize abs_err
scalar mae_imp = r(mean)

di _newline
di "--- Imputacao nos pontos missing ---"
di "n_missing = " `n_miss'
di "RMSE      = " rmse_imp
di "MAE       = " mae_imp

* --------------------------------------------------------------------------
* 8. Exportar resultados
* --------------------------------------------------------------------------

* 8a. Serie + smoothed + imputacoes
export delimited mdate ly_full ly_miss is_missing ///
    mu_full ly_hat_full mu_miss ly_hat_miss mu_miss_se ///
    using "results_missing_data_stata.csv", replace

* 8b. Parametros
preserve
    clear
    set obs 4
    gen str20 parameter   = ""
    gen double complete   = .
    gen double with_miss  = .

    replace parameter = "sigma2_obs"    in 1
    replace complete  = scalar(H_full)  in 1
    replace with_miss = scalar(H_miss)  in 1

    replace parameter = "sigma2_level"     in 2
    replace complete  = scalar(Qlvl_full)  in 2
    replace with_miss = scalar(Qlvl_miss)  in 2

    replace parameter = "sigma2_trend"     in 3
    replace complete  = scalar(Qtrd_full)  in 3
    replace with_miss = scalar(Qtrd_miss)  in 3

    replace parameter = "sigma2_seasonal"  in 4
    replace complete  = scalar(Qsea_full)  in 4
    replace with_miss = scalar(Qsea_miss)  in 4

    gen double abs_diff     = abs(complete - with_miss)
    gen double rel_diff_pct = 100 * abs_diff / abs(complete)

    export delimited using "results_missing_data_params_stata.csv", replace
restore

* 8c. Resumo agregado
preserve
    clear
    set obs 7
    gen str30 metric = ""
    gen double value = .

    replace metric = "n_obs"            in 1
    replace value  = `n_total'          in 1

    replace metric = "n_missing"        in 2
    replace value  = `n_miss'           in 2

    replace metric = "pct_missing"      in 3
    replace value  = 100*`n_miss'/`n_total' in 3

    replace metric = "loglik_complete"  in 4
    replace value  = scalar(ll_full)    in 4

    replace metric = "loglik_missing"   in 5
    replace value  = scalar(ll_miss)    in 5

    replace metric = "rmse_imputation"  in 6
    replace value  = scalar(rmse_imp)   in 6

    replace metric = "mae_imputation"   in 7
    replace value  = scalar(mae_imp)    in 7

    export delimited using "results_missing_data_summary_stata.csv", replace
restore

di _newline
di "Missing data (Stata) completo."
di "  results_missing_data_stata.csv"
di "  results_missing_data_params_stata.csv"
di "  results_missing_data_summary_stata.csv"
