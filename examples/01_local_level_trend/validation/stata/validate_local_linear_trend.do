* ===========================================================================
* Validacao: Local Linear Trend Model - Airline Passengers data
* Comando: sspace (State-Space Models)
* ===========================================================================
*
* Modelo Local Linear Trend:
*   Observacao:  y_t = mu_t + eps_t,          eps_t ~ N(0, sigma2_eps)
*   Estado 1:    mu_t = mu_{t-1} + nu_{t-1} + xi_t,  xi_t ~ N(0, sigma2_xi)
*   Estado 2:    nu_t = nu_{t-1} + zeta_t,    zeta_t ~ N(0, sigma2_zeta)
*
* Na sintaxe sspace:
*   - Equacao de observacao: "log_passengers = {mu}" define que a variavel
*     observada depende do estado latente mu (level). "mle" indica que a
*     variancia do erro de observacao sera estimada via MLE.
*   - Equacao de estado 1 (level): "mu = L.mu + L.nu" define que o level
*     depende do seu valor defasado e do slope defasado. "state" marca como
*     equacao de estado. "noconstant" remove o intercepto.
*   - Equacao de estado 2 (slope/trend): "nu = L.nu" define que o slope
*     segue um random walk. "state noconstant" marca como equacao de estado.
*
* Nota: Trabalhamos com log(passengers) para estabilizar a variancia,
*       seguindo a pratica padrao para esta serie (Box-Jenkins, 1976).
*
* Referencia: Harvey (1989), Cap. 3; Durbin & Koopman (2012), Cap. 3
* ===========================================================================

clear all
set more off

* --------------------------------------------------------------------------
* 1. Carregar e preparar dados
* --------------------------------------------------------------------------
* O dataset Airline Passengers contem passageiros mensais (1949-1960)
import delimited "../../data/airline.csv", clear

* Criar variavel temporal mensal
gen mdate = mofd(date(date, "YMD"))
format mdate %tm
tsset mdate

* Transformacao log para estabilizar variancia
gen log_passengers = ln(passengers)

* --------------------------------------------------------------------------
* 2. Especificar e estimar o modelo Local Linear Trend via sspace
* --------------------------------------------------------------------------
* sspace (obs_eq) (state_eq_level) (state_eq_slope)
*   obs_eq:         log_passengers = {mu}     -> y_t = mu_t + eps_t
*   state_eq_level: mu = L.mu + L.nu          -> mu_t = mu_{t-1} + nu_{t-1} + xi_t
*   state_eq_slope: nu = L.nu                 -> nu_t = nu_{t-1} + zeta_t
*
* O modelo tem 2 equacoes de estado:
*   1) Level (mu): random walk com drift (drift = slope defasado)
*   2) Slope (nu): random walk (tendencia estocastica)
sspace (log_passengers = {mu}, mle) ///
       (mu = L.mu + L.nu, state noconstant) ///
       (nu = L.nu, state noconstant)

* --------------------------------------------------------------------------
* 3. Extrair parametros estimados
* --------------------------------------------------------------------------
matrix params = e(b)
matrix list params

* Variancias dos erros (armazenadas em log na matriz e(b))
scalar sigma2_eps  = exp(params[1, "/:var(log_passengers.log_passengers)"])
scalar sigma2_xi   = exp(params[1, "/:var(mu.mu)"])
scalar sigma2_zeta = exp(params[1, "/:var(nu.nu)"])

* Log-verossimilhanca
scalar loglik = e(ll)

di _newline
di "=========================================="
di "Resultados: Local Linear Trend (Airline)"
di "=========================================="
di "sigma2_eps  (obs)   = " sigma2_eps
di "sigma2_xi   (level) = " sigma2_xi
di "sigma2_zeta (slope) = " sigma2_zeta
di "Log-likelihood      = " loglik
di "=========================================="

* --------------------------------------------------------------------------
* 4. Predizer estados filtrados e suavizados
* --------------------------------------------------------------------------
* Level (mu) - filtrado e suavizado
predict mu_filtered, state equation(mu) smethod(filter)
predict mu_smoothed, state equation(mu) smethod(smooth)

* Slope (nu) - filtrado e suavizado
predict nu_filtered, state equation(nu) smethod(filter)
predict nu_smoothed, state equation(nu) smethod(smooth)

* Standard errors dos estados suavizados
predict mu_smoothed_se, state equation(mu) smethod(smooth) rmse
predict nu_smoothed_se, state equation(nu) smethod(smooth) rmse

* --------------------------------------------------------------------------
* 5. Forecast (12 meses apos o fim da amostra)
* --------------------------------------------------------------------------
* Estender a serie temporal para o periodo de forecast
tsappend, add(12)

* Predizer estados no periodo estendido (forecast usa smethod(filter))
predict mu_forecast, state equation(mu) smethod(filter) dynamic(tm(1961m1))
predict nu_forecast, state equation(nu) smethod(filter) dynamic(tm(1961m1))

* --------------------------------------------------------------------------
* 6. Exportar resultados para comparacao
* --------------------------------------------------------------------------
export delimited mdate log_passengers mu_filtered mu_smoothed ///
    nu_filtered nu_smoothed mu_smoothed_se nu_smoothed_se ///
    mu_forecast nu_forecast ///
    using "results_llt_stata.csv", replace

di _newline
di "Resultados exportados para results_llt_stata.csv"
