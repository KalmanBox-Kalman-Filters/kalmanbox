* ===========================================================================
* Validacao: Local Level Model - Nile data
* Comando: sspace (State-Space Models)
* ===========================================================================
*
* Modelo Local Level:
*   Observacao:  y_t = mu_t + eps_t,   eps_t ~ N(0, sigma2_eps)
*   Estado:      mu_t = mu_{t-1} + eta_t, eta_t ~ N(0, sigma2_eta)
*
* Na sintaxe sspace:
*   - Primeira equacao: observacao. "flow = {mu}" define que a variavel
*     observada (flow) depende do estado latente mu. "mle" indica que a
*     variancia do erro de observacao sera estimada via MLE.
*   - Segunda equacao: transicao. "mu = L.mu" define que o estado mu segue
*     um random walk (mu_t = mu_{t-1} + eta_t). "state" marca como equacao
*     de estado. "noconstant" remove o intercepto.
*
* Referencia: Durbin & Koopman (2012), Cap. 2
* ===========================================================================

clear all
set more off

* --------------------------------------------------------------------------
* 1. Carregar dados
* --------------------------------------------------------------------------
* O dataset Nile contem a vazao anual do rio Nilo (1871-1970)
import delimited "../../data/nile.csv", clear

* Declarar serie temporal (dados anuais)
tsset year

* --------------------------------------------------------------------------
* 2. Especificar e estimar o modelo Local Level via sspace
* --------------------------------------------------------------------------
* sspace (obs_eq) (state_eq)
*   obs_eq:   flow = {mu}         -> y_t = mu_t + eps_t
*   state_eq: mu = L.mu           -> mu_t = mu_{t-1} + eta_t
*   mle: variancia do erro de observacao estimada por MLE
*   state: marca equacao como equacao de estado
*   noconstant: sem intercepto na equacao de estado
sspace (flow = {mu}, mle) ///
       (mu = L.mu, state noconstant)

* --------------------------------------------------------------------------
* 3. Extrair parametros estimados
* --------------------------------------------------------------------------
* Os parametros de variancia sao armazenados em log na matriz e(b)
* Para obter as variancias, aplicamos exp()
matrix params = e(b)
matrix list params

* Variancia do erro de observacao (measurement noise)
scalar sigma2_eps = exp(params[1, "/:var(flow.flow)"])

* Variancia do erro de estado (state noise / signal)
scalar sigma2_eta = exp(params[1, "/:var(mu.mu)"])

* Log-verossimilhanca
scalar loglik = e(ll)

di _newline
di "=========================================="
di "Resultados: Local Level Model (Nile)"
di "=========================================="
di "sigma2_eps (obs)   = " sigma2_eps
di "sigma2_eta (state) = " sigma2_eta
di "q = sigma2_eta/sigma2_eps = " sigma2_eta/sigma2_eps
di "Log-likelihood     = " loglik
di "=========================================="

* --------------------------------------------------------------------------
* 4. Predizer estados filtrados e suavizados
* --------------------------------------------------------------------------
* smethod(filter): Kalman filter (usa info ate t)
* smethod(smooth): Kalman smoother (usa toda a amostra)
predict mu_filtered, state equation(mu) smethod(filter)
predict mu_smoothed, state equation(mu) smethod(smooth)

* Variancia do estado filtrado e suavizado
predict mu_filtered_se, state equation(mu) smethod(filter) rmse
predict mu_smoothed_se, state equation(mu) smethod(smooth) rmse

* --------------------------------------------------------------------------
* 5. Exportar resultados para comparacao
* --------------------------------------------------------------------------
export delimited year flow mu_filtered mu_smoothed mu_filtered_se mu_smoothed_se ///
    using "results_local_level_stata.csv", replace

di _newline
di "Resultados exportados para results_local_level_stata.csv"
