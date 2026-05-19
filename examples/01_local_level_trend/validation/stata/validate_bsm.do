* ===========================================================================
* Validacao: Basic Structural Model (BSM) - UK Drivers data
* Comando: sspace (State-Space Models)
* ===========================================================================
*
* Modelo BSM (Harvey, 1989):
*   y_t = mu_t + gamma_t + eps_t
*
*   Level:     mu_t = mu_{t-1} + nu_{t-1} + xi_t,    xi_t ~ N(0, sigma2_xi)
*   Slope:     nu_t = nu_{t-1} + zeta_t,              zeta_t ~ N(0, sigma2_zeta)
*   Seasonal:  gamma_t = -sum(gamma_{t-j}, j=1..s-1) + omega_t, omega_t ~ N(0, sigma2_omega)
*
* Com s=12 (mensal), a sazonalidade requer s-1=11 equacoes de estado
* usando a representacao dummy sazonal:
*   gamma_t   = -gamma_{t-1} - gamma_{t-2} - ... - gamma_{t-11} + omega_t
*   gamma*_1  = gamma_{t}     (shift register)
*   gamma*_2  = gamma*_1_{t-1}
*   ...
*   gamma*_10 = gamma*_9_{t-1}
*
* Na sintaxe sspace, cada estado sazonal e uma equacao separada:
*   - g1 = -L.g1 - L.g2 - ... - L.g11  (com inovacao)
*   - g2 = L.g1, g3 = L.g2, ..., g11 = L.g10  (shift sem inovacao)
*
* O dataset UK Drivers contem mortes/feridos graves em acidentes
* de transito no Reino Unido (jan/1969 - dez/1984).
* Em fev/1983 a lei do cinto de seguranca entrou em vigor (intervencao).
*
* Referencia: Harvey (1989), Cap. 2-3; Harvey & Durbin (1986)
* ===========================================================================

clear all
set more off

* --------------------------------------------------------------------------
* 1. Carregar e preparar dados
* --------------------------------------------------------------------------
import delimited "../../data/uk_drivers.csv", clear

* Criar variavel temporal mensal
gen mdate = mofd(date(date, "YMD"))
format mdate %tm
tsset mdate

* Transformacao log (padrao para esta serie)
gen log_deaths = ln(deaths)

* Variavel dummy de intervencao: lei do cinto de seguranca (fev/1983)
* 0 antes, 1 a partir de fev/1983
gen seatbelt = (mdate >= tm(1983m2))

* --------------------------------------------------------------------------
* 2. Modelo BSM SEM intervencao
* --------------------------------------------------------------------------
* Especificacao sspace com:
*   - 1 equacao de observacao: log_deaths = mu + g1
*   - 2 equacoes de estado para level+slope (mu, nu)
*   - 11 equacoes de estado para sazonalidade dummy (g1..g11)
*
* Total: 13 equacoes de estado
*
* Equacao de observacao:
*   log_deaths = {mu} + {g1}    (level + seasonal component)
*
* Equacoes de estado - Level e Slope:
*   mu = L.mu + L.nu            (level: random walk com drift)
*   nu = L.nu                   (slope: random walk)
*
* Equacoes de estado - Sazonalidade (dummy seasonal, s=12):
*   g1 = -L.g1 -L.g2 -L.g3 -L.g4 -L.g5 -L.g6 ///
*        -L.g7 -L.g8 -L.g9 -L.g10 -L.g11       (componente sazonal principal)
*   g2 = L.g1   (shift register: g2 armazena g1 defasado)
*   g3 = L.g2   (shift register)
*   g4 = L.g3
*   g5 = L.g4
*   g6 = L.g5
*   g7 = L.g6
*   g8 = L.g7
*   g9 = L.g8
*   g10 = L.g9
*   g11 = L.g10 (shift register: ultimo componente)
*
* Nota: Apenas g1 tem inovacao (omega_t). g2..g11 sao deterministicos
* (noconstant + noisily free of error, pois nao especificamos variancia).

di _newline
di "=========================================="
di "Modelo 1: BSM sem intervencao"
di "=========================================="

sspace (log_deaths = {mu} + {g1}, mle)          ///
       (mu = L.mu + L.nu, state noconstant)      ///
       (nu = L.nu, state noconstant)              ///
       (g1 = -L.g1 -L.g2 -L.g3 -L.g4 -L.g5     ///
             -L.g6 -L.g7 -L.g8 -L.g9 -L.g10     ///
             -L.g11, state noconstant)            ///
       (g2 = L.g1, state noconstant noisily)      ///
       (g3 = L.g2, state noconstant noisily)      ///
       (g4 = L.g3, state noconstant noisily)      ///
       (g5 = L.g4, state noconstant noisily)      ///
       (g6 = L.g5, state noconstant noisily)      ///
       (g7 = L.g6, state noconstant noisily)      ///
       (g8 = L.g7, state noconstant noisily)      ///
       (g9 = L.g8, state noconstant noisily)      ///
       (g10 = L.g9, state noconstant noisily)     ///
       (g11 = L.g10, state noconstant noisily)

* Extrair parametros
matrix params1 = e(b)
matrix list params1

scalar sigma2_eps_1   = exp(params1[1, "/:var(log_deaths.log_deaths)"])
scalar sigma2_xi_1    = exp(params1[1, "/:var(mu.mu)"])
scalar sigma2_zeta_1  = exp(params1[1, "/:var(nu.nu)"])
scalar sigma2_omega_1 = exp(params1[1, "/:var(g1.g1)"])
scalar loglik_1       = e(ll)

di _newline
di "=========================================="
di "Resultados: BSM sem intervencao"
di "=========================================="
di "sigma2_eps   (obs)      = " sigma2_eps_1
di "sigma2_xi    (level)    = " sigma2_xi_1
di "sigma2_zeta  (slope)    = " sigma2_zeta_1
di "sigma2_omega (seasonal) = " sigma2_omega_1
di "Log-likelihood          = " loglik_1
di "=========================================="

* Estados suavizados (Modelo 1 - sem intervencao)
predict mu_sm_1, state equation(mu) smethod(smooth)
predict nu_sm_1, state equation(nu) smethod(smooth)
predict g1_sm_1, state equation(g1) smethod(smooth)

* --------------------------------------------------------------------------
* 3. Modelo BSM COM intervencao (seatbelt law)
* --------------------------------------------------------------------------
* Adiciona a variavel dummy de intervencao na equacao de observacao:
*   log_deaths = {mu} + {g1} + {beta_seat}*seatbelt
*
* Isso permite estimar o efeito da lei do cinto de seguranca como um
* shift de nivel na serie, controlando por tendencia e sazonalidade.

di _newline
di "=========================================="
di "Modelo 2: BSM com intervencao (seatbelt)"
di "=========================================="

sspace (log_deaths = {mu} + {g1} + {beta_seat}*seatbelt, mle) ///
       (mu = L.mu + L.nu, state noconstant)      ///
       (nu = L.nu, state noconstant)              ///
       (g1 = -L.g1 -L.g2 -L.g3 -L.g4 -L.g5     ///
             -L.g6 -L.g7 -L.g8 -L.g9 -L.g10     ///
             -L.g11, state noconstant)            ///
       (g2 = L.g1, state noconstant noisily)      ///
       (g3 = L.g2, state noconstant noisily)      ///
       (g4 = L.g3, state noconstant noisily)      ///
       (g5 = L.g4, state noconstant noisily)      ///
       (g6 = L.g5, state noconstant noisily)      ///
       (g7 = L.g6, state noconstant noisily)      ///
       (g8 = L.g7, state noconstant noisily)      ///
       (g9 = L.g8, state noconstant noisily)      ///
       (g10 = L.g9, state noconstant noisily)     ///
       (g11 = L.g10, state noconstant noisily)

* Extrair parametros
matrix params2 = e(b)
matrix list params2

scalar sigma2_eps_2   = exp(params2[1, "/:var(log_deaths.log_deaths)"])
scalar sigma2_xi_2    = exp(params2[1, "/:var(mu.mu)"])
scalar sigma2_zeta_2  = exp(params2[1, "/:var(nu.nu)"])
scalar sigma2_omega_2 = exp(params2[1, "/:var(g1.g1)"])
scalar beta_seat      = params2[1, "/:{beta_seat}"]
scalar loglik_2       = e(ll)

di _newline
di "=========================================="
di "Resultados: BSM com intervencao"
di "=========================================="
di "sigma2_eps   (obs)      = " sigma2_eps_2
di "sigma2_xi    (level)    = " sigma2_xi_2
di "sigma2_zeta  (slope)    = " sigma2_zeta_2
di "sigma2_omega (seasonal) = " sigma2_omega_2
di "beta_seat    (interv.)  = " beta_seat
di "Efeito %    = " (exp(beta_seat)-1)*100 "%"
di "Log-likelihood          = " loglik_2
di "=========================================="

* Estados suavizados (Modelo 2 - com intervencao)
predict mu_sm_2, state equation(mu) smethod(smooth)
predict nu_sm_2, state equation(nu) smethod(smooth)
predict g1_sm_2, state equation(g1) smethod(smooth)

* --------------------------------------------------------------------------
* 4. Exportar resultados para comparacao
* --------------------------------------------------------------------------
* Exporta decomposicao completa de ambos os modelos
export delimited mdate log_deaths seatbelt ///
    mu_sm_1 nu_sm_1 g1_sm_1 ///
    mu_sm_2 nu_sm_2 g1_sm_2 ///
    using "results_bsm_stata.csv", replace

di _newline
di "Resultados exportados para results_bsm_stata.csv"
