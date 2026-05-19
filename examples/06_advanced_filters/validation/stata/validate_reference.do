* =============================================================================
* Validacao: Pendulo nao-linear LINEARIZADO via sspace (Stata)
*
* LIMITACOES FUNDAMENTAIS DO STATA PARA FILTROS NAO-LINEARES
* --------------------------------------------------------------------------
* O comando `sspace` do Stata implementa APENAS o filtro de Kalman linear
* e Gaussiano (Durbin-Koopman, 2012). Nao existem implementacoes nativas
* de EKF, UKF, ou EnKF no core do Stata (ate o Stata 18, 2024).
*
* Em particular:
*   - Nao ha suporte para transicao nao-linear f(x_t)
*   - Nao ha suporte para observacao nao-linear h(x_t)
*   - Nao ha propagacao de sigma points (unscented transform)
*   - Nao ha ensemble stochastico para dinamicas caoticas
*
* Alternativas possiveis em Stata para problemas nao-lineares:
*   1. LINEARIZACAO analitica (Jacobianos) -> sspace, valido so localmente
*   2. `nl` / `ml` para MLE em problemas bem-comportados sem states latentes
*   3. `gsem` para modelos estruturais (mas nao com states dinamicos NL)
*   4. Mata custom: usar Mata programacao para implementar EKF/UKF
*      manualmente (fora do escopo de uma validacao de referencia)
*
* REFERENCIA: Stata Time-Series Reference Manual, [TS] sspace,
* secao "Remarks and examples" - "sspace estimates parameters of linear
* state-space models." (grifo nosso)
*
* --------------------------------------------------------------------------
* MODELO DO PENDULO (original, nao-linear; Sarkka 2013, Cap. 5)
* --------------------------------------------------------------------------
*   theta_dot_{k+1} = theta_dot_k - dt * (g/L) * sin(theta_k) + w_k
*   theta_{k+1}     = theta_k + dt * theta_dot_{k+1}
*   y_k             = sin(theta_k) + v_k
*
* LINEARIZACAO em torno de theta = 0 (aproximacao de pequenas oscilacoes):
*   sin(theta) ~ theta
*   => theta_dot_{k+1} ~ theta_dot_k - dt * omega2 * theta_k + w_k
*      theta_{k+1}     ~ theta_k + dt * theta_dot_{k+1}
*      y_k             ~ theta_k + v_k
*
* Em forma de espaco de estados linear:
*   x_t = [theta_t, theta_dot_t]'
*   F = [[1 - dt^2*omega2,  dt],
*        [   -dt*omega2,     1]]
*   Z = [1, 0]
*
* A aproximacao SO e valida para |theta| pequeno (~ < 0.3 rad).
* Para theta ~ 1.2 rad (condicao inicial do dataset) a linearizacao
* produz erro substancial, confirmando a necessidade de EKF/UKF.
* =============================================================================

clear all
set more off

display "============================================================"
display "  Pendulo: modelo LINEARIZADO via sspace (Stata)"
display "  Referencia limitada - Stata nao suporta EKF/UKF/EnKF"
display "============================================================"
display ""

* ---------------------------------------------------------------------------
* Dados
* ---------------------------------------------------------------------------
import delimited using "../../data/pendulum.csv", clear

generate t_idx = _n
tsset t_idx

local n_obs = _N
* dt = t[2] - t[1] (timestamp uniforme do gerador)
local dt = t[2] - t[1]
local g  = 9.81
local L  = 1.0
local omega2 = `g' / `L'

display "Observations  : `n_obs'"
display "dt            : `dt'"
display "omega2 (g/L)  : `omega2'"
display ""

* y observado (sin(theta) + ruido). Para a LINEARIZACAO, interpretamos
* y ~ theta + ruido, o que e exato apenas quando |theta| e pequeno.
rename y_sin_theta y_obs

* Verificacao rapida: magnitude de theta no dataset
summarize x_theta, detail
display ""
display "NOTA: se |theta| excede ~0.3 rad, a aproximacao sin(theta)~theta"
display "      deixa de ser razoavel. O dataset inicia em theta=1.2 rad"
display "      (oscilacao grande), entao espera-se erro de linearizacao."
display ""

* ---------------------------------------------------------------------------
* sspace: pendulo LINEARIZADO
*
* Especificamos os dois estados (theta, theta_dot) explicitamente.
* F depende de omega2 * dt^2 e dt, que sao pinados via constraints.
* ---------------------------------------------------------------------------
display "--- sspace: pendulo linearizado (sin(theta) ~ theta) ---"
display ""

local F11 = 1 - (`dt')^2 * `omega2'
local F12 = `dt'
local F21 = -`dt' * `omega2'
local F22 = 1

display "F_linear:"
display "   [[ " %8.4f `F11' ", " %8.4f `F12' " ],"
display "    [ " %8.4f `F21' ", " %8.4f `F22' " ]]"
display ""

* Pinagem dos elementos de F (transicao) e Z (medida)
constraint 1 [theta]L.theta       = `F11'
constraint 2 [theta]L.theta_dot   = `F12'
constraint 3 [theta_dot]L.theta     = `F21'
constraint 4 [theta_dot]L.theta_dot = `F22'
constraint 5 [y_obs]theta     = 1
constraint 6 [y_obs]theta_dot = 0

* Ruido de processo: apenas theta_dot tem perturbacao (consistente
* com o gerador Python: Q = diag(0, (dt*sigma_proc)^2)).
* Ruido de medida: estimado livre.
capture noisily sspace                                                   ///
    (theta     = L.theta + L.theta_dot, state noconstant noerror)        ///
    (theta_dot = L.theta + L.theta_dot, state noconstant)                ///
    (y_obs     = theta + theta_dot, noconstant),                         ///
    constraints(1 2 3 4 5 6) covstate(diagonal)                          ///
    difficult iterate(500)

if _rc != 0 {
    display as error "sspace falhou (codigo: " _rc ")."
    display as error "Isto e esperado se a linearizacao for instavel."
}

local loglik_lin = .
local sigma2_obs = .
local sigma2_td  = .
capture {
    local loglik_lin = e(ll)
    local sigma2_obs = exp(2 * _b[/ln_var(y_obs)])
    local sigma2_td  = exp(2 * _b[/ln_var(theta_dot)])
}

display ""
display "--- Parametros estimados (modelo linearizado) ---"
display "logLik         = " %12.4f `loglik_lin'
display "sigma2_obs     = " %12.6f `sigma2_obs'
display "sigma2_td      = " %12.6f `sigma2_td'
display ""

* ---------------------------------------------------------------------------
* Estados filtrados e suavizados
* ---------------------------------------------------------------------------
capture {
    predict theta_f,     state equation(theta)     smethod(onestep)
    predict theta_dot_f, state equation(theta_dot) smethod(onestep)
    predict theta_s,     state equation(theta)     smethod(smooth)
    predict theta_dot_s, state equation(theta_dot) smethod(smooth)
}

* RMSE contra estados verdadeiros
capture {
    generate err_theta_f     = (theta_f     - x_theta)^2
    generate err_theta_dot_f = (theta_dot_f - x_theta_dot)^2
    generate err_theta_s     = (theta_s     - x_theta)^2
    generate err_theta_dot_s = (theta_dot_s - x_theta_dot)^2
    quietly summarize err_theta_f
    local rmse_theta_f = sqrt(r(mean))
    quietly summarize err_theta_dot_f
    local rmse_theta_dot_f = sqrt(r(mean))
    quietly summarize err_theta_s
    local rmse_theta_s = sqrt(r(mean))
    quietly summarize err_theta_dot_s
    local rmse_theta_dot_s = sqrt(r(mean))

    display "--- RMSE (sspace linearizado vs. verdadeiro) ---"
    display "RMSE theta (filt) = " %10.6f `rmse_theta_f'
    display "RMSE theta_dot (filt) = " %10.6f `rmse_theta_dot_f'
    display "RMSE theta (smooth) = " %10.6f `rmse_theta_s'
    display "RMSE theta_dot (smooth) = " %10.6f `rmse_theta_dot_s'
    display ""
    display "ATENCAO: estes RMSE DEVEM ser substancialmente piores do que"
    display "os obtidos com EKF/UKF (kalmanbox), porque a linearizacao"
    display "global falha para theta grande. Isto documenta a necessidade"
    display "de filtros nao-lineares, que Stata nao oferece nativamente."
}

* ---------------------------------------------------------------------------
* Exportacao dos estados para comparacao cruzada
* ---------------------------------------------------------------------------
capture {
    preserve
    keep t x_theta x_theta_dot theta_f theta_dot_f theta_s theta_dot_s
    rename theta_f     lin_theta_filt
    rename theta_dot_f lin_theta_dot_filt
    rename theta_s     lin_theta_smooth
    rename theta_dot_s lin_theta_dot_smooth
    order t x_theta x_theta_dot lin_theta_filt lin_theta_dot_filt ///
          lin_theta_smooth lin_theta_dot_smooth
    export delimited using "results_reference_stata_states.csv", replace
    display "Saved: results_reference_stata_states.csv"
    restore
}

* ---------------------------------------------------------------------------
* Sumario
* ---------------------------------------------------------------------------
preserve
clear
set obs 20
generate str40 metric = ""
generate double value = .

replace metric = "loglik_lin_sspace"   in 1
replace value  = `loglik_lin'          in 1
replace metric = "sigma2_obs_stata"    in 2
replace value  = `sigma2_obs'          in 2
replace metric = "sigma2_theta_dot"    in 3
replace value  = `sigma2_td'           in 3
replace metric = "rmse_theta_filt_lin" in 4
capture replace value = `rmse_theta_f'     in 4
replace metric = "rmse_theta_dot_filt_lin" in 5
capture replace value = `rmse_theta_dot_f' in 5
replace metric = "rmse_theta_smooth_lin" in 6
capture replace value = `rmse_theta_s'     in 6
replace metric = "rmse_theta_dot_smooth_lin" in 7
capture replace value = `rmse_theta_dot_s' in 7
replace metric = "dt"                  in 8
replace value  = `dt'                  in 8
replace metric = "omega2"              in 9
replace value  = `omega2'              in 9
replace metric = "n_obs"               in 10
replace value  = `n_obs'               in 10
replace metric = "supports_ekf"        in 11
replace value  = 0                     in 11
replace metric = "supports_ukf"        in 12
replace value  = 0                     in 12
replace metric = "supports_enkf"       in 13
replace value  = 0                     in 13

drop if missing(metric) | metric == ""
export delimited using "results_reference_stata_summary.csv", replace
display "Saved: results_reference_stata_summary.csv"
restore

display ""
display "============================================================"
display "  Validacao de referencia linearizada concluida."
display ""
display "  LEMBRETE: Stata nao substitui EKF/UKF/EnKF. Para o pendulo"
display "  nao-linear, use kalmanbox (Python) ou implementacao manual."
display "============================================================"
