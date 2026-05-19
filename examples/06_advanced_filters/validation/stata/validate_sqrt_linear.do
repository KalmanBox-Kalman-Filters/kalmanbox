* =============================================================================
* Validacao: caso LINEAR puro como referencia para SQRT / Information filter
*
* Contexto
* --------------------------------------------------------------------------
* Square-root (SQRT) e Information filters sao reformulacoes numericamente
* equivalentes do filtro de Kalman linear. No caso linear-Gaussiano, todas
* as tres formulacoes (KF padrao, SQRT, Information) devem produzir
* estimativas IDENTICAS (ate precisao de ponto flutuante).
*
* Stata `sspace` implementa o KF padrao (Durbin-Koopman) e, portanto, serve
* como referencia exogena para o caso linear. Stata NAO oferece:
*   - Forma square-root/Cholesky explicita (embora o algoritmo interno possa
*     usar atualizacao square-root)
*   - Forma de informacao (precisao vs. covariancia)
*
* REFERENCIA: [TS] sspace, Stata Reference Manual. Como sspace expoe apenas
* a interface de momento, usamos seus outputs (estados filtrados,
* suavizados, logLik) como "verdade linear" a ser comparada com a saida
* do SQRT/Information filter do kalmanbox (Python) e FKF (R).
*
* --------------------------------------------------------------------------
* MODELO: constant-velocity 2D (target tracking sintetico)
*   state x = [px, vx, py, vy]'
*   F = [[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]]
*   H = [[1,0,0,0],[0,0,1,0]]          (observa posicao)
*   Q = CWNA, bloco diagonal
*   R = diag(sigma_pos^2, sigma_pos^2)
*
* Nota: dataset target_tracking.csv tem obs nao-linear (range/bearing).
* Aqui substituimos a observacao por (px, py) + ruido Gaussiano, para que
* o problema fique estritamente LINEAR-Gaussiano e as tres formulacoes
* possam ser confrontadas.
* =============================================================================

clear all
set more off
set seed 42

display "============================================================"
display "  SQRT / Info filter: referencia LINEAR via sspace (Stata)"
display "  Stata oferece apenas forma de momento; usamos como benchmark"
display "============================================================"
display ""

* ---------------------------------------------------------------------------
* Dados
* ---------------------------------------------------------------------------
import delimited using "../../data/target_tracking.csv", clear

local n_obs = _N
local dt    = t[2] - t[1]

display "Observations: `n_obs'"
display "dt          : `dt'"
display ""

* Constroi observacao linear sintetica: y = posicao + ruido N(0, sigma_pos^2)
* Seed fixa em 42; DEVE bater com validate_sqrt_info.R (mesma seed/sigma).
local sigma_pos = 1.0
generate noise_x = rnormal(0, `sigma_pos')
generate noise_y = rnormal(0, `sigma_pos')
generate y1 = x_x + noise_x
generate y2 = x_y + noise_y

* Indice temporal
generate t_idx = _n
tsset t_idx

* ---------------------------------------------------------------------------
* Parametros do sistema (coincidem com o gerador Python)
* ---------------------------------------------------------------------------
local sigma_proc = 0.1
local q = `sigma_proc'^2
local F12 = `dt'
local F34 = `dt'
local Q11 = `q' * (`dt')^3 / 3
local Q12 = `q' * (`dt')^2 / 2
local Q22 = `q' * `dt'
local Q33 = `Q11'
local Q34 = `Q12'
local Q44 = `Q22'

display "--- Parametros do sistema ---"
display "F12 = F34 = dt = " %8.4f `F12'
display "Q diagonal blocks (q * [dt^3/3, dt^2/2, dt])"
display "R = sigma_pos^2 * I = " %8.4f `sigma_pos'^2
display ""

* ---------------------------------------------------------------------------
* sspace: constant-velocity 2D
*
* Observamos y1 (medida de px) e y2 (medida de py). Os estados sao
* px, vx, py, vy. As transicoes entre px/vx e py/vy sao blocos 2x2.
*
* Pinagem:
*   F[px, L.px] = 1
*   F[px, L.vx] = dt
*   F[vx, L.vx] = 1
*   (analogamente para py, vy)
*
* Usamos `covstate(diagonal)` como simplificacao. Isso introduz uma
* aproximacao (o Q correto e bloco-nao-diagonal via Q12, Q34), entao
* a comparacao com SQRT/Info deve ser entendida como CONSISTENTE em
* ORDEM DE MAGNITUDE, nao como igualdade numerica. Para igualdade
* exata, o teste canonico e feito em R (FKF) contra Python.
* ---------------------------------------------------------------------------
display "--- sspace: 2D constant-velocity, obs linear ---"
display ""

constraint 1  [px]L.px = 1
constraint 2  [px]L.vx = `F12'
constraint 3  [vx]L.vx = 1
constraint 4  [py]L.py = 1
constraint 5  [py]L.vy = `F34'
constraint 6  [vy]L.vy = 1
constraint 7  [y1]px = 1
constraint 8  [y1]vx = 0
constraint 9  [y1]py = 0
constraint 10 [y1]vy = 0
constraint 11 [y2]px = 0
constraint 12 [y2]vx = 0
constraint 13 [y2]py = 1
constraint 14 [y2]vy = 0

capture noisily sspace                                                    ///
    (px = L.px + L.vx, state noconstant noerror)                          ///
    (vx = L.vx, state noconstant)                                         ///
    (py = L.py + L.vy, state noconstant noerror)                          ///
    (vy = L.vy, state noconstant)                                         ///
    (y1 = px + vx + py + vy, noconstant)                                  ///
    (y2 = px + vx + py + vy, noconstant),                                 ///
    constraints(1 2 3 4 5 6 7 8 9 10 11 12 13 14)                         ///
    covstate(diagonal) covobserved(diagonal)                              ///
    difficult iterate(500)

if _rc != 0 {
    display as error "sspace falhou (codigo: " _rc ")."
}

local loglik_sspace = .
capture local loglik_sspace = e(ll)

display ""
display "logLik (sspace) = " %12.4f `loglik_sspace'
display ""

* ---------------------------------------------------------------------------
* Estados filtrados e suavizados
* ---------------------------------------------------------------------------
capture {
    predict px_f, state equation(px) smethod(onestep)
    predict vx_f, state equation(vx) smethod(onestep)
    predict py_f, state equation(py) smethod(onestep)
    predict vy_f, state equation(vy) smethod(onestep)
    predict px_s, state equation(px) smethod(smooth)
    predict vx_s, state equation(vx) smethod(smooth)
    predict py_s, state equation(py) smethod(smooth)
    predict vy_s, state equation(vy) smethod(smooth)
}

capture {
    generate e_px = (px_f - x_x)^2
    generate e_vx = (vx_f - x_vx)^2
    generate e_py = (py_f - x_y)^2
    generate e_vy = (vy_f - x_vy)^2
    quietly summarize e_px
    local rmse_px = sqrt(r(mean))
    quietly summarize e_vx
    local rmse_vx = sqrt(r(mean))
    quietly summarize e_py
    local rmse_py = sqrt(r(mean))
    quietly summarize e_vy
    local rmse_vy = sqrt(r(mean))

    display "--- RMSE (sspace linear vs. estado verdadeiro) ---"
    display "RMSE px = " %10.6f `rmse_px'
    display "RMSE vx = " %10.6f `rmse_vx'
    display "RMSE py = " %10.6f `rmse_py'
    display "RMSE vy = " %10.6f `rmse_vy'
    display ""
}

* ---------------------------------------------------------------------------
* Exporta estados filtrados/suavizados
* ---------------------------------------------------------------------------
capture {
    preserve
    keep t x_x x_vx x_y x_vy px_f vx_f py_f vy_f px_s vx_s py_s vy_s
    rename px_f sspace_px_filt
    rename vx_f sspace_vx_filt
    rename py_f sspace_py_filt
    rename vy_f sspace_vy_filt
    rename px_s sspace_px_smooth
    rename vx_s sspace_vx_smooth
    rename py_s sspace_py_smooth
    rename vy_s sspace_vy_smooth
    order t x_x x_vx x_y x_vy ///
          sspace_px_filt sspace_vx_filt sspace_py_filt sspace_vy_filt ///
          sspace_px_smooth sspace_vx_smooth sspace_py_smooth sspace_vy_smooth
    export delimited using "results_sqrt_linear_stata_states.csv", replace
    display "Saved: results_sqrt_linear_stata_states.csv"
    restore
}

* ---------------------------------------------------------------------------
* Sumario para comparacao cross-language
* ---------------------------------------------------------------------------
preserve
clear
set obs 20
generate str40 metric = ""
generate double value = .

replace metric = "loglik_sspace"       in 1
replace value  = `loglik_sspace'       in 1
replace metric = "rmse_px_stata"       in 2
capture replace value = `rmse_px'      in 2
replace metric = "rmse_vx_stata"       in 3
capture replace value = `rmse_vx'      in 3
replace metric = "rmse_py_stata"       in 4
capture replace value = `rmse_py'      in 4
replace metric = "rmse_vy_stata"       in 5
capture replace value = `rmse_vy'      in 5
replace metric = "dt"                  in 6
replace value  = `dt'                  in 6
replace metric = "sigma_pos"           in 7
replace value  = `sigma_pos'           in 7
replace metric = "sigma_proc"          in 8
replace value  = `sigma_proc'          in 8
replace metric = "n_obs"               in 9
replace value  = `n_obs'               in 9
replace metric = "supports_sqrt_form"  in 10
replace value  = 0                     in 10
replace metric = "supports_info_form"  in 11
replace value  = 0                     in 11
replace metric = "form_exposed"        in 12
replace value  = 1                     in 12
* 1 = momento (mean/covariance)

drop if missing(metric) | metric == ""
export delimited using "results_sqrt_linear_stata_summary.csv", replace
display "Saved: results_sqrt_linear_stata_summary.csv"
restore

display ""
display "============================================================"
display "  Caso linear concluido. Use este CSV como referencia para"
display "  confrontar SQRT / Information filter do kalmanbox (Python)"
display "  e FKF (R) no mesmo dataset."
display ""
display "  LEMBRETE: Stata nao expoe forma square-root ou informacional."
display "============================================================"
