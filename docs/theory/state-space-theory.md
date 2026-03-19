# State-Space Models

## General formulation

A linear Gaussian state-space model consists of two equations:

**Observation equation:**

$$y_t = Z_t \alpha_t + d_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, H_t)$$

**State transition equation:**

$$\alpha_{t+1} = T_t \alpha_t + c_t + R_t \eta_t, \quad \eta_t \sim N(0, Q_t)$$

## System matrices

| Matrix | Dimension | Name |
|:-------|:----------|:-----|
| $Z_t$ | $p \times m$ | Design (observation) matrix |
| $T_t$ | $m \times m$ | Transition matrix |
| $R_t$ | $m \times r$ | Selection matrix |
| $H_t$ | $p \times p$ | Observation noise covariance |
| $Q_t$ | $r \times r$ | State noise covariance |
| $d_t$ | $p \times 1$ | Observation intercept |
| $c_t$ | $m \times 1$ | State intercept |

where $p$ = number of observables, $m$ = number of states, $r$ = number of
state disturbances.

## Initial conditions

The filter requires initial values $\alpha_1 \sim N(a_1, P_1)$:

- **Exact diffuse**: $P_1 = \kappa I$ with $\kappa \to \infty$
  (used when prior information is unavailable)
- **Stationary**: $P_1$ is the solution to the discrete Lyapunov equation
  $P = T P T' + R Q R'$

## Examples of state-space models

### Local Level
- $m = 1$, $p = 1$, $r = 1$
- $Z = [1]$, $T = [1]$, $R = [1]$

### Local Linear Trend
- $m = 2$, $p = 1$, $r = 2$
- $Z = [1, 0]$, $T = [[1,1],[0,1]]$, $R = I_2$

### ARIMA(p,d,q)
- Can be represented in state-space form
- Allows unified estimation and missing data handling

## References

- Durbin, J. & Koopman, S.J. (2012). *Time Series Analysis by State Space Methods*. Chapter 2.
- Harvey, A.C. (1989). *Forecasting, Structural Time Series and the Kalman Filter*. Chapter 3.
