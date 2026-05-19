# ARIMA via State-Space Models

## Overview

Any ARIMA(p,d,q) model can be cast into state-space form, enabling estimation
via the Kalman filter. This equivalence is fundamental because it allows:

- **Maximum likelihood estimation** via the Kalman filter prediction error decomposition
- **Natural handling of missing observations** without modifications to the algorithm
- **Seamless extensions** to include regression effects, interventions, and structural breaks

## State-Space Representation of ARIMA

Following Harvey (1989, Chapter 4), an ARIMA(p,d,q) model can be written in
state-space form with state dimension `r = max(p, q+1)`.

### Observation equation

$$
y_t = Z \alpha_t + \varepsilon_t, \qquad \varepsilon_t \sim N(0, 0)
$$

where `Z = [1, 0, ..., 0]` is a `1 x r` selection vector. Note that the
observation noise variance is zero because all stochastic variation is captured
in the state equation.

### State equation

$$
\alpha_t = T \alpha_{t-1} + R \eta_t, \qquad \eta_t \sim N(0, Q)
$$

where:

- **T** is the `r x r` companion matrix constructed from the AR polynomial:

$$
T = \begin{bmatrix}
\phi_1 & 1 & 0 & \cdots & 0 \\
\phi_2 & 0 & 1 & \cdots & 0 \\
\vdots & & & \ddots & \vdots \\
\phi_{r-1} & 0 & 0 & \cdots & 1 \\
\phi_r & 0 & 0 & \cdots & 0
\end{bmatrix}
$$

- **R** is the `r x 1` vector encoding the MA polynomial:

$$
R = \begin{bmatrix} 1 \\ \theta_1 \\ \theta_2 \\ \vdots \\ \theta_{r-1} \end{bmatrix}
$$

- **Q** = `sigma^2` is the scalar innovation variance.

### Integration (differencing)

For integrated models (d > 0), the differencing operator is absorbed into the
state transition matrix. For ARIMA(p,d,q), the AR polynomial in the transition
matrix includes the differencing factors, i.e., `phi(B)(1-B)^d`.

## Datasets

| File | Description | Typical Model |
|------|-------------|---------------|
| `data/airline.csv` | Monthly international airline passengers (1949-1960, 144 obs) | ARIMA(0,1,1)x(0,1,1)_12 |
| `data/nile.csv` | Annual flow of the Nile at Aswan (1871-1970, 100 obs) | ARIMA(0,1,1) / Local Level |

## Notebooks

1. **ARIMA(0,1,1) for Nile data** - Simple exponential smoothing as a state-space model.
   Demonstrates the equivalence between the local level model and ARIMA(0,1,1).

2. **Seasonal ARIMA for Airline data** - Multiplicative seasonal ARIMA(0,1,1)x(0,1,1)_12
   in state-space form. Shows how seasonal differencing and MA terms map to
   the state vector and system matrices.

## References

- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*.
  Cambridge University Press. **Chapter 4: Time Series Models**.

- Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*
  (2nd ed.). Oxford University Press. **Chapter 3: Linear State Space Models**.

- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
  **Chapter 13: The Kalman Filter**.
