# Key Concepts

## State-Space Models

A state-space model represents a time series as the output of an unobserved
(latent) state process:

**Observation equation:**

$$y_t = Z_t \alpha_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, H_t)$$

**State equation:**

$$\alpha_{t+1} = T_t \alpha_t + R_t \eta_t, \quad \eta_t \sim N(0, Q_t)$$

where:

- $y_t$ is the observed data at time $t$
- $\alpha_t$ is the unobserved state vector
- $Z_t$ is the observation (design) matrix
- $T_t$ is the state transition matrix
- $R_t$ is the selection matrix
- $H_t$ is the observation noise covariance
- $Q_t$ is the state noise covariance

## The Kalman Filter

The Kalman filter is a recursive algorithm that estimates the unobserved
states $\alpha_t$ given the observations $y_1, \ldots, y_t$. It operates
in two steps at each time point:

1. **Prediction**: Project the state forward using the transition equation
2. **Update**: Incorporate the new observation to refine the estimate

## Models in KalmanBox

| Model | States | Use Case |
|:------|:-------|:---------|
| LocalLevel | Level | Slowly changing mean |
| LocalLinearTrend | Level + Trend | Mean with drift |
| BSM | Level + Trend + Seasonal | Seasonal time series |
| UCM | Configurable components | Flexible decomposition |
| DynamicFactor | Common factors | Multivariate co-movement |
| TVPRegression | Time-varying coefficients | Regression with instability |
| ARIMASSM | ARIMA states | ARIMA via state space |

## Estimation

Parameters (noise variances) are estimated by **Maximum Likelihood Estimation (MLE)**,
which maximizes the log-likelihood computed as a by-product of the Kalman filter.
