# Dynamic Factor Models (DFM)

## Overview

Dynamic Factor Models extract a small number of latent common factors from a
large panel of observed time series. They are widely used in macroeconomic
nowcasting, forecasting, and index construction.

## Model Specification

### Observation Equation

$$y_t = \Lambda f_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, R)$$

where:
- $y_t$ is an $N \times 1$ vector of observed series
- $f_t$ is a $K \times 1$ vector of latent factors ($K \ll N$)
- $\Lambda$ is the $N \times K$ factor loading matrix
- $R$ is the $N \times N$ idiosyncratic covariance (typically diagonal)

### State (Factor) Transition Equation

$$f_t = \Phi f_{t-1} + \eta_t, \quad \eta_t \sim N(0, Q)$$

where:
- $\Phi$ is the $K \times K$ factor transition matrix (VAR dynamics)
- $Q$ is the $K \times K$ factor innovation covariance

### Identification

The model requires identification restrictions because the factors and loadings
are only jointly identified. Common strategies:
1. **Normalization**: Fix $Q = I_K$ and impose upper-triangular structure on
   the first $K$ rows of $\Lambda$
2. **PCA rotation**: Estimate factors via PCA, then fit the state-space model
3. **Bayesian priors**: Shrinkage priors on loadings provide soft identification

### Estimation

1. **Two-step (Stock & Watson)**: PCA on standardized panel to extract factors,
   then fit VAR to estimated factors
2. **Quasi-ML (Doz, Giannone, Reichlin)**: PCA initialization followed by
   EM algorithm on the state-space representation
3. **Full ML**: Direct numerical optimization of the Kalman filter likelihood
4. **Bayesian MCMC**: Gibbs sampling alternating between factors, loadings,
   and variance parameters

## Datasets

### us_macro_panel.csv

Panel of 15 synthetic US macroeconomic series, monthly, 2000-01 to 2023-12
(288 observations). All series are standardized (mean 0, std 1). The data is
calibrated to exhibit 2-3 common factors:

| Series | Description | Primary Factor |
|--------|-------------|----------------|
| gdp_growth | Real GDP growth | Real activity |
| industrial_production | Industrial production index | Real activity |
| unemployment | Unemployment rate | Real activity |
| payrolls | Nonfarm payrolls | Real activity |
| retail_sales | Retail sales growth | Real activity |
| housing_starts | Housing starts | Real activity |
| consumer_confidence | Consumer confidence index | Sentiment |
| pmi_manufacturing | PMI manufacturing | Sentiment |
| cpi_inflation | CPI inflation | Prices/financial |
| pce_inflation | PCE inflation | Prices/financial |
| fed_funds_rate | Federal funds rate | Prices/financial |
| sp500_returns | S&P 500 returns | Prices/financial |
| term_spread | Term spread (10y - 2y) | Prices/financial |
| credit_spread | Credit spread (BAA - AAA) | Prices/financial |
| oil_price_change | Oil price change | Sentiment |

### mixed_freq_macro.csv

Mixed-frequency dataset for DFM-MF estimation, 2000-01 to 2023-12. Monthly
series (industrial_production, unemployment, cpi, pmi) are fully observed.
The quarterly series (gdp_growth) is observed only in quarter-end months
(March, June, September, December); all other months contain NaN.

## Notebooks

1. **Notebook 1 - Static DFM**: Extract factors via PCA and state-space ML
   from the US macro panel
2. **Notebook 2 - Dynamic DFM with EM**: EM algorithm estimation following
   Doz, Giannone & Reichlin (2012)
3. **Notebook 3 - Mixed-Frequency DFM**: Handle missing observations for
   nowcasting with mixed-frequency data

## References

- Stock, J.H. and Watson, M.W. (2002). "Forecasting Using Principal Components
  from a Large Number of Predictors." *Journal of the American Statistical
  Association*, 97(460), 1167-1179.
- Banbura, M., Giannone, D. and Reichlin, L. (2011). "Nowcasting."
  *Oxford Handbook of Economic Forecasting*, Chapter 7.
- Doz, C., Giannone, D. and Reichlin, L. (2012). "A Quasi-Maximum Likelihood
  Approach for Large, Approximate Dynamic Factor Models."
  *Review of Economics and Statistics*, 94(4), 1014-1024.
- Bai, J. and Ng, S. (2002). "Determining the Number of Factors in Approximate
  Factor Models." *Econometrica*, 70(1), 191-221.
