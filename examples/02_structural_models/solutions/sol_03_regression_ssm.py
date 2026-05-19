"""Solution: Regression SSM on UK gas consumption data.

Validates kalmanbox RegressionSSM (fixed coefficients) against statsmodels OLS,
and fits TimeVaryingParameters (TVP) model for structural change analysis.

Expected behavior:
  - Fixed-coefficient SSM recovers OLS coefficients (tolerance < 1%)
  - sigma2 estimate matches OLS MLE sigma2
  - TVP model estimates coefficient evolution over time
  - AIC/BIC comparison between fixed and TVP models

References:
  - Harvey, A.C. (1989). Forecasting, Structural Time Series Models.
  - Durbin, J. & Koopman, S.J. (2012). Time Series Analysis by State Space Methods.
  - Stock, J.H. & Watson, M.W. (1996). Evidence on Structural Instability.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Ensure kalmanbox is importable
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from kalmanbox.models.regression_ssm import RegressionSSM
from kalmanbox.models.tvp import TimeVaryingParameters

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COEF_TOL = 0.01          # 1% relative tolerance for fixed coefs vs OLS
SIGMA2_TOL = 0.05        # 5% relative tolerance for sigma2

# ---------------------------------------------------------------------------
# 1. Load data and construct regressors
# ---------------------------------------------------------------------------
print("=" * 60)
print("Solution 03: Regression SSM — UK Gas Consumption")
print("=" * 60)

gas = pd.read_csv(DATA_DIR / "uk_gas.csv")
gas["date"] = pd.to_datetime(gas["date"])
y = gas["gas_consumption"].values.astype(np.float64)
n = len(y)

# Create date index and extract quarters
dates = gas["date"]
quarters = dates.dt.quarter.values

# Construct regressors: intercept + trend + 3 seasonal dummies (Q2, Q3, Q4)
intercept = np.ones(n)
trend = np.arange(1, n + 1, dtype=np.float64)
dummy_q2 = (quarters == 2).astype(np.float64)
dummy_q3 = (quarters == 3).astype(np.float64)
dummy_q4 = (quarters == 4).astype(np.float64)

X = np.column_stack([intercept, trend, dummy_q2, dummy_q3, dummy_q4])
regressor_names = ["intercept", "trend", "Q2", "Q3", "Q4"]

print(f"Loaded {n} observations (UK gas consumption, 1960-1986)")
print(f"Regressors: {regressor_names}")
print(f"Gas stats: mean={y.mean():.1f}, std={y.std():.1f}, range=[{y.min():.1f}, {y.max():.1f}]")


def rel_err(actual: float, expected: float) -> float:
    if abs(expected) < 1e-10:
        return abs(actual - expected)
    return abs(actual - expected) / abs(expected)


# ---------------------------------------------------------------------------
# 2. Fit kalmanbox RegressionSSM (fixed coefficients)
# ---------------------------------------------------------------------------
print("\n--- kalmanbox RegressionSSM (fixed coefficients) ---")
model_fixed = RegressionSSM(y, X)
results_fixed = model_fixed.fit()

print("  Estimated coefficients:")
for i, name in enumerate(regressor_names):
    val = results_fixed.params[i]
    se = results_fixed.se[i]
    t_stat = val / se if se > 0 else np.nan
    print(f"    {name:12s}: {val:10.4f} (SE={se:.4f}, t={t_stat:.2f})")
sigma2_ssm = results_fixed.params[-1]
print(f"    {'sigma2':12s}: {sigma2_ssm:10.4f}")
print(f"  Log-likelihood: {results_fixed.loglike:.4f}")

# ---------------------------------------------------------------------------
# 3. Fit OLS reference (statsmodels)
# ---------------------------------------------------------------------------
print("\n--- OLS reference (statsmodels) ---")
ols_model = sm.OLS(y, X)
ols_results = ols_model.fit()

print("  OLS coefficients:")
for i, name in enumerate(regressor_names):
    print(f"    {name:12s}: {ols_results.params[i]:10.4f}")
sigma2_ols_mle = float(np.sum(ols_results.resid**2) / n)  # MLE (not unbiased)
print(f"    {'sigma2 (MLE)':12s}: {sigma2_ols_mle:10.4f}")
print(f"  Log-likelihood: {ols_results.llf:.4f}")

# ---------------------------------------------------------------------------
# 4. Validate: fixed SSM coefficients ~= OLS (tolerance < 1%)
# ---------------------------------------------------------------------------
print("\n--- Validation: SSM vs OLS coefficients ---")
max_coef_diff = 0.0
for i, name in enumerate(regressor_names):
    ssm_val = results_fixed.params[i]
    ols_val = ols_results.params[i]
    err = rel_err(ssm_val, ols_val)
    status = "PASS" if err < COEF_TOL else "FAIL"
    print(f"    {name:12s}: SSM={ssm_val:10.4f}  OLS={ols_val:10.4f}  "
          f"rel_err={err:.6f}  [{status}]")
    assert err < COEF_TOL, f"{name} coefficient mismatch: rel_err={err:.6f} > {COEF_TOL}"
    max_coef_diff = max(max_coef_diff, err)

print(f"  Max relative error: {max_coef_diff:.8f}")
print("  [PASS] All fixed coefficients match OLS within 1% tolerance")

# ---------------------------------------------------------------------------
# 5. Validate: sigma2 matches OLS MLE
# ---------------------------------------------------------------------------
print("\n--- Validation: sigma2 ---")
err_sigma2 = rel_err(sigma2_ssm, sigma2_ols_mle)
print(f"  SSM sigma2:     {sigma2_ssm:.4f}")
print(f"  OLS sigma2 MLE: {sigma2_ols_mle:.4f}")
print(f"  relative error: {err_sigma2:.6f} (tol {SIGMA2_TOL})")
assert err_sigma2 < SIGMA2_TOL, f"sigma2 mismatch: rel_err={err_sigma2:.6f}"
print("  [PASS] sigma2 matches OLS MLE")

# ---------------------------------------------------------------------------
# 6. Fit Time-Varying Parameters model
# ---------------------------------------------------------------------------
print("\n--- kalmanbox TVP (time-varying coefficients) ---")
model_tvp = TimeVaryingParameters(y, X, q_type="diagonal")
results_tvp = model_tvp.fit()

print("  TVP parameters:")
for name, val in zip(results_tvp.param_names, results_tvp.params):
    print(f"    {name:20s} = {val:.6f}")
print(f"  Log-likelihood: {results_tvp.loglike:.4f}")

# Validate TVP converged and produced reasonable results
assert results_tvp.smoothed_state is not None, "TVP smoother output missing"
print("  [PASS] TVP model fitted successfully")

# ---------------------------------------------------------------------------
# 7. TVP coefficient evolution analysis
# ---------------------------------------------------------------------------
print("\n--- TVP coefficient evolution ---")
smoothed_states = results_tvp.smoothed_state

for i, name in enumerate(regressor_names):
    tvp_coef = smoothed_states[:, i]
    ols_val = ols_results.params[i]
    tvp_range = tvp_coef.max() - tvp_coef.min()
    print(f"  {name:12s}: range=[{tvp_coef.min():.2f}, {tvp_coef.max():.2f}], "
          f"OLS={ols_val:.2f}, span={tvp_range:.2f}")

# ---------------------------------------------------------------------------
# 8. Model comparison: AIC/BIC
# ---------------------------------------------------------------------------
print("\n--- Model comparison: AIC / BIC ---")
n_params_fixed = len(results_fixed.params)
n_params_tvp = len(results_tvp.params)

aic_fixed = results_fixed.aic
bic_fixed = results_fixed.bic
aic_tvp = results_tvp.aic
bic_tvp = results_tvp.bic

print(f"  {'Model':<30} {'LL':>10} {'AIC':>10} {'BIC':>10}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
print(f"  {'Fixed Coeff (SSM)':<30} {results_fixed.loglike:>10.2f} {aic_fixed:>10.2f} {bic_fixed:>10.2f}")
print(f"  {'Time-Varying (TVP)':<30} {results_tvp.loglike:>10.2f} {aic_tvp:>10.2f} {bic_tvp:>10.2f}")
print(f"  {'OLS (statsmodels)':<30} {ols_results.llf:>10.2f} {ols_results.aic:>10.2f} {ols_results.bic:>10.2f}")

if aic_tvp < aic_fixed:
    print(f"\n  AIC prefers: TVP (delta = {aic_fixed - aic_tvp:.1f})")
else:
    print(f"\n  AIC prefers: Fixed (delta = {aic_tvp - aic_fixed:.1f})")

if bic_tvp < bic_fixed:
    print(f"  BIC prefers: TVP (delta = {bic_fixed - bic_tvp:.1f})")
else:
    print(f"  BIC prefers: Fixed (delta = {bic_tvp - bic_fixed:.1f})")

# ---------------------------------------------------------------------------
# 9. Significance test: are TVP coefficients really time-varying?
# ---------------------------------------------------------------------------
print("\n--- Signal-to-noise test ---")
sigma2_tvp_obs = results_tvp.params[0]
for i, name in enumerate(regressor_names):
    sigma2_beta_i = results_tvp.params[1 + i]
    ratio = sigma2_beta_i / sigma2_tvp_obs if sigma2_tvp_obs > 0 else np.inf
    print(f"  sigma2_{name:12s} / sigma2_obs = {ratio:.6f}")

# ---------------------------------------------------------------------------
# 10. Generate diagnostic plots
# ---------------------------------------------------------------------------
print("\n--- Generating figures ---")

# Figure 1: TVP coefficient evolution (6-panel)
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

for i, (name, ax) in enumerate(zip(regressor_names, axes.flat)):
    tvp_coef = smoothed_states[:, i]
    tvp_se = np.sqrt(results_tvp.smoothed_cov[:, i, i])

    ax.plot(dates, tvp_coef, "b-", linewidth=1.2, label="TVP (smoothed)")
    ax.fill_between(dates, tvp_coef - 1.96 * tvp_se, tvp_coef + 1.96 * tvp_se,
                    alpha=0.2, color="blue")
    ax.axhline(ols_results.params[i], color="red", linestyle="--",
               linewidth=1, label=f"OLS = {ols_results.params[i]:.2f}")
    ax.set_title(f"Coefficient: {name}")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)

# Hide extra subplot
if len(regressor_names) < len(axes.flat):
    axes.flat[-1].set_visible(False)

plt.suptitle("Time-Varying vs Fixed Coefficients", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "regression_ssm_tvp_coefficients.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'regression_ssm_tvp_coefficients.png'}")

# Figure 2: Forecast comparison
n_forecast = 8

# Fixed forecast: y_hat = X_future @ beta_hat
last_q = quarters[-1]
future_quarters = np.array([(last_q + i) % 4 + 1 for i in range(1, n_forecast + 1)])
future_trend = np.arange(n + 1, n + 1 + n_forecast, dtype=np.float64)
future_X = np.column_stack([
    np.ones(n_forecast),
    future_trend,
    (future_quarters == 2).astype(np.float64),
    (future_quarters == 3).astype(np.float64),
    (future_quarters == 4).astype(np.float64),
])

beta_fixed = results_fixed.params[:-1]
fc_fixed = future_X @ beta_fixed

# TVP forecast
beta_last = results_tvp.smoothed_state[-1]
P_last = results_tvp.smoothed_cov[-1]
sigma2_tvp_obs_val = results_tvp.params[0]

k = len(regressor_names)
Q_tvp = np.diag(results_tvp.params[1:1 + k])

fc_tvp = np.zeros(n_forecast)
fc_se_tvp = np.zeros(n_forecast)
beta_h = beta_last.copy().ravel()
P_h = P_last.copy()

for h in range(n_forecast):
    if h > 0:
        P_h = P_h + Q_tvp
    X_h = future_X[h, :]
    fc_tvp[h] = X_h @ beta_h
    fc_se_tvp[h] = np.sqrt(float(X_h @ P_h @ X_h.T + sigma2_tvp_obs_val))

# Create forecast dates
last_date = dates.iloc[-1]
fc_dates = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=n_forecast, freq="QS")

fig, ax = plt.subplots(figsize=(14, 6))
n_show = 40
ax.plot(dates.values[-n_show:], y[-n_show:], "k-", linewidth=1, label="Observed")
ax.plot(fc_dates, fc_fixed, "r-", linewidth=1.5, label="Forecast (Fixed)", marker="o", markersize=4)
ax.plot(fc_dates, fc_tvp, "b-", linewidth=1.5, label="Forecast (TVP)", marker="s", markersize=4)
ax.fill_between(fc_dates, fc_tvp - 1.96 * fc_se_tvp, fc_tvp + 1.96 * fc_se_tvp,
                alpha=0.15, color="blue")
ax.axvline(dates.iloc[-1], color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Date")
ax.set_ylabel("Gas Consumption")
ax.set_title("Forecast Comparison: Fixed vs Time-Varying Coefficients")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "regression_ssm_forecast.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'regression_ssm_forecast.png'}")

# Figure 3: Data + fitted values comparison
fitted_fixed = X @ beta_fixed
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates, y, "k-", linewidth=0.8, alpha=0.6, label="Observed")
ax.plot(dates, fitted_fixed, "r-", linewidth=1.2, label="SSM (fixed)")
ax.set_xlabel("Date")
ax.set_ylabel("Gas Consumption")
ax.set_title("UK Gas Consumption: Observed vs Fitted (Fixed Regression SSM)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "regression_ssm_fitted.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'regression_ssm_fitted.png'}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Solution 03: ALL CHECKS PASSED")
print("=" * 60)
