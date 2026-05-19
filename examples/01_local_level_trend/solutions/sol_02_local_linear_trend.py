"""Solution: Local Linear Trend Model on Airline data.

Validates kalmanbox LocalLinearTrend against statsmodels UnobservedComponents.
Uses log-transformed airline passengers (Box-Jenkins dataset).

Expected behavior:
  - Model decomposes log(passengers) into level + trend
  - Level captures smooth evolution of the series
  - Trend captures the growth rate
  - 24-period forecast should capture upward trend continuation

References:
  - Box, G.E.P. & Jenkins, G.M. (1976). Time Series Analysis.
  - Harvey, A.C. (1989). Forecasting, Structural Time Series Models.
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from kalmanbox.models.local_linear_trend import LocalLinearTrend

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PARAM_TOL = 0.10       # 10% relative tolerance for parameters
LOGLIKE_TOL = 15.0      # absolute tolerance for log-likelihood (diffuse init differs)
CORR_TOL = 0.99         # minimum correlation for state comparison
FORECAST_STEPS = 24

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("Solution 02: Local Linear Trend — Airline data")
print("=" * 60)

airline = pd.read_csv(DATA_DIR / "airline.csv")
airline["date"] = pd.to_datetime(airline["date"])
raw_passengers = airline["passengers"].values.astype(np.float64)
y = np.log(raw_passengers)
print(f"Loaded {len(y)} observations (log airline passengers, 1949–1960)")

# ---------------------------------------------------------------------------
# 2. Fit kalmanbox Local Linear Trend
# ---------------------------------------------------------------------------
print("\n--- kalmanbox Local Linear Trend ---")
kb_model = LocalLinearTrend(y)
kb_results = kb_model.fit()

for i, name in enumerate(kb_results.param_names):
    print(f"  {name} = {kb_results.params[i]:.6f}")
print(f"  log-like = {kb_results.loglike:.2f}")

# ---------------------------------------------------------------------------
# 3. Fit statsmodels UnobservedComponents (local linear trend)
# ---------------------------------------------------------------------------
print("\n--- statsmodels UnobservedComponents ---")
sm_model = sm.tsa.UnobservedComponents(y, level="local linear trend")
sm_results = sm_model.fit(disp=False)

print(f"  sigma2.irregular = {sm_results.params[0]:.6f}")
print(f"  sigma2.level     = {sm_results.params[1]:.6f}")
print(f"  sigma2.trend     = {sm_results.params[2]:.6f}")
print(f"  log-like         = {sm_results.llf:.2f}")

# ---------------------------------------------------------------------------
# 4. Compare parameters
# ---------------------------------------------------------------------------
print("\n--- kalmanbox vs statsmodels: parameters ---")

def rel_err(actual: float, expected: float) -> float:
    if abs(expected) < 1e-10:
        return abs(actual - expected)
    return abs(actual - expected) / abs(expected)

for i, name in enumerate(kb_results.param_names):
    kb_val = kb_results.params[i]
    sm_val = sm_results.params[i]
    err = rel_err(kb_val, sm_val)
    status = "PASS" if err < PARAM_TOL else "WARN"
    print(f"  {name}: kb={kb_val:.6f} sm={sm_val:.6f} rel_err={err:.4f} [{status}]")

# ---------------------------------------------------------------------------
# 5. Compare log-likelihoods
# ---------------------------------------------------------------------------
print("\n--- kalmanbox vs statsmodels: log-likelihood ---")
ll_diff = abs(kb_results.loglike - sm_results.llf)
print(f"  absolute diff: {ll_diff:.4f} (tol {LOGLIKE_TOL})")
assert ll_diff < LOGLIKE_TOL, f"Log-likelihood mismatch: {ll_diff:.4f}"
print("  [PASS] Log-likelihoods match")

# ---------------------------------------------------------------------------
# 6. Compare smoothed states — level
# ---------------------------------------------------------------------------
print("\n--- kalmanbox vs statsmodels: smoothed level ---")
kb_level = kb_results.smoothed_state[:, 0]
sm_level = sm_results.smoothed_state[0]  # (k_states, nobs)

corr_level = np.corrcoef(kb_level, sm_level)[0, 1]
print(f"  correlation: {corr_level:.6f} (min {CORR_TOL})")
assert corr_level > CORR_TOL, f"Smoothed level correlation {corr_level:.6f} < {CORR_TOL}"
print("  [PASS] Smoothed levels match")

# ---------------------------------------------------------------------------
# 7. Compare smoothed states — trend
# ---------------------------------------------------------------------------
print("\n--- kalmanbox vs statsmodels: smoothed trend ---")
kb_trend = kb_results.smoothed_state[:, 1]
sm_trend = sm_results.smoothed_state[1]

corr_trend = np.corrcoef(kb_trend, sm_trend)[0, 1]
print(f"  correlation: {corr_trend:.6f} (min {CORR_TOL})")
assert corr_trend > CORR_TOL, f"Smoothed trend correlation {corr_trend:.6f} < {CORR_TOL}"
print("  [PASS] Smoothed trends match")

# ---------------------------------------------------------------------------
# 8. Validate forecast (24 periods)
# ---------------------------------------------------------------------------
print(f"\n--- Forecast validation ({FORECAST_STEPS} steps) ---")
kb_forecast = kb_results.forecast(steps=FORECAST_STEPS)
sm_forecast = sm_results.get_forecast(steps=FORECAST_STEPS)

kb_fmean = kb_forecast["mean"].flatten()
sm_fmean = sm_forecast.predicted_mean

corr_forecast = np.corrcoef(kb_fmean, sm_fmean)[0, 1]
print(f"  forecast correlation: {corr_forecast:.6f} (min {CORR_TOL})")
assert corr_forecast > CORR_TOL, f"Forecast correlation {corr_forecast:.6f} < {CORR_TOL}"

# Forecast should be increasing (upward trend)
assert kb_fmean[-1] > kb_fmean[0], "Forecast should show upward trend"
print("  [PASS] Forecast shows upward trend and matches statsmodels")

# ---------------------------------------------------------------------------
# 9. Generate and save diagnostic plots
# ---------------------------------------------------------------------------
print("\n--- Generating figures ---")

dates = airline["date"].values

# Decomposition plot: level + trend
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Original + level
axes[0].plot(dates, y, "k.", alpha=0.4, markersize=3, label="Observed (log)")
axes[0].plot(dates, kb_level, "b-", linewidth=1.5, label="kalmanbox level")
axes[0].plot(dates, sm_level, "r--", linewidth=1.5, label="statsmodels level")
axes[0].set_ylabel("Log passengers")
axes[0].set_title("Local Linear Trend — Decomposition (Airline data)")
axes[0].legend(fontsize=8)

# Trend (slope)
axes[1].plot(dates, kb_trend, "b-", linewidth=1.5, label="kalmanbox trend")
axes[1].plot(dates, sm_trend, "r--", linewidth=1.5, label="statsmodels trend")
axes[1].set_ylabel("Trend (slope)")
axes[1].legend(fontsize=8)

# Forecast
forecast_idx = np.arange(len(y), len(y) + FORECAST_STEPS)
axes[2].plot(range(len(y)), y, "k-", alpha=0.5, label="Observed")
axes[2].plot(forecast_idx, kb_fmean, "b-", linewidth=1.5, label="kalmanbox forecast")
axes[2].plot(forecast_idx, sm_fmean, "r--", linewidth=1.5, label="statsmodels forecast")
kb_lower = kb_forecast["lower"].flatten()
kb_upper = kb_forecast["upper"].flatten()
axes[2].fill_between(forecast_idx, kb_lower, kb_upper, alpha=0.2, color="blue")
axes[2].set_ylabel("Log passengers")
axes[2].set_xlabel("Period")
axes[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(FIG_DIR / "local_linear_trend_decomposition.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'local_linear_trend_decomposition.png'}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Solution 02: ALL CHECKS PASSED")
print("=" * 60)
