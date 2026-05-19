"""Solution: Stochastic Cycle Model on Brazil IPCA inflation data.

Validates kalmanbox local level + stochastic cycle model on monthly IPCA.
Compares extracted cycle with classical filters (HP, CF).

Expected behavior:
  - Damping factor rho in (0, 1)
  - Cycle period reasonable (12-120 months = 1-10 years)
  - Extracted cycle correlated with HP and CF band-pass filters
  - Log-likelihood comparable to statsmodels UCM

References:
  - Harvey, A.C. (1989). Forecasting, Structural Time Series Models.
  - Hodrick, R.J. & Prescott, E.C. (1997). Postwar U.S. Business Cycles.
  - Christiano, L.J. & Fitzgerald, T.J. (2003). The Band Pass Filter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.filters.cf_filter import cffilter

# Ensure kalmanbox is importable
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from kalmanbox import UnobservedComponents
from kalmanbox.filters.kalman import KalmanFilter
from kalmanbox.smoothers.rts import RTSSmoother

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LOGLIKE_TOL = 30.0       # tolerance for log-likelihood comparison
CORR_TOL = 0.50          # minimum correlation with classical filters
CYCLE_PERIOD_MIN = 12    # months (1 year)
CYCLE_PERIOD_MAX = 120   # months (10 years)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("Solution 02: Stochastic Cycle — Brazil IPCA data")
print("=" * 60)

ipca = pd.read_csv(DATA_DIR / "brazil_ipca.csv")
ipca["date"] = pd.to_datetime(ipca["date"])
dates = ipca["date"]
y = ipca["ipca_12m"].values.astype(np.float64)
nobs = len(y)
print(f"Loaded {nobs} observations (IPCA 12-month rate, 2000-2023)")
print(f"  Mean: {y.mean():.3f}%")
print(f"  Std:  {y.std():.3f}%")
print(f"  Range: [{y.min():.3f}%, {y.max():.3f}%]")


def rel_err(actual: float, expected: float) -> float:
    if abs(expected) < 1e-10:
        return abs(actual - expected)
    return abs(actual - expected) / abs(expected)


# ---------------------------------------------------------------------------
# 2. Fit kalmanbox UCM: local level + stochastic cycle
# ---------------------------------------------------------------------------
print("\n--- kalmanbox: Local Level + Stochastic Cycle ---")
ucm = UnobservedComponents(
    y,
    level=True,
    trend="none",
    cycle=True,
)
print(f"  States: {ucm._k_states}")
print(f"  Params: {ucm.param_names}")
print(f"  Layout: {ucm._layout}")

# Multi-start L-BFGS-B optimization
bounds_opt = [
    (-8, 3),      # log(sigma2_obs)
    (-10, 3),     # log(sigma2_level)
    (-1.5, 3.0),  # arctanh(2*rho-1) -> rho in ~(0.1, 0.995)
    (-3.5, 0.5),  # logit(lambda_c/pi) -> wide frequency range
    (-10, 3),     # log(sigma2_cycle)
]


def neg_loglike_kb(unc: np.ndarray) -> float:
    try:
        constrained = ucm.transform_params(unc)
        return -ucm.loglike(constrained)
    except Exception:
        return 1e10


best_result = None
best_fun = np.inf

# Grid over rho and frequency starting points
start_configs = [
    (0.5, -2.0), (0.5, -1.0),
    (0.8, -2.0), (0.8, -1.0),
    (0.9, -1.5), (0.95, -0.5),
]

for rho_init, lam_init in start_configs:
    x0 = np.array([-2, -4, np.arctanh(2 * rho_init - 1), lam_init, -4])
    result = optimize.minimize(
        neg_loglike_kb, x0, method="L-BFGS-B", bounds=bounds_opt,
        options={"maxiter": 200, "ftol": 1e-8},
    )
    if result.fun < best_fun:
        best_fun = result.fun
        best_result = result

refined = best_result

kb_params = ucm.transform_params(refined.x)
kb_loglike = -refined.fun

print("\n  Estimated parameters (MLE):")
for name, val in zip(ucm.param_names, kb_params):
    print(f"    {name:20s} = {val:.6f}")
print(f"  Log-likelihood: {kb_loglike:.4f}")

# Extract cycle parameters
rho_hat = kb_params[ucm.param_names.index("rho")]
lambda_c_hat = kb_params[ucm.param_names.index("lambda_c")]
period_months = 2 * np.pi / lambda_c_hat
period_years = period_months / 12

print(f"\n  Cycle period: {period_months:.1f} months = {period_years:.1f} years")
print(f"  Damping (rho): {rho_hat:.4f}")

# ---------------------------------------------------------------------------
# 3. Validate rho in (0, 1)
# ---------------------------------------------------------------------------
print("\n--- Validation: rho in (0, 1) ---")
assert 0 < rho_hat < 1, f"rho = {rho_hat:.4f} not in (0, 1)"
print(f"  [PASS] rho = {rho_hat:.4f} in (0, 1)")

# Half-life
if rho_hat > 0 and rho_hat < 1:
    half_life = -np.log(2) / np.log(rho_hat)
    print(f"  Half-life: {half_life:.1f} months = {half_life/12:.1f} years")

# ---------------------------------------------------------------------------
# 4. Validate cycle periodicity
# ---------------------------------------------------------------------------
print("\n--- Validation: cycle periodicity ---")
print(f"  Period: {period_months:.1f} months ({period_years:.1f} years)")
assert CYCLE_PERIOD_MIN <= period_months <= CYCLE_PERIOD_MAX, (
    f"Cycle period {period_months:.1f} months not in [{CYCLE_PERIOD_MIN}, {CYCLE_PERIOD_MAX}]"
)
print(f"  [PASS] Period in reasonable range [{CYCLE_PERIOD_MIN}, {CYCLE_PERIOD_MAX}] months")

# ---------------------------------------------------------------------------
# 5. Extract smoothed cycle component
# ---------------------------------------------------------------------------
print("\n--- Extracting smoothed components ---")
ssm = ucm._build_ssm(kb_params)
kf = KalmanFilter()
smoother = RTSSmoother()
fo = kf.filter(y.reshape(-1, 1), ssm)
so = smoother.smooth(fo, ssm)

layout = ucm._layout
smoothed = so.smoothed_state

kb_level = smoothed[:, layout["level"]["start"]]
kb_cycle = smoothed[:, layout["cycle"]["start"]]
kb_cycle_star = smoothed[:, layout["cycle"]["start"] + 1]
kb_irregular = y - kb_level - kb_cycle

print(f"  Level range:   [{kb_level.min():.3f}, {kb_level.max():.3f}]")
print(f"  Cycle range:   [{kb_cycle.min():.3f}, {kb_cycle.max():.3f}]")
print(f"  Irregular std: {np.std(kb_irregular):.3f}")

# Validate cycle was actually extracted (non-trivial amplitude)
cycle_std = np.std(kb_cycle)
assert cycle_std > 0.01, f"Cycle std too small: {cycle_std:.6f}"
print(f"  [PASS] Cycle extracted with std = {cycle_std:.3f}")

# ---------------------------------------------------------------------------
# 6. Variance decomposition
# ---------------------------------------------------------------------------
print("\n--- Variance decomposition ---")
sig2_obs = kb_params[ucm.param_names.index("sigma2_obs")]
sig2_level = kb_params[ucm.param_names.index("sigma2_level")]
sig2_cycle = kb_params[ucm.param_names.index("sigma2_cycle")]
total_sig2 = sig2_obs + sig2_level + sig2_cycle
print(f"  sigma2_obs   / total = {sig2_obs/total_sig2*100:.1f}% (irregular)")
print(f"  sigma2_level / total = {sig2_level/total_sig2*100:.1f}% (level shifts)")
print(f"  sigma2_cycle / total = {sig2_cycle/total_sig2*100:.1f}% (cyclical)")

# ---------------------------------------------------------------------------
# 7. Comparison with classical filters: HP and CF
# ---------------------------------------------------------------------------
print("\n--- Comparison with HP and CF filters ---")

# HP filter (lambda=14400 for monthly data)
cycle_hp, trend_hp = hpfilter(y, lamb=14400)

# CF band-pass filter (18-96 months)
cycle_cf, trend_cf = cffilter(y, low=18, high=96)

# Correlations
n_min = min(len(kb_cycle), len(cycle_hp), len(cycle_cf))
corr_hp = np.corrcoef(kb_cycle[:n_min], cycle_hp[:n_min])[0, 1]
corr_cf = np.corrcoef(kb_cycle[:n_min], cycle_cf[:n_min])[0, 1]
corr_hp_cf = np.corrcoef(cycle_hp[:n_min], cycle_cf[:n_min])[0, 1]

print(f"  Kalman vs HP:  {corr_hp:.3f}")
print(f"  Kalman vs CF:  {corr_cf:.3f}")
print(f"  HP vs CF:      {corr_hp_cf:.3f}")

# Validate reasonable correlation with at least one classical filter
max_corr = max(abs(corr_hp), abs(corr_cf))
assert max_corr > CORR_TOL, (
    f"Cycle not correlated with any classical filter: "
    f"HP={corr_hp:.3f}, CF={corr_cf:.3f} (min {CORR_TOL})"
)
print(f"  [PASS] Cycle correlated with classical filters (max |corr| = {max_corr:.3f})")

# ---------------------------------------------------------------------------
# 8. Comparison with statsmodels
# ---------------------------------------------------------------------------
print("\n--- statsmodels UCM ---")
sm_model = sm.tsa.UnobservedComponents(
    y,
    level="local level",
    cycle=True,
    stochastic_cycle=True,
    damped_cycle=True,
)
sm_results = sm_model.fit(disp=False)

sm_freq = sm_results.params[sm_results.param_names.index("frequency.cycle")]
sm_rho = sm_results.params[sm_results.param_names.index("damping.cycle")]
sm_period = 2 * np.pi / sm_freq
sm_loglike = sm_results.llf

print(f"  Log-likelihood: {sm_loglike:.4f}")
print(f"  Cycle period:   {sm_period:.1f} months = {sm_period/12:.1f} years")
print(f"  Damping (rho):  {sm_rho:.4f}")

# Compare log-likelihoods
ll_diff = abs(kb_loglike - sm_loglike)
print(f"\n  Log-likelihood diff: {ll_diff:.4f} (tol {LOGLIKE_TOL})")
assert ll_diff < LOGLIKE_TOL, f"Log-likelihood mismatch: {ll_diff:.4f}"
print("  [PASS] Log-likelihoods comparable")

# statsmodels rho should also be in (0, 1)
assert 0 < sm_rho < 1, f"statsmodels rho = {sm_rho:.4f} not in (0, 1)"
print(f"  [PASS] statsmodels rho = {sm_rho:.4f} in (0, 1)")

# ---------------------------------------------------------------------------
# 9. Generate diagnostic plots
# ---------------------------------------------------------------------------
print("\n--- Generating figures ---")

# Figure 1: Decomposition (3 panels: level, cycle, irregular)
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Observed + level
axes[0].plot(dates, y, "k-", linewidth=0.6, alpha=0.6, label="IPCA observed")
axes[0].plot(dates, kb_level, "b-", linewidth=1.5, label="Level (smoothed)")
axes[0].set_ylabel("12-month IPCA (%)")
axes[0].set_title("IPCA Decomposition: Level + Stochastic Cycle")
axes[0].legend(fontsize=8)

# Extracted cycle
axes[1].plot(dates, kb_cycle, "r-", linewidth=1.2, label="Stochastic cycle")
axes[1].fill_between(dates, kb_cycle, 0, alpha=0.15, color="red")
axes[1].axhline(0, color="gray", linewidth=0.5)
axes[1].set_ylabel("Cycle (%)")
axes[1].set_title(f"Extracted Cycle (Period = {period_months:.0f}m = {period_years:.1f}y, "
                   f"rho = {rho_hat:.3f})")
axes[1].legend(fontsize=8)

# Irregular
axes[2].plot(dates, kb_irregular, "gray", linewidth=0.6)
axes[2].axhline(0, color="black", linewidth=0.5)
axes[2].set_ylabel("Irregular (%)")
axes[2].set_title("Irregular Component")
axes[2].set_xlabel("Date")

fig.tight_layout()
fig.savefig(FIG_DIR / "stochastic_cycle_decomposition.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'stochastic_cycle_decomposition.png'}")

# Figure 2: Comparison with classical filters
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(dates, kb_cycle, "r-", linewidth=1.2, label="Kalman stochastic cycle")
axes[0].axhline(0, color="gray", linewidth=0.5)
axes[0].set_ylabel("Cycle")
axes[0].set_title("Stochastic Cycle (kalmanbox UCM)")
axes[0].legend(fontsize=8)

axes[1].plot(dates, cycle_hp, "g-", linewidth=1.2, label="HP filter (lambda=14400)")
axes[1].axhline(0, color="gray", linewidth=0.5)
axes[1].set_ylabel("Cycle")
axes[1].set_title("HP Filter Cycle")
axes[1].legend(fontsize=8)

n_cf = len(cycle_cf)
axes[2].plot(dates[:n_cf], cycle_cf, "b-", linewidth=1.2,
             label="CF band-pass (18-96 months)")
axes[2].axhline(0, color="gray", linewidth=0.5)
axes[2].set_ylabel("Cycle")
axes[2].set_title("Christiano-Fitzgerald Band-Pass Filter")
axes[2].set_xlabel("Date")
axes[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(FIG_DIR / "stochastic_cycle_filter_comparison.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'stochastic_cycle_filter_comparison.png'}")

# Figure 3: Cycle amplitude and phase
amplitude = np.sqrt(kb_cycle**2 + kb_cycle_star**2)
phase = np.arctan2(kb_cycle_star, kb_cycle)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(dates, amplitude, "m-", linewidth=1.2)
axes[0].fill_between(dates, 0, amplitude, alpha=0.15, color="purple")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Cycle Amplitude Over Time")

axes[1].plot(dates, phase, "c-", linewidth=0.8)
axes[1].set_ylabel("Phase (radians)")
axes[1].set_title("Cycle Phase Over Time")
axes[1].set_xlabel("Date")

fig.tight_layout()
fig.savefig(FIG_DIR / "stochastic_cycle_amplitude_phase.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'stochastic_cycle_amplitude_phase.png'}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Solution 02: ALL CHECKS PASSED")
print("=" * 60)
