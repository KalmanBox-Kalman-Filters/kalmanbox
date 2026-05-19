"""Solution: Local Level Model on Nile data.

Validates kalmanbox local level against statsmodels UnobservedComponents.
Expected results (Durbin & Koopman 2012):
  - sigma2_eps (observation) ~ 15099
  - sigma2_eta (level)       ~ 1469.1
  - log-likelihood           ~ -632.54

References:
  - Durbin, J. & Koopman, S.J. (2012). Time Series Analysis by State Space Methods.
  - Harvey, A.C. (1989). Forecasting, Structural Time Series Models and the Kalman Filter.
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

from kalmanbox.models.local_level import LocalLevel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Reference values (Durbin & Koopman 2012, Table 2.1)
REF_SIGMA2_EPS = 15099.0
REF_SIGMA2_ETA = 1469.1
REF_LOGLIKE = -632.54

PARAM_TOL = 0.05       # 5% relative tolerance for parameters
LOGLIKE_TOL = 2.0       # absolute tolerance for log-likelihood
CORR_TOL = 0.99         # minimum correlation for state comparison

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("Solution 01: Local Level Model — Nile data")
print("=" * 60)

nile = pd.read_csv(DATA_DIR / "nile.csv")
y = nile["flow"].values.astype(np.float64)
print(f"Loaded {len(y)} observations (Nile river flow, 1871–1970)")

# ---------------------------------------------------------------------------
# 2. Fit kalmanbox Local Level
# ---------------------------------------------------------------------------
print("\n--- kalmanbox Local Level ---")
kb_model = LocalLevel(y)
kb_results = kb_model.fit()

kb_sigma2_obs = kb_results.params[0]
kb_sigma2_level = kb_results.params[1]
kb_loglike = kb_results.loglike

print(f"  sigma2_obs   = {kb_sigma2_obs:.2f}")
print(f"  sigma2_level = {kb_sigma2_level:.2f}")
print(f"  log-like     = {kb_loglike:.2f}")

# ---------------------------------------------------------------------------
# 3. Fit statsmodels UnobservedComponents (local level)
# ---------------------------------------------------------------------------
print("\n--- statsmodels UnobservedComponents ---")
sm_model = sm.tsa.UnobservedComponents(y, level="local level")
sm_results = sm_model.fit(disp=False)

sm_sigma2_obs = sm_results.params[0]    # sigma2.irregular
sm_sigma2_level = sm_results.params[1]  # sigma2.level
sm_loglike = sm_results.llf

print(f"  sigma2_obs   = {sm_sigma2_obs:.2f}")
print(f"  sigma2_level = {sm_sigma2_level:.2f}")
print(f"  log-like     = {sm_loglike:.2f}")

# ---------------------------------------------------------------------------
# 4. Compare parameters against classical references
# ---------------------------------------------------------------------------
print("\n--- Validation against Durbin & Koopman (2012) ---")

def rel_err(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected)

err_eps = rel_err(kb_sigma2_obs, REF_SIGMA2_EPS)
err_eta = rel_err(kb_sigma2_level, REF_SIGMA2_ETA)
err_ll = abs(kb_loglike - REF_LOGLIKE)

print(f"  sigma2_eps rel error: {err_eps:.4f} (tol {PARAM_TOL})")
print(f"  sigma2_eta rel error: {err_eta:.4f} (tol {PARAM_TOL})")
print(f"  loglike abs error:    {err_ll:.2f}  (tol {LOGLIKE_TOL})")

assert err_eps < PARAM_TOL, f"sigma2_eps relative error {err_eps:.4f} exceeds {PARAM_TOL}"
assert err_eta < PARAM_TOL, f"sigma2_eta relative error {err_eta:.4f} exceeds {PARAM_TOL}"
assert err_ll < LOGLIKE_TOL, f"loglike absolute error {err_ll:.2f} exceeds {LOGLIKE_TOL}"
print("  [PASS] Parameters match classical references")

# ---------------------------------------------------------------------------
# 5. Compare kalmanbox vs statsmodels — parameters
# ---------------------------------------------------------------------------
print("\n--- kalmanbox vs statsmodels: parameters ---")
rel_obs = rel_err(kb_sigma2_obs, sm_sigma2_obs)
rel_lev = rel_err(kb_sigma2_level, sm_sigma2_level)
print(f"  sigma2_obs   rel diff: {rel_obs:.6f}")
print(f"  sigma2_level rel diff: {rel_lev:.6f}")

assert rel_obs < PARAM_TOL, f"sigma2_obs mismatch: {rel_obs:.6f}"
assert rel_lev < PARAM_TOL, f"sigma2_level mismatch: {rel_lev:.6f}"
print("  [PASS] Parameters match between libraries")

# ---------------------------------------------------------------------------
# 6. Compare kalmanbox vs statsmodels — log-likelihood
# ---------------------------------------------------------------------------
print("\n--- kalmanbox vs statsmodels: log-likelihood ---")
ll_diff = abs(kb_loglike - sm_loglike)
print(f"  absolute diff: {ll_diff:.4f} (tol {LOGLIKE_TOL})")
assert ll_diff < LOGLIKE_TOL, f"Log-likelihood mismatch: {ll_diff:.4f}"
print("  [PASS] Log-likelihoods match")

# ---------------------------------------------------------------------------
# 7. Compare filtered states
# ---------------------------------------------------------------------------
print("\n--- kalmanbox vs statsmodels: filtered states ---")
kb_filtered = kb_results.filtered_state[:, 0]
sm_filtered = sm_results.filtered_state[0]  # shape (k_states, nobs)

corr_filtered = np.corrcoef(kb_filtered, sm_filtered)[0, 1]
print(f"  correlation: {corr_filtered:.6f} (min {CORR_TOL})")
assert corr_filtered > CORR_TOL, f"Filtered state correlation {corr_filtered:.6f} < {CORR_TOL}"
print("  [PASS] Filtered states match")

# ---------------------------------------------------------------------------
# 8. Compare smoothed states
# ---------------------------------------------------------------------------
print("\n--- kalmanbox vs statsmodels: smoothed states ---")
kb_smoothed = kb_results.smoothed_state[:, 0]
sm_smoothed = sm_results.smoothed_state[0]

corr_smoothed = np.corrcoef(kb_smoothed, sm_smoothed)[0, 1]
print(f"  correlation: {corr_smoothed:.6f} (min {CORR_TOL})")
assert corr_smoothed > CORR_TOL, f"Smoothed state correlation {corr_smoothed:.6f} < {CORR_TOL}"
print("  [PASS] Smoothed states match")

# ---------------------------------------------------------------------------
# 9. Generate and save diagnostic plots
# ---------------------------------------------------------------------------
print("\n--- Generating figures ---")

years = nile["year"].values

# Filtered state plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(years, y, "k.", alpha=0.5, label="Observed (Nile flow)")
ax.plot(years, kb_filtered, "b-", linewidth=1.5, label="kalmanbox filtered")
ax.plot(years, sm_filtered, "r--", linewidth=1.5, label="statsmodels filtered")
ax.set_xlabel("Year")
ax.set_ylabel("Flow")
ax.set_title("Local Level — Filtered State (Nile)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "local_level_filtered.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'local_level_filtered.png'}")

# Smoothed state plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(years, y, "k.", alpha=0.5, label="Observed (Nile flow)")
ax.plot(years, kb_smoothed, "b-", linewidth=1.5, label="kalmanbox smoothed")
ax.plot(years, sm_smoothed, "r--", linewidth=1.5, label="statsmodels smoothed")
ax.set_xlabel("Year")
ax.set_ylabel("Flow")
ax.set_title("Local Level — Smoothed State (Nile)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "local_level_smoothed.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'local_level_smoothed.png'}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Solution 01: ALL CHECKS PASSED")
print("=" * 60)
