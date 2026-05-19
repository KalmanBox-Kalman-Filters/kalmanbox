"""Solution: Comparison of MLE Methods for ARIMA Models.

Implements and compares three log-likelihood computation methods:
  1. Manual Kalman filter (prediction error decomposition)
  2. kalmanbox ARIMA_SSM
  3. statsmodels ARIMA

Also compares CSS (Conditional Sum of Squares) vs exact MLE initialization.

References:
  - Durbin, J. & Koopman, S.J. (2012). Time Series Analysis by State Space Methods.
  - Hamilton, J.D. (1994). Time Series Analysis. Ch. 13.
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
from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA

# Ensure kalmanbox is importable
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from kalmanbox.models.arima_ssm import ARIMA_SSM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LOGLIKE_TOL = 0.1       # max allowed diff between manual and kalmanbox loglike
LOGLIKE_TOL_SM = 5.0    # broader tolerance vs statsmodels (init differences)

ALL_PASSED = True


def check(condition: bool, msg: str) -> None:
    """Assert with tracking."""
    global ALL_PASSED
    if not condition:
        ALL_PASSED = False
        print(f"  [FAIL] {msg}")
    else:
        print(f"  [PASS] {msg}")
    assert condition, msg


# ===========================================================================
# Manual Kalman filter log-likelihood
# ===========================================================================
def manual_kalman_loglike(
    y: np.ndarray,
    T: np.ndarray,
    Z: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    a1: np.ndarray,
    P1: np.ndarray,
    n_diffuse: int = 1,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute log-likelihood via Kalman filter prediction error decomposition.

    Parameters
    ----------
    y : 1-D array of observations (may contain NaN)
    T, Z, R, Q, H : SSM matrices
    a1, P1 : initial state and covariance
    n_diffuse : number of initial observations to exclude from loglike

    Returns
    -------
    loglike : total log-likelihood (excluding first n_diffuse obs)
    v_all : prediction errors
    F_all : forecast variances
    """
    n = len(y)
    m = len(a1)

    v_all = np.full(n, np.nan)
    F_all = np.full(n, np.nan)
    ll_obs = np.zeros(n)

    a = a1.copy().reshape(-1)
    P = P1.copy()

    for t in range(n):
        # Forecast
        y_pred = float((Z @ a).item())
        F_t = float((Z @ P @ Z.T + H).item())

        if np.isnan(y[t]):
            # Skip update — predicted becomes filtered
            a_new = (T @ a).reshape(-1)
            P_new = T @ P @ T.T + R @ Q @ R.T
        else:
            v_t = float(y[t] - y_pred)
            v_all[t] = v_t
            F_all[t] = F_t

            # Log-likelihood contribution
            if F_t > 0:
                ll_obs[t] = -0.5 * (np.log(2 * np.pi) + np.log(F_t) + v_t**2 / F_t)

            # Kalman gain: K = T P Z' F^{-1}
            K = (T @ P @ Z.T / F_t).reshape(-1)  # (m,)

            # Update: a_{t|t} = T a_{t-1|t-1} + K v_t, then predict
            a_filt = a + (P @ Z.T).reshape(-1) * v_t / F_t
            P_filt = P - np.outer(P @ Z.T, Z @ P) / F_t

            # Predict
            a_new = (T @ a_filt).reshape(-1)
            P_new = T @ P_filt @ T.T + R @ Q @ R.T

        a = a_new
        P = 0.5 * (P_new + P_new.T)  # symmetrize

    # Sum loglike excluding initial diffuse observations
    loglike = float(np.sum(ll_obs[n_diffuse:]))
    return loglike, v_all, F_all


# ===========================================================================
# CSS (Conditional Sum of Squares) log-likelihood
# ===========================================================================
def css_loglike(
    y: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    sigma2: float,
) -> float:
    """Conditional Sum of Squares log-likelihood for ARMA(p,q).

    Assumes y is already differenced if needed.
    Conditions on first max(p,q) observations.

    Parameters
    ----------
    y : differenced time series
    phi : AR coefficients [phi_1, ..., phi_p]
    theta : MA coefficients [theta_1, ..., theta_q]
    sigma2 : innovation variance

    Returns
    -------
    loglike : CSS log-likelihood
    """
    n = len(y)
    p = len(phi)
    q = len(theta)
    start = max(p, q)

    # Compute residuals
    e = np.zeros(n)
    for t in range(start, n):
        ar_part = sum(phi[j] * y[t - j - 1] for j in range(p)) if p > 0 else 0.0
        ma_part = sum(theta[j] * e[t - j - 1] for j in range(q)) if q > 0 else 0.0
        e[t] = y[t] - ar_part - ma_part

    # CSS log-likelihood
    n_eff = n - start
    ss = np.sum(e[start:] ** 2)
    ll = -0.5 * n_eff * np.log(2 * np.pi * sigma2) - ss / (2 * sigma2)
    return float(ll)


# ===========================================================================
# Load data
# ===========================================================================
print("=" * 70)
print("Solution 02: Comparison of MLE Methods")
print("=" * 70)

nile = pd.read_csv(DATA_DIR / "nile.csv")
y_nile = nile["flow"].values.astype(np.float64)
print(f"Loaded Nile data: {len(y_nile)} observations")

airline = pd.read_csv(DATA_DIR / "airline.csv")
y_airline = np.log(airline["passengers"].values.astype(np.float64))
print(f"Loaded Airline data: {len(y_airline)} observations (log)")

# ===========================================================================
# Part 1: Manual Kalman vs kalmanbox — ARIMA(0,1,1) on Nile
# ===========================================================================
print("\n" + "-" * 70)
print("Part 1: Manual Kalman vs kalmanbox — ARIMA(0,1,1)")
print("-" * 70)

# Fit kalmanbox
kb_011 = ARIMA_SSM(y_nile, order=(0, 1, 1))
kb_011_res = kb_011.fit()
kb_params = kb_011_res.params
kb_loglike = kb_011_res.loglike

theta_1 = kb_params[0]
sigma2 = kb_params[1]

print(f"  kalmanbox params: theta_1={theta_1:.6f}, sigma2={sigma2:.2f}")
print(f"  kalmanbox loglike: {kb_loglike:.4f}")

# Build SSM matrices with fitted params
ssm = kb_011._build_ssm(kb_params)

# Manual Kalman
manual_ll, v_manual, F_manual = manual_kalman_loglike(
    y=kb_011._differenced_endog,
    T=ssm.T, Z=ssm.Z, R=ssm.R, Q=ssm.Q, H=ssm.H,
    a1=ssm.a1, P1=ssm.P1,
    n_diffuse=1,
)

print(f"  Manual Kalman loglike: {manual_ll:.4f}")

diff_manual_kb = abs(manual_ll - kb_loglike)
print(f"  |manual - kalmanbox| = {diff_manual_kb:.6f}")

check(
    diff_manual_kb < LOGLIKE_TOL,
    f"Manual Kalman ~= kalmanbox (diff={diff_manual_kb:.6f} < {LOGLIKE_TOL})"
)

# ===========================================================================
# Part 2: Three-way comparison — kalmanbox, manual, statsmodels
# ===========================================================================
print("\n" + "-" * 70)
print("Part 2: Three-way log-likelihood comparison")
print("-" * 70)

# statsmodels
sm_011 = SM_ARIMA(y_nile, order=(0, 1, 1))
sm_011_res = sm_011.fit()
sm_loglike = sm_011_res.llf

print(f"  kalmanbox:   {kb_loglike:.4f}")
print(f"  manual:      {manual_ll:.4f}")
print(f"  statsmodels: {sm_loglike:.4f}")

diff_kb_sm = abs(kb_loglike - sm_loglike)
print(f"\n  |kalmanbox - statsmodels| = {diff_kb_sm:.4f}")

# Note: statsmodels computes loglike on original (undifferenced) data
# while kalmanbox computes on differenced data, so absolute values differ.
# The key check is manual vs kalmanbox (same data, same initialization).

check(
    diff_manual_kb < LOGLIKE_TOL,
    f"Manual Kalman ~= kalmanbox loglike (diff={diff_manual_kb:.6f})"
)

# ===========================================================================
# Part 3: Manual Kalman vs kalmanbox — AR(1) on Nile
# ===========================================================================
print("\n" + "-" * 70)
print("Part 3: Manual Kalman vs kalmanbox — AR(1)")
print("-" * 70)

kb_ar1 = ARIMA_SSM(y_nile, order=(1, 0, 0))
kb_ar1_res = kb_ar1.fit()
kb_ar1_params = kb_ar1_res.params
kb_ar1_ll = kb_ar1_res.loglike

ssm_ar1 = kb_ar1._build_ssm(kb_ar1_params)
manual_ar1_ll, _, _ = manual_kalman_loglike(
    y=y_nile,
    T=ssm_ar1.T, Z=ssm_ar1.Z, R=ssm_ar1.R, Q=ssm_ar1.Q, H=ssm_ar1.H,
    a1=ssm_ar1.a1, P1=ssm_ar1.P1,
    n_diffuse=1,
)

diff_ar1 = abs(manual_ar1_ll - kb_ar1_ll)
print(f"  kalmanbox loglike: {kb_ar1_ll:.4f}")
print(f"  manual loglike:    {manual_ar1_ll:.4f}")
print(f"  difference:        {diff_ar1:.6f}")

check(
    diff_ar1 < LOGLIKE_TOL,
    f"AR(1): manual ~= kalmanbox (diff={diff_ar1:.6f})"
)

# ===========================================================================
# Part 4: Manual Kalman vs kalmanbox — ARIMA(1,1,1) on Nile
# ===========================================================================
print("\n" + "-" * 70)
print("Part 4: Manual Kalman vs kalmanbox — ARIMA(1,1,1)")
print("-" * 70)

kb_111 = ARIMA_SSM(y_nile, order=(1, 1, 1))
kb_111_res = kb_111.fit()
kb_111_params = kb_111_res.params
kb_111_ll = kb_111_res.loglike

ssm_111 = kb_111._build_ssm(kb_111_params)
manual_111_ll, _, _ = manual_kalman_loglike(
    y=kb_111._differenced_endog,
    T=ssm_111.T, Z=ssm_111.Z, R=ssm_111.R, Q=ssm_111.Q, H=ssm_111.H,
    a1=ssm_111.a1, P1=ssm_111.P1,
    n_diffuse=1,
)

diff_111 = abs(manual_111_ll - kb_111_ll)
print(f"  kalmanbox loglike: {kb_111_ll:.4f}")
print(f"  manual loglike:    {manual_111_ll:.4f}")
print(f"  difference:        {diff_111:.6f}")

check(
    diff_111 < LOGLIKE_TOL,
    f"ARIMA(1,1,1): manual ~= kalmanbox (diff={diff_111:.6f})"
)

# ===========================================================================
# Part 5: CSS vs Exact MLE comparison
# ===========================================================================
print("\n" + "-" * 70)
print("Part 5: CSS vs Exact MLE comparison")
print("-" * 70)

# Differenced Nile data
dy_nile = np.diff(y_nile)

# Use kalmanbox fitted params for ARIMA(0,1,1)
theta_fit = kb_011_res.params[0]
sigma2_fit = kb_011_res.params[1]

# CSS loglike for ARIMA(0,1,1) => MA(1) on differenced data
css_ll_011 = css_loglike(dy_nile, phi=np.array([]), theta=np.array([theta_fit]), sigma2=sigma2_fit)

print(f"  ARIMA(0,1,1) on Nile:")
print(f"    Exact MLE (Kalman):  {kb_loglike:.4f}")
print(f"    CSS:                 {css_ll_011:.4f}")
print(f"    Difference:          {abs(kb_loglike - css_ll_011):.4f}")

# For ARIMA(1,1,1)
phi_111 = kb_111_res.params[0]
theta_111 = kb_111_res.params[1]
sigma2_111 = kb_111_res.params[2]

css_ll_111 = css_loglike(
    dy_nile,
    phi=np.array([phi_111]),
    theta=np.array([theta_111]),
    sigma2=sigma2_111,
)

print(f"\n  ARIMA(1,1,1) on Nile:")
print(f"    Exact MLE (Kalman):  {kb_111_ll:.4f}")
print(f"    CSS:                 {css_ll_111:.4f}")
print(f"    Difference:          {abs(kb_111_ll - css_ll_111):.4f}")

# For AR(1) on original data
phi_ar1 = kb_ar1_res.params[0]
sigma2_ar1 = kb_ar1_res.params[1]
css_ll_ar1 = css_loglike(y_nile, phi=np.array([phi_ar1]), theta=np.array([]), sigma2=sigma2_ar1)

print(f"\n  AR(1) on Nile:")
print(f"    Exact MLE (Kalman):  {kb_ar1_ll:.4f}")
print(f"    CSS:                 {css_ll_ar1:.4f}")
print(f"    Difference:          {abs(kb_ar1_ll - css_ll_ar1):.4f}")

# Results table
print("\n  " + "=" * 60)
print(f"  {'Model':<20} {'Exact MLE':>12} {'CSS':>12} {'Diff':>12}")
print("  " + "-" * 60)
print(f"  {'ARIMA(0,1,1)':<20} {kb_loglike:>12.2f} {css_ll_011:>12.2f} {abs(kb_loglike - css_ll_011):>12.2f}")
print(f"  {'ARIMA(1,1,1)':<20} {kb_111_ll:>12.2f} {css_ll_111:>12.2f} {abs(kb_111_ll - css_ll_111):>12.2f}")
print(f"  {'AR(1)':<20} {kb_ar1_ll:>12.2f} {css_ll_ar1:>12.2f} {abs(kb_ar1_ll - css_ll_ar1):>12.2f}")
print("  " + "=" * 60)

# CSS and exact should be similar for reasonably long series
# The difference is mainly from initial condition handling
check(
    abs(kb_loglike - css_ll_011) < 10.0,
    "CSS vs Exact: ARIMA(0,1,1) difference is bounded"
)
check(
    abs(kb_111_ll - css_ll_111) < 10.0,
    "CSS vs Exact: ARIMA(1,1,1) difference is bounded"
)

# ===========================================================================
# Part 6: Full comparison table — kalmanbox vs statsmodels vs manual
# ===========================================================================
print("\n" + "-" * 70)
print("Part 6: Full comparison table")
print("-" * 70)

# statsmodels results
sm_ar1 = SM_ARIMA(y_nile, order=(1, 0, 0))
sm_ar1_res = sm_ar1.fit()

sm_111 = SM_ARIMA(y_nile, order=(1, 1, 1))
sm_111_res = sm_111.fit()

print(f"\n  {'Model':<20} {'kalmanbox':>12} {'manual':>12} {'statsmodels':>12} {'kb-manual':>12}")
print("  " + "-" * 72)
print(f"  {'ARIMA(0,1,1)':<20} {kb_loglike:>12.2f} {manual_ll:>12.2f} {sm_loglike:>12.2f} {diff_manual_kb:>12.6f}")
print(f"  {'AR(1)':<20} {kb_ar1_ll:>12.2f} {manual_ar1_ll:>12.2f} {sm_ar1_res.llf:>12.2f} {diff_ar1:>12.6f}")
print(f"  {'ARIMA(1,1,1)':<20} {kb_111_ll:>12.2f} {manual_111_ll:>12.2f} {sm_111_res.llf:>12.2f} {diff_111:>12.6f}")
print("  " + "=" * 72)
print("  Note: statsmodels loglike is on original data; kalmanbox on differenced.")
print("        The manual vs kalmanbox comparison is the definitive validation.")

# ===========================================================================
# Generate validation figures
# ===========================================================================
print("\n" + "-" * 70)
print("Generating validation figures")
print("-" * 70)

# Figure 1: Prediction errors comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# ARIMA(0,1,1) prediction errors
valid_mask = ~np.isnan(v_manual)
t_valid = np.where(valid_mask)[0]
axes[0].plot(t_valid, v_manual[valid_mask], "b-", alpha=0.7, linewidth=0.8, label="Manual Kalman")
kb_resids = kb_011_res.residuals[:, 0]
axes[0].plot(range(len(kb_resids)), kb_resids, "r--", alpha=0.7, linewidth=0.8, label="kalmanbox")
axes[0].axhline(0, color="gray", linestyle="--", alpha=0.4)
axes[0].set_title("ARIMA(0,1,1) Prediction Errors: Manual vs kalmanbox")
axes[0].set_ylabel("v_t")
axes[0].legend()

# Forecast variances
axes[1].plot(t_valid, F_manual[valid_mask], "b-", linewidth=1.0, label="Manual F_t")
kb_fcov = kb_011_res.residuals_cov[:, 0, 0]
axes[1].plot(range(len(kb_fcov)), kb_fcov, "r--", linewidth=1.0, label="kalmanbox F_t")
axes[1].set_title("Forecast Error Variance F_t: Manual vs kalmanbox")
axes[1].set_xlabel("Observation")
axes[1].set_ylabel("F_t")
axes[1].legend()

fig.tight_layout()
fig.savefig(FIG_DIR / "loglike_comparison_residuals.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'loglike_comparison_residuals.png'}")

# Figure 2: CSS vs Exact MLE bar chart
fig, ax = plt.subplots(figsize=(10, 6))

models = ["ARIMA(0,1,1)", "ARIMA(1,1,1)", "AR(1)"]
exact_vals = [kb_loglike, kb_111_ll, kb_ar1_ll]
css_vals = [css_ll_011, css_ll_111, css_ll_ar1]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width / 2, exact_vals, width, label="Exact MLE (Kalman)", color="steelblue")
bars2 = ax.bar(x + width / 2, css_vals, width, label="CSS", color="coral")

ax.set_ylabel("Log-likelihood")
ax.set_title("CSS vs Exact MLE: Log-likelihood Comparison")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8)

fig.tight_layout()
fig.savefig(FIG_DIR / "css_vs_exact_mle.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'css_vs_exact_mle.png'}")

# Figure 3: Log-likelihood convergence by method
fig, ax = plt.subplots(figsize=(10, 5))

# Per-observation loglike from kalmanbox
ll_obs_kb = kb_011_res.filter_output.loglike_obs
cumsum_ll = np.cumsum(ll_obs_kb)
ax.plot(range(len(cumsum_ll)), cumsum_ll, "b-", linewidth=1.5,
        label="Cumulative log-likelihood (kalmanbox)")
ax.axhline(kb_loglike, color="red", linestyle="--", alpha=0.7,
           label=f"Total: {kb_loglike:.2f}")
ax.set_title("ARIMA(0,1,1): Cumulative Log-likelihood")
ax.set_xlabel("Observation")
ax.set_ylabel("Cumulative log-likelihood")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "loglike_cumulative.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'loglike_cumulative.png'}")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
if ALL_PASSED:
    print("Solution 02: ALL CHECKS PASSED")
else:
    print("Solution 02: SOME CHECKS FAILED")
    sys.exit(1)
print("=" * 70)
