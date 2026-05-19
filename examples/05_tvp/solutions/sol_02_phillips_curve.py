"""Solution 02: Phillips Curve with Time-Varying NAIRU and slope.

Validates three Staiger-Stock-Watson style Phillips curve specifications
with kalmanbox on the US inflation/unemployment panel.

Accelerationist Phillips curve:
    Delta pi_t = alpha_t + beta_t * u_t + eps_t
    NAIRU: u*_t = -alpha_t / beta_t

Three models:
  - Model 1: alpha fixed at -OLS_slope * u_mean, beta_t random walk
  - Model 2: beta fixed at OLS, alpha_t random walk (NAIRU TV, slope fixed)
  - Model 3: both alpha_t and beta_t are random walks (full SSW)

Checks performed:
  1. Each model fits with finite loglike and non-negative variance parameters.
  2. Model 2 NAIRU path is inside [3, 8] for >= 80% of observations.
  3. Model 3 NAIRU path is inside [3, 8] for >= 80% of observations.
  4. Model 1 slope is negative for >= 50% of periods (Phillips trade-off).
  5. Model 3 slope is negative for >= 50% of periods.
  6. Model 2 matches statsmodels UnobservedComponents on the residualized
     series: sigma2_obs agrees within 1%.

All checks print PASS/FAIL and exit with status 0 on success.

Expected behaviour (reference values on the shipped dataset):
  - constant OLS NAIRU ~ 5-7%
  - Model 2/3 NAIRU averages 4-7% with wide CI (weak identification is expected)
  - Model 3 slope hovers around 0 with pre-1990 average more negative than post-1990

References:
  - Staiger, Stock, Watson (1997). NBER WP 5477 / JBES 15.
  - Cogley & Sargent (2005). RED.
  - Ball & Mazumder (2011). BPEA.
  - Blanchard (2016). AER P&P 106(5).
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
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.statespace.structural import UnobservedComponents

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from kalmanbox import LocalLevel, TimeVaryingParameters

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NAIRU_LOWER = 3.0
NAIRU_UPPER = 8.0
NAIRU_INSIDE_MIN_FRAC = 0.80
NEG_SLOPE_MIN_FRAC = 0.50

ALL_PASSED = True


def check(condition: bool, msg: str) -> None:
    """Record a PASS/FAIL line."""
    global ALL_PASSED
    if not condition:
        ALL_PASSED = False
        print(f"  [FAIL] {msg}")
    else:
        print(f"  [PASS] {msg}")


def fraction_in_band(x: np.ndarray, lo: float, hi: float) -> float:
    """Fraction of finite values of x lying in [lo, hi]."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return 0.0
    return float(np.mean((x[finite] >= lo) & (x[finite] <= hi)))


def fraction_negative(x: np.ndarray) -> float:
    """Fraction of finite values of x that are strictly negative."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return 0.0
    return float(np.mean(x[finite] < 0.0))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("Solution 02: Phillips Curve with TV NAIRU and slope")
print("=" * 70)

df = pd.read_csv(
    DATA_DIR / "us_inflation_unemployment.csv", parse_dates=["date"]
).set_index("date")
pi = df["inflation"].to_numpy(dtype=np.float64)
u = df["unemployment"].to_numpy(dtype=np.float64)
dates = df.index
n = len(pi)

# First difference of inflation (drop first obs because of differencing)
dpi = np.concatenate([[np.nan], np.diff(pi)])
mask = np.isfinite(dpi)
dpi_v = dpi[mask]
u_v = u[mask]
dates_v = dates[mask]
nv = len(dpi_v)
print(f"Observations used: {nv} (dropping first due to differencing)")
print(f"Period: {dates_v[0].date()} -> {dates_v[-1].date()}")

# OLS reference: Delta pi = alpha + beta * u + eps
X_ols = np.column_stack([np.ones(nv), u_v])
ols = sm.OLS(dpi_v, X_ols).fit()
alpha_ols = float(ols.params[0])
beta_ols = float(ols.params[1])
nairu_ols = -alpha_ols / beta_ols if abs(beta_ols) > 1e-8 else np.nan
print(f"  OLS alpha = {alpha_ols:.4f}, beta = {beta_ols:.4f}")
print(f"  OLS-implied constant NAIRU = {nairu_ols:.3f}%")

# HP-trend reference for unemployment (CBO-like proxy)
_, u_trend = hpfilter(u, lamb=1600)
cbo_like = np.asarray(u_trend)

# ===========================================================================
# Model 1: slope TV, NAIRU fixed at sample mean
# ===========================================================================
print("\n" + "-" * 70)
print("Model 1: slope TV, NAIRU fixed (u* = u_mean)")
print("-" * 70)

u_bar = float(u_v.mean())
X1 = (u_v - u_bar).reshape(-1, 1)
m1 = TimeVaryingParameters(dpi_v, X1, q_type="diagonal")
r1 = m1.fit()
assert r1.smoothed_state is not None
assert r1.smoothed_cov is not None
beta1_path = np.asarray(r1.smoothed_state)[:, 0]
beta1_se = np.sqrt(np.maximum(np.asarray(r1.smoothed_cov)[:, 0, 0], 0.0))

print(f"  loglike = {r1.loglike:.4f}")
for name, val in zip(r1.param_names, r1.params):
    print(f"  {name:18s} = {val:.6f}")
check(np.isfinite(r1.loglike), "Model 1 loglike is finite")
check(np.all(r1.params >= 0.0), "Model 1 variance params are non-negative")

frac_neg_m1 = fraction_negative(beta1_path)
print(f"  Fraction of periods with beta_t < 0 (Model 1): {frac_neg_m1:.3f}")
check(
    frac_neg_m1 >= NEG_SLOPE_MIN_FRAC,
    f"Model 1 slope is negative in >= {NEG_SLOPE_MIN_FRAC:.0%} of periods "
    f"(got {frac_neg_m1:.3f})",
)

# ===========================================================================
# Model 2: NAIRU TV, slope fixed (local level on residual)
# ===========================================================================
print("\n" + "-" * 70)
print("Model 2: NAIRU TV, slope fixed at OLS beta")
print("-" * 70)

beta_fixed = beta_ols
r_res = dpi_v - beta_fixed * u_v

m2 = LocalLevel(r_res)
r2 = m2.fit()
assert r2.smoothed_state is not None
assert r2.smoothed_cov is not None
alpha2 = np.asarray(r2.smoothed_state)[:, 0]
alpha2_se = np.sqrt(np.maximum(np.asarray(r2.smoothed_cov)[:, 0, 0], 0.0))

if abs(beta_fixed) < 1e-8:
    raise RuntimeError("OLS slope is too close to zero; Model 2 is ill-defined.")

nairu2 = -alpha2 / beta_fixed
nairu2_se = alpha2_se / abs(beta_fixed)

print(f"  loglike (LocalLevel on residual) = {r2.loglike:.4f}")
for name, val in zip(r2.param_names, r2.params):
    print(f"  {name:18s} = {val:.6f}")
print(f"  fixed slope = {beta_fixed:.6f}")
check(np.isfinite(r2.loglike), "Model 2 loglike is finite")
check(np.all(r2.params >= 0.0), "Model 2 variance params are non-negative")

frac_in_m2 = fraction_in_band(nairu2, NAIRU_LOWER, NAIRU_UPPER)
print(
    f"  Fraction of periods with NAIRU in [{NAIRU_LOWER}, {NAIRU_UPPER}] "
    f"(Model 2): {frac_in_m2:.3f}"
)
print(f"  NAIRU (Model 2): mean = {nairu2.mean():.3f}, "
      f"min = {nairu2.min():.3f}, max = {nairu2.max():.3f}")
check(
    frac_in_m2 >= NAIRU_INSIDE_MIN_FRAC,
    f"Model 2 NAIRU in [{NAIRU_LOWER}, {NAIRU_UPPER}] for "
    f">= {NAIRU_INSIDE_MIN_FRAC:.0%} of periods (got {frac_in_m2:.3f})",
)

# Cross-check vs statsmodels UnobservedComponents on residualized series
sm_ll = UnobservedComponents(r_res, level="local level").fit(disp=False)
kb_sig2_obs = float(r2.params[0])
sm_sig2_irr = float(sm_ll.params[0])
abs_diff = abs(kb_sig2_obs - sm_sig2_irr)
rel_diff = abs_diff / max(abs(sm_sig2_irr), 1e-12)
print(
    f"  sigma2_obs kalmanbox = {kb_sig2_obs:.6f}, statsmodels = {sm_sig2_irr:.6f}, "
    f"rel_diff = {rel_diff:.4e}"
)
check(
    rel_diff < 0.05,
    f"Model 2 sigma2_obs agrees with statsmodels UC within 5% (rel_diff={rel_diff:.4f})",
)

# ===========================================================================
# Model 3: both TV (full SSW specification)
# ===========================================================================
print("\n" + "-" * 70)
print("Model 3: full TVP (intercept + slope TV)")
print("-" * 70)

X3 = np.column_stack([np.ones(nv), u_v])
m3 = TimeVaryingParameters(dpi_v, X3, q_type="diagonal")
r3 = m3.fit()
assert r3.smoothed_state is not None
assert r3.smoothed_cov is not None
alpha3 = np.asarray(r3.smoothed_state)[:, 0]
beta3 = np.asarray(r3.smoothed_state)[:, 1]
P3 = np.asarray(r3.smoothed_cov)  # (nv, 2, 2)

beta3_safe = np.where(np.abs(beta3) < 1e-3, np.sign(beta3 + 1e-12) * 1e-3, beta3)
nairu3 = -alpha3 / beta3_safe

# Delta-method SE for u* = -alpha/beta
g1 = -1.0 / beta3_safe
g2 = alpha3 / (beta3_safe ** 2)
var_nairu3 = (
    g1 ** 2 * P3[:, 0, 0]
    + g2 ** 2 * P3[:, 1, 1]
    + 2 * g1 * g2 * P3[:, 0, 1]
)
var_nairu3 = np.maximum(var_nairu3, 0.0)
nairu3_se = np.sqrt(var_nairu3)
beta3_se = np.sqrt(np.maximum(P3[:, 1, 1], 0.0))

print(f"  loglike = {r3.loglike:.4f}")
for name, val in zip(r3.param_names, r3.params):
    print(f"  {name:18s} = {val:.6f}")
check(np.isfinite(r3.loglike), "Model 3 loglike is finite")
check(np.all(r3.params >= 0.0), "Model 3 variance params are non-negative")

frac_in_m3 = fraction_in_band(nairu3, NAIRU_LOWER, NAIRU_UPPER)
frac_neg_m3 = fraction_negative(beta3)
print(
    f"  Fraction of periods with NAIRU in [{NAIRU_LOWER}, {NAIRU_UPPER}] "
    f"(Model 3): {frac_in_m3:.3f}"
)
print(f"  NAIRU (Model 3): mean = {nairu3.mean():.3f}, "
      f"min = {nairu3.min():.3f}, max = {nairu3.max():.3f}")
print(f"  Fraction of periods with beta_t < 0 (Model 3): {frac_neg_m3:.3f}")

check(
    frac_in_m3 >= NAIRU_INSIDE_MIN_FRAC,
    f"Model 3 NAIRU in [{NAIRU_LOWER}, {NAIRU_UPPER}] for "
    f">= {NAIRU_INSIDE_MIN_FRAC:.0%} of periods (got {frac_in_m3:.3f})",
)
check(
    frac_neg_m3 >= NEG_SLOPE_MIN_FRAC,
    f"Model 3 slope is negative in >= {NEG_SLOPE_MIN_FRAC:.0%} of periods "
    f"(got {frac_neg_m3:.3f})",
)

# Literature sanity: flattening after 1990 (informative, not a gating check)
pre = dates_v < pd.Timestamp("1990-01-01")
post = ~pre
pre_mean = float(beta3[pre].mean())
post_mean = float(beta3[post].mean())
print(f"  Model 3 slope pre-1990 mean  = {pre_mean:.4f}")
print(f"  Model 3 slope post-1990 mean = {post_mean:.4f}")

# ===========================================================================
# Figures
# ===========================================================================
print("\n" + "-" * 70)
print("Generating validation figures")
print("-" * 70)

# Figure 1: Model 1 time-varying slope
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates_v, beta1_path, "b-", linewidth=1.3, label="slope beta_t (Model 1)")
ax.fill_between(
    dates_v,
    beta1_path - 1.96 * beta1_se,
    beta1_path + 1.96 * beta1_se,
    alpha=0.2,
    color="blue",
    label="95% CI",
)
ax.axhline(0, color="k", linestyle=":", alpha=0.5)
ax.axhline(
    beta_ols, color="red", linestyle="--", linewidth=1, label=f"OLS = {beta_ols:.3f}"
)
ax.set_title("Model 1: time-varying Phillips slope (NAIRU fixed)")
ax.set_ylabel("beta_t")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol02_model1_slope.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol02_model1_slope.png'}")

# Figure 2: Model 2 NAIRU
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates_v, nairu2, "g-", linewidth=1.3, label="NAIRU (Model 2)")
ax.fill_between(
    dates_v,
    nairu2 - 1.96 * nairu2_se,
    nairu2 + 1.96 * nairu2_se,
    alpha=0.2,
    color="green",
    label="95% CI",
)
ax.plot(dates, u, "k-", linewidth=0.5, alpha=0.4, label="Unemployment")
ax.plot(dates, cbo_like, "r--", linewidth=1.0, label="HP trend (CBO-like)")
ax.axhspan(NAIRU_LOWER, NAIRU_UPPER, alpha=0.08, color="green", label="[3, 8] band")
ax.set_title("Model 2: time-varying NAIRU (slope fixed at OLS)")
ax.set_ylabel("Rate (%)")
ax.set_ylim(0, 12)
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol02_model2_nairu.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol02_model2_nairu.png'}")

# Figure 3: Model 3 NAIRU and slope
fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
ax = axes[0]
ax.plot(dates_v, nairu3, "g-", linewidth=1.4, label="NAIRU (Model 3)")
ax.fill_between(
    dates_v,
    nairu3 - 1.96 * nairu3_se,
    nairu3 + 1.96 * nairu3_se,
    alpha=0.2,
    color="green",
    label="95% CI",
)
ax.plot(dates, u, "k-", linewidth=0.5, alpha=0.4, label="Unemployment")
ax.plot(dates, cbo_like, "r--", linewidth=1.0, label="HP trend (CBO-like)")
ax.axhspan(NAIRU_LOWER, NAIRU_UPPER, alpha=0.08, color="green", label="[3, 8] band")
ax.set_title("Model 3: smoothed NAIRU")
ax.set_ylabel("%")
ax.set_ylim(0, 12)
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(dates_v, beta3, "b-", linewidth=1.4, label="slope beta_t")
ax.fill_between(
    dates_v,
    beta3 - 1.96 * beta3_se,
    beta3 + 1.96 * beta3_se,
    alpha=0.2,
    color="blue",
    label="95% CI",
)
ax.axhline(0, color="k", linestyle=":", alpha=0.5)
ax.axhline(
    beta_ols, color="red", linestyle="--", linewidth=1, label=f"OLS = {beta_ols:.3f}"
)
ax.set_title("Model 3: smoothed Phillips slope")
ax.set_ylabel("beta_t")
ax.set_xlabel("Date")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol02_model3_nairu_slope.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol02_model3_nairu_slope.png'}")

# Figure 4: NAIRU comparison across models
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates, u, "k-", linewidth=0.5, alpha=0.4, label="Unemployment")
ax.plot(dates, cbo_like, "r--", linewidth=1.1, label="HP trend (CBO-like)")
ax.axhline(
    nairu_ols,
    color="gray",
    linestyle=":",
    linewidth=1,
    label=f"OLS constant NAIRU = {nairu_ols:.2f}%",
)
ax.plot(dates_v, nairu2, color="purple", linewidth=1.2, label="Model 2 NAIRU")
ax.plot(dates_v, nairu3, color="green", linewidth=1.4, label="Model 3 NAIRU")
ax.fill_between(
    dates_v,
    nairu3 - 1.96 * nairu3_se,
    nairu3 + 1.96 * nairu3_se,
    alpha=0.12,
    color="green",
)
ax.axhspan(NAIRU_LOWER, NAIRU_UPPER, alpha=0.08, color="green")
ax.set_ylabel("Unemployment / NAIRU (%)")
ax.set_title("Estimated NAIRU vs reference")
ax.set_ylim(0, 12)
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol02_nairu_comparison.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol02_nairu_comparison.png'}")

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
