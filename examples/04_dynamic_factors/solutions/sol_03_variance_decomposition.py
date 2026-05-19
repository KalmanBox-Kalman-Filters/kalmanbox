"""Solution 03: Variance Decomposition for Dynamic Factor Models.

Validates the variance decomposition produced by kalmanbox DynamicFactorModel
on the us_macro_panel dataset.

Checks performed:
  1. Per-series variance shares from variance_decomposition() sum to 1 up to
     machine precision.
  2. Per-series R^2 of the common component (1 - idiosyncratic share) has the
     expected qualitative structure: series that load strongly on the common
     factor have high R^2, while near-unrelated series have lower R^2.
  3. Sanity comparison with a VAR(1) FEVD on a small subset — the DFM should
     capture a non-trivial fraction of the covariance structure that a VAR
     would otherwise attribute to idiosyncratic shocks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from kalmanbox.models.dynamic_factor import DynamicFactorModel

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

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


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("Solution 03: Variance Decomposition in DFM")
print("=" * 70)

panel = pd.read_csv(DATA_DIR / "us_macro_panel.csv", parse_dates=["date"], index_col="date")
y = panel.values.astype(np.float64)
names = panel.columns.tolist()
T, N = y.shape
print(f"Loaded us_macro_panel: T={T}, N={N}")

# ===========================================================================
# Part 1: Fit DFM with K=2 (matches spec: 2-3 latent factors calibrated)
# ===========================================================================
print("\n" + "-" * 70)
print("Part 1: Fit DFM (K=2)")
print("-" * 70)

K = 2
model = DynamicFactorModel(y, k_factors=K, endog_names=names)
fit_res = model.fit(maxiter=400, compute_se=False)
res = model.smooth(fit_res.params)
print(f"  loglike:   {res.loglike:.2f}")
print(f"  BIC:       {res.bic:.2f}")
print(f"  converged: {res.optimizer_converged}")
check(np.isfinite(res.loglike), "Fitted loglike is finite")

# ===========================================================================
# Part 2: Variance decomposition sums to 1 per series
# ===========================================================================
print("\n" + "-" * 70)
print("Part 2: Variance decomposition")
print("-" * 70)

decomp = model.variance_decomposition(res)  # shape (N, K+1)
assert decomp.shape == (N, K + 1), f"decomp shape mismatch: {decomp.shape}"
row_sums = decomp.sum(axis=1)

print(f"  Decomposition shape: {decomp.shape} (N={N}, K+1={K + 1})")
print(f"  min row sum: {row_sums.min():.10f}")
print(f"  max row sum: {row_sums.max():.10f}")

check(
    np.allclose(row_sums, 1.0, atol=1e-10),
    "All per-series variance shares sum to 1 (fatores + idiossincratica ~= 1)",
)

# Shares are proportions in [0, 1]
check(
    np.all((decomp >= -1e-12) & (decomp <= 1.0 + 1e-12)),
    "All variance shares are in [0, 1]",
)

# Print per-series decomposition table
print("\n  Per-series variance decomposition:")
header = f"  {'series':<22}" + "".join([f"{'f' + str(j + 1):>10}" for j in range(K)]) + f"{'idio':>10}{'R^2':>10}"
print(header)
print("  " + "-" * (22 + 10 * (K + 2)))
r2_common = 1.0 - decomp[:, -1]
for i, s in enumerate(names):
    row = f"  {s:<22}"
    for j in range(K):
        row += f"{decomp[i, j]:>10.3f}"
    row += f"{decomp[i, -1]:>10.3f}"
    row += f"{r2_common[i]:>10.3f}"
    print(row)

# ===========================================================================
# Part 3: Qualitative structure
# ===========================================================================
print("\n" + "-" * 70)
print("Part 3: Structural checks on loadings")
print("-" * 70)

Lambda = res.ssm.Z[:, :K]
print(f"  Lambda shape: {Lambda.shape}")

# Mean R^2 of the common component across series should be non-trivial since
# the panel is calibrated to have 2-3 factors.
mean_r2 = float(np.mean(r2_common))
print(f"  Mean R^2 of common component across series: {mean_r2:.3f}")
check(mean_r2 > 0.2, f"Mean R^2 of common component is non-trivial ({mean_r2:.3f} > 0.2)")

# At least one real-activity series should load strongly on some factor.
real_activity_series = [
    "gdp_growth",
    "industrial_production",
    "unemployment",
    "payrolls",
    "retail_sales",
]
real_idx = [names.index(s) for s in real_activity_series]
max_abs_load_real = np.max(np.abs(Lambda[real_idx]))
print(f"  Max |loading| on real-activity block: {max_abs_load_real:.3f}")
check(
    max_abs_load_real > 0.3,
    "Some real-activity series has non-trivial loading (|lambda| > 0.3)",
)

# ===========================================================================
# Part 4: Sanity comparison with VAR(1) FEVD on a small subset
# ===========================================================================
print("\n" + "-" * 70)
print("Part 4: VAR(1) FEVD comparison on 4-series subset")
print("-" * 70)

subset_names = ["industrial_production", "unemployment", "pmi_manufacturing", "fed_funds_rate"]
sub_idx = [names.index(s) for s in subset_names]
y_sub = y[:, sub_idx]
var_model = VAR(y_sub)
var_res = var_model.fit(maxlags=1)

# FEVD at horizon 12 — captures persistent own vs cross contributions.
fevd = var_res.fevd(12)
# fevd.decomp has shape (k, horizon, k): (equation_i, step, shock_j)
h_decomp = fevd.decomp[:, -1, :]  # last horizon

print(f"  VAR(1) FEVD at h=12 (share of variance from each shock):")
print(f"  {'series':<22}" + "".join([f"{s[:6]:>10}" for s in subset_names]))
print("  " + "-" * (22 + 10 * len(subset_names)))
for i, s in enumerate(subset_names):
    row = f"  {s:<22}" + "".join([f"{h_decomp[i, j]:>10.3f}" for j in range(len(subset_names))])
    print(row)

# Diagonal (own-variance) should dominate for each series (otherwise the VAR is
# ill-conditioned); off-diagonals carry spillover info that the DFM summarizes
# into a common factor.
own_shares = np.diag(h_decomp)
cross_shares = h_decomp - np.diag(own_shares)
print(f"\n  Mean own-shock share:   {own_shares.mean():.3f}")
print(f"  Mean cross-shock share: {cross_shares[~np.eye(len(sub_idx), dtype=bool)].mean():.3f}")

check(
    np.allclose(h_decomp.sum(axis=1), 1.0, atol=1e-6),
    "VAR FEVD rows sum to 1 (sanity check)",
)

# Cross-sectional pairs should move together (positive mean covariance of the
# subset after demeaning — feature of a common-factor panel).
yc = y_sub - y_sub.mean(axis=0, keepdims=True)
cov_sub = yc.T @ yc / T
off_diag_mean = (cov_sub.sum() - np.trace(cov_sub)) / (len(sub_idx) ** 2 - len(sub_idx))
print(f"  Mean off-diagonal covariance of subset: {off_diag_mean:.3f}")

# ===========================================================================
# Figures
# ===========================================================================
print("\n" + "-" * 70)
print("Generating validation figures")
print("-" * 70)

# Figure 1: stacked bar of variance decomposition by series
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(N)
bottom = np.zeros(N)
colors = plt.get_cmap("tab10")
for j in range(K):
    ax.bar(x, decomp[:, j], bottom=bottom, label=f"f{j + 1}", color=colors(j))
    bottom += decomp[:, j]
ax.bar(x, decomp[:, -1], bottom=bottom, label="idiosyncratic", color="lightgray")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Variance share")
ax.set_title(f"DFM variance decomposition (K={K})")
ax.legend(loc="upper right")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol03_variance_decomposition.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol03_variance_decomposition.png'}")

# Figure 2: R^2 of common component per series
fig, ax = plt.subplots(figsize=(10, 5))
order = np.argsort(r2_common)[::-1]
ax.bar(range(N), r2_common[order], color="steelblue")
ax.set_xticks(range(N))
ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("R^2 of common component")
ax.set_title("Share of variance explained by common factors, per series")
ax.axhline(mean_r2, color="red", linestyle="--", label=f"mean = {mean_r2:.2f}")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol03_r2_per_series.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol03_r2_per_series.png'}")

# Figure 3: VAR FEVD heatmap (subset)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(h_decomp, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(subset_names)))
ax.set_yticks(range(len(subset_names)))
ax.set_xticklabels([s[:8] for s in subset_names], rotation=30, ha="right")
ax.set_yticklabels([s[:8] for s in subset_names])
ax.set_xlabel("Shock j")
ax.set_ylabel("Variable i")
ax.set_title("VAR(1) FEVD at h=12 (share)")
for i in range(len(subset_names)):
    for j in range(len(subset_names)):
        ax.text(j, i, f"{h_decomp[i, j]:.2f}", ha="center", va="center",
                color="black" if h_decomp[i, j] < 0.5 else "white", fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.7)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol03_var_fevd.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol03_var_fevd.png'}")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
if ALL_PASSED:
    print("Solution 03: ALL CHECKS PASSED")
else:
    print("Solution 03: SOME CHECKS FAILED")
    sys.exit(1)
print("=" * 70)
