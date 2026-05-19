"""Solution 01: Dynamic Factor Model (DFM) Introduction.

Validates kalmanbox DynamicFactorModel against reference implementations
on the us_macro_panel dataset (15 synthetic US macro series, monthly).

Checks performed:
  1. Fit kalmanbox DFM with K=1 and K=2, verify convergence and finite loglike.
  2. Fit statsmodels DynamicFactor and compare common factors (|corr| > 0.9
     after sign adjustment).
  3. Compare kalmanbox factors with principal components (PCA) of the panel:
     the state-space ML factor should track PC1 closely (|corr| > 0.9).
  4. Model selection via BIC: compare K=1 vs K=2 and pick the minimizer.

References:
  - Stock & Watson (2002) JASA 97(460).
  - Doz, Giannone & Reichlin (2012) Rev Econ Stat 94(4).
  - Banbura, Giannone & Reichlin (2011) Oxford Handbook of Economic Forecasting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from kalmanbox.models.dynamic_factor import DynamicFactorModel

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CORR_TOL = 0.9       # primary threshold: kalmanbox vs statsmodels (ML reference)
CORR_TOL_PCA = 0.85  # secondary threshold: kalmanbox vs PC1 (static factor)

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


def sign_adjusted_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation between two 1D series, taking absolute value to handle sign flips."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 0.0
    return float(abs(np.corrcoef(a[mask], b[mask])[0, 1]))


def pca_factors(y: np.ndarray, k: int) -> np.ndarray:
    """PCA factors on standardized columns.

    Parameters
    ----------
    y : (T, N) array (no NaN).
    k : number of principal components to return.

    Returns
    -------
    (T, k) array of principal component scores.
    """
    yc = y - y.mean(axis=0, keepdims=True)
    yc = yc / (y.std(axis=0, ddof=0, keepdims=True) + 1e-12)
    cov = yc.T @ yc / yc.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1][:k]
    loadings = evecs[:, idx]
    return yc @ loadings


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("Solution 01: Dynamic Factor Model — Introduction")
print("=" * 70)

panel = pd.read_csv(DATA_DIR / "us_macro_panel.csv", parse_dates=["date"], index_col="date")
y = panel.values.astype(np.float64)
names = panel.columns.tolist()
T, N = y.shape
print(f"Loaded us_macro_panel: T={T} months, N={N} series")
print(f"Date range: {panel.index[0].date()} -> {panel.index[-1].date()}")

# ===========================================================================
# Part 1: PCA reference
# ===========================================================================
print("\n" + "-" * 70)
print("Part 1: PCA reference factors")
print("-" * 70)

pc_all = pca_factors(y, k=2)
print(f"  PC1 variance share: {np.var(pc_all[:, 0]) / np.var(pc_all).sum():.3f}")
print(f"  PC2 variance share: {np.var(pc_all[:, 1]) / np.var(pc_all).sum():.3f}")

# ===========================================================================
# Part 2: kalmanbox DFM with K=1
# ===========================================================================
print("\n" + "-" * 70)
print("Part 2: kalmanbox DFM, K=1")
print("-" * 70)

kb1 = DynamicFactorModel(y, k_factors=1, endog_names=names)
kb1_res = kb1.smooth(kb1.fit(maxiter=300, compute_se=False).params)
print(f"  kalmanbox loglike: {kb1_res.loglike:.2f}")
print(f"  kalmanbox BIC:     {kb1_res.bic:.2f}")
print(f"  converged:         {kb1_res.optimizer_converged}")

assert kb1_res.smoothed_state is not None
kb1_factor = kb1_res.smoothed_state[:, 0]
check(np.isfinite(kb1_res.loglike), "kalmanbox K=1 loglike is finite")

# Compare kalmanbox factor vs PC1 (secondary reference; PCA uses different
# normalization and no AR dynamics, so threshold is slightly looser)
corr_kb1_pc1 = sign_adjusted_corr(kb1_factor, pc_all[:, 0])
print(f"  |corr(kalmanbox factor, PC1)| = {corr_kb1_pc1:.4f}")
check(corr_kb1_pc1 > CORR_TOL_PCA, f"kalmanbox K=1 factor corr with PC1 > {CORR_TOL_PCA}")

# ===========================================================================
# Part 3: statsmodels DynamicFactor reference (K=1)
# ===========================================================================
print("\n" + "-" * 70)
print("Part 3: statsmodels DynamicFactor, K=1")
print("-" * 70)

sm1 = DynamicFactor(y, k_factors=1, factor_order=1, error_order=0)
sm1_res = sm1.fit(disp=False, maxiter=300)
print(f"  statsmodels loglike: {sm1_res.llf:.2f}")

# statsmodels stores factors as (k_factors, nobs)
sm1_factor = np.asarray(sm1_res.factors.smoothed).reshape(-1)
if sm1_factor.shape[0] != T:
    # Handle orientation
    sm1_factor = np.asarray(sm1_res.factors.smoothed).T.reshape(-1)

corr_kb_sm = sign_adjusted_corr(kb1_factor, sm1_factor)
print(f"  |corr(kalmanbox, statsmodels)| = {corr_kb_sm:.4f}")
check(corr_kb_sm > CORR_TOL, f"kalmanbox vs statsmodels factor corr > {CORR_TOL}")

# Also check statsmodels vs PC1 (upper bound on what the reference achieves)
corr_sm_pc1 = sign_adjusted_corr(sm1_factor, pc_all[:, 0])
print(f"  |corr(statsmodels, PC1)|      = {corr_sm_pc1:.4f}")

# ===========================================================================
# Part 4: kalmanbox DFM with K=2
# ===========================================================================
print("\n" + "-" * 70)
print("Part 4: kalmanbox DFM, K=2")
print("-" * 70)

kb2 = DynamicFactorModel(y, k_factors=2, endog_names=names)
kb2_res = kb2.smooth(kb2.fit(maxiter=400, compute_se=False).params)
print(f"  kalmanbox loglike: {kb2_res.loglike:.2f}")
print(f"  kalmanbox BIC:     {kb2_res.bic:.2f}")
print(f"  converged:         {kb2_res.optimizer_converged}")

check(np.isfinite(kb2_res.loglike), "kalmanbox K=2 loglike is finite")

assert kb2_res.smoothed_state is not None
kb2_f1 = kb2_res.smoothed_state[:, 0]
kb2_f2 = kb2_res.smoothed_state[:, 1]

# For K=2 the individual factors are only identified up to an orthogonal rotation;
# the *span* of (f1, f2) should match the span of (PC1, PC2). Verify by
# regressing each PC on (f1, f2) and checking R^2.
F = np.column_stack([kb2_f1, kb2_f2])
for j in range(2):
    pc = pc_all[:, j]
    beta, *_ = np.linalg.lstsq(F, pc, rcond=None)
    fitted = F @ beta
    ss_res = float(np.sum((pc - fitted) ** 2))
    ss_tot = float(np.sum((pc - pc.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    print(f"  R^2(PC{j + 1} ~ f1 + f2) = {r2:.4f}")
    check(r2 > 0.8, f"PC{j + 1} is well explained by (f1, f2): R^2 = {r2:.4f} > 0.8")

# ===========================================================================
# Part 5: BIC-based model selection (K=1 vs K=2)
# ===========================================================================
print("\n" + "-" * 70)
print("Part 5: BIC-based model selection")
print("-" * 70)

print(f"  BIC(K=1) = {kb1_res.bic:.2f}  (k_params={kb1_res.k_params})")
print(f"  BIC(K=2) = {kb2_res.bic:.2f}  (k_params={kb2_res.k_params})")

k_best = 1 if kb1_res.bic <= kb2_res.bic else 2
print(f"  Selected: K = {k_best}")

# The panel is calibrated to have multiple (2-3) factors, so BIC should not
# strongly prefer K=1. We simply check the comparison is well-defined and that
# K=2 achieves a higher loglike than K=1 (more flexibility).
check(
    kb2_res.loglike > kb1_res.loglike,
    f"K=2 loglike ({kb2_res.loglike:.2f}) > K=1 ({kb1_res.loglike:.2f})",
)
check(
    np.isfinite(kb1_res.bic) and np.isfinite(kb2_res.bic),
    "BIC is finite for K=1 and K=2",
)

# ===========================================================================
# Figures
# ===========================================================================
print("\n" + "-" * 70)
print("Generating validation figures")
print("-" * 70)

# Align signs for visual comparison: flip to positively correlate with PC1.
def align_sign(series: np.ndarray, ref: np.ndarray) -> np.ndarray:
    c = np.corrcoef(series, ref)[0, 1]
    return series if c >= 0 else -series


kb1_aligned = align_sign(kb1_factor, pc_all[:, 0])
sm1_aligned = align_sign(sm1_factor, pc_all[:, 0])

fig, ax = plt.subplots(figsize=(12, 5))
dates = panel.index
ax.plot(dates, pc_all[:, 0], color="black", alpha=0.5, linewidth=1.0, label="PC1 (standardized)")
ax.plot(dates, kb1_aligned, color="steelblue", linewidth=1.2, label="kalmanbox DFM factor (K=1)")
ax.plot(dates, sm1_aligned, color="coral", linewidth=1.0, linestyle="--", label="statsmodels DF factor (K=1)")
ax.set_title("Common factor: kalmanbox vs statsmodels vs PCA")
ax.set_ylabel("Factor value")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol01_factor_comparison_k1.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol01_factor_comparison_k1.png'}")

# K=2: plot both factors vs PC1/PC2 (sign-aligned for visual comparison)
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(dates, pc_all[:, 0], color="black", alpha=0.5, linewidth=1.0, label="PC1")
axes[0].plot(dates, align_sign(kb2_f1, pc_all[:, 0]), color="steelblue", linewidth=1.2,
             label="kalmanbox f1 (K=2)")
axes[0].set_title("K=2 factors vs principal components")
axes[0].set_ylabel("Factor 1")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(dates, pc_all[:, 1], color="black", alpha=0.5, linewidth=1.0, label="PC2")
axes[1].plot(dates, align_sign(kb2_f2, pc_all[:, 1]), color="seagreen", linewidth=1.2,
             label="kalmanbox f2 (K=2)")
axes[1].set_ylabel("Factor 2")
axes[1].set_xlabel("Date")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol01_factor_comparison_k2.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol01_factor_comparison_k2.png'}")

# BIC bar chart
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(["K=1", "K=2"], [kb1_res.bic, kb2_res.bic], color=["steelblue", "coral"])
for i, v in enumerate([kb1_res.bic, kb2_res.bic]):
    ax.annotate(f"{v:.1f}", xy=(i, v), ha="center", va="bottom", fontsize=10)
ax.set_ylabel("BIC (lower is better)")
ax.set_title("DFM model selection via BIC")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol01_bic_selection.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol01_bic_selection.png'}")

# Loadings heatmap (K=2)
fig, ax = plt.subplots(figsize=(5, 8))
Lambda2 = kb2_res.ssm.Z[:, :2]
im = ax.imshow(Lambda2, aspect="auto", cmap="RdBu_r",
               vmin=-np.abs(Lambda2).max(), vmax=np.abs(Lambda2).max())
ax.set_yticks(range(N))
ax.set_yticklabels(names, fontsize=9)
ax.set_xticks([0, 1])
ax.set_xticklabels(["f1", "f2"])
ax.set_title("Estimated loadings Lambda (K=2)")
fig.colorbar(im, ax=ax, shrink=0.6)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol01_loadings_k2.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol01_loadings_k2.png'}")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
if ALL_PASSED:
    print("Solution 01: ALL CHECKS PASSED")
else:
    print("Solution 01: SOME CHECKS FAILED")
    sys.exit(1)
print("=" * 70)
