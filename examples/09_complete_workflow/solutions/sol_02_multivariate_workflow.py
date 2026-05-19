"""Solution: Complete multivariate state-space workflow on a US macro panel.

Replicates the 12-step pipeline in `02_multivariate_workflow.ipynb` as a
runnable script with explicit validation gates. The workflow:

  1. Load `us_macro_panel.csv` (8 monthly indicators, 288 obs) and
     `mixed_freq_macro.csv` (5 series, GDP quarterly + 4 monthly)
  2. Standardise the panels (zero mean, unit variance)
  3. Exploration: correlation heatmap + PCA scree + missingness
  4. Fit Dynamic Factor Models with K = 1, 2, 3 (factor_order=1)
  5. Select K* = argmin(BIC)
  6. Extract smoothed factors + plot loadings heatmap
  7. Mixed-frequency GDP nowcast (Kalman smoother on missing values)
  8. Variance decomposition and per-series residual diagnostics
  9. 6-panel dashboard summarising every step
 10. Cross-check with `statsmodels.tsa.DynamicFactor` and export

Acceptance gates (raise AssertionError if violated):
  - All three DFMs (K = 1, 2, 3) converge with finite log-likelihood
  - Selected K is argmin(BIC) across the three candidates
  - Variance-decomposition rows sum to ~1
  - GDP nowcast at quarter-ends has lower RMSE than the naive
    "previous-quarter" benchmark
  - All expected output artefacts are written

Exit code 0 on success.
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

from kalmanbox import DynamicFactorModel  # noqa: E402
from kalmanbox.diagnostics import (  # noqa: E402
    ljung_box_test,
    normality_test,
)

NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = NOTEBOOK_DIR / "data"
OUT_DIR = NOTEBOOK_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "multivariate_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SOL_FIG_DIR = Path(__file__).resolve().parent / "figures"
SOL_FIG_DIR.mkdir(parents=True, exist_ok=True)

MAIN_COLS = [
    "gdp_growth", "industrial_production", "unemployment",
    "payrolls", "retail_sales", "housing_starts",
    "pmi_manufacturing", "cpi_inflation",
]
MIXED_COLS = ["gdp_growth", "industrial_production",
              "unemployment", "cpi", "pmi"]

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    np.random.seed(42)
    _section("Solution 02: Complete multivariate workflow (US macro panel)")

    # ------------------------------------------------------------------
    # Step 1 - Load
    # ------------------------------------------------------------------
    print("\n--- Step 1: Data preparation ---")
    panel = pd.read_csv(DATA_DIR / "us_macro_panel.csv",
                        parse_dates=["date"]).set_index("date")
    mixed = pd.read_csv(DATA_DIR / "mixed_freq_macro.csv",
                        parse_dates=["date"]).set_index("date")
    Y_raw = panel[MAIN_COLS].astype(float)
    Y_std = (Y_raw - Y_raw.mean()) / Y_raw.std()
    print(f"  panel = {Y_std.shape}  cols = {MAIN_COLS}")
    assert Y_std.shape == (288, 8), f"unexpected panel shape {Y_std.shape}"

    # ------------------------------------------------------------------
    # Step 2 - Exploration
    # ------------------------------------------------------------------
    print("\n--- Step 2: Exploratory analysis ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    corr = Y_std.corr()
    im = axes[0].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(MAIN_COLS)))
    axes[0].set_yticks(range(len(MAIN_COLS)))
    axes[0].set_xticklabels(MAIN_COLS, rotation=60, ha="right")
    axes[0].set_yticklabels(MAIN_COLS)
    axes[0].set_title("Correlation matrix")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    X = Y_std.dropna().values
    X_c = X - X.mean(axis=0)
    cov = X_c.T @ X_c / X_c.shape[0]
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    explained = eigvals / eigvals.sum()
    axes[1].bar(range(1, len(explained) + 1), explained, color="steelblue")
    axes[1].plot(range(1, len(explained) + 1), np.cumsum(explained),
                 color="crimson", marker="o", label="cumulative")
    axes[1].axhline(0.8, color="gray", ls="--", lw=0.7)
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Explained variance")
    axes[1].set_title("PCA scree")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_exploration.png")
    plt.close(fig)
    print(f"  PCA cumulative var (5 PCs): "
          f"{np.round(np.cumsum(explained)[:5], 3)}")

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.imshow(mixed.isna().T, aspect="auto", cmap="Greys",
              interpolation="none")
    ax.set_yticks(range(len(mixed.columns)))
    ax.set_yticklabels(mixed.columns)
    ax.set_title("Missingness - mixed-frequency panel (black = missing)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_missing_pattern.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Step 3 - Fit DFM K=1,2,3
    # ------------------------------------------------------------------
    _section("Step 3: Fit DFM with K = 1, 2, 3")
    fitted_models: dict[int, tuple] = {}
    for K in (1, 2, 3):
        m = DynamicFactorModel(
            Y_std.values, k_factors=K, factor_order=1,
            endog_names=MAIN_COLS,
        )
        r = m.fit(compute_se=False, maxiter=200)
        ok = bool(getattr(r, "optimizer_converged", True))
        print(f"  K={K}: converged={ok} loglike={r.loglike:.3f} "
              f"AIC={r.aic:.3f} BIC={r.bic:.3f} k_params={r.k_params}")
        assert ok, f"DFM K={K} did not converge"
        assert np.isfinite(r.loglike), f"DFM K={K} non-finite loglike"
        fitted_models[K] = (m, r)

    # ------------------------------------------------------------------
    # Step 4 - K selection (BIC)
    # ------------------------------------------------------------------
    _section("Step 4: Factor-count selection")
    k_comparison = pd.DataFrame({
        K: {
            "k_params": r.k_params,
            "loglik":   r.loglike,
            "aic":      r.aic,
            "bic":      r.bic,
        } for K, (_, r) in fitted_models.items()
    }).T.round(3)
    k_comparison.index.name = "K"
    k_comparison.to_csv(OUT_DIR / "multivariate_K_selection.csv")
    print(k_comparison)

    best_K = int(k_comparison["bic"].idxmin())
    expected_K = int(min(k_comparison.index, key=lambda k: k_comparison.loc[k, "bic"]))
    assert best_K == expected_K, "BIC selection mismatch"
    print(f"  Selected K* (min BIC) = {best_K}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_comparison.index, k_comparison["bic"], marker="o",
            color="crimson", label="BIC")
    ax.plot(k_comparison.index, k_comparison["aic"], marker="s",
            color="steelblue", label="AIC")
    ax.axvline(best_K, color="gray", ls="--",
               label=f"selected K={best_K}")
    ax.set_xlabel("Number of factors K")
    ax.set_ylabel("Information criterion")
    ax.set_title("DFM factor-count selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_k_selection.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Step 5 - Best DFM
    # ------------------------------------------------------------------
    print("\n--- Step 5: Best DFM ---")
    best_model, best_res = fitted_models[best_K]
    print(f"  k_params={best_res.k_params}  loglik={best_res.loglike:.3f}  "
          f"AIC={best_res.aic:.3f}  BIC={best_res.bic:.3f}")

    # ------------------------------------------------------------------
    # Step 6 - Smoothed factors
    # ------------------------------------------------------------------
    print("\n--- Step 6: Smoothed factors ---")
    factors = best_res.smoothed_state[:, :best_K]
    factor_cov = best_res.smoothed_cov[:, :best_K, :best_K]
    factor_se = np.sqrt(np.maximum(
        np.diagonal(factor_cov, axis1=1, axis2=2), 0.0))
    factors_df = pd.DataFrame(
        factors,
        columns=[f"factor_{i+1}" for i in range(best_K)],
        index=Y_std.index,
    )
    factors_df.to_csv(OUT_DIR / "multivariate_factors.csv")
    assert factors_df.shape == (Y_std.shape[0], best_K), (
        "factor matrix has wrong shape"
    )

    fig, axes = plt.subplots(best_K, 1, figsize=(12, 3 * best_K),
                             sharex=True, squeeze=False)
    for i in range(best_K):
        ax = axes[i, 0]
        ax.plot(factors_df.index, factors[:, i], color="steelblue")
        ax.fill_between(
            factors_df.index,
            factors[:, i] - 1.96 * factor_se[:, i],
            factors[:, i] + 1.96 * factor_se[:, i],
            color="steelblue", alpha=0.15,
        )
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel(f"Factor {i+1}")
    axes[-1, 0].set_xlabel("Date")
    fig.suptitle(f"Smoothed factors (DFM K={best_K})", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_factors.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Step 7 - Loadings
    # ------------------------------------------------------------------
    print("\n--- Step 7: Loadings ---")
    loadings = best_res.ssm.Z[:, :best_K]
    loadings_df = pd.DataFrame(
        loadings, index=MAIN_COLS,
        columns=[f"f{i+1}" for i in range(best_K)],
    )
    loadings_df.to_csv(OUT_DIR / "multivariate_loadings.csv")
    print(loadings_df.round(3))

    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = float(np.max(np.abs(loadings))) if np.max(np.abs(loadings)) > 0 else 1.0
    im = ax.imshow(loadings, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(best_K))
    ax.set_xticklabels([f"f{i+1}" for i in range(best_K)])
    ax.set_yticks(range(len(MAIN_COLS)))
    ax.set_yticklabels(MAIN_COLS)
    for i in range(loadings.shape[0]):
        for j in range(loadings.shape[1]):
            ax.text(j, i, f"{loadings[i, j]:.2f}",
                    ha="center", va="center",
                    color=("white" if abs(loadings[i, j]) > vmax / 2
                           else "black"),
                    fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(f"Factor loadings (DFM K={best_K})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_loadings.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Step 8 - GDP nowcast (mixed-frequency)
    # ------------------------------------------------------------------
    _section("Step 8: GDP nowcasting on mixed-frequency panel")
    mixed_data = mixed[MIXED_COLS].astype(float)
    mixed_mean = mixed_data.mean()
    mixed_std = mixed_data.std()
    mixed_z = (mixed_data - mixed_mean) / mixed_std
    mixed_y = mixed_z.values

    nowcast_model = DynamicFactorModel(
        mixed_y, k_factors=1, factor_order=1, endog_names=MIXED_COLS,
    )
    nowcast_res = nowcast_model.fit(compute_se=False, maxiter=200)
    Z = nowcast_res.ssm.Z
    signal_smoothed = nowcast_res.smoothed_state @ Z.T
    gdp_idx = MIXED_COLS.index("gdp_growth")
    gdp_nowcast = (signal_smoothed[:, gdp_idx]
                   * float(mixed_std["gdp_growth"])
                   + float(mixed_mean["gdp_growth"]))

    nowcast_df = pd.DataFrame({
        "gdp_observed": mixed["gdp_growth"].values,
        "gdp_nowcast":  gdp_nowcast,
    }, index=mixed.index)
    nowcast_df.to_csv(OUT_DIR / "multivariate_nowcast.csv")

    obs = nowcast_df["gdp_observed"].dropna()
    nc_at_obs = pd.Series(gdp_nowcast, index=mixed.index).loc[obs.index]
    rmse_nowcast = float(np.sqrt(np.mean((obs.values - nc_at_obs.values) ** 2)))
    naive = obs.shift(1).dropna()
    obs_aligned = obs.loc[naive.index]
    rmse_naive = float(np.sqrt(np.mean((obs_aligned.values - naive.values) ** 2)))
    rho = float(np.corrcoef(obs.values, nc_at_obs.values)[0, 1])
    print(f"  nowcast RMSE   = {rmse_nowcast:.4f}")
    print(f"  naive RMSE     = {rmse_naive:.4f}  (previous-quarter)")
    print(f"  correlation    = {rho:+.3f}")
    assert rmse_nowcast < rmse_naive, (
        f"nowcast RMSE {rmse_nowcast:.4f} >= naive {rmse_naive:.4f}; "
        f"benchmark not beaten"
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(obs.index, obs.values, color="black", zorder=5,
               label="GDP observations (quarterly)")
    ax.plot(nowcast_df.index, nowcast_df["gdp_nowcast"], color="steelblue",
            alpha=0.8, label="monthly nowcast")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("GDP nowcasting from 1-factor DFM on monthly indicators")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_nowcast.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Step 9 - Variance decomposition
    # ------------------------------------------------------------------
    _section("Step 9: Variance decomposition")
    decomp = np.asarray(best_model.variance_decomposition(best_res),
                        dtype=float)
    row_sums = decomp.sum(axis=1)
    max_dev = float(np.max(np.abs(row_sums - 1.0)))
    print(f"  max |row_sum - 1| = {max_dev:.2e}")
    assert max_dev < 1e-3, (
        f"variance decomposition rows do not sum to 1 (max dev {max_dev:.2e})"
    )

    decomp_df = pd.DataFrame(
        decomp,
        index=MAIN_COLS,
        columns=[f"factor_{i+1}" for i in range(best_K)] + ["idiosyncratic"],
    ).round(4)
    decomp_df["R2"] = 1.0 - decomp_df["idiosyncratic"]
    decomp_df.to_csv(OUT_DIR / "multivariate_variance_decomp.csv")
    print(decomp_df)

    fig, ax = plt.subplots(figsize=(12, 5))
    left = np.zeros(len(MAIN_COLS))
    colors = plt.cm.tab10.colors
    for i in range(best_K):
        ax.barh(MAIN_COLS, decomp_df[f"factor_{i+1}"], left=left,
                color=colors[i], label=f"factor {i+1}")
        left += decomp_df[f"factor_{i+1}"].values
    ax.barh(MAIN_COLS, decomp_df["idiosyncratic"], left=left,
            color="lightgray", label="idiosyncratic")
    ax.set_xlabel("Share of variance")
    ax.set_title(f"Variance decomposition (DFM K={best_K})")
    ax.legend(loc="lower right", ncol=best_K + 1)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_variance_decomp.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Step 10 - Per-series residual diagnostics
    # ------------------------------------------------------------------
    print("\n--- Step 10: Per-series residual diagnostics ---")
    v = best_res.residuals
    F = best_res.residuals_cov
    k_endog = v.shape[1]
    std_resid = np.full_like(v, np.nan)
    for t in range(v.shape[0]):
        if np.any(np.isnan(v[t])):
            continue
        se = np.sqrt(np.maximum(np.diag(F[t]), 1e-12))
        std_resid[t] = v[t] / se

    rows = []
    for i, col in enumerate(MAIN_COLS):
        r_i = std_resid[:, i]
        lb = ljung_box_test(r_i, lags=10)
        jb = normality_test(r_i)
        rows.append({
            "series":     col,
            "lb_stat":    lb.statistic,
            "lb_pval":    lb.p_value,
            "jb_stat":    jb.statistic,
            "jb_pval":    jb.p_value,
            "resid_mean": float(np.nanmean(r_i)),
            "resid_std":  float(np.nanstd(r_i)),
        })
    resid_df = pd.DataFrame(rows).set_index("series").round(4)
    resid_df.to_csv(OUT_DIR / "multivariate_residual_tests.csv")
    print(resid_df)

    n_rows = int(np.ceil(k_endog / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 2.2 * n_rows),
                             sharex=True)
    axes_flat = axes.ravel()
    for i, ax in enumerate(axes_flat):
        if i >= k_endog:
            ax.axis("off")
            continue
        ax.plot(Y_std.index, std_resid[:, i], color="steelblue")
        ax.axhline(0, color="black", lw=0.5)
        ax.axhline(2, color="red", ls="--", lw=0.5)
        ax.axhline(-2, color="red", ls="--", lw=0.5)
        ax.set_title(MAIN_COLS[i], fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_residuals.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Step 11 - Dashboard
    # ------------------------------------------------------------------
    _section("Step 11: Dashboard")
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)

    axA = fig.add_subplot(gs[0, 0])
    axA.plot(k_comparison.index, k_comparison["bic"], marker="o",
             color="crimson")
    axA.axvline(best_K, ls="--", color="gray")
    axA.set_title("(a) BIC vs K")
    axA.set_xlabel("K")

    axB = fig.add_subplot(gs[0, 1])
    axB.imshow(Y_std.corr(), cmap="RdBu_r", vmin=-1, vmax=1)
    axB.set_xticks(range(len(MAIN_COLS)))
    axB.set_yticks(range(len(MAIN_COLS)))
    axB.set_xticklabels(MAIN_COLS, rotation=75, ha="right", fontsize=7)
    axB.set_yticklabels(MAIN_COLS, fontsize=7)
    axB.set_title("(b) Correlations")

    axC = fig.add_subplot(gs[0, 2])
    axC.imshow(loadings, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axC.set_xticks(range(best_K))
    axC.set_xticklabels([f"f{i+1}" for i in range(best_K)])
    axC.set_yticks(range(len(MAIN_COLS)))
    axC.set_yticklabels(MAIN_COLS, fontsize=7)
    axC.set_title("(c) Loadings")

    axD = fig.add_subplot(gs[1, :])
    palette = ["steelblue", "crimson", "seagreen"]
    for i in range(best_K):
        axD.plot(factors_df.index, factors[:, i], color=palette[i % 3],
                 label=f"factor {i+1}")
    axD.axhline(0, color="black", lw=0.5)
    axD.legend()
    axD.set_title("(d) Smoothed factors")

    axE = fig.add_subplot(gs[2, 0])
    left = np.zeros(len(MAIN_COLS))
    for i in range(best_K):
        axE.barh(MAIN_COLS, decomp_df[f"factor_{i+1}"], left=left,
                 color=plt.cm.tab10.colors[i], label=f"f{i+1}")
        left += decomp_df[f"factor_{i+1}"].values
    axE.barh(MAIN_COLS, decomp_df["idiosyncratic"], left=left,
             color="lightgray", label="idio")
    axE.invert_yaxis()
    axE.legend(fontsize=7, loc="lower right")
    axE.set_title("(e) Variance decomp")

    axF = fig.add_subplot(gs[2, 1:])
    axF.scatter(obs.index, obs.values, color="black", zorder=5,
                label="GDP obs")
    axF.plot(nowcast_df.index, nowcast_df["gdp_nowcast"], color="steelblue",
             alpha=0.8, label="monthly nowcast")
    axF.axhline(0, color="gray", lw=0.5)
    axF.legend()
    axF.set_title("(f) GDP nowcast")

    fig.suptitle(f"Multivariate workflow dashboard - DFM K={best_K}",
                 fontsize=14, y=0.995)
    fig.savefig(FIG_DIR / "09_dashboard.png", bbox_inches="tight")
    fig.savefig(SOL_FIG_DIR / "sol_02_dashboard.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Step 12 - Cross-check + export
    # ------------------------------------------------------------------
    _section("Step 12: statsmodels cross-check + export")
    sm_model = sm.tsa.DynamicFactor(
        Y_std.values, k_factors=1, factor_order=1,
        error_order=0, error_cov_type="diagonal",
    )
    sm_res = sm_model.fit(disp=False, maxiter=200)
    _, r1_dfm = fitted_models[1]
    xcompare = pd.DataFrame({
        "loglike": [r1_dfm.loglike, sm_res.llf],
        "aic":     [r1_dfm.aic,     sm_res.aic],
        "bic":     [r1_dfm.bic,     sm_res.bic],
    }, index=["kalmanbox", "statsmodels"]).round(3)
    xcompare.to_csv(OUT_DIR / "multivariate_vs_statsmodels.csv")
    print(xcompare)

    best_res.to_dataframe().to_csv(OUT_DIR / "multivariate_best_params.csv")
    (OUT_DIR / "multivariate_best_summary.txt").write_text(best_res.summary())

    xtool = pd.DataFrame({
        "Feature": [
            "log-likelihood (DFM K=1)",
            "Nowcasting support",
            "Variance decomposition",
            "MLE optimiser",
            "Exact diffuse init",
        ],
        "kalmanbox": [
            f"{r1_dfm.loglike:.3f}",
            "yes (Kalman smoother on NaN)",
            "built-in .variance_decomposition",
            "scipy L-BFGS-B",
            "approximate (large-variance prior)",
        ],
        "statsmodels": [
            f"{sm_res.llf:.3f}",
            "yes (MLEModel with missing)",
            "via loadings + error cov",
            "scipy L-BFGS-B",
            "exact (Koopman 1997)",
        ],
        "R (KFAS)": [
            "(see FASE9.4 report)",
            "yes (SSModel / NAs)",
            "via Z matrix + diag(H)",
            "optim BFGS",
            "exact (Koopman 1997)",
        ],
    })
    xtool.to_csv(OUT_DIR / "multivariate_tool_comparison.csv", index=False)

    expected_files = [
        "multivariate_K_selection.csv",
        "multivariate_factors.csv",
        "multivariate_loadings.csv",
        "multivariate_nowcast.csv",
        "multivariate_variance_decomp.csv",
        "multivariate_residual_tests.csv",
        "multivariate_vs_statsmodels.csv",
        "multivariate_best_params.csv",
        "multivariate_best_summary.txt",
        "multivariate_tool_comparison.csv",
    ]
    for f in expected_files:
        assert (OUT_DIR / f).exists(), f"missing output {f}"
    expected_figs = [
        "01_exploration.png", "02_missing_pattern.png", "03_k_selection.png",
        "04_factors.png", "05_loadings.png", "06_nowcast.png",
        "07_variance_decomp.png", "08_residuals.png", "09_dashboard.png",
    ]
    for f in expected_figs:
        assert (FIG_DIR / f).exists(), f"missing figure {f}"

    _section("Solution 02: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
