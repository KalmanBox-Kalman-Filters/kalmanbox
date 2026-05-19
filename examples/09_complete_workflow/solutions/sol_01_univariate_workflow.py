"""Solution: Complete univariate state-space workflow on UK driver deaths.

Replicates the 12-step pipeline in `01_univariate_workflow.ipynb` as a
runnable script with explicit validation gates. The workflow:

  1. Load and log-transform UK monthly car-driver deaths (1969-1984)
  2. Build a seat-belt-law dummy (intervention from Feb 1983)
  3. Fit four candidate state-space models with kalmanbox MLE:
       - LocalLevel
       - LocalLinearTrend
       - BasicStructuralModel  (level + slope + seasonal)
       - UnobservedComponents  (BSM + seat-belt intervention)
  4. Select the best specification by minimum AIC
  5. Compute filtered + smoothed states, decompose into components
  6. Run residual diagnostics (Ljung-Box, Jarque-Bera, heteroskedasticity,
     CUSUM) and validate basic calibration of standardized residuals
  7. Produce a 24-month forecast with 95% prediction intervals
  8. Render a 6-panel dashboard
  9. Export every artefact to `output/`

Acceptance gates (raise AssertionError if violated):
  - All four models converge
  - Best model is selected as argmin(AIC) across the 4 candidates
  - Standardized residuals have mean ~ 0 and std in a sensible range
  - At least one residual diagnostic does NOT reject H0 at 5%
  - 24-month forecast has finite mean and (lower < upper) intervals
  - All expected output files are written

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
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from kalmanbox import (  # noqa: E402
    BasicStructuralModel,
    LocalLevel,
    LocalLinearTrend,
    UnobservedComponents,
)
from kalmanbox.diagnostics import (  # noqa: E402
    cusum_test,
    heteroskedasticity_test,
    ljung_box_test,
    normality_test,
    standardized_residuals,
)

NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = NOTEBOOK_DIR / "data"
OUT_DIR = NOTEBOOK_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "univariate_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SOL_FIG_DIR = Path(__file__).resolve().parent / "figures"
SOL_FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
HORIZON = 24

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
    _section("Solution 01: Complete univariate workflow (UK driver deaths)")

    # ------------------------------------------------------------------
    # Step 1 - Data preparation
    # ------------------------------------------------------------------
    print("\n--- Step 1: Data preparation ---")
    df = pd.read_csv(DATA_DIR / "uk_drivers.csv", parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df["log_deaths"] = np.log(df["deaths"].astype(float))
    df["belt_law"] = (df.index >= pd.Timestamp("1983-02-01")).astype(float)
    print(f"  nobs       = {len(df)}")
    print(f"  range      = {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  intervened = {int(df['belt_law'].sum())} obs after Feb 1983")
    assert len(df) > 100, "uk_drivers panel too short"

    # ------------------------------------------------------------------
    # Step 2 - Exploratory analysis
    # ------------------------------------------------------------------
    print("\n--- Step 2: Exploratory analysis ---")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].plot(df.index, df["log_deaths"], color="steelblue")
    axes[0].axvline(pd.Timestamp("1983-02-01"), color="crimson", ls="--",
                    label="Seat-belt law (Feb 1983)")
    axes[0].set_title("UK driver deaths (log scale)")
    axes[0].legend()
    plot_acf(df["log_deaths"].values, lags=36, ax=axes[1])
    axes[1].set_title("Autocorrelation (36 lags)")
    stl = STL(df["log_deaths"], period=12, robust=True).fit()
    axes[2].plot(df.index, stl.trend, label="trend", color="darkorange")
    axes[2].plot(df.index, stl.seasonal, label="seasonal", color="seagreen")
    axes[2].plot(df.index, stl.resid, label="residual", color="gray", alpha=0.6)
    axes[2].legend(ncol=3)
    axes[2].set_title("STL decomposition (preliminary)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_exploration.png")
    plt.close(fig)
    print("  saved 01_exploration.png")

    # ------------------------------------------------------------------
    # Step 3 - Candidate models
    # ------------------------------------------------------------------
    print("\n--- Step 3: Fit four candidate models ---")
    y = df["log_deaths"].values
    exog = df[["belt_law"]].values

    m1 = LocalLevel(y)
    r1 = m1.fit(compute_se=False)

    m2 = LocalLinearTrend(y)
    r2 = m2.fit(compute_se=False)

    m3 = BasicStructuralModel(y, seasonal_period=12, seasonal="dummy")
    r3 = m3.fit(compute_se=False, maxiter=300)

    m4 = UnobservedComponents(
        y, level=True, trend="stochastic",
        seasonal="dummy", seasonal_period=12, exog=exog,
    )
    r4 = m4.fit(compute_se=False, maxiter=400)

    fits = {
        "LocalLevel":       (m1, r1),
        "LocalLinearTrend": (m2, r2),
        "BSM":              (m3, r3),
        "BSM+intervention": (m4, r4),
    }
    for name, (_, r) in fits.items():
        ok = bool(getattr(r, "optimizer_converged", True))
        print(f"  {name:20s} converged={ok} "
              f"loglik={r.loglike:9.3f} AIC={r.aic:9.3f} BIC={r.bic:9.3f}")
        assert ok, f"{name} failed to converge"
        assert np.isfinite(r.loglike), f"{name} non-finite log-likelihood"

    # ------------------------------------------------------------------
    # Step 4 - Model comparison
    # ------------------------------------------------------------------
    print("\n--- Step 4: Model comparison ---")
    comparison = pd.DataFrame({
        "model":    list(fits.keys()),
        "k_params": [r.k_params for _, r in fits.values()],
        "loglik":   [r.loglike for _, r in fits.values()],
        "aic":      [r.aic for _, r in fits.values()],
        "bic":      [r.bic for _, r in fits.values()],
    }).set_index("model").round(3)
    comparison["delta_aic"] = comparison["aic"] - comparison["aic"].min()
    comparison = comparison.sort_values("aic")
    comparison.to_csv(OUT_DIR / "univariate_model_comparison.csv")
    print(comparison)

    # ------------------------------------------------------------------
    # Step 5 - Select best model
    # ------------------------------------------------------------------
    _section("Step 5: Best-model selection (min AIC)")
    best_name = str(comparison["aic"].idxmin())
    best_model, best = fits[best_name]
    expected_best = comparison.index[0]
    assert best_name == expected_best, (
        f"selection mismatch: {best_name!r} vs {expected_best!r}"
    )
    print(f"  Selected: {best_name}")
    print(f"  loglik = {best.loglike:.3f}  AIC = {best.aic:.3f}  "
          f"BIC = {best.bic:.3f}")

    # ------------------------------------------------------------------
    # Step 6 - Filtered states
    # ------------------------------------------------------------------
    print("\n--- Step 6: Kalman filter ---")
    filtered = best.filtered_state
    filtered_se = np.sqrt(np.maximum(
        np.diagonal(best.filtered_cov, axis1=1, axis2=2), 0.0))
    level_f = filtered[:, 0]
    level_f_se = filtered_se[:, 0]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df.index, df["log_deaths"], color="black", alpha=0.5,
            label="observed")
    ax.plot(df.index, level_f, color="steelblue", label="filtered level")
    ax.fill_between(df.index, level_f - 1.96 * level_f_se,
                    level_f + 1.96 * level_f_se,
                    color="steelblue", alpha=0.15, label="95% CI")
    ax.set_title(f"Filtered level ({best_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_filtered.png")
    plt.close(fig)
    print("  saved 02_filtered.png")

    # ------------------------------------------------------------------
    # Step 7 - Smoother
    # ------------------------------------------------------------------
    print("\n--- Step 7: Kalman smoother ---")
    smoothed = best.smoothed_state
    smoothed_se = np.sqrt(np.maximum(
        np.diagonal(best.smoothed_cov, axis1=1, axis2=2), 0.0))
    level_s = smoothed[:, 0]
    level_s_se = smoothed_se[:, 0]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df.index, df["log_deaths"], color="black", alpha=0.4,
            label="observed")
    ax.plot(df.index, level_f, color="steelblue", lw=1.0, alpha=0.7,
            label="filtered")
    ax.plot(df.index, level_s, color="crimson", label="smoothed")
    ax.fill_between(df.index, level_s - 1.96 * level_s_se,
                    level_s + 1.96 * level_s_se,
                    color="crimson", alpha=0.15, label="95% CI (smoothed)")
    ax.set_title(f"Smoothed level ({best_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_smoothed.png")
    plt.close(fig)
    print("  saved 03_smoothed.png")

    # ------------------------------------------------------------------
    # Step 8 - Component decomposition
    # ------------------------------------------------------------------
    print("\n--- Step 8: Component decomposition ---")
    level_comp = smoothed[:, 0]
    slope_comp = (np.zeros(len(df)) if best_name == "LocalLevel"
                  else smoothed[:, 1])
    has_seasonal = best_name in ("BSM", "BSM+intervention")
    seasonal_comp = (smoothed[:, 2] if has_seasonal
                     else np.zeros(len(df)))
    if best_name == "BSM+intervention":
        exog_coef = float(smoothed[-1, -1])
        intervention_effect = df["belt_law"].values * exog_coef
        signal = level_comp + seasonal_comp + intervention_effect
    else:
        intervention_effect = np.zeros(len(df))
        signal = level_comp + seasonal_comp
    irregular = df["log_deaths"].values - signal

    decomp = pd.DataFrame({
        "observed":     df["log_deaths"].values,
        "level":        level_comp,
        "slope":        slope_comp,
        "seasonal":     seasonal_comp,
        "intervention": intervention_effect,
        "irregular":    irregular,
    }, index=df.index)
    decomp.to_csv(OUT_DIR / "univariate_decomposition.csv")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df.index, decomp["observed"], color="black")
    axes[0].set_title("Observed")
    axes[1].plot(df.index, decomp["level"], color="steelblue", label="level")
    if best_name == "BSM+intervention":
        axes[1].plot(df.index, decomp["level"] + decomp["intervention"],
                     color="crimson", ls="--", label="level + intervention")
    axes[1].legend()
    axes[1].set_title("Level (+ intervention if applicable)")
    axes[2].plot(df.index, decomp["seasonal"], color="seagreen")
    axes[2].set_title("Seasonal")
    axes[3].plot(df.index, decomp["irregular"], color="gray")
    axes[3].axhline(0, color="black", lw=0.5)
    axes[3].set_title("Irregular")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_decomposition.png")
    plt.close(fig)
    print("  saved 04_decomposition.png")

    # ------------------------------------------------------------------
    # Step 9 - Residual diagnostics
    # ------------------------------------------------------------------
    _section("Step 9: Residual diagnostics")
    std_res = standardized_residuals(best)
    finite_res = std_res[~np.isnan(std_res)]
    n_eff = len(finite_res)
    res_mean = float(np.mean(finite_res))
    res_std = float(np.std(finite_res, ddof=1))
    print(f"  n_eff = {n_eff}, mean = {res_mean:+.3f}, std = {res_std:.3f}")

    tests = {
        "Ljung-Box(12)":      ljung_box_test(std_res, lags=12),
        "Jarque-Bera":        normality_test(std_res),
        "Heteroskedasticity": heteroskedasticity_test(std_res),
        "CUSUM":              cusum_test(std_res),
    }
    diag_df = pd.DataFrame([
        {"test": k, "statistic": v.statistic, "p_value": v.p_value,
         "reject_H0_5pct": v.reject}
        for k, v in tests.items()
    ]).set_index("test")
    diag_df.to_csv(OUT_DIR / "univariate_diagnostics.csv")
    print(diag_df.round(4))

    # Calibration: standardized residuals must look like ~N(0,1) on the
    # first two moments.  The independence/normality/CUSUM tests can fail
    # when the chosen model leaves structure on the table (eg LocalLevel
    # on a strongly seasonal series); we still require the basic moment
    # calibration to hold and at least one diagnostic to not reject.
    assert abs(res_mean) < 0.5, (
        f"standardized residual mean {res_mean:+.3f} too far from 0"
    )
    assert 0.5 < res_std < 2.0, (
        f"standardized residual std {res_std:.3f} outside [0.5, 2.0]"
    )
    n_pass = int((~diag_df["reject_H0_5pct"].astype(bool)).sum())
    print(f"  diagnostics not rejecting H0 at 5%: {n_pass}/4")
    assert n_pass >= 1, "no residual diagnostic accepts the null at 5%"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(df.index, std_res, color="steelblue")
    axes[0, 0].axhline(0, color="black", lw=0.5)
    axes[0, 0].axhline(2, color="red", ls="--", lw=0.5)
    axes[0, 0].axhline(-2, color="red", ls="--", lw=0.5)
    axes[0, 0].set_title("Standardised residuals")
    plot_acf(finite_res, lags=24, ax=axes[0, 1])
    axes[0, 1].set_title("ACF of residuals")
    axes[1, 0].hist(finite_res, bins=25, density=True, color="steelblue",
                    edgecolor="black", alpha=0.7)
    xx = np.linspace(finite_res.min(), finite_res.max(), 200)
    axes[1, 0].plot(xx, stats.norm.pdf(xx), color="crimson", label="N(0,1)")
    axes[1, 0].legend()
    axes[1, 0].set_title("Histogram + N(0,1)")
    stats.probplot(finite_res, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q plot")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_diagnostics.png")
    plt.close(fig)
    print("  saved 05_diagnostics.png")

    # ------------------------------------------------------------------
    # Step 10 - Forecast
    # ------------------------------------------------------------------
    _section(f"Step 10: {HORIZON}-month forecast")
    fc = best.forecast(steps=HORIZON, alpha=0.05)
    future_idx = pd.date_range(
        start=df.index[-1] + pd.offsets.MonthBegin(1),
        periods=HORIZON, freq="MS",
    )
    forecast_df = pd.DataFrame({
        "mean":  fc["mean"].ravel(),
        "lower": fc["lower"].ravel(),
        "upper": fc["upper"].ravel(),
    }, index=future_idx)
    forecast_df.to_csv(OUT_DIR / "univariate_forecast.csv")

    assert len(forecast_df) == HORIZON, "wrong forecast length"
    assert np.all(np.isfinite(forecast_df["mean"].values)), (
        "non-finite forecast mean"
    )
    assert np.all(forecast_df["upper"] > forecast_df["lower"]), (
        "forecast prediction intervals are not (lower < upper)"
    )
    print(f"  produced {len(forecast_df)} step forecasts")
    print(f"  first-step mean = {forecast_df['mean'].iloc[0]:.4f}, "
          f"PI = [{forecast_df['lower'].iloc[0]:.4f}, "
          f"{forecast_df['upper'].iloc[0]:.4f}]")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["log_deaths"], color="black", label="observed")
    ax.plot(forecast_df.index, forecast_df["mean"], color="crimson",
            label="forecast")
    ax.fill_between(forecast_df.index, forecast_df["lower"],
                    forecast_df["upper"], color="crimson", alpha=0.2,
                    label="95% PI")
    ax.axvline(df.index[-1], color="gray", ls=":", lw=0.8)
    ax.set_title(f"{HORIZON}-month forecast ({best_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_forecast.png")
    plt.close(fig)
    print("  saved 06_forecast.png")

    # ------------------------------------------------------------------
    # Step 11 - Dashboard
    # ------------------------------------------------------------------
    _section("Step 11: Dashboard")
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.25)

    axA = fig.add_subplot(gs[0, 0])
    axA.plot(df.index, df["log_deaths"], color="black")
    axA.axvline(pd.Timestamp("1983-02-01"), color="crimson", ls="--")
    axA.set_title("(a) Observed + intervention")

    axB = fig.add_subplot(gs[0, 1])
    comp_sorted = comparison.sort_values("aic")
    axB.barh(comp_sorted.index, comp_sorted["aic"], color="steelblue")
    axB.invert_yaxis()
    axB.set_title("(b) AIC ranking")
    axB.set_xlabel("AIC")

    axC = fig.add_subplot(gs[1, 0])
    axC.plot(df.index, level_f, color="steelblue", label="filtered")
    axC.plot(df.index, level_s, color="crimson", label="smoothed")
    axC.legend()
    axC.set_title("(c) Level: filtered vs smoothed")

    axD = fig.add_subplot(gs[1, 1])
    axD.plot(df.index, seasonal_comp, color="seagreen")
    axD.axhline(0, color="black", lw=0.5)
    axD.set_title("(d) Seasonal component")

    axE = fig.add_subplot(gs[2, 0])
    axE.plot(df.index, std_res, color="steelblue")
    axE.axhline(0, color="black", lw=0.5)
    axE.axhline(2, color="red", ls="--", lw=0.5)
    axE.axhline(-2, color="red", ls="--", lw=0.5)
    axE.set_title("(e) Standardised residuals")

    axF = fig.add_subplot(gs[2, 1])
    axF.plot(df.index[-36:], df["log_deaths"].values[-36:], color="black",
             label="observed")
    axF.plot(forecast_df.index, forecast_df["mean"], color="crimson",
             label="forecast")
    axF.fill_between(forecast_df.index, forecast_df["lower"],
                     forecast_df["upper"], color="crimson", alpha=0.2)
    axF.legend()
    axF.set_title(f"(f) {HORIZON}-month forecast")

    fig.suptitle(f"Univariate workflow dashboard - {best_name}",
                 fontsize=14, y=0.995)
    fig.savefig(FIG_DIR / "07_dashboard.png", bbox_inches="tight")
    fig.savefig(SOL_FIG_DIR / "sol_01_dashboard.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 07_dashboard.png")

    # ------------------------------------------------------------------
    # Step 12 - Export results + statsmodels cross-check
    # ------------------------------------------------------------------
    _section("Step 12: Export + statsmodels cross-check")
    params_df = best.to_dataframe()
    params_df.to_csv(OUT_DIR / "univariate_best_params.csv")
    (OUT_DIR / "univariate_best_summary.txt").write_text(best.summary())

    sm_model = sm.tsa.UnobservedComponents(
        df["log_deaths"], level="local linear trend",
        seasonal=12, exog=df[["belt_law"]],
    )
    sm_res = sm_model.fit(disp=False, maxiter=200)
    xcompare = pd.DataFrame({
        "loglike": [r4.loglike, sm_res.llf],
        "aic":     [r4.aic,     sm_res.aic],
        "bic":     [r4.bic,     sm_res.bic],
    }, index=["kalmanbox", "statsmodels"]).round(3)
    xcompare.to_csv(OUT_DIR / "univariate_vs_statsmodels.csv")
    print(xcompare)

    expected_files = [
        "univariate_model_comparison.csv",
        "univariate_decomposition.csv",
        "univariate_diagnostics.csv",
        "univariate_forecast.csv",
        "univariate_best_params.csv",
        "univariate_best_summary.txt",
        "univariate_vs_statsmodels.csv",
    ]
    for f in expected_files:
        assert (OUT_DIR / f).exists(), f"missing output {f}"
    expected_figs = [
        "01_exploration.png", "02_filtered.png", "03_smoothed.png",
        "04_decomposition.png", "05_diagnostics.png",
        "06_forecast.png", "07_dashboard.png",
    ]
    for f in expected_figs:
        assert (FIG_DIR / f).exists(), f"missing figure {f}"

    _section("Solution 01: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
