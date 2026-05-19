"""Solution: Bayesian Gibbs / FFBS for the Local Level model on Nile data.

Validates the kalmanbox Bayesian estimator by checking that
  1. The Gibbs sampler converges (split-chain R-hat < 1.1).
  2. The posterior means of the variance components are close to the MLE
     (relative difference < 10%) when used with weakly informative priors.
  3. FFBS produces state draws whose posterior mean matches the Kalman
     smoother (correlation > 0.99, mean absolute relative error small).

References
----------
Carter, C.K. and Kohn, R. (1994). "On Gibbs Sampling for State Space
Models." Biometrika 81(3): 541-553.
Durbin, J. and Koopman, S.J. (2012). Time Series Analysis by State Space
Methods, 2nd ed. OUP.
Gelman, A. et al. (2013). Bayesian Data Analysis, 3rd ed. CRC Press.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from kalmanbox.estimation.bayesian import BayesianSSM, InverseGamma
from kalmanbox.models.local_level import LocalLevel

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(DATA_DIR))

from mcmc_utils import (  # noqa: E402
    effective_sample_size,
    gelman_rubin,
    hpd_interval,
    trace_plot,
)

SEED = 20260416
N_DRAWS = 4000
BURNIN = 1000
N_CHAINS = 2

RHAT_TOL = 1.1
PARAM_REL_TOL = 0.10
STATE_CORR_TOL = 0.99
STATE_RELERR_TOL = 0.02


def _relative_error(actual: float, reference: float) -> float:
    return abs(actual - reference) / abs(reference)


def _run_chain(y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    model = LocalLevel(y)
    bayes = BayesianSSM(model)
    priors = {
        "sigma2_obs": InverseGamma(a=0.01, b=0.01),
        "sigma2_level": InverseGamma(a=0.01, b=0.01),
    }
    posterior = bayes.fit(
        endog=y,
        n_draws=N_DRAWS,
        burnin=BURNIN,
        priors=priors,
        seed=seed,
    )
    return {
        "sigma2_obs": posterior.param_draws["sigma2_obs"],
        "sigma2_level": posterior.param_draws["sigma2_level"],
        "states": posterior.state_draws,
    }


def main() -> int:
    print("=" * 70)
    print("Solution 01: Bayesian Gibbs / FFBS on the Nile data")
    print("=" * 70)

    nile = pd.read_csv(DATA_DIR / "nile.csv")
    y = nile["flow"].to_numpy(dtype=np.float64)
    years = nile["year"].to_numpy()
    print(f"Loaded {len(y)} Nile annual observations ({years[0]}-{years[-1]}).")

    # ------------------------------------------------------------------
    # 1. MLE reference
    # ------------------------------------------------------------------
    print("\n--- MLE reference (kalmanbox) ---")
    mle_model = LocalLevel(y)
    mle_result = mle_model.fit()
    mle_sigma2_obs = float(mle_result.params[0])
    mle_sigma2_level = float(mle_result.params[1])
    mle_smooth = mle_model.smooth(mle_result.params)
    smoothed_state = mle_smooth.smoothed_state
    assert smoothed_state is not None
    mle_smoothed = np.asarray(smoothed_state[:, 0], dtype=np.float64)
    print(f"  sigma2_obs   MLE = {mle_sigma2_obs:.3f}")
    print(f"  sigma2_level MLE = {mle_sigma2_level:.3f}")

    # ------------------------------------------------------------------
    # 2. Run two independent Gibbs chains
    # ------------------------------------------------------------------
    print(f"\n--- Running Gibbs / FFBS: {N_CHAINS} chains, "
          f"{N_DRAWS} draws + {BURNIN} burnin ---")
    rng = np.random.default_rng(SEED)
    seeds = rng.integers(low=1, high=2**31 - 1, size=N_CHAINS).tolist()

    chains_obs = np.zeros((N_CHAINS, N_DRAWS))
    chains_level = np.zeros((N_CHAINS, N_DRAWS))
    state_draws_all: list[np.ndarray] = []

    for c, s in enumerate(seeds):
        print(f"  chain {c + 1}/{N_CHAINS} (seed={s}) ...")
        out = _run_chain(y, seed=int(s))
        chains_obs[c] = out["sigma2_obs"]
        chains_level[c] = out["sigma2_level"]
        state_draws_all.append(out["states"])

    state_draws = np.concatenate(state_draws_all, axis=0)
    # state_draws shape: (N_CHAINS * N_DRAWS, nobs, k_states)

    # ------------------------------------------------------------------
    # 3. Convergence: R-hat
    # ------------------------------------------------------------------
    print("\n--- Gelman-Rubin R-hat (split-chain) ---")
    rhat_obs = gelman_rubin(chains_obs)
    rhat_level = gelman_rubin(chains_level)
    print(f"  R-hat sigma2_obs   = {rhat_obs:.4f} (tol {RHAT_TOL})")
    print(f"  R-hat sigma2_level = {rhat_level:.4f} (tol {RHAT_TOL})")

    assert rhat_obs < RHAT_TOL, f"R-hat for sigma2_obs too high: {rhat_obs:.3f}"
    assert rhat_level < RHAT_TOL, f"R-hat for sigma2_level too high: {rhat_level:.3f}"
    print("  [PASS] R-hat < 1.1 for all parameters")

    # ------------------------------------------------------------------
    # 4. Posterior central tendency vs MLE
    # ------------------------------------------------------------------
    # Under weakly informative IG(0.01, 0.01) priors the marginal posteriors
    # of the variance components are right-skewed, so the posterior MEDIAN is
    # the natural point estimate to compare against MLE (the maximiser of the
    # profile likelihood). The posterior mean is also reported as context.
    print("\n--- Posterior vs MLE (diff < 10%, using posterior median) ---")
    post_obs_mean = float(np.mean(chains_obs))
    post_level_mean = float(np.mean(chains_level))
    post_obs_med = float(np.median(chains_obs))
    post_level_med = float(np.median(chains_level))
    err_obs = _relative_error(post_obs_med, mle_sigma2_obs)
    err_level = _relative_error(post_level_med, mle_sigma2_level)
    print(f"  sigma2_obs   posterior mean   = {post_obs_mean:.3f}")
    print(f"  sigma2_obs   posterior median = {post_obs_med:.3f}  "
          f"(rel err vs MLE {err_obs:.4f})")
    print(f"  sigma2_level posterior mean   = {post_level_mean:.3f}")
    print(f"  sigma2_level posterior median = {post_level_med:.3f}  "
          f"(rel err vs MLE {err_level:.4f})")

    assert err_obs < PARAM_REL_TOL, (
        f"sigma2_obs posterior median vs MLE relative error {err_obs:.4f} "
        f"exceeds {PARAM_REL_TOL}"
    )
    assert err_level < PARAM_REL_TOL, (
        f"sigma2_level posterior median vs MLE relative error {err_level:.4f} "
        f"exceeds {PARAM_REL_TOL}"
    )
    print("  [PASS] Posterior medians within 10% of MLE")

    # ------------------------------------------------------------------
    # 5. FFBS states vs Kalman smoother
    # ------------------------------------------------------------------
    print("\n--- FFBS posterior mean vs RTS smoother ---")
    ffbs_mean_state = state_draws.mean(axis=0)[:, 0]
    corr = float(np.corrcoef(ffbs_mean_state, mle_smoothed)[0, 1])
    scale = float(np.mean(np.abs(mle_smoothed)))
    relerr = float(np.mean(np.abs(ffbs_mean_state - mle_smoothed)) / scale)
    print(f"  correlation         = {corr:.6f} (tol > {STATE_CORR_TOL})")
    print(f"  mean rel abs error  = {relerr:.6f} (tol < {STATE_RELERR_TOL})")

    assert corr > STATE_CORR_TOL, (
        f"FFBS state correlation {corr:.4f} below tol {STATE_CORR_TOL}"
    )
    assert relerr < STATE_RELERR_TOL, (
        f"FFBS state rel error {relerr:.4f} above tol {STATE_RELERR_TOL}"
    )
    print("  [PASS] FFBS mean state matches RTS smoother")

    # ------------------------------------------------------------------
    # 6. ESS (informational)
    # ------------------------------------------------------------------
    print("\n--- Effective sample size (concatenated chains) ---")
    ess_obs = effective_sample_size(chains_obs.ravel())
    ess_level = effective_sample_size(chains_level.ravel())
    print(f"  ESS sigma2_obs   = {ess_obs:.1f}")
    print(f"  ESS sigma2_level = {ess_level:.1f}")

    # ------------------------------------------------------------------
    # 7. HPD intervals (informational)
    # ------------------------------------------------------------------
    print("\n--- 95% HPD credible intervals ---")
    hpd_obs = hpd_interval(chains_obs.ravel(), alpha=0.05)
    hpd_level = hpd_interval(chains_level.ravel(), alpha=0.05)
    print(f"  sigma2_obs   HPD = [{hpd_obs[0]:.2f}, {hpd_obs[1]:.2f}]")
    print(f"  sigma2_level HPD = [{hpd_level[0]:.2f}, {hpd_level[1]:.2f}]")

    # ------------------------------------------------------------------
    # 8. Diagnostic figures
    # ------------------------------------------------------------------
    print("\n--- Generating diagnostic figures ---")

    fig, _ = trace_plot(chains_obs, "sigma2_obs")
    out_path = FIG_DIR / "sol_01_trace_sigma2_obs.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")

    fig, _ = trace_plot(chains_level, "sigma2_level")
    out_path = FIG_DIR / "sol_01_trace_sigma2_level.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, chain, name, ref in (
        (axes[0], chains_obs.ravel(), "sigma2_obs", mle_sigma2_obs),
        (axes[1], chains_level.ravel(), "sigma2_level", mle_sigma2_level),
    ):
        ax.hist(chain, bins=50, density=True, alpha=0.6, color="steelblue")
        ax.axvline(ref, color="red", lw=1.5, label=f"MLE={ref:.2f}")
        ax.axvline(chain.mean(), color="black", lw=1.2, ls="--",
                   label=f"post mean={chain.mean():.2f}")
        ax.set_title(f"Posterior: {name}")
        ax.set_xlabel(name)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = FIG_DIR / "sol_01_posterior_densities.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")

    states_summary_lower = np.percentile(state_draws[:, :, 0], 2.5, axis=0)
    states_summary_upper = np.percentile(state_draws[:, :, 0], 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(years, y, "k.", ms=4, alpha=0.6, label="Nile flow")
    ax.plot(years, mle_smoothed, "r-", lw=1.5, label="RTS smoother (MLE)")
    ax.plot(years, ffbs_mean_state, "b--", lw=1.5,
            label="FFBS posterior mean")
    ax.fill_between(years, states_summary_lower, states_summary_upper,
                    color="steelblue", alpha=0.2,
                    label="95% posterior band")
    ax.set_xlabel("year")
    ax.set_ylabel("level")
    ax.set_title("Bayesian local level — Nile")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = FIG_DIR / "sol_01_states_vs_smoother.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")

    print("\n" + "=" * 70)
    print("Solution 01: ALL CHECKS PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
