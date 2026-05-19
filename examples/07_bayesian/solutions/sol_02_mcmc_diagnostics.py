"""Solution: MCMC diagnostics for the Bayesian local level model.

Runs four independent Gibbs / FFBS chains on the Nile data and validates
standard convergence diagnostics:

  * Gelman-Rubin R-hat < 1.1 for every parameter.
  * Effective sample size (ESS) > 500 for every parameter.
  * Prior sensitivity: rerun with a second, genuinely different prior and
    verify that the posterior medians are qualitatively consistent with the
    baseline run (relative change < 25%), i.e. that the data still dominate.

Produces diagnostic figures (trace, autocorrelation, running mean, prior
comparison) under ``solutions/figures/``.

References
----------
Gelman, A. and Rubin, D.B. (1992). "Inference from Iterative Simulation
Using Multiple Sequences." Statistical Science 7(4): 457-472.
Geyer, C.J. (1992). "Practical Markov Chain Monte Carlo." Statistical
Science 7(4): 473-483.
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
    autocorrelation_plot,
    effective_sample_size,
    gelman_rubin,
    hpd_interval,
    trace_plot,
)

SEED = 20260416
N_DRAWS = 6500
BURNIN = 1500
N_CHAINS = 4

RHAT_TOL = 1.1
ESS_TOL = 500.0
PRIOR_SENS_TOL = 0.25  # relative posterior-median change across priors

PARAM_NAMES = ["sigma2_obs", "sigma2_level"]


def _run_chain(
    y: np.ndarray,
    priors: dict[str, InverseGamma],
    seed: int,
    n_draws: int = N_DRAWS,
    burnin: int = BURNIN,
) -> dict[str, np.ndarray]:
    model = LocalLevel(y)
    bayes = BayesianSSM(model)
    posterior = bayes.fit(
        endog=y,
        n_draws=n_draws,
        burnin=burnin,
        priors=priors,
        seed=seed,
    )
    return {name: posterior.param_draws[name] for name in PARAM_NAMES}


def _chain_matrix(
    y: np.ndarray,
    priors: dict[str, InverseGamma],
    seeds: list[int],
    n_draws: int,
    burnin: int,
    label: str,
) -> dict[str, np.ndarray]:
    n_chains = len(seeds)
    out = {name: np.zeros((n_chains, n_draws)) for name in PARAM_NAMES}
    for c, s in enumerate(seeds):
        print(f"  [{label}] chain {c + 1}/{n_chains} (seed={s}) ...")
        draws = _run_chain(y, priors, seed=int(s), n_draws=n_draws, burnin=burnin)
        for name in PARAM_NAMES:
            out[name][c] = draws[name]
    return out


def main() -> int:
    print("=" * 70)
    print("Solution 02: MCMC diagnostics (4 chains, Nile local level)")
    print("=" * 70)

    nile = pd.read_csv(DATA_DIR / "nile.csv")
    y = nile["flow"].to_numpy(dtype=np.float64)
    print(f"Loaded {len(y)} Nile observations.")

    rng = np.random.default_rng(SEED)
    base_seeds = rng.integers(low=1, high=2**31 - 1, size=N_CHAINS).tolist()
    base_seeds = [int(s) for s in base_seeds]

    # ------------------------------------------------------------------
    # 1. Baseline: weakly informative IG(0.01, 0.01)
    # ------------------------------------------------------------------
    print("\n--- Baseline priors: IG(0.01, 0.01) on both variances ---")
    priors_baseline = {
        "sigma2_obs": InverseGamma(a=0.01, b=0.01),
        "sigma2_level": InverseGamma(a=0.01, b=0.01),
    }
    chains_base = _chain_matrix(
        y, priors_baseline, base_seeds, N_DRAWS, BURNIN, "baseline"
    )

    # ------------------------------------------------------------------
    # 2. Convergence (R-hat)
    # ------------------------------------------------------------------
    print("\n--- Gelman-Rubin R-hat (split-chain) ---")
    rhat_values: dict[str, float] = {}
    for name in PARAM_NAMES:
        rhat_values[name] = gelman_rubin(chains_base[name])
        print(f"  R-hat {name:<14} = {rhat_values[name]:.4f} (tol {RHAT_TOL})")

    for name, r in rhat_values.items():
        assert r < RHAT_TOL, f"R-hat for {name} too high: {r:.3f}"
    print("  [PASS] R-hat < 1.1 for all parameters")

    # ------------------------------------------------------------------
    # 3. Effective sample size
    # ------------------------------------------------------------------
    print("\n--- Effective sample size (pooled across 4 chains) ---")
    ess_values: dict[str, float] = {}
    for name in PARAM_NAMES:
        pooled = chains_base[name].ravel()
        ess_values[name] = effective_sample_size(pooled)
        total = chains_base[name].size
        print(
            f"  ESS {name:<14} = {ess_values[name]:>8.1f} "
            f"(of {total} pooled draws)"
        )

    for name, ess in ess_values.items():
        assert ess > ESS_TOL, (
            f"ESS for {name} = {ess:.1f} below tolerance {ESS_TOL}"
        )
    print(f"  [PASS] ESS > {ESS_TOL:.0f} for all parameters")

    # ------------------------------------------------------------------
    # 4. Summaries (mean, median, HPD)
    # ------------------------------------------------------------------
    print("\n--- Posterior summaries (baseline) ---")
    baseline_median: dict[str, float] = {}
    for name in PARAM_NAMES:
        pooled = chains_base[name].ravel()
        mean = float(np.mean(pooled))
        med = float(np.median(pooled))
        lo, hi = hpd_interval(pooled, alpha=0.05)
        baseline_median[name] = med
        print(
            f"  {name:<14} mean={mean:>11.2f}  median={med:>11.2f}  "
            f"HPD95=[{lo:>10.2f}, {hi:>10.2f}]"
        )

    # ------------------------------------------------------------------
    # 5. Prior sensitivity
    # ------------------------------------------------------------------
    # Alternative priors that are *different* from the baseline:
    # finite-mean IG(a=3, b=...) centred on a priori reasonable scales.
    # The data likelihood should dominate so posterior medians stay close.
    var_data = float(np.nanvar(y))
    priors_alt = {
        "sigma2_obs": InverseGamma(a=3.0, b=2.0 * var_data),
        "sigma2_level": InverseGamma(a=3.0, b=2.0 * (var_data / 10.0)),
    }
    print("\n--- Prior sensitivity: alt priors ---")
    print(f"  sigma2_obs   prior: IG(a=3, b={2.0 * var_data:.1f}) "
          f"(prior mean {var_data:.1f})")
    print(f"  sigma2_level prior: IG(a=3, b={2.0 * (var_data / 10.0):.1f}) "
          f"(prior mean {var_data / 10.0:.1f})")
    alt_seeds = rng.integers(low=1, high=2**31 - 1, size=N_CHAINS).tolist()
    alt_seeds = [int(s) for s in alt_seeds]
    n_draws_alt = 2000
    burnin_alt = 500
    chains_alt = _chain_matrix(
        y, priors_alt, alt_seeds, n_draws_alt, burnin_alt, "alt"
    )

    print("\n--- Prior sensitivity: diagnostics for alt run ---")
    for name in PARAM_NAMES:
        r = gelman_rubin(chains_alt[name])
        pooled = chains_alt[name].ravel()
        ess = effective_sample_size(pooled)
        print(f"  {name:<14} R-hat={r:.4f}, pooled ESS={ess:.1f}")
        assert r < RHAT_TOL, f"Alt run R-hat for {name} = {r:.3f} >= {RHAT_TOL}"

    print("\n--- Prior sensitivity: posterior median comparison ---")
    all_pass = True
    for name in PARAM_NAMES:
        m_base = baseline_median[name]
        m_alt = float(np.median(chains_alt[name].ravel()))
        rel = abs(m_alt - m_base) / abs(m_base)
        flag = "OK" if rel < PRIOR_SENS_TOL else "FAIL"
        print(
            f"  {name:<14} baseline median={m_base:.2f}, "
            f"alt median={m_alt:.2f}, rel diff={rel:.4f} [{flag}]"
        )
        if rel >= PRIOR_SENS_TOL:
            all_pass = False
    assert all_pass, (
        "Posterior medians under alt priors differ too much from baseline "
        "(data does not dominate prior)."
    )
    print(f"  [PASS] Posterior medians consistent across priors "
          f"(rel diff < {PRIOR_SENS_TOL})")

    # ------------------------------------------------------------------
    # 6. Diagnostic figures
    # ------------------------------------------------------------------
    print("\n--- Generating diagnostic figures ---")
    for name in PARAM_NAMES:
        fig, _ = trace_plot(chains_base[name], name)
        out_path = FIG_DIR / f"sol_02_trace_{name}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"  saved {out_path}")

        fig, _ = autocorrelation_plot(chains_base[name][0], max_lag=80)
        out_path = FIG_DIR / f"sol_02_acf_{name}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"  saved {out_path}")

    # Running mean comparison (baseline)
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(12, 4))
    for ax, name in zip(axes, PARAM_NAMES):
        for c in range(N_CHAINS):
            run_mean = np.cumsum(chains_base[name][c]) / np.arange(
                1, N_DRAWS + 1
            )
            ax.plot(run_mean, lw=0.8, alpha=0.8, label=f"chain {c + 1}")
        ax.set_xlabel("iteration")
        ax.set_ylabel(f"running mean {name}")
        ax.set_title(f"Running mean: {name}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = FIG_DIR / "sol_02_running_means.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")

    # Prior-vs-posterior density comparison
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(12, 4))
    for ax, name in zip(axes, PARAM_NAMES):
        pooled_base = chains_base[name].ravel()
        pooled_alt = chains_alt[name].ravel()
        ax.hist(
            pooled_base, bins=60, density=True, alpha=0.5,
            color="steelblue", label="baseline IG(0.01, 0.01)",
        )
        ax.hist(
            pooled_alt, bins=60, density=True, alpha=0.5,
            color="darkorange", label="alt IG(3, .)",
        )
        ax.set_title(f"Posterior under two priors: {name}")
        ax.set_xlabel(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = FIG_DIR / "sol_02_prior_sensitivity.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")

    print("\n" + "=" * 70)
    print("Solution 02: ALL CHECKS PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
