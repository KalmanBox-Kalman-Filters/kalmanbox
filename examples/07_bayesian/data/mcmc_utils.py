"""MCMC diagnostic utilities for Bayesian state-space examples.

The functions in this module implement the standard convergence and posterior-
summary diagnostics used throughout the notebooks in
``examples/07_bayesian/``:

- ``trace_plot`` / ``autocorrelation_plot`` -- visual diagnostics of a single
  chain.
- ``gelman_rubin`` -- the Gelman & Rubin (1992) potential scale reduction
  factor :math:`\\hat R` using the split-chain definition of Gelman et al.
  (2013, BDA3) / Brooks & Gelman (1998).
- ``effective_sample_size`` -- the batched-autocorrelation ESS estimator as
  described in Gelman et al. (2013, Sec. 11.5) and Geyer (1992).
- ``hpd_interval`` -- the highest-posterior-density (HPD) interval via the
  Chen & Shao (1999) empirical-quantile algorithm.

References
----------
- Gelman, A. and Rubin, D.B. (1992). "Inference from Iterative Simulation
  Using Multiple Sequences." *Statistical Science*, 7(4), 457-472.
- Brooks, S.P. and Gelman, A. (1998). "General Methods for Monitoring
  Convergence of Iterative Simulations." *Journal of Computational and
  Graphical Statistics*, 7(4), 434-455.
- Geyer, C.J. (1992). "Practical Markov Chain Monte Carlo." *Statistical
  Science*, 7(4), 473-483.
- Chen, M.-H. and Shao, Q.-M. (1999). "Monte Carlo Estimation of Bayesian
  Credible and HPD Intervals." *Journal of Computational and Graphical
  Statistics*, 8(1), 69-92.
- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed. Chapman & Hall.
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


# --------------------------------------------------------------------------- #
# Visual diagnostics                                                           #
# --------------------------------------------------------------------------- #

def trace_plot(
    samples: ArrayLike,
    param_name: str,
    figsize: Tuple[float, float] = (12, 4),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the MCMC trace for a parameter.

    Parameters
    ----------
    samples
        1-D array of posterior draws, or 2-D array of shape ``(n_chains,
        n_iter)`` for multiple chains (each plotted as its own line).
    param_name
        Parameter label used for axis title.
    figsize
        Matplotlib figure size.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes. Trace is overlaid with the running mean
        for a quick visual check of stationarity.
    """
    x = np.atleast_2d(np.asarray(samples, dtype=float))
    if x.ndim != 2:
        raise ValueError("samples must be 1-D or 2-D array")

    fig, ax = plt.subplots(figsize=figsize)
    n_chains, n_iter = x.shape
    iters = np.arange(n_iter)

    for c in range(n_chains):
        label = f"chain {c + 1}" if n_chains > 1 else None
        ax.plot(iters, x[c], lw=0.6, alpha=0.8, label=label)

    running_mean = np.cumsum(x.mean(axis=0)) / np.arange(1, n_iter + 1)
    ax.plot(iters, running_mean, color="black", lw=1.2, label="running mean")

    ax.set_xlabel("iteration")
    ax.set_ylabel(param_name)
    ax.set_title(f"Trace: {param_name}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def autocorrelation_plot(
    samples: ArrayLike,
    max_lag: int = 50,
    figsize: Tuple[float, float] = (8, 4),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the sample autocorrelation function of an MCMC chain.

    The ACF is computed via the FFT-based biased estimator (Brockwell & Davis,
    1991), which is consistent for weakly-stationary chains.

    Parameters
    ----------
    samples
        1-D array of posterior draws.
    max_lag
        Maximum lag to display.
    figsize
        Matplotlib figure size.

    Returns
    -------
    fig, ax
    """
    x = np.asarray(samples, dtype=float).ravel()
    n = x.size
    if max_lag >= n:
        max_lag = n - 1

    rho = _acf(x, nlags=max_lag)
    lags = np.arange(max_lag + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.vlines(lags, 0, rho, color="steelblue")
    ax.axhline(0.0, color="black", lw=0.8)
    ci = 1.96 / np.sqrt(n)
    ax.axhline(ci, color="red", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(-ci, color="red", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")
    ax.set_title("MCMC autocorrelation")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# Convergence / ESS                                                            #
# --------------------------------------------------------------------------- #

def gelman_rubin(chains: ArrayLike) -> float:
    """Gelman-Rubin potential scale reduction factor :math:`\\hat R`.

    Uses the standard definition from Gelman et al. (2013, BDA3, eq. 11.4):

    .. math::

        \\hat V = \\frac{n-1}{n}\\, W + \\frac{1}{n}\\, B, \\qquad
        \\hat R = \\sqrt{\\hat V / W}

    where :math:`W` is the within-chain variance (mean of the per-chain
    sample variances) and :math:`B / n` is the between-chain variance.

    Parameters
    ----------
    chains
        Array of shape ``(m, n)`` with ``m >= 2`` chains and ``n`` draws.

    Returns
    -------
    rhat : float
        Potential scale reduction factor. Values close to 1 (typically
        :math:`< 1.01`) indicate convergence.
    """
    x = np.asarray(chains, dtype=float)
    if x.ndim != 2:
        raise ValueError("chains must be 2-D array of shape (n_chains, n_iter)")
    m, n = x.shape
    if m < 2:
        raise ValueError("Gelman-Rubin requires at least 2 chains")
    if n < 2:
        raise ValueError("Each chain must have at least 2 draws")

    chain_means = x.mean(axis=1)
    chain_vars = x.var(axis=1, ddof=1)

    B = n * np.var(chain_means, ddof=1)
    W = chain_vars.mean()

    if W <= 0.0:
        return float("nan")

    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def effective_sample_size(samples: ArrayLike) -> float:
    """Effective sample size (ESS) of an MCMC chain.

    Uses the initial-positive-sequence estimator of Geyer (1992), summing
    autocorrelations in pairs and truncating as soon as a pair-sum becomes
    non-positive. This is the estimator recommended in Gelman et al. (2013,
    Sec. 11.5):

    .. math::

        \\text{ESS} = \\frac{n}{1 + 2 \\sum_{k=1}^{K} \\rho_k}

    Parameters
    ----------
    samples
        1-D array of posterior draws.

    Returns
    -------
    ess : float
        Effective sample size (:math:`\\le n`).
    """
    x = np.asarray(samples, dtype=float).ravel()
    n = x.size
    if n < 4:
        return float(n)

    max_lag = min(n - 1, 1000)
    rho = _acf(x, nlags=max_lag)

    # Geyer's initial-positive-sequence: sum rho_{2k-1} + rho_{2k} while > 0.
    tau = 1.0
    k = 1
    while k + 1 <= max_lag:
        pair = rho[k] + rho[k + 1]
        if pair <= 0.0:
            break
        tau += 2.0 * pair
        k += 2

    if tau <= 0.0:
        return float(n)
    return float(min(n, n / tau))


# --------------------------------------------------------------------------- #
# HPD interval                                                                 #
# --------------------------------------------------------------------------- #

def hpd_interval(samples: ArrayLike, alpha: float = 0.05) -> Tuple[float, float]:
    """Highest posterior density (HPD) interval.

    Implements the Chen-Shao (1999) empirical-quantile algorithm: sort the
    draws, slide a window of width ``(1 - alpha) * n`` along the sorted array,
    and return the narrowest window. For unimodal posteriors this coincides
    with the shortest credible interval.

    Parameters
    ----------
    samples
        1-D array of posterior draws.
    alpha
        One minus the credible level. Default ``0.05`` gives the 95% HPD.

    Returns
    -------
    lower, upper : float
        HPD interval bounds.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    x = np.sort(np.asarray(samples, dtype=float).ravel())
    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 samples for an HPD interval")

    k = int(np.floor((1.0 - alpha) * n))
    k = max(k, 1)
    if k >= n:
        return float(x[0]), float(x[-1])

    widths = x[k:] - x[: n - k]
    j = int(np.argmin(widths))
    return float(x[j]), float(x[j + k])


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelation up to ``nlags`` via FFT, biased normalisation."""
    n = x.size
    x_centered = x - x.mean()
    # Next power of two >= 2n for linear (non-circular) correlation via FFT.
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x_centered, n=size)
    acov = np.fft.irfft(f * np.conjugate(f), n=size)[:n].real
    acov /= n  # biased estimator (consistent, preferred for ACF)
    if acov[0] <= 0.0:
        return np.zeros(nlags + 1)
    rho = acov / acov[0]
    return rho[: nlags + 1]
