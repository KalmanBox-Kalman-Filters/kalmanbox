"""Bayesian estimation for state-space models via Gibbs sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from kalmanbox._logging import get_logger
from kalmanbox.filters.kalman import KalmanFilter
from kalmanbox.utils.matrix_ops import ensure_symmetric

if TYPE_CHECKING:
    from kalmanbox.core.model import StateSpaceModel

logger = get_logger("bayesian")


@dataclass
class InverseGamma:
    """Inverse-Gamma prior distribution.

    Parametrization: p(sigma2) propto sigma2^{-(a+1)} exp(-b/sigma2)
    Mean = b / (a - 1) for a > 1.
    Variance = b^2 / ((a-1)^2 * (a-2)) for a > 2.

    Parameters
    ----------
    a : float
        Shape parameter (alpha). Must be > 0.
    b : float
        Scale parameter (beta). Must be > 0.
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a <= 0:
            msg = f"Shape parameter a must be > 0, got {self.a}"
            raise ValueError(msg)
        if self.b <= 0:
            msg = f"Scale parameter b must be > 0, got {self.b}"
            raise ValueError(msg)

    @property
    def mean(self) -> float:
        """Expected value E[sigma2] = b / (a - 1)."""
        if self.a <= 1:
            return np.inf
        return self.b / (self.a - 1)

    @property
    def variance(self) -> float:
        """Variance Var[sigma2] = b^2 / ((a-1)^2 * (a-2))."""
        if self.a <= 2:
            return np.inf
        return self.b**2 / ((self.a - 1) ** 2 * (self.a - 2))

    def sample(self, size: int = 1, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        """Draw samples from InverseGamma(a, b).

        Uses the fact that if X ~ Gamma(a, 1/b), then 1/X ~ InverseGamma(a, b).
        """
        if rng is None:
            rng = np.random.default_rng()
        gamma_samples = rng.gamma(shape=self.a, scale=1.0 / self.b, size=size)
        return 1.0 / gamma_samples

    def posterior(self, n: int, sum_sq: float) -> InverseGamma:
        """Compute posterior given conjugate update.

        posterior(sigma2 | data) = InverseGamma(a + n/2, b + sum_sq/2)

        Parameters
        ----------
        n : int
            Number of observations.
        sum_sq : float
            Sum of squared residuals.

        Returns
        -------
        InverseGamma
            Posterior distribution.
        """
        return InverseGamma(
            a=self.a + n / 2.0,
            b=self.b + sum_sq / 2.0,
        )


@dataclass
class PosteriorResult:
    """Container for Bayesian posterior results.

    Attributes
    ----------
    param_draws : dict[str, NDArray[np.float64]]
        MCMC draws for each parameter, shape (n_draws,) each.
    state_draws : NDArray[np.float64]
        MCMC draws for states, shape (n_draws, nobs, k_states).
    param_names : list[str]
        Names of estimated parameters.
    n_draws : int
        Number of MCMC draws (after burnin).
    burnin : int
        Number of burnin draws discarded.
    """

    param_draws: dict[str, NDArray[np.float64]]
    state_draws: NDArray[np.float64]
    param_names: list[str]
    n_draws: int
    burnin: int

    def summary(self) -> str:
        """Generate summary table of posterior distributions.

        Returns
        -------
        str
            Formatted summary with mean, median, std, HPD 95%.
        """
        lines = [
            "Bayesian Estimation Summary",
            "=" * 70,
            f"Draws: {self.n_draws} (burnin: {self.burnin})",
            "",
            f"{'Parameter':<20} {'Mean':>10} {'Median':>10} {'Std':>10} "
            f"{'HPD 2.5%':>10} {'HPD 97.5%':>10} {'ESS':>8}",
            "-" * 70,
        ]

        for name in self.param_names:
            draws = self.param_draws[name]
            mean = np.mean(draws)
            median = np.median(draws)
            std = np.std(draws, ddof=1)
            hpd_lo, hpd_hi = _hpd_interval(draws, alpha=0.05)
            ess = effective_sample_size(draws)

            lines.append(
                f"{name:<20} {mean:>10.4f} {median:>10.4f} {std:>10.4f} "
                f"{hpd_lo:>10.4f} {hpd_hi:>10.4f} {ess:>8.1f}"
            )

        return "\n".join(lines)

    def trace_plot_data(self) -> dict[str, NDArray[np.float64]]:
        """Return data for trace plots.

        Returns
        -------
        dict[str, NDArray]
            Parameter name -> MCMC chain values.
        """
        return {name: self.param_draws[name].copy() for name in self.param_names}

    def posterior_density_data(
        self, n_points: int = 200
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return data for posterior density plots (KDE).

        Parameters
        ----------
        n_points : int
            Number of points for KDE evaluation.

        Returns
        -------
        dict[str, tuple[NDArray, NDArray]]
            Parameter name -> (x_grid, density).
        """
        result: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for name in self.param_names:
            draws = self.param_draws[name]
            try:
                kde = stats.gaussian_kde(draws)
                x_min = np.min(draws)
                x_max = np.max(draws)
                margin = (x_max - x_min) * 0.1
                x_grid = np.linspace(x_min - margin, x_max + margin, n_points)
                density = kde(x_grid)
                result[name] = (x_grid, density)
            except Exception:  # noqa: BLE001
                result[name] = (np.array([]), np.array([]))
        return result

    def states_summary(self) -> dict[str, NDArray[np.float64]]:
        """Compute posterior mean and credible intervals for states.

        Returns
        -------
        dict with keys:
            'mean': shape (nobs, k_states)
            'lower': shape (nobs, k_states) - 2.5th percentile
            'upper': shape (nobs, k_states) - 97.5th percentile
        """
        mean = np.mean(self.state_draws, axis=0)
        lower = np.percentile(self.state_draws, 2.5, axis=0)
        upper = np.percentile(self.state_draws, 97.5, axis=0)
        return {"mean": mean, "lower": lower, "upper": upper}


class BayesianSSM:
    """Bayesian estimation for state-space models via Gibbs sampling.

    Uses the Forward Filtering Backward Sampling (FFBS) algorithm
    of Carter & Kohn (1994) combined with conjugate priors for
    variance parameters.

    Parameters
    ----------
    model : StateSpaceModel
        The state-space model to estimate.

    Examples
    --------
    >>> model = LocalLevel(nile)
    >>> bayes = BayesianSSM(model)
    >>> posterior = bayes.fit(
    ...     endog=nile,
    ...     n_draws=5000,
    ...     burnin=1000,
    ...     priors={
    ...         'sigma2_obs': InverseGamma(a=3, b=1),
    ...         'sigma2_level': InverseGamma(a=3, b=1),
    ...     }
    ... )
    >>> print(posterior.summary())
    """

    def __init__(self, model: StateSpaceModel) -> None:
        self.model = model

    def fit(
        self,
        endog: NDArray[np.float64],
        n_draws: int = 5000,
        burnin: int = 1000,
        priors: dict[str, InverseGamma] | None = None,
        seed: int | None = None,
    ) -> PosteriorResult:
        """Run Gibbs sampler for Bayesian estimation.

        Parameters
        ----------
        endog : NDArray[np.float64]
            Observed data, shape (nobs,) or (nobs, k_endog).
        n_draws : int
            Number of MCMC draws (after burnin). Default 5000.
        burnin : int
            Number of initial draws to discard. Default 1000.
        priors : dict[str, InverseGamma] or None
            Prior distributions for variance parameters.
            Keys should match parameter names from model.param_names.
            If None, uses weakly informative InverseGamma(a=0.01, b=0.01).
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        PosteriorResult
            Posterior draws and summary statistics.
        """
        rng = np.random.default_rng(seed)

        if endog.ndim == 1:
            endog = endog.reshape(-1, 1)

        total_draws = n_draws + burnin

        # Initialize with MLE or start_params
        try:
            mle_results = self.model.fit()
            params = mle_results.params.copy()
        except Exception:  # noqa: BLE001
            params = self.model.start_params.copy()

        param_names = list(self.model.param_names)

        # Set up priors
        if priors is None:
            priors = {}
        for name in param_names:
            if name not in priors:
                priors[name] = InverseGamma(a=0.01, b=0.01)

        # Storage for draws
        param_storage: dict[str, list[float]] = {name: [] for name in param_names}
        state_storage: list[NDArray[np.float64]] = []

        for draw in range(total_draws):
            # --- Build current SSM ---
            ssm = self.model._build_ssm(params)

            # --- Step 1: Sample states via FFBS ---
            sampled_states = self._ffbs(endog, ssm, rng)

            # --- Step 2: Sample parameters given states ---
            params = self._sample_params(
                endog, sampled_states, ssm, params, priors, param_names, rng
            )

            # Store draws after burnin
            if draw >= burnin:
                for i, name in enumerate(param_names):
                    param_storage[name].append(float(params[i]))
                state_storage.append(sampled_states.copy())

            if (draw + 1) % 1000 == 0:
                logger.debug("Gibbs iteration %d/%d", draw + 1, total_draws)

        # Convert to arrays
        param_draws = {name: np.array(param_storage[name]) for name in param_names}
        state_draws = np.array(state_storage)

        return PosteriorResult(
            param_draws=param_draws,
            state_draws=state_draws,
            param_names=param_names,
            n_draws=n_draws,
            burnin=burnin,
        )

    def _ffbs(
        self,
        endog: NDArray[np.float64],
        ssm: object,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Forward Filtering Backward Sampling (Carter & Kohn 1994).

        1. Forward: run Kalman filter to get a_{t|t}, P_{t|t}
        2. Backward: sample states from t=T to t=1

        Parameters
        ----------
        endog : NDArray, shape (nobs, k_endog)
        ssm : StateSpaceRepresentation
        rng : numpy random generator

        Returns
        -------
        NDArray, shape (nobs, k_states)
            Sampled states.
        """
        kf = KalmanFilter()
        filter_output = kf.filter(endog, ssm)

        nobs = endog.shape[0]
        k_states = ssm.k_states  # type: ignore[union-attr]
        T_mat = ssm.T  # type: ignore[union-attr]
        c_vec = ssm.c  # type: ignore[union-attr]

        sampled = np.zeros((nobs, k_states))

        # --- Last time point: sample from N(a_{T|T}, P_{T|T}) ---
        a_T = filter_output.filtered_state[-1]
        P_T = filter_output.filtered_cov[-1]
        P_T = ensure_symmetric(P_T)

        try:
            L = np.linalg.cholesky(P_T)
            sampled[-1] = a_T + L @ rng.standard_normal(k_states)
        except np.linalg.LinAlgError:
            std = np.sqrt(np.maximum(np.diag(P_T), 1e-10))
            sampled[-1] = a_T + std * rng.standard_normal(k_states)

        # --- Backward sampling ---
        for t in range(nobs - 2, -1, -1):
            a_tt = filter_output.filtered_state[t]
            P_tt = filter_output.filtered_cov[t]

            # Predicted state and covariance for t+1
            a_pred = T_mat @ a_tt + c_vec
            P_pred = T_mat @ P_tt @ T_mat.T + ssm.R @ ssm.Q @ ssm.R.T  # type: ignore[union-attr]
            P_pred = ensure_symmetric(P_pred)

            # Smoother gain: J_t = P_{t|t} @ T' @ inv(P_{t+1|t})
            try:
                P_pred_inv = np.linalg.inv(P_pred)
            except np.linalg.LinAlgError:
                P_pred_inv = np.linalg.pinv(P_pred)

            J_t = P_tt @ T_mat.T @ P_pred_inv

            # Conditional mean and covariance
            m_t = a_tt + J_t @ (sampled[t + 1] - a_pred)
            V_t = P_tt - J_t @ P_pred @ J_t.T
            V_t = ensure_symmetric(V_t)

            # Ensure positive definite
            eigvals = np.linalg.eigvalsh(V_t)
            if np.any(eigvals < 0):
                V_t = V_t + np.eye(k_states) * (abs(float(np.min(eigvals))) + 1e-10)
                V_t = ensure_symmetric(V_t)

            try:
                L = np.linalg.cholesky(V_t)
                sampled[t] = m_t + L @ rng.standard_normal(k_states)
            except np.linalg.LinAlgError:
                std = np.sqrt(np.maximum(np.diag(V_t), 1e-10))
                sampled[t] = m_t + std * rng.standard_normal(k_states)

        return sampled

    def _sample_params(
        self,
        endog: NDArray[np.float64],
        states: NDArray[np.float64],
        ssm: object,
        current_params: NDArray[np.float64],
        priors: dict[str, InverseGamma],
        param_names: list[str],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample parameters given states using conjugate priors.

        For variance parameters with InverseGamma priors:
            posterior = InverseGamma(a + n/2, b + SSR/2)

        Parameters
        ----------
        endog : NDArray, shape (nobs, k_endog)
        states : NDArray, shape (nobs, k_states)
        ssm : StateSpaceRepresentation
        current_params : NDArray
        priors : dict of InverseGamma priors
        param_names : list of parameter names
        rng : numpy random generator

        Returns
        -------
        NDArray
            New parameter values.
        """
        nobs = endog.shape[0]
        new_params = current_params.copy()
        Z = ssm.Z  # type: ignore[union-attr]
        T_mat = ssm.T  # type: ignore[union-attr]
        d_vec = ssm.d  # type: ignore[union-attr]
        c_vec = ssm.c  # type: ignore[union-attr]

        for i, name in enumerate(param_names):
            prior = priors.get(name, InverseGamma(a=0.01, b=0.01))

            if "obs" in name or "irregular" in name:
                # Observation variance: SSR = sum (y_t - Z @ alpha_t - d)^2
                obs_resid = np.zeros(nobs)
                for t in range(nobs):
                    if not np.any(np.isnan(endog[t])):
                        e_t = endog[t] - Z @ states[t] - d_vec
                        obs_resid[t] = float(e_t @ e_t)

                n_valid = int(np.sum(~np.any(np.isnan(endog), axis=1)))
                ssr = float(np.sum(obs_resid))

                posterior_dist = prior.posterior(n_valid, ssr)
                new_params[i] = float(posterior_dist.sample(size=1, rng=rng)[0])

            elif (
                "level" in name
                or "trend" in name
                or "slope" in name
                or "state" in name
                or "seasonal" in name
            ):
                # State variance: SSE = sum (alpha_{t+1} - T @ alpha_t - c)^2
                state_resid = np.zeros(nobs - 1)
                for t in range(nobs - 1):
                    e_t = states[t + 1] - T_mat @ states[t] - c_vec
                    state_resid[t] = float(e_t @ e_t)

                sse = float(np.sum(state_resid))
                posterior_dist = prior.posterior(nobs - 1, sse)
                new_params[i] = float(posterior_dist.sample(size=1, rng=rng)[0])

            else:
                logger.debug(
                    "Parameter '%s' not recognized as variance, keeping current value",
                    name,
                )

        return new_params


def effective_sample_size(chain: NDArray[np.float64]) -> float:
    """Compute Effective Sample Size (ESS) for an MCMC chain.

    ESS = n / (1 + 2 * sum_{k=1}^{K} rho_k)

    where rho_k is the autocorrelation at lag k. The sum is truncated
    when the autocorrelation first becomes negative.

    Parameters
    ----------
    chain : NDArray[np.float64]
        MCMC chain values, shape (n_draws,).

    Returns
    -------
    float
        Effective sample size.
    """
    n = len(chain)
    if n < 4:
        return float(n)

    chain_centered = chain - np.mean(chain)
    var = np.var(chain_centered, ddof=0)

    if var < 1e-15:
        return float(n)

    # Compute autocorrelations
    max_lag = min(n // 2, 1000)
    acf = np.zeros(max_lag)

    for k in range(max_lag):
        acf[k] = np.mean(chain_centered[: n - k] * chain_centered[k:]) / var

    # Sum autocorrelations until first negative pair
    # (Geyer's initial positive sequence estimator)
    tau = 1.0
    for k in range(1, max_lag - 1, 2):
        pair_sum = acf[k] + acf[k + 1]
        if pair_sum < 0:
            break
        tau += 2.0 * pair_sum

    ess = n / tau
    return max(1.0, ess)


def _hpd_interval(samples: NDArray[np.float64], alpha: float = 0.05) -> tuple[float, float]:
    """Compute Highest Posterior Density (HPD) interval.

    Parameters
    ----------
    samples : NDArray[np.float64]
        Posterior samples.
    alpha : float
        Significance level (1 - credibility level). Default 0.05.

    Returns
    -------
    tuple[float, float]
        (lower, upper) bounds of the HPD interval.
    """
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    credibility = 1.0 - alpha

    interval_size = int(np.ceil(credibility * n))
    if interval_size >= n:
        return float(sorted_samples[0]), float(sorted_samples[-1])

    # Find shortest interval
    widths = sorted_samples[interval_size:] - sorted_samples[: n - interval_size]
    best_idx = int(np.argmin(widths))

    return float(sorted_samples[best_idx]), float(sorted_samples[best_idx + interval_size])
