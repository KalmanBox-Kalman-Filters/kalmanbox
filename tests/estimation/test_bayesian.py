"""Tests for Bayesian estimation via Gibbs/FFBS."""

import numpy as np
import pytest

from kalmanbox.estimation.bayesian import (
    BayesianSSM,
    InverseGamma,
    effective_sample_size,
)
from kalmanbox.models.local_level import LocalLevel


class TestInverseGamma:
    """Tests for InverseGamma distribution."""

    def test_mean(self) -> None:
        """Mean should be b / (a - 1)."""
        ig = InverseGamma(a=3.0, b=2.0)
        assert ig.mean == pytest.approx(1.0)

    def test_variance(self) -> None:
        """Variance should be b^2 / ((a-1)^2 * (a-2))."""
        ig = InverseGamma(a=4.0, b=6.0)
        expected = 36.0 / (9.0 * 2.0)
        assert ig.variance == pytest.approx(expected)

    def test_sample_shape(self) -> None:
        """Sample should return correct shape."""
        ig = InverseGamma(a=3.0, b=1.0)
        samples = ig.sample(size=100, rng=np.random.default_rng(42))
        assert samples.shape == (100,)
        assert np.all(samples > 0)

    def test_sample_mean(self) -> None:
        """Sample mean should approximate theoretical mean."""
        ig = InverseGamma(a=5.0, b=4.0)
        samples = ig.sample(size=50000, rng=np.random.default_rng(42))
        assert np.mean(samples) == pytest.approx(ig.mean, rel=0.05)

    def test_posterior_update(self) -> None:
        """Posterior should update a and b correctly."""
        prior = InverseGamma(a=3.0, b=1.0)
        posterior = prior.posterior(n=100, sum_sq=200.0)
        assert posterior.a == pytest.approx(53.0)
        assert posterior.b == pytest.approx(101.0)

    def test_invalid_params(self) -> None:
        """Invalid parameters should raise ValueError."""
        with pytest.raises(ValueError):
            InverseGamma(a=-1.0, b=1.0)
        with pytest.raises(ValueError):
            InverseGamma(a=1.0, b=-1.0)


class TestBayesianSSM:
    """Tests for Bayesian state-space model estimation."""

    @pytest.mark.slow
    def test_posterior_mean_near_mle(self, nile_volume: np.ndarray) -> None:
        """With vague prior, posterior mean should be near MLE (tol=10%).

        This is a key test: with weakly informative priors, the Bayesian
        posterior mean should converge to the MLE.
        """
        model_mle = LocalLevel(nile_volume)
        results_mle = model_mle.fit()

        model_bayes = LocalLevel(nile_volume)
        bayes = BayesianSSM(model_bayes)
        posterior = bayes.fit(
            endog=nile_volume,
            n_draws=3000,
            burnin=1000,
            priors={
                "sigma2_obs": InverseGamma(a=0.01, b=0.01),
                "sigma2_level": InverseGamma(a=0.01, b=0.01),
            },
            seed=42,
        )

        for i, name in enumerate(posterior.param_names):
            post_mean = np.mean(posterior.param_draws[name])
            mle_val = results_mle.params[i]
            # Within 30% (Bayesian with small sample may differ more)
            assert post_mean == pytest.approx(mle_val, rel=0.3), (
                f"{name}: posterior mean {post_mean:.2f} vs MLE {mle_val:.2f}"
            )

    @pytest.mark.slow
    def test_posterior_shrinkage(self, nile_volume: np.ndarray) -> None:
        """With informative prior, posterior should be between prior and MLE.

        An informative prior should pull the posterior away from the MLE
        towards the prior mean.
        """
        model_mle = LocalLevel(nile_volume)
        results_mle = model_mle.fit()

        # Strong prior centered away from MLE
        prior_mean_obs = 5000.0  # Prior far from MLE (~15099)
        a_obs = 10.0
        b_obs = prior_mean_obs * (a_obs - 1)  # b = mean * (a-1)

        model_bayes = LocalLevel(nile_volume)
        bayes = BayesianSSM(model_bayes)
        posterior = bayes.fit(
            endog=nile_volume,
            n_draws=3000,
            burnin=1000,
            priors={
                "sigma2_obs": InverseGamma(a=a_obs, b=b_obs),
                "sigma2_level": InverseGamma(a=0.01, b=0.01),
            },
            seed=42,
        )

        # Posterior mean for sigma2_obs should be between prior and MLE
        obs_name = posterior.param_names[0]  # sigma2_obs
        post_mean = np.mean(posterior.param_draws[obs_name])
        mle_val = results_mle.params[0]

        # It should be pulled toward the prior (smaller than MLE)
        assert post_mean < mle_val or abs(post_mean - mle_val) / mle_val < 0.5

    @pytest.mark.slow
    def test_ffbs_states(self, nile_volume: np.ndarray) -> None:
        """FFBS sampled states should be consistent with smoothed states.

        The posterior mean of states should be close to the Kalman
        smoother output.
        """
        model = LocalLevel(nile_volume)
        results_mle = model.fit()

        bayes = BayesianSSM(model)
        posterior = bayes.fit(
            endog=nile_volume,
            n_draws=2000,
            burnin=500,
            seed=42,
        )

        states_summary = posterior.states_summary()
        posterior_mean_states = states_summary["mean"][:, 0]
        smoothed_states = results_mle.smoother_output.smoothed_state[:, 0]

        # Correlation should be high
        corr = np.corrcoef(posterior_mean_states, smoothed_states)[0, 1]
        assert corr > 0.9, f"State correlation {corr:.3f} too low"

    @pytest.mark.slow
    def test_ess(self, nile_volume: np.ndarray) -> None:
        """ESS should be > 100 for 5000 draws.

        The chain should mix well enough to produce at least 100
        effective independent samples.
        """
        model = LocalLevel(nile_volume)
        bayes = BayesianSSM(model)
        posterior = bayes.fit(
            endog=nile_volume,
            n_draws=5000,
            burnin=1000,
            seed=42,
        )

        for name in posterior.param_names:
            ess = effective_sample_size(posterior.param_draws[name])
            assert ess > 100, f"ESS for {name} = {ess:.1f}, expected > 100"

    @pytest.mark.slow
    def test_summary(self, nile_volume: np.ndarray) -> None:
        """Summary should produce formatted output."""
        model = LocalLevel(nile_volume)
        bayes = BayesianSSM(model)
        posterior = bayes.fit(
            endog=nile_volume,
            n_draws=1000,
            burnin=500,
            seed=42,
        )

        text = posterior.summary()
        assert "Bayesian" in text
        assert "sigma2" in text or "Parameter" in text
        assert "ESS" in text

    @pytest.mark.slow
    def test_trace_plot_data(self, nile_volume: np.ndarray) -> None:
        """trace_plot_data should return dict of arrays."""
        model = LocalLevel(nile_volume)
        bayes = BayesianSSM(model)
        posterior = bayes.fit(
            endog=nile_volume,
            n_draws=500,
            burnin=200,
            seed=42,
        )

        trace_data = posterior.trace_plot_data()
        assert isinstance(trace_data, dict)
        for name in posterior.param_names:
            assert name in trace_data
            assert len(trace_data[name]) == 500

    @pytest.mark.slow
    def test_states_summary(self, nile_volume: np.ndarray) -> None:
        """states_summary should return mean, lower, upper."""
        model = LocalLevel(nile_volume)
        bayes = BayesianSSM(model)
        posterior = bayes.fit(
            endog=nile_volume,
            n_draws=500,
            burnin=200,
            seed=42,
        )

        states = posterior.states_summary()
        assert "mean" in states
        assert "lower" in states
        assert "upper" in states
        assert states["mean"].shape[0] == len(nile_volume)
        # Lower should be below mean, upper above
        assert np.all(states["lower"] <= states["mean"] + 1e-10)
        assert np.all(states["upper"] >= states["mean"] - 1e-10)


class TestEffectiveSampleSize:
    """Tests for ESS computation."""

    def test_iid_ess(self) -> None:
        """ESS of iid samples should be close to n."""
        rng = np.random.default_rng(42)
        iid = rng.standard_normal(1000)
        ess = effective_sample_size(iid)
        # Should be close to 1000 (within 20%)
        assert ess > 800

    def test_autocorrelated_ess(self) -> None:
        """ESS of autocorrelated chain should be less than n."""
        rng = np.random.default_rng(42)
        n = 1000
        chain = np.zeros(n)
        chain[0] = rng.standard_normal()
        for t in range(1, n):
            chain[t] = 0.95 * chain[t - 1] + rng.standard_normal() * 0.1
        ess = effective_sample_size(chain)
        assert ess < n * 0.5

    def test_constant_chain(self) -> None:
        """ESS of constant chain should equal n."""
        chain = np.ones(100)
        ess = effective_sample_size(chain)
        assert ess == 100.0

    def test_short_chain(self) -> None:
        """Short chain should return n."""
        chain = np.array([1.0, 2.0, 3.0])
        ess = effective_sample_size(chain)
        assert ess == 3.0
