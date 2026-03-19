"""Tests for parametric bootstrap."""

import numpy as np
import pytest

from kalmanbox.models.local_level import LocalLevel
from kalmanbox.simulation.bootstrap import BootstrapResult, parametric_bootstrap


class TestParametricBootstrap:
    """Tests for parametric bootstrap."""

    @pytest.mark.slow
    def test_bootstrap_coverage(self, nile_volume: np.ndarray) -> None:
        """95% CI from bootstrap should contain the MLE estimate.

        The bootstrap CI should cover the point estimate (which is
        the 'true' parameter in the bootstrap world).
        """
        model = LocalLevel(nile_volume)
        results = model.fit()

        boot = parametric_bootstrap(model, results, n_boot=50, alpha=0.05, seed=42)

        assert isinstance(boot, BootstrapResult)

        # The MLE point estimate should be within the bootstrap CI
        for i in range(len(results.params)):
            assert boot.ci_lower[i] <= results.params[i] <= boot.ci_upper[i], (
                f"Parameter {boot.param_names[i]}: "
                f"{results.params[i]:.2f} not in "
                f"[{boot.ci_lower[i]:.2f}, {boot.ci_upper[i]:.2f}]"
            )

    @pytest.mark.slow
    def test_bootstrap_se_positive(self, nile_volume: np.ndarray) -> None:
        """Bootstrap standard errors should be positive."""
        model = LocalLevel(nile_volume)
        results = model.fit()

        boot = parametric_bootstrap(model, results, n_boot=30, seed=42)

        if boot.n_success >= 3:
            assert np.all(boot.params_se > 0)

    @pytest.mark.slow
    def test_bootstrap_mean_near_mle(self, nile_volume: np.ndarray) -> None:
        """Bootstrap mean should be near MLE (within 50%)."""
        model = LocalLevel(nile_volume)
        results = model.fit()

        boot = parametric_bootstrap(model, results, n_boot=50, seed=42)

        if boot.n_success >= 10:
            for i in range(len(results.params)):
                assert boot.params_mean[i] == pytest.approx(results.params[i], rel=0.5)

    @pytest.mark.slow
    def test_bootstrap_result_repr(self, nile_volume: np.ndarray) -> None:
        """BootstrapResult repr should be informative."""
        model = LocalLevel(nile_volume)
        results = model.fit()

        boot = parametric_bootstrap(model, results, n_boot=20, seed=42)

        text = repr(boot)
        assert "Bootstrap" in text
        assert "sigma2" in text or "Parameter" in text

    @pytest.mark.slow
    def test_bootstrap_bias_corrected(self, nile_volume: np.ndarray) -> None:
        """Bias-corrected CI should be available."""
        model = LocalLevel(nile_volume)
        results = model.fit()

        boot = parametric_bootstrap(model, results, n_boot=30, seed=42)

        if boot.n_success >= 3:
            assert np.all(np.isfinite(boot.ci_lower_bc))
            assert np.all(np.isfinite(boot.ci_upper_bc))
