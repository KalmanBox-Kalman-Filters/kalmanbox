"""Tests for CustomStateSpace."""

import numpy as np
import pytest

from kalmanbox.models.custom import CustomStateSpace


class SimpleCustomModel(CustomStateSpace):
    """Local Level reimplemented as CustomStateSpace for testing."""

    def __init__(self, endog: np.ndarray) -> None:
        super().__init__(endog, k_states=1, k_endog=1, k_posdef=1)

    def _build_ssm(self, params):
        ssm = self.create_ssm()
        ssm.T = np.array([[1.0]])
        ssm.Z = np.array([[1.0]])
        ssm.R = np.array([[1.0]])
        ssm.H = np.array([[params[0]]])
        ssm.Q = np.array([[params[1]]])
        return ssm

    @property
    def start_params(self):
        var = float(np.nanvar(self.endog))
        return np.array([var / 2, var / 2])

    @property
    def param_names(self):
        return ["sigma2_obs", "sigma2_level"]


class TwoStateCustomModel(CustomStateSpace):
    """Custom model with 2 states and mixed transforms."""

    def __init__(self, endog: np.ndarray) -> None:
        super().__init__(endog, k_states=2, k_endog=1, k_posdef=1)

    def _build_ssm(self, params):
        ssm = self.create_ssm()
        ssm.T = np.array([[1.0, 1.0], [0.0, params[0]]])  # phi
        ssm.Z = np.array([[1.0, 0.0]])
        ssm.R = np.array([[1.0], [0.0]])
        ssm.Q = np.array([[params[1]]])
        ssm.H = np.array([[params[2]]])
        return ssm

    @property
    def start_params(self):
        return np.array([0.9, 1.0, 1.0])

    @property
    def param_names(self):
        return ["phi", "sigma2_state", "sigma2_obs"]

    def transform_params(self, unconstrained):
        constrained = np.empty_like(unconstrained)
        constrained[0] = np.tanh(unconstrained[0])  # phi in (-1, 1)
        constrained[1] = np.exp(unconstrained[1])  # sigma2 > 0
        constrained[2] = np.exp(unconstrained[2])  # sigma2 > 0
        return constrained

    def untransform_params(self, constrained):
        unconstrained = np.empty_like(constrained)
        unconstrained[0] = np.arctanh(constrained[0])
        unconstrained[1] = np.log(constrained[1])
        unconstrained[2] = np.log(constrained[2])
        return unconstrained


class TestCustomStateSpace:
    """Tests for CustomStateSpace."""

    def test_simple_model_fit(self, nile_volume: np.ndarray) -> None:
        """Custom local level should match built-in LocalLevel."""
        from kalmanbox.models.local_level import LocalLevel

        custom = SimpleCustomModel(nile_volume)
        builtin = LocalLevel(nile_volume)

        custom_results = custom.fit()
        builtin_results = builtin.fit()

        assert custom_results.loglike == pytest.approx(builtin_results.loglike, abs=1.0)

    def test_create_ssm_dimensions(self) -> None:
        """create_ssm() should return SSM with correct dimensions."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 50)
        model = SimpleCustomModel(y)
        ssm = model.create_ssm()
        assert ssm.k_states == 1
        assert ssm.k_endog == 1
        assert ssm.k_posdef == 1
        assert ssm.P1[0, 0] > 1e5  # diffuse

    def test_two_state_model(self, nile_volume: np.ndarray) -> None:
        """Two-state custom model should converge."""
        model = TwoStateCustomModel(nile_volume)
        results = model.fit()
        assert results.optimizer_converged
        assert len(results.params) == 3

    def test_custom_transform(self, nile_volume: np.ndarray) -> None:
        """Custom transform/untransform should be inverses."""
        model = TwoStateCustomModel(nile_volume)
        params = np.array([0.9, 1.0, 1.0])
        unconstrained = model.untransform_params(params)
        roundtrip = model.transform_params(unconstrained)
        np.testing.assert_allclose(roundtrip, params, atol=1e-10)

    def test_default_transform(self, nile_volume: np.ndarray) -> None:
        """Default transform (exp/log) should work."""
        model = SimpleCustomModel(nile_volume)
        params = np.array([100.0, 50.0])
        unconstrained = model.untransform_params(params)
        roundtrip = model.transform_params(unconstrained)
        np.testing.assert_allclose(roundtrip, params, atol=1e-10)

    def test_param_names(self) -> None:
        """param_names should be as defined."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 50)
        model = SimpleCustomModel(y)
        assert model.param_names == ["sigma2_obs", "sigma2_level"]

    def test_forecast(self, nile_volume: np.ndarray) -> None:
        """Forecast should work on custom model."""
        model = SimpleCustomModel(nile_volume)
        results = model.fit()
        fc = results.forecast(steps=10)
        assert fc["mean"].shape == (10, 1)

    def test_summary(self, nile_volume: np.ndarray) -> None:
        """summary() should work."""
        model = SimpleCustomModel(nile_volume)
        results = model.fit()
        s = results.summary()
        assert "sigma2_obs" in s

    def test_simulate(self, nile_volume: np.ndarray) -> None:
        """simulate() should work on custom model."""
        model = SimpleCustomModel(nile_volume)
        y, states = model.simulate(50, seed=42)
        assert y.shape == (50, 1)
        assert states.shape == (50, 1)

    def test_custom_k_posdef(self) -> None:
        """Custom k_posdef should work."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 50)
        model = TwoStateCustomModel(y)
        ssm = model.create_ssm()
        assert ssm.k_posdef == 1
        assert ssm.R.shape == (2, 1)
