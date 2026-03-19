"""Tests for StateSpaceModel ABC."""

import numpy as np
import pytest

from kalmanbox.core.model import StateSpaceModel
from kalmanbox.core.representation import StateSpaceRepresentation


class SimpleModel(StateSpaceModel):
    """Minimal concrete model for testing the ABC."""

    def _build_ssm(self, params):
        ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
        ssm.T = np.array([[1.0]])
        ssm.Z = np.array([[1.0]])
        ssm.R = np.array([[1.0]])
        ssm.H = np.array([[params[0]]])
        ssm.Q = np.array([[params[1]]])
        ssm.a1 = np.array([0.0])
        ssm.P1 = np.array([[1e7]])
        return ssm

    @property
    def start_params(self):
        var = np.var(self.endog)
        return np.array([var / 2, var / 2])

    @property
    def param_names(self):
        return ["sigma2_obs", "sigma2_level"]

    def transform_params(self, unconstrained):
        return np.exp(unconstrained)

    def untransform_params(self, constrained):
        return np.log(constrained)


class TestStateSpaceModel:
    """Tests for StateSpaceModel ABC."""

    def test_init_1d(self, nile_volume: np.ndarray) -> None:
        """Test initialization with 1D data."""
        model = SimpleModel(nile_volume)
        assert model.nobs == 100
        assert model.k_endog == 1
        assert model.endog.shape == (100, 1)

    def test_loglike(self, nile_volume: np.ndarray) -> None:
        """Test loglike computation."""
        model = SimpleModel(nile_volume)
        params = np.array([15099.0, 1469.0])
        ll = model.loglike(params)
        assert ll == pytest.approx(-641.59, abs=0.5)

    def test_filter_returns_results(self, nile_volume: np.ndarray) -> None:
        """Test filter returns StateSpaceResults."""
        model = SimpleModel(nile_volume)
        params = np.array([15099.0, 1469.0])
        results = model.filter(params)
        assert results.loglike == pytest.approx(-641.59, abs=0.5)
        assert results.smoother_output is None

    def test_smooth_returns_results(self, nile_volume: np.ndarray) -> None:
        """Test smooth returns results with smoother output."""
        model = SimpleModel(nile_volume)
        params = np.array([15099.0, 1469.0])
        results = model.smooth(params)
        assert results.smoother_output is not None
        assert results.smoothed_state is not None

    def test_simulate(self, nile_volume: np.ndarray) -> None:
        """Test simulation."""
        model = SimpleModel(nile_volume)
        params = np.array([15099.0, 1469.0])
        y, states = model.simulate(50, params, seed=42)
        assert y.shape == (50, 1)
        assert states.shape == (50, 1)

    def test_start_params(self, nile_volume: np.ndarray) -> None:
        """Test start params are reasonable."""
        model = SimpleModel(nile_volume)
        sp = model.start_params
        assert len(sp) == 2
        assert all(p > 0 for p in sp)

    def test_transform_roundtrip(self, nile_volume: np.ndarray) -> None:
        """Test transform/untransform are inverses."""
        model = SimpleModel(nile_volume)
        params = np.array([15099.0, 1469.0])
        unconstrained = model.untransform_params(params)
        roundtrip = model.transform_params(unconstrained)
        np.testing.assert_allclose(roundtrip, params)
