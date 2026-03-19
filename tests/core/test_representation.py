"""Tests for StateSpaceRepresentation."""

import numpy as np
import pytest

from kalmanbox.core.representation import StateSpaceRepresentation


class TestStateSpaceRepresentation:
    """Tests for StateSpaceRepresentation."""

    def test_init_defaults(self) -> None:
        """Test default initialization with zeros."""
        ssm = StateSpaceRepresentation(k_states=2, k_endog=1)
        assert ssm.k_states == 2
        assert ssm.k_endog == 1
        assert ssm.k_posdef == 2  # defaults to k_states
        assert ssm.T.shape == (2, 2)
        assert ssm.Z.shape == (1, 2)
        assert ssm.R.shape == (2, 2)
        assert ssm.H.shape == (1, 1)
        assert ssm.Q.shape == (2, 2)
        assert ssm.c.shape == (2,)
        assert ssm.d.shape == (1,)
        assert ssm.a1.shape == (2,)
        assert ssm.P1.shape == (2, 2)

    def test_init_custom_k_posdef(self) -> None:
        """Test initialization with custom k_posdef."""
        ssm = StateSpaceRepresentation(k_states=3, k_endog=2, k_posdef=1)
        assert ssm.k_posdef == 1
        assert ssm.R.shape == (3, 1)
        assert ssm.Q.shape == (1, 1)

    def test_init_invalid_k_states(self) -> None:
        """Test that k_states < 1 raises ValueError."""
        with pytest.raises(ValueError, match="k_states"):
            StateSpaceRepresentation(k_states=0, k_endog=1)

    def test_init_invalid_k_endog(self) -> None:
        """Test that k_endog < 1 raises ValueError."""
        with pytest.raises(ValueError, match="k_endog"):
            StateSpaceRepresentation(k_states=1, k_endog=0)

    def test_validate_correct(self) -> None:
        """Test validation passes for correctly configured SSM."""
        ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
        ssm.T = np.array([[0.9]])
        ssm.Z = np.array([[1.0]])
        ssm.R = np.array([[1.0]])
        ssm.H = np.array([[1.0]])
        ssm.Q = np.array([[0.5]])
        ssm.P1 = np.array([[1e7]])
        ssm.validate()  # should not raise

    def test_validate_wrong_shape(self) -> None:
        """Test validation fails for wrong matrix shape."""
        ssm = StateSpaceRepresentation(k_states=2, k_endog=1)
        ssm.T = np.zeros((3, 3))  # wrong shape
        with pytest.raises(ValueError, match="Matrix T"):
            ssm.validate()

    def test_validate_not_symmetric(self) -> None:
        """Test validation fails for non-symmetric Q."""
        ssm = StateSpaceRepresentation(k_states=2, k_endog=1)
        ssm.Q = np.array([[1.0, 0.5], [0.3, 1.0]])  # not symmetric
        with pytest.raises(ValueError, match="not symmetric"):
            ssm.validate()

    def test_validate_not_psd(self) -> None:
        """Test validation fails for non-PSD matrix."""
        ssm = StateSpaceRepresentation(k_states=2, k_endog=1)
        ssm.Q = np.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric but not PSD
        with pytest.raises(ValueError, match="not positive semi-definite"):
            ssm.validate()

    def test_clone(self) -> None:
        """Test deep copy."""
        ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
        ssm.T = np.array([[0.9]])
        clone = ssm.clone()
        clone.T[0, 0] = 0.1
        assert ssm.T[0, 0] == 0.9  # original unchanged

    def test_is_stable(self) -> None:
        """Test stability check."""
        ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
        ssm.T = np.array([[0.9]])
        assert ssm.is_stable()

        ssm.T = np.array([[1.0]])
        assert not ssm.is_stable()

        ssm.T = np.array([[1.1]])
        assert not ssm.is_stable()

    def test_repr(self) -> None:
        """Test string representation."""
        ssm = StateSpaceRepresentation(k_states=2, k_endog=1)
        assert "k_states=2" in repr(ssm)
        assert "k_endog=1" in repr(ssm)

    def test_local_level_setup(self) -> None:
        """Test setting up a local level model (simplest case)."""
        ssm = StateSpaceRepresentation(k_states=1, k_endog=1)
        ssm.T = np.array([[1.0]])
        ssm.Z = np.array([[1.0]])
        ssm.R = np.array([[1.0]])
        ssm.H = np.array([[15099.0]])
        ssm.Q = np.array([[1469.0]])
        ssm.P1 = np.array([[1e7]])
        ssm.validate()

        # Local level has T=1 so it's a random walk (not stable)
        assert not ssm.is_stable()
