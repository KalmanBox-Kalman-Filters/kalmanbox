"""Global configuration for kalmanbox."""

from dataclasses import dataclass


@dataclass
class KalmanBoxConfig:
    """Global configuration."""

    default_optimizer: str = "L-BFGS-B"
    max_iterations: int = 500
    tolerance: float = 1e-8
    diffuse_initial_variance: float = 1e7
    symmetry_enforcement: bool = True
    cholesky_fallback_eps: float = 1e-8


# Singleton global config
config = KalmanBoxConfig()
