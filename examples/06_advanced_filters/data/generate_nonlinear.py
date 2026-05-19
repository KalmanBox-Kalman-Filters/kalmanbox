"""Generate synthetic nonlinear data for advanced filter examples.

The three generators in this module produce canonical test problems used in the
nonlinear filtering literature:

- ``generate_pendulum``: a damped nonlinear pendulum with a sine-of-angle
  observation. Used by e.g. Sarkka (2013, Chap. 5) as a running EKF/UKF
  example.
- ``generate_target_tracking``: a 2-D constant-velocity target observed in
  range/bearing coordinates. The standard nonlinear-observation benchmark from
  the tracking literature (Bar-Shalom et al., 2001).
- ``generate_lorenz63``: the Lorenz (1963) chaotic ODE system with partial
  linear observations. A classical high-sensitivity test case for ensemble
  and sigma-point filters (Evensen, 2009).

All generators are reproducible via a fixed seed, return NumPy float arrays,
and use explicit Euler / RK4 integration to make the dynamics transparent for
teaching. The script, when run as ``__main__``, writes the three datasets to
CSV in the current directory alongside this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Return container                                                             #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class NonlinearDataset:
    """Container for a simulated nonlinear time series."""

    name: str
    t: np.ndarray            # (n,)  time grid
    x_true: np.ndarray       # (n, d_x) true hidden state
    y: np.ndarray            # (n, d_y) noisy observations
    dt: float
    meta: dict

    @property
    def n(self) -> int:
        return self.t.shape[0]


# --------------------------------------------------------------------------- #
# 1. Nonlinear pendulum                                                        #
# --------------------------------------------------------------------------- #

def generate_pendulum(
    n: int = 200,
    dt: float = 0.1,
    sigma_obs: float = 0.3,
    sigma_proc: float = 0.01,
    g: float = 9.81,
    length: float = 1.0,
    theta0: float = 1.2,
    theta_dot0: float = 0.0,
    seed: int = 42,
) -> NonlinearDataset:
    """Nonlinear pendulum with additive process and observation noise.

    Continuous-time dynamics::

        theta''(t) + (g/L) * sin(theta(t)) = w(t)

    discretised with a symplectic (semi-implicit) Euler step of size ``dt``:

        theta_dot_{k+1} = theta_dot_k - dt * (g/L) * sin(theta_k) + dt * w_k
        theta_{k+1}     = theta_k + dt * theta_dot_{k+1}

    A symplectic scheme is used instead of forward Euler because the latter
    injects spurious energy into the oscillator and drives the trajectory
    unbounded within a few periods — undesirable for filtering demos.

    The observation equation is

        y_k = sin(theta_k) + v_k,   v_k ~ N(0, sigma_obs^2),

    which mimics observing the horizontal projection of the bob.

    Parameters
    ----------
    n : int
        Number of time steps.
    dt : float
        Integration step size (seconds).
    sigma_obs : float
        Standard deviation of the observation noise.
    sigma_proc : float
        Standard deviation of the angular-acceleration process noise.
    g : float
        Gravitational acceleration.
    length : float
        Pendulum rod length.
    theta0, theta_dot0 : float
        Initial angle (rad) and angular velocity (rad/s).
    seed : int
        Seed for the ``numpy`` RNG.

    Returns
    -------
    NonlinearDataset
        ``x_true`` has shape ``(n, 2)`` (columns: theta, theta_dot) and ``y``
        has shape ``(n, 1)``.
    """
    rng = np.random.default_rng(seed)

    omega2 = g / length
    x = np.empty((n, 2), dtype=np.float64)
    x[0] = (theta0, theta_dot0)

    w = rng.normal(0.0, sigma_proc, size=n)
    for k in range(n - 1):
        theta, theta_dot = x[k]
        theta_dot_next = theta_dot - dt * omega2 * np.sin(theta) + dt * w[k]
        x[k + 1, 1] = theta_dot_next
        x[k + 1, 0] = theta + dt * theta_dot_next

    v = rng.normal(0.0, sigma_obs, size=n)
    y = np.sin(x[:, 0]) + v
    y = y.reshape(-1, 1)

    t = np.arange(n, dtype=np.float64) * dt

    meta = {
        "sigma_obs": sigma_obs,
        "sigma_proc": sigma_proc,
        "g": g,
        "length": length,
        "state_names": ["theta", "theta_dot"],
        "obs_names": ["sin_theta"],
    }
    return NonlinearDataset("pendulum", t, x, y, dt, meta)


# --------------------------------------------------------------------------- #
# 2. 2-D target tracking with range/bearing observations                       #
# --------------------------------------------------------------------------- #

def generate_target_tracking(
    n: int = 100,
    dt: float = 1.0,
    sigma_range: float = 5.0,
    sigma_bearing: float = 0.05,
    sigma_proc: float = 0.1,
    x0: tuple[float, float, float, float] = (0.0, 1.0, 100.0, 0.5),
    seed: int = 42,
) -> NonlinearDataset:
    """Constant-velocity 2-D target with nonlinear range/bearing observations.

    Linear state transition (near-constant-velocity model)::

        [x, vx, y, vy]_{k+1} = F @ [x, vx, y, vy]_k + G @ w_k

    with

        F = [[1, dt, 0,  0],
             [0,  1, 0,  0],
             [0,  0, 1, dt],
             [0,  0, 0,  1]]

    and piecewise-constant white-noise acceleration input on ``(vx, vy)`` with
    standard deviation ``sigma_proc``.

    Nonlinear observation equation (sensor at the origin)::

        y_k = [ sqrt(x^2 + y^2),  atan2(y, x) ] + noise

    with independent Gaussian noise on range and bearing.

    Parameters
    ----------
    n : int
        Number of time steps.
    dt : float
        Time step.
    sigma_range : float
        Std. dev. of range measurements.
    sigma_bearing : float
        Std. dev. of bearing measurements (radians).
    sigma_proc : float
        Std. dev. of the acceleration-driven process noise.
    x0 : tuple of float
        Initial ``(x, vx, y, vy)``.
    seed : int
        Seed for the ``numpy`` RNG.

    Returns
    -------
    NonlinearDataset
        ``x_true`` has shape ``(n, 4)`` and ``y`` has shape ``(n, 2)``
        (columns: range, bearing).
    """
    rng = np.random.default_rng(seed)

    F = np.array(
        [
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # Continuous white-noise acceleration discretisation (van Loan form).
    q = sigma_proc**2
    Q = q * np.array(
        [
            [dt**3 / 3.0, dt**2 / 2.0, 0.0, 0.0],
            [dt**2 / 2.0, dt,          0.0, 0.0],
            [0.0, 0.0, dt**3 / 3.0, dt**2 / 2.0],
            [0.0, 0.0, dt**2 / 2.0, dt],
        ]
    )
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(4))

    x = np.empty((n, 4), dtype=np.float64)
    x[0] = x0
    for k in range(n - 1):
        eps = rng.standard_normal(4)
        x[k + 1] = F @ x[k] + L @ eps

    r = np.hypot(x[:, 0], x[:, 2])
    phi = np.arctan2(x[:, 2], x[:, 0])

    y = np.column_stack(
        [
            r + rng.normal(0.0, sigma_range, size=n),
            phi + rng.normal(0.0, sigma_bearing, size=n),
        ]
    )

    t = np.arange(n, dtype=np.float64) * dt

    meta = {
        "sigma_range": sigma_range,
        "sigma_bearing": sigma_bearing,
        "sigma_proc": sigma_proc,
        "state_names": ["x", "vx", "y", "vy"],
        "obs_names": ["range", "bearing"],
    }
    return NonlinearDataset("target_tracking", t, x, y, dt, meta)


# --------------------------------------------------------------------------- #
# 3. Lorenz 63 chaotic system                                                  #
# --------------------------------------------------------------------------- #

def _lorenz_rhs(state: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = state
    return np.array(
        [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ]
    )


def generate_lorenz63(
    n: int = 500,
    dt: float = 0.01,
    sigma_obs: float = 2.0,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    x0: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int = 42,
) -> NonlinearDataset:
    """Lorenz (1963) chaotic system integrated by classical RK4.

    Continuous-time dynamics::

        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x*y - beta * z

    Integrated with a fixed-step RK4 scheme of step ``dt``. Only the ``x`` and
    ``z`` components are observed, with additive Gaussian noise of standard
    deviation ``sigma_obs`` each. The default parameters ``(sigma, rho, beta)``
    are the canonical chaotic regime.

    Parameters
    ----------
    n : int
        Number of observations (including the initial condition).
    dt : float
        RK4 step size.
    sigma_obs : float
        Std. dev. of observation noise (per component).
    sigma, rho, beta : float
        Lorenz system parameters.
    x0 : tuple of float
        Initial state.
    seed : int
        Seed for the ``numpy`` RNG.

    Returns
    -------
    NonlinearDataset
        ``x_true`` has shape ``(n, 3)`` and ``y`` has shape ``(n, 2)``
        (columns: x, z).
    """
    rng = np.random.default_rng(seed)

    x = np.empty((n, 3), dtype=np.float64)
    x[0] = x0
    for k in range(n - 1):
        s = x[k]
        k1 = _lorenz_rhs(s, sigma, rho, beta)
        k2 = _lorenz_rhs(s + 0.5 * dt * k1, sigma, rho, beta)
        k3 = _lorenz_rhs(s + 0.5 * dt * k2, sigma, rho, beta)
        k4 = _lorenz_rhs(s + dt * k3, sigma, rho, beta)
        x[k + 1] = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    noise = rng.normal(0.0, sigma_obs, size=(n, 2))
    y = x[:, [0, 2]] + noise

    t = np.arange(n, dtype=np.float64) * dt

    meta = {
        "sigma_obs": sigma_obs,
        "sigma": sigma,
        "rho": rho,
        "beta": beta,
        "state_names": ["x", "y", "z"],
        "obs_names": ["x_obs", "z_obs"],
    }
    return NonlinearDataset("lorenz63", t, x, y, dt, meta)


# --------------------------------------------------------------------------- #
# CSV writer and CLI entry point                                               #
# --------------------------------------------------------------------------- #

def _save_csv(ds: NonlinearDataset, path: Path) -> None:
    """Write a NonlinearDataset to CSV with columns t, state..., obs..."""
    state_names = ds.meta["state_names"]
    obs_names = ds.meta["obs_names"]
    header = ["t"] + [f"x_{c}" for c in state_names] + [f"y_{c}" for c in obs_names]
    data = np.column_stack([ds.t, ds.x_true, ds.y])
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        np.savetxt(fh, data, delimiter=",", fmt="%.10g")


def main(out_dir: Path | None = None) -> dict[str, NonlinearDataset]:
    """Generate all three datasets and save them as CSV next to this file."""
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "pendulum": generate_pendulum(),
        "target_tracking": generate_target_tracking(),
        "lorenz63": generate_lorenz63(),
    }
    for name, ds in datasets.items():
        _save_csv(ds, out_dir / f"{name}.csv")
        print(
            f"[{name}] n={ds.n}  x_true={ds.x_true.shape}  y={ds.y.shape}  "
            f"-> {out_dir / (name + '.csv')}"
        )
    return datasets


if __name__ == "__main__":
    main()
