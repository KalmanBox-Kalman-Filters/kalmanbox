"""Solution 03: Ensemble Kalman Filter on the Lorenz 63 system.

Validates kalmanbox `EnsembleKalmanFilter` on the canonical Lorenz (1963)
chaotic ODE:

    dx/dt = sigma * (y - x),
    dy/dt = x * (rho - z) - y,
    dz/dt = x*y - beta * z,

integrated with RK4 and observed partially (only x and z components) with
additive Gaussian noise. The dataset is produced by the shipped CSV
generator with the canonical chaotic parameters (sigma, rho, beta) =
(10, 28, 8/3).

Checks performed:

*   EnKF RMSE beats the raw-observation baseline for every tested
    ensemble size ``N in {10, 50, 200, 500}``.
*   EnKF RMSE decreases (or stabilises) with ensemble size — i.e. the
    RMSE at N=500 is no worse than at N=10 by more than a small
    statistical margin, and the spread across seeds shrinks with N.
*   EnKF at the largest tested ensemble size approaches the RMSE of the
    analytic EKF on the same problem.

Figures showing the filtered trajectory, the RMSE-vs-N curve, and a 3-D
plot of the true and filtered attractor are written to
``solutions/figures``.

All checks print PASS/FAIL and the script exits with status 0 on success.

References
----------
-   Evensen, G. (2003). *The Ensemble Kalman Filter: theoretical
    formulation and practical implementation*. Ocean Dynamics.
-   Lorenz, E. N. (1963). Deterministic nonperiodic flow. J. Atmos. Sci.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from kalmanbox.filters import EnsembleKalmanFilter, ExtendedKalmanFilter

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALL_PASSED = True


def check(condition: bool, msg: str) -> None:
    """Record a PASS/FAIL line and flip ``ALL_PASSED`` on failure."""
    global ALL_PASSED
    if not condition:
        ALL_PASSED = False
        print(f"  [FAIL] {msg}")
    else:
        print(f"  [PASS] {msg}")


def rmse3d(a: np.ndarray, b: np.ndarray) -> float:
    """Per-step 3-D RMSE, averaged over time."""
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


# ---------------------------------------------------------------------------
# Lorenz 63 dynamics
# ---------------------------------------------------------------------------
SIGMA_L = 10.0
RHO_L = 28.0
BETA_L = 8.0 / 3.0
SIGMA_OBS_L = 2.0
SIGMA_PROC_L = 0.2


def lorenz_rhs(x: np.ndarray) -> np.ndarray:
    """Continuous-time Lorenz right-hand side."""
    xx, yy, zz = x
    return np.array(
        [
            SIGMA_L * (yy - xx),
            xx * (RHO_L - zz) - yy,
            xx * yy - BETA_L * zz,
        ]
    )


def lorenz_jac(x: np.ndarray) -> np.ndarray:
    """Jacobian of the continuous-time Lorenz RHS."""
    xx, _, zz = x
    return np.array(
        [
            [-SIGMA_L, SIGMA_L, 0.0],
            [RHO_L - zz, -1.0, -xx],
            [x[1], xx, -BETA_L],
        ]
    )


def make_rk4(dt: float):
    """Factory: a one-step RK4 integrator for the current ``dt``."""

    def rk4(x: np.ndarray) -> np.ndarray:
        k1 = lorenz_rhs(x)
        k2 = lorenz_rhs(x + 0.5 * dt * k1)
        k3 = lorenz_rhs(x + 0.5 * dt * k2)
        k4 = lorenz_rhs(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return rk4


def make_rk4_jacobian(dt: float):
    """Factory: Jacobian of the one-step RK4 map w.r.t. ``x``."""

    def rk4_jac(x: np.ndarray) -> np.ndarray:
        k1 = lorenz_rhs(x)
        dk1 = lorenz_jac(x)

        y2 = x + 0.5 * dt * k1
        k2 = lorenz_rhs(y2)
        dk2 = lorenz_jac(y2) @ (np.eye(3) + 0.5 * dt * dk1)

        y3 = x + 0.5 * dt * k2
        k3 = lorenz_rhs(y3)
        dk3 = lorenz_jac(y3) @ (np.eye(3) + 0.5 * dt * dk2)

        y4 = x + dt * k3
        _k4 = lorenz_rhs(y4)  # value not used in Jacobian; evaluate for clarity
        dk4 = lorenz_jac(y4) @ (np.eye(3) + dt * dk3)

        return np.eye(3) + (dt / 6.0) * (dk1 + 2.0 * dk2 + 2.0 * dk3 + dk4)

    return rk4_jac


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class LorenzEnKFModel:
    """RK4 Lorenz transition + linear observation of (x, z)."""

    def __init__(self, dt: float, a1: np.ndarray, P1: np.ndarray) -> None:
        self._rk4 = make_rk4(dt)
        self.k_states = 3
        self.k_endog = 2
        self.R = np.eye(3)
        self.Q = (SIGMA_PROC_L**2) * np.eye(3)
        self.H = (SIGMA_OBS_L**2) * np.eye(2)
        self.a1 = a1
        self.P1 = P1

    def transition(self, alpha: np.ndarray, t: int) -> np.ndarray:
        return self._rk4(alpha)

    def observation(self, alpha: np.ndarray, t: int) -> np.ndarray:
        return np.array([alpha[0], alpha[2]])


class LorenzEKFModel(LorenzEnKFModel):
    """Lorenz EKF: analytic Jacobian of the RK4 map and linear observation."""

    def __init__(self, dt: float, a1: np.ndarray, P1: np.ndarray) -> None:
        super().__init__(dt, a1, P1)
        self._rk4_jac = make_rk4_jacobian(dt)

    def transition_jacobian(self, alpha: np.ndarray, t: int) -> np.ndarray:
        return self._rk4_jac(alpha)

    def observation_jacobian(self, alpha: np.ndarray, t: int) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Load data and set up initial condition
# ---------------------------------------------------------------------------
print("=" * 70)
print("Solution 03: Ensemble Kalman filter on Lorenz 63")
print("=" * 70)

lorenz = pd.read_csv(DATA_DIR / "lorenz63.csv")
print(f"Loaded lorenz63: n={len(lorenz)}  columns={list(lorenz.columns)}")

x_true = lorenz[["x_x", "x_y", "x_z"]].to_numpy(dtype=np.float64)
y_obs = lorenz[["y_x_obs", "y_z_obs"]].to_numpy(dtype=np.float64)
t_grid = lorenz["t"].to_numpy(dtype=np.float64)
DT_L = float(t_grid[1] - t_grid[0])

# Initial condition: bias the truth slightly so the filters have to work.
a1 = x_true[0] + np.array([2.0, -1.5, 1.0])
P1 = 4.0 * np.eye(3)

# Raw-observation baseline: naively treat (x_obs, z_obs) as (x, z) estimates
# and set y-hat = 0 (since y is unobserved, the uninformed estimate is the
# unconditional mean of the attractor, which is close to 0 in the chaotic
# regime). We compare on the observed components only — that is what the
# "raw observation" actually provides.
rmse_raw_xz = float(
    np.sqrt(np.mean(np.sum((y_obs - x_true[:, [0, 2]]) ** 2, axis=1)))
)
print(f"Baseline RMSE on (x, z) from raw obs = {rmse_raw_xz:.4f}")


# ===========================================================================
# Part 1 — EnKF sanity run at a moderate N
# ===========================================================================
print("\n" + "-" * 70)
print("Part 1: EnKF sanity run at N=100")
print("-" * 70)

enkf100 = EnsembleKalmanFilter(n_ensemble=100, random_state=0)
out100 = enkf100.filter(y_obs, LorenzEnKFModel(DT_L, a1, P1))
rmse100 = rmse3d(out100.filtered_state, x_true)
rmse100_xz = float(
    np.sqrt(
        np.mean(
            np.sum(
                (out100.filtered_state[:, [0, 2]] - x_true[:, [0, 2]]) ** 2, axis=1
            )
        )
    )
)
print(f"  EnKF (N=100) RMSE full 3-D = {rmse100:.3f}")
print(f"  EnKF (N=100) RMSE on (x,z) = {rmse100_xz:.3f}")
check(rmse100_xz < rmse_raw_xz, "EnKF(N=100) beats raw observation RMSE on (x, z)")


# ===========================================================================
# Part 2 — RMSE convergence with ensemble size
# ===========================================================================
print("\n" + "-" * 70)
print("Part 2: RMSE convergence vs ensemble size")
print("-" * 70)

ensemble_sizes = [10, 50, 200, 500]
seeds = [0, 1, 2]

conv_rows: list[dict[str, float]] = []
for N in ensemble_sizes:
    rmse_runs: list[float] = []
    rmse_xz_runs: list[float] = []
    for s in seeds:
        f = EnsembleKalmanFilter(n_ensemble=N, random_state=s)
        out = f.filter(y_obs, LorenzEnKFModel(DT_L, a1, P1))
        rmse_runs.append(rmse3d(out.filtered_state, x_true))
        rmse_xz_runs.append(
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (out.filtered_state[:, [0, 2]] - x_true[:, [0, 2]]) ** 2,
                            axis=1,
                        )
                    )
                )
            )
        )
    conv_rows.append(
        {
            "N": float(N),
            "rmse_mean": float(np.mean(rmse_runs)),
            "rmse_std": float(np.std(rmse_runs)),
            "rmse_xz_mean": float(np.mean(rmse_xz_runs)),
        }
    )

conv = pd.DataFrame(conv_rows)
print(conv.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# --- Convergence checks ---------------------------------------------------
# (a) Every tested size beats the raw-observation baseline on (x, z).
for row in conv_rows:
    check(
        row["rmse_xz_mean"] < rmse_raw_xz,
        f"EnKF(N={int(row['N'])}) beats raw observation RMSE on (x, z)",
    )

# (b) RMSE spread across seeds shrinks with N — sample variance of the
# ensemble-mean estimate is O(1/N), so std at N=500 must be < std at N=10.
std_small = conv_rows[0]["rmse_std"]
std_large = conv_rows[-1]["rmse_std"]
print(f"  RMSE std across seeds: N=10 -> {std_small:.4f}   N=500 -> {std_large:.4f}")
check(std_large < std_small, "EnKF RMSE std across seeds decreases with N (convergence)")

# (c) Large-N RMSE is no worse than small-N by more than a generous
# statistical margin (2 sigma of the small-N spread). This is the core
# "convergence in N" statement — the mean RMSE at the largest tested
# ensemble is at least as good as at the smallest.
margin = max(2.0 * std_small, 0.1)
rmse_small = conv_rows[0]["rmse_mean"]
rmse_large = conv_rows[-1]["rmse_mean"]
print(f"  RMSE mean: N=10 -> {rmse_small:.4f}   N=500 -> {rmse_large:.4f}"
      f"   (margin = {margin:.4f})")
check(
    rmse_large <= rmse_small + margin,
    "EnKF(N=500) mean RMSE <= EnKF(N=10) mean RMSE + 2 sigma",
)


# ===========================================================================
# Part 3 — EnKF vs EKF (analytic Jacobian)
# ===========================================================================
print("\n" + "-" * 70)
print("Part 3: EnKF (N=500) vs EKF (analytic Jacobian)")
print("-" * 70)

ekf = ExtendedKalmanFilter()
out_ekf = ekf.filter(y_obs, LorenzEKFModel(DT_L, a1, P1))
rmse_ekf = rmse3d(out_ekf.filtered_state, x_true)

enkf500 = EnsembleKalmanFilter(n_ensemble=500, random_state=0)
out_enkf500 = enkf500.filter(y_obs, LorenzEnKFModel(DT_L, a1, P1))
rmse_enkf500 = rmse3d(out_enkf500.filtered_state, x_true)

print(f"  EKF        RMSE = {rmse_ekf:.4f}")
print(f"  EnKF (500) RMSE = {rmse_enkf500:.4f}")
# EnKF at large N should be close to EKF. We accept a 50% relative gap:
# the EnKF's sampling noise and inflation tuning mean we do not expect
# strict parity, only the same order of magnitude.
gap = abs(rmse_enkf500 - rmse_ekf) / max(rmse_ekf, 1e-12)
print(f"  relative gap (EnKF vs EKF) = {gap:.3f}")
check(gap < 0.50, "EnKF(N=500) and EKF RMSE are within 50% of each other")


# ===========================================================================
# Figures
# ===========================================================================
print("\n" + "-" * 70)
print("Saving figures")
print("-" * 70)

# Figure 1: time series of x/y/z for truth, EnKF(N=100), EKF
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for j, name in enumerate(["x", "y", "z"]):
    axes[j].plot(t_grid, x_true[:, j], "k-", lw=1.2, label="true")
    axes[j].plot(t_grid, out100.filtered_state[:, j], "C1-", lw=1.0, label="EnKF(N=100)")
    axes[j].plot(t_grid, out_ekf.filtered_state[:, j], "C0--", lw=1.0, label="EKF")
    if j == 0:
        axes[j].scatter(t_grid, y_obs[:, 0], c="k", s=3, alpha=0.3, label="x_obs")
    if j == 2:
        axes[j].scatter(t_grid, y_obs[:, 1], c="k", s=3, alpha=0.3, label="z_obs")
    axes[j].set_ylabel(name)
    axes[j].grid(alpha=0.3)
    axes[j].legend(loc="upper right")
axes[-1].set_xlabel("time")
fig.suptitle(
    f"Lorenz 63 filtering — EnKF(N=100) RMSE={rmse100:.2f}  EKF RMSE={rmse_ekf:.2f}",
    y=1.02,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol03_lorenz_timeseries.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol03_lorenz_timeseries.png'}")

# Figure 2: RMSE vs ensemble size
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].errorbar(
    conv["N"],
    conv["rmse_mean"],
    yerr=conv["rmse_std"],
    fmt="o-",
    capsize=4,
    color="C1",
    label="EnKF (3-D RMSE)",
)
axes[0].axhline(rmse_ekf, color="C0", ls="--", lw=1.0, label=f"EKF = {rmse_ekf:.3f}")
axes[0].set_xscale("log")
axes[0].set_xlabel("ensemble size N")
axes[0].set_ylabel("RMSE")
axes[0].set_title("EnKF RMSE vs N (mean +/- 1 sigma across seeds)")
axes[0].legend(loc="best")
axes[0].grid(alpha=0.3)

axes[1].plot(conv["N"], conv["rmse_std"], "s-", color="C2")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("ensemble size N")
axes[1].set_ylabel("RMSE std across seeds")
axes[1].set_title(r"Sampling-noise convergence (expect $\sim N^{-1/2}$)")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol03_rmse_vs_N.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol03_rmse_vs_N.png'}")

# Figure 3: 3-D attractor — true vs filtered (EnKF N=500)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], "k-", lw=0.7, alpha=0.8, label="true")
ax.plot(
    out_enkf500.filtered_state[:, 0],
    out_enkf500.filtered_state[:, 1],
    out_enkf500.filtered_state[:, 2],
    "C1-",
    lw=0.7,
    alpha=0.8,
    label="EnKF(N=500)",
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Lorenz 63 attractor — true vs filtered")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "sol03_attractor.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol03_attractor.png'}")


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
if ALL_PASSED:
    print("Solution 03: ALL CHECKS PASSED")
    print("=" * 70)
    sys.exit(0)
else:
    print("Solution 03: SOME CHECKS FAILED")
    print("=" * 70)
    sys.exit(1)
