"""Solution 01: Extended and Unscented Kalman filters.

Validates kalmanbox `ExtendedKalmanFilter` and `UnscentedKalmanFilter` on
two canonical nonlinear problems:

1.  Damped nonlinear pendulum with ``y = sin(theta) + noise``.
2.  2-D constant-velocity target observed in ``(range, bearing)`` polar
    coordinates.

Checks performed:

*   EKF / UKF RMSE beats the raw-observation baseline on the pendulum.
*   UKF RMSE <= EKF RMSE on the highly nonlinear target-tracking problem.
*   Filtered states from kalmanbox are highly correlated (>0.95) with a
    reference implementation built on top of ``filterpy``.
*   The pendulum validation is run both for the angle (``theta``) and for
    its sine observation as a sanity check.

All checks print PASS/FAIL and the script exits with status 0 on success.
Figures are saved to ``solutions/figures``.

References
----------
-   Sarkka, S. (2013). *Bayesian Filtering and Smoothing*.
-   Julier, S. & Uhlmann, J. (1997). A new extension of the Kalman
    filter to nonlinear systems.
-   Bar-Shalom, Y., Li, X. & Kirubarajan, T. (2001). *Estimation with
    Applications to Tracking and Navigation*.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from kalmanbox.filters import ExtendedKalmanFilter, UnscentedKalmanFilter

from filterpy.kalman import ExtendedKalmanFilter as FPExtendedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as FPUnscentedKalmanFilter

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


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square error on the last axis (sum over state dims)."""
    diff = np.asarray(a) - np.asarray(b)
    if diff.ndim == 1:
        return float(np.sqrt(np.mean(diff**2)))
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=-1))))


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
DT_PEND = 0.1
G = 9.81
L_ROD = 1.0
OMEGA2 = G / L_ROD
SIGMA_PROC_PEND = 0.01
SIGMA_OBS_PEND = 0.3

DT_TARG = 1.0
SIGMA_RANGE = 5.0
SIGMA_BEARING = 0.05
SIGMA_PROC_T = 0.1

F_TARG = np.array(
    [
        [1.0, DT_TARG, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, DT_TARG],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
q_c = SIGMA_PROC_T**2
Q_TARG = q_c * np.array(
    [
        [DT_TARG**3 / 3.0, DT_TARG**2 / 2.0, 0.0, 0.0],
        [DT_TARG**2 / 2.0, DT_TARG, 0.0, 0.0],
        [0.0, 0.0, DT_TARG**3 / 3.0, DT_TARG**2 / 2.0],
        [0.0, 0.0, DT_TARG**2 / 2.0, DT_TARG],
    ]
)


# ---------------------------------------------------------------------------
# Pendulum models (kalmanbox)
# ---------------------------------------------------------------------------
class PendulumEKFModel:
    """Pendulum EKFModel: symplectic Euler transition + sin(theta) observation."""

    def __init__(self, dt: float = DT_PEND) -> None:
        self.dt = dt
        self.k_states = 2
        self.k_endog = 1
        self.R = np.array([[dt], [1.0]])
        self.Q = np.array([[(dt * SIGMA_PROC_PEND) ** 2]])
        self.H = np.array([[SIGMA_OBS_PEND**2]])
        self.a1 = np.array([1.2, 0.0])
        self.P1 = np.diag([0.5, 0.5])

    def transition(self, alpha: np.ndarray, t: int) -> np.ndarray:
        theta, theta_dot = alpha
        theta_dot_new = theta_dot - self.dt * OMEGA2 * np.sin(theta)
        return np.array([theta + self.dt * theta_dot_new, theta_dot_new])

    def transition_jacobian(self, alpha: np.ndarray, t: int) -> np.ndarray:
        theta, _ = alpha
        dt2 = self.dt**2
        return np.array(
            [
                [1.0 - dt2 * OMEGA2 * np.cos(theta), self.dt],
                [-self.dt * OMEGA2 * np.cos(theta), 1.0],
            ]
        )

    def observation(self, alpha: np.ndarray, t: int) -> np.ndarray:
        return np.array([np.sin(alpha[0])])

    def observation_jacobian(self, alpha: np.ndarray, t: int) -> np.ndarray:
        return np.array([[np.cos(alpha[0]), 0.0]])


class PendulumUKFModel(PendulumEKFModel):
    """UKF version: same dynamics/observation, Jacobians ignored."""


# ---------------------------------------------------------------------------
# Target tracking models (kalmanbox)
# ---------------------------------------------------------------------------
class TargetEKFModel:
    """Linear transition, nonlinear range/bearing observation."""

    def __init__(self, a1: np.ndarray, P1: np.ndarray) -> None:
        self.k_states = 4
        self.k_endog = 2
        self.R = np.eye(4)
        self.Q = Q_TARG
        self.H = np.diag([SIGMA_RANGE**2, SIGMA_BEARING**2])
        self.a1 = a1
        self.P1 = P1

    def transition(self, alpha: np.ndarray, t: int) -> np.ndarray:
        return F_TARG @ alpha

    def transition_jacobian(self, alpha: np.ndarray, t: int) -> np.ndarray:
        return F_TARG

    def observation(self, alpha: np.ndarray, t: int) -> np.ndarray:
        xp, _, yp, _ = alpha
        return np.array([np.hypot(xp, yp), np.arctan2(yp, xp)])

    def observation_jacobian(self, alpha: np.ndarray, t: int) -> np.ndarray:
        xp, _, yp, _ = alpha
        r2 = xp * xp + yp * yp
        r = np.sqrt(max(r2, 1e-12))
        return np.array(
            [
                [xp / r, 0.0, yp / r, 0.0],
                [-yp / r2, 0.0, xp / r2, 0.0],
            ]
        )


class TargetUKFModel(TargetEKFModel):
    """UKF variant — only transition/observation are used."""


# ---------------------------------------------------------------------------
# filterpy reference runners
# ---------------------------------------------------------------------------
def _filterpy_ekf_pendulum(y: np.ndarray, a1: np.ndarray, P1: np.ndarray) -> np.ndarray:
    """Run filterpy EKF on the pendulum; returns filtered state (n, 2)."""
    n = y.shape[0]
    ekf = FPExtendedKalmanFilter(dim_x=2, dim_z=1)
    ekf.x = a1.copy()
    ekf.P = P1.copy()
    ekf.R = np.array([[SIGMA_OBS_PEND**2]])
    # Process noise: (dt * sigma_proc)^2 injected on theta_dot; corresponding
    # diffusion on theta scales with dt^2 (symplectic). We build the 2x2 Q as
    # R Q R^T with R = [[dt], [1.0]] for parity with the kalmanbox model.
    R_sel = np.array([[DT_PEND], [1.0]])
    Qscalar = (DT_PEND * SIGMA_PROC_PEND) ** 2
    ekf.Q = R_sel @ np.array([[Qscalar]]) @ R_sel.T

    def fx(x: np.ndarray) -> np.ndarray:
        theta, theta_dot = x
        theta_dot_new = theta_dot - DT_PEND * OMEGA2 * np.sin(theta)
        return np.array([theta + DT_PEND * theta_dot_new, theta_dot_new])

    def Fjac(x: np.ndarray) -> np.ndarray:
        theta, _ = x
        dt2 = DT_PEND**2
        return np.array(
            [
                [1.0 - dt2 * OMEGA2 * np.cos(theta), DT_PEND],
                [-DT_PEND * OMEGA2 * np.cos(theta), 1.0],
            ]
        )

    def Hjac(x: np.ndarray) -> np.ndarray:
        return np.array([[np.cos(x[0]), 0.0]])

    def hx(x: np.ndarray) -> np.ndarray:
        return np.array([np.sin(x[0])])

    out = np.zeros((n, 2))
    for k in range(n):
        if k > 0:
            ekf.F = Fjac(ekf.x)
            ekf.x = fx(ekf.x)
            ekf.P = ekf.F @ ekf.P @ ekf.F.T + ekf.Q
        ekf.update(np.atleast_1d(y[k]), HJacobian=lambda x: Hjac(x), Hx=lambda x: hx(x))
        out[k] = ekf.x
    return out


def _filterpy_ukf_target(y: np.ndarray, a1: np.ndarray, P1: np.ndarray) -> np.ndarray:
    """Run filterpy UKF on the target; returns filtered state (n, 4)."""

    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        return F_TARG @ x

    def hx(x: np.ndarray) -> np.ndarray:
        xp, _, yp, _ = x
        return np.array([np.hypot(xp, yp), np.arctan2(yp, xp)])

    points = MerweScaledSigmaPoints(n=4, alpha=1e-1, beta=2.0, kappa=0.0)
    ukf = FPUnscentedKalmanFilter(dim_x=4, dim_z=2, dt=DT_TARG, hx=hx, fx=fx, points=points)
    ukf.x = a1.copy()
    ukf.P = P1.copy()
    ukf.R = np.diag([SIGMA_RANGE**2, SIGMA_BEARING**2])
    ukf.Q = Q_TARG

    n = y.shape[0]
    out = np.zeros((n, 4))
    for k in range(n):
        if k > 0:
            ukf.predict()
        ukf.update(y[k])
        out[k] = ukf.x
    return out


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("Solution 01: EKF / UKF validation")
print("=" * 70)

pend = pd.read_csv(DATA_DIR / "pendulum.csv")
targ = pd.read_csv(DATA_DIR / "target_tracking.csv")
print(f"Loaded pendulum: n={len(pend)}  columns={list(pend.columns)}")
print(f"Loaded target_tracking: n={len(targ)}  columns={list(targ.columns)}")


# ===========================================================================
# Part 1 — Pendulum
# ===========================================================================
print("\n" + "-" * 70)
print("Part 1: pendulum (y = sin(theta) + noise)")
print("-" * 70)

y_pend = pend[["y_sin_theta"]].to_numpy(dtype=np.float64)
x_true_pend = pend[["x_theta", "x_theta_dot"]].to_numpy(dtype=np.float64)
t_pend = pend["t"].to_numpy(dtype=np.float64)

ekf_kbx = ExtendedKalmanFilter()
ukf_kbx = UnscentedKalmanFilter(alpha=1e-1, beta=2.0, kappa=0.0)

out_ekf_pend = ekf_kbx.filter(y_pend, PendulumEKFModel())
out_ukf_pend = ukf_kbx.filter(y_pend, PendulumUKFModel())

theta_hat_ekf = out_ekf_pend.filtered_state[:, 0]
theta_hat_ukf = out_ukf_pend.filtered_state[:, 0]

rmse_ekf_theta = rmse(theta_hat_ekf, x_true_pend[:, 0])
rmse_ukf_theta = rmse(theta_hat_ukf, x_true_pend[:, 0])

# Baseline RMSE for sin(theta): RMSE of raw observation vs true sin(theta).
# Converting to theta via arcsin is ambiguous, so we evaluate in the
# observation space where the comparison is well defined.
obs_baseline_rmse = rmse(y_pend[:, 0], np.sin(x_true_pend[:, 0]))
sin_ekf_rmse = rmse(np.sin(theta_hat_ekf), np.sin(x_true_pend[:, 0]))
sin_ukf_rmse = rmse(np.sin(theta_hat_ukf), np.sin(x_true_pend[:, 0]))

print(f"  RMSE(theta) EKF = {rmse_ekf_theta:.4f}")
print(f"  RMSE(theta) UKF = {rmse_ukf_theta:.4f}")
print(f"  RMSE(sin theta) raw obs = {obs_baseline_rmse:.4f}")
print(f"  RMSE(sin theta) EKF     = {sin_ekf_rmse:.4f}")
print(f"  RMSE(sin theta) UKF     = {sin_ukf_rmse:.4f}")

check(sin_ekf_rmse < obs_baseline_rmse, "EKF RMSE < raw observation RMSE (pendulum)")
check(sin_ukf_rmse < obs_baseline_rmse, "UKF RMSE < raw observation RMSE (pendulum)")

# Compare against filterpy EKF (the UKF sigma-point comparison happens on target).
ekf_fp = _filterpy_ekf_pendulum(
    y_pend, a1=np.array([1.2, 0.0]), P1=np.diag([0.5, 0.5])
)
corr_ekf_theta = float(np.corrcoef(theta_hat_ekf, ekf_fp[:, 0])[0, 1])
print(f"  corr(kalmanbox EKF theta, filterpy EKF theta) = {corr_ekf_theta:.6f}")
check(corr_ekf_theta > 0.95, "kalmanbox EKF vs filterpy EKF correlation > 0.95 (pendulum)")


# ===========================================================================
# Part 2 — Target tracking
# ===========================================================================
print("\n" + "-" * 70)
print("Part 2: target tracking (range / bearing observations)")
print("-" * 70)

y_targ = targ[["y_range", "y_bearing"]].to_numpy(dtype=np.float64)
x_true_targ = targ[["x_x", "x_vx", "x_y", "x_vy"]].to_numpy(dtype=np.float64)
t_targ = targ["t"].to_numpy(dtype=np.float64)

r0, phi0 = y_targ[0]

# Observation-space baseline: reconstruct (x, y) from (range, bearing) and
# compute RMSE against the true position. This is the naive no-filter estimate.
x_raw = y_targ[:, 0] * np.cos(y_targ[:, 1])
y_raw = y_targ[:, 0] * np.sin(y_targ[:, 1])
xy_raw = np.column_stack([x_raw, y_raw])
xy_true = x_true_targ[:, [0, 2]]
rmse_raw_pos = rmse(xy_raw, xy_true)

# --- Tight prior: EKF and UKF both stay within the linearisation basin.
# This is the regime where kalmanbox and filterpy must agree at very high
# correlation (> 0.95). It also confirms that both filters beat the raw
# observation baseline.
a1_tight = np.array([r0 * np.cos(phi0), 0.0, r0 * np.sin(phi0), 0.0])
P1_tight = np.diag([SIGMA_RANGE**2, 10.0, SIGMA_RANGE**2, 10.0])

out_ekf_tight = ekf_kbx.filter(y_targ, TargetEKFModel(a1_tight, P1_tight))
out_ukf_tight = ukf_kbx.filter(y_targ, TargetUKFModel(a1_tight, P1_tight))
xy_ekf_tight = out_ekf_tight.filtered_state[:, [0, 2]]
xy_ukf_tight = out_ukf_tight.filtered_state[:, [0, 2]]

rmse_ekf_tight = rmse(xy_ekf_tight, xy_true)
rmse_ukf_tight = rmse(xy_ukf_tight, xy_true)

print("Tight prior (linearisation valid):")
print(f"  RMSE(position) raw (range->xy) = {rmse_raw_pos:.4f}")
print(f"  RMSE(position) EKF             = {rmse_ekf_tight:.4f}")
print(f"  RMSE(position) UKF             = {rmse_ukf_tight:.4f}")
check(rmse_ekf_tight < rmse_raw_pos, "EKF RMSE < raw observation RMSE (target)")
check(rmse_ukf_tight < rmse_raw_pos, "UKF RMSE < raw observation RMSE (target)")

# filterpy UKF reference on the same nonlinear observation model.
xhat_fp_ukf = _filterpy_ukf_target(y_targ, a1_tight, P1_tight)
corr_ukf_x = float(np.corrcoef(xy_ukf_tight[:, 0], xhat_fp_ukf[:, 0])[0, 1])
corr_ukf_y = float(np.corrcoef(xy_ukf_tight[:, 1], xhat_fp_ukf[:, 2])[0, 1])
print(f"  corr(kalmanbox UKF x, filterpy UKF x) = {corr_ukf_x:.6f}")
print(f"  corr(kalmanbox UKF y, filterpy UKF y) = {corr_ukf_y:.6f}")
check(corr_ukf_x > 0.95, "kalmanbox UKF vs filterpy UKF x correlation > 0.95 (target)")
check(corr_ukf_y > 0.95, "kalmanbox UKF vs filterpy UKF y correlation > 0.95 (target)")

# --- Wide prior: biased and uncertain initialisation. Here the initial
# sigma points span a wide arc so the linearised Jacobian of EKF evaluated
# around the (wrong) prior mean is a poor approximation of the true
# range/bearing map. UKF captures the 2nd-order behaviour and wins on RMSE.
print("\nWide prior (biased init, UKF's 2nd-order sigma-point transform helps):")
a1_wide = (
    np.array([r0 * np.cos(phi0), 0.0, r0 * np.sin(phi0), 0.0])
    + np.array([50.0, 5.0, -30.0, -3.0])
)
P1_wide = 1000.0 * np.eye(4)

out_ekf_wide = ekf_kbx.filter(y_targ, TargetEKFModel(a1_wide, P1_wide))
out_ukf_wide = ukf_kbx.filter(y_targ, TargetUKFModel(a1_wide, P1_wide))
xy_ekf_wide = out_ekf_wide.filtered_state[:, [0, 2]]
xy_ukf_wide = out_ukf_wide.filtered_state[:, [0, 2]]
rmse_ekf_wide = rmse(xy_ekf_wide, xy_true)
rmse_ukf_wide = rmse(xy_ukf_wide, xy_true)

print(f"  RMSE(position) EKF = {rmse_ekf_wide:.4f}")
print(f"  RMSE(position) UKF = {rmse_ukf_wide:.4f}")
check(rmse_ukf_wide <= rmse_ekf_wide + 1e-9, "UKF RMSE <= EKF RMSE (target tracking, wide prior)")

# For downstream plotting, use the tight-prior result (matches notebook).
xy_ekf = xy_ekf_tight
xy_ukf = xy_ukf_tight
rmse_ekf_pos = rmse_ekf_tight
rmse_ukf_pos = rmse_ukf_tight


# ===========================================================================
# Figures
# ===========================================================================
print("\n" + "-" * 70)
print("Saving figures")
print("-" * 70)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(t_pend, x_true_pend[:, 0], "k-", label=r"true $\theta$", lw=1.3)
axes[0].plot(t_pend, theta_hat_ekf, "C0--", label="EKF", lw=1.1)
axes[0].plot(t_pend, theta_hat_ukf, "C3-.", label="UKF", lw=1.1)
axes[0].set_ylabel(r"$\theta$ (rad)")
axes[0].legend(loc="best")
axes[0].grid(alpha=0.3)
axes[0].set_title(
    f"Pendulum — EKF vs UKF (RMSE theta: EKF={rmse_ekf_theta:.3f}, UKF={rmse_ukf_theta:.3f})"
)
axes[1].plot(t_pend, y_pend[:, 0], "k.", ms=3, alpha=0.4, label=r"observed $\sin\theta$")
axes[1].plot(t_pend, np.sin(theta_hat_ekf), "C0-", label="EKF forecast")
axes[1].plot(t_pend, np.sin(theta_hat_ukf), "C3-", label="UKF forecast", lw=0.8)
axes[1].set_xlabel("time (s)")
axes[1].set_ylabel(r"$\sin\theta$")
axes[1].legend(loc="best")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol01_pendulum.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol01_pendulum.png'}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(xy_true[:, 0], xy_true[:, 1], "k-", label="true path", lw=1.3)
axes[0].plot(xy_ekf[:, 0], xy_ekf[:, 1], "C0--", label="EKF", lw=1.0)
axes[0].plot(xy_ukf[:, 0], xy_ukf[:, 1], "C3-.", label="UKF", lw=1.0)
axes[0].plot(xhat_fp_ukf[:, 0], xhat_fp_ukf[:, 2], "C2:", label="filterpy UKF", lw=0.9)
axes[0].scatter([0], [0], marker="s", color="k", label="sensor")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("Target trajectory (kalmanbox vs filterpy)")
axes[0].legend(loc="best")
axes[0].grid(alpha=0.3)
axes[0].set_aspect("equal", adjustable="datalim")

err_ekf = np.sqrt(np.sum((xy_ekf - xy_true) ** 2, axis=1))
err_ukf = np.sqrt(np.sum((xy_ukf - xy_true) ** 2, axis=1))
err_raw = np.sqrt(np.sum((xy_raw - xy_true) ** 2, axis=1))
axes[1].plot(t_targ, err_raw, "k.", ms=3, alpha=0.4, label="raw obs")
axes[1].plot(t_targ, err_ekf, "C0--", label="EKF", lw=1.1)
axes[1].plot(t_targ, err_ukf, "C3-.", label="UKF", lw=1.1)
axes[1].set_xlabel("time")
axes[1].set_ylabel(r"position error $\|\hat x - x\|$")
axes[1].set_title(
    f"Position error — RMSE: raw={rmse_raw_pos:.2f}, EKF={rmse_ekf_pos:.2f}, UKF={rmse_ukf_pos:.2f}"
)
axes[1].legend(loc="best")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "sol01_target.png", dpi=150)
plt.close(fig)
print(f"  Saved {FIG_DIR / 'sol01_target.png'}")


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
if ALL_PASSED:
    print("Solution 01: ALL CHECKS PASSED")
    print("=" * 70)
    sys.exit(0)
else:
    print("Solution 01: SOME CHECKS FAILED")
    print("=" * 70)
    sys.exit(1)
