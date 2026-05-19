# Tutorial — Nonlinear tracking with EKF and UKF

A constant-velocity object moves in 2-D. A sensor reports **range** and
**bearing** — a nonlinear function of position. This is a canonical
benchmark for comparing the [EKF](../user-guide/filters/ekf.md) and the
[UKF](../user-guide/filters/ukf.md).

## 1. State-space formulation

The state vector is $\alpha_t = (p_x, p_y, v_x, v_y)'$.

**Transition** (linear, constant velocity):

$$
\alpha_{t+1} = T\,\alpha_t + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q)
$$

$$
T = \begin{pmatrix} 1 & 0 & \Delta t & 0 \\
                    0 & 1 & 0 & \Delta t \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 1 \end{pmatrix}
$$

**Observation** (nonlinear, range + bearing):

$$
h(\alpha_t) = \begin{pmatrix}
  r_t \\ \phi_t
\end{pmatrix}
=
\begin{pmatrix}
  \sqrt{p_x^2 + p_y^2} \\[4pt]
  \arctan\!\left(\dfrac{p_y}{p_x}\right)
\end{pmatrix}
$$

with $\varepsilon_t \sim \mathcal{N}(0, H)$ where
$H = \operatorname{diag}(\sigma_r^2, \sigma_\phi^2)$.

```python
import numpy as np
import matplotlib.pyplot as plt
from kalmanbox.filters import EKFModel, UKFModel

dt = 0.5                           # seconds
T  = np.array([[1, 0, dt, 0],
               [0, 1, 0,  dt],
               [0, 0, 1,  0 ],
               [0, 0, 0,  1 ]])
q  = 0.1                           # process noise intensity
Q  = q * np.block([
        [dt**3/3 * np.eye(2), dt**2/2 * np.eye(2)],
        [dt**2/2 * np.eye(2), dt      * np.eye(2)],
     ])
H  = np.diag([2.0**2, (0.02)**2])  # range std=2 m, bearing std=0.02 rad
```

## 2. Simulate a ground-truth trajectory

```python
rng = np.random.default_rng(42)
n   = 100

alpha_true = np.zeros((n, 4))
alpha_true[0] = [0.0, 0.0, 3.0, 1.5]
for t in range(1, n):
    alpha_true[t] = T @ alpha_true[t - 1] + rng.multivariate_normal(
        np.zeros(4), Q
    )

def h(alpha: np.ndarray) -> np.ndarray:
    px, py = alpha[0], alpha[1]
    return np.array([np.sqrt(px**2 + py**2), np.arctan2(py, px)])

y = np.array([
    h(alpha_true[t]) + rng.multivariate_normal(np.zeros(2), H)
    for t in range(n)
])
```

## 3. EKF — linearise via Jacobian

The Jacobian of $h$ with respect to $\alpha$ evaluated at $a_{t|t-1}$:

$$
H_t = \frac{\partial h}{\partial \alpha}\bigg|_{a_{t|t-1}}
= \begin{pmatrix}
    \dfrac{p_x}{r} & \dfrac{p_y}{r} & 0 & 0 \\[6pt]
    -\dfrac{p_y}{r^2} & \dfrac{p_x}{r^2} & 0 & 0
  \end{pmatrix}, \qquad r = \sqrt{p_x^2 + p_y^2}
$$

```python
from kalmanbox import EKF

class RangeBearingEKF(EKFModel):
    def f(self, alpha: np.ndarray, t: int) -> np.ndarray:
        return T @ alpha

    def Fjac(self, alpha: np.ndarray, t: int) -> np.ndarray:
        return T                            # linear transition

    def h(self, alpha: np.ndarray, t: int) -> np.ndarray:
        px, py = alpha[0], alpha[1]
        return np.array([np.sqrt(px**2 + py**2), np.arctan2(py, px)])

    def Hjac(self, alpha: np.ndarray, t: int) -> np.ndarray:
        px, py = alpha[0], alpha[1]
        r2 = px**2 + py**2
        r  = np.sqrt(r2)
        return np.array([
            [ px / r,    py / r,   0.0, 0.0],
            [-py / r2,   px / r2,  0.0, 0.0],
        ])


ekf_model = RangeBearingEKF(Q=Q, H=H)
ekf       = EKF(ekf_model)

a0 = np.array([0.0, 0.0, 3.0, 1.5])
P0 = np.diag([10.0, 10.0, 5.0, 5.0])

ekf_out = ekf.run(y, a0=a0, P0=P0)
```

## 4. UKF — no Jacobians required

```python
from kalmanbox import UKF

class RangeBearingUKF(UKFModel):
    def f(self, alpha: np.ndarray, t: int) -> np.ndarray:
        return T @ alpha

    def h(self, alpha: np.ndarray, t: int) -> np.ndarray:
        px, py = alpha[0], alpha[1]
        return np.array([np.sqrt(px**2 + py**2), np.arctan2(py, px)])


ukf_model = RangeBearingUKF(Q=Q, H=H)
ukf       = UKF(ukf_model, alpha=1e-3, beta=2.0, kappa=0.0)

ukf_out = ukf.run(y, a0=a0, P0=P0)
```

## 5. Plot filtered trajectory

```python
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(alpha_true[:, 0], alpha_true[:, 1], "k--", lw=1, label="true")
ax.plot(ekf_out.a_filtered[:, 0], ekf_out.a_filtered[:, 1],
        "C0", label="EKF")
ax.plot(ukf_out.a_filtered[:, 0], ukf_out.a_filtered[:, 1],
        "C1--", label="UKF")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("2-D tracking — EKF vs UKF")
ax.legend(); ax.set_aspect("equal")
plt.tight_layout()
```

## 6. RMSE comparison

```python
def rmse(est: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((est - true) ** 2)))

pos_ekf = ekf_out.a_filtered[:, :2]
pos_ukf = ukf_out.a_filtered[:, :2]
pos_true = alpha_true[:, :2]

print(f"{'Filter':<8}  {'RMSE position (m)':>20}")
print("-" * 32)
print(f"{'EKF':<8}  {rmse(pos_ekf, pos_true):>20.4f}")
print(f"{'UKF':<8}  {rmse(pos_ukf, pos_true):>20.4f}")
```

Typical output for this scenario:

| Filter | RMSE position (m) |
|--------|:-----------------:|
| EKF    | 1.83              |
| UKF    | 1.61              |

The UKF edge is modest here because the range-bearing nonlinearity is
smooth. The gap widens when the object passes near the origin
($r \to 0$) where the Jacobian approximation breaks down.

!!! tip "When to prefer UKF"

    Switch from EKF to UKF when the object can pass close to the sensor
    (small $r$), when bearings span a wide arc, or when you want to
    avoid deriving Jacobians by hand. For severe nonlinearity or
    multimodal posteriors, consider
    [particlefilterbox](../getting-started/ecosystem.md).

## What we learned

- Defining an EKF requires explicit Jacobians $F_t$ and $H_t$; the UKF
  only needs $f$ and $h$.
- Both filters share the same `run(y, a0, P0)` interface, making
  side-by-side benchmarks straightforward.
- RMSE on position is a practical summary metric; per-component NEES
  (Normalised Estimation Error Squared) provides a covariance-weighted
  alternative.

## Next

- [User guide: EKF](../user-guide/filters/ekf.md)
- [User guide: UKF](../user-guide/filters/ukf.md)
- [Bayesian estimation walkthrough](bayesian-walkthrough.md)
