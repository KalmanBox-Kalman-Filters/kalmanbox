# FAQ — Advanced

## How do I define a custom state-space model?

Any state-space model in kalmanbox is characterised by four system matrices
$(T, Z, H, Q)$ and an initial state distribution $(a_1, P_1)$:

$$
\alpha_{t+1} = T_t \alpha_t + R_t \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_t)
$$
$$
y_t = Z_t \alpha_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, H_t)
$$

The simplest approach is to subclass `StateSpaceModel`:

```python
import numpy as np
from kalmanbox.models.base import StateSpaceModel
from kalmanbox import KalmanFilter

class DampedLocalLevel(StateSpaceModel):
    """Level that decays toward zero at rate `phi`."""

    def __init__(self, endog: np.ndarray, phi: float = 0.95):
        super().__init__(endog=endog, k_states=1)
        self.phi = phi

    def build_system_matrices(self, params: np.ndarray) -> dict:
        sigma_eta, sigma_eps = params
        return dict(
            T=np.array([[self.phi]]),
            Z=np.array([[1.0]]),
            H=np.array([[sigma_eps**2]]),
            Q=np.array([[sigma_eta**2]]),
            R=np.array([[1.0]]),
        )

    @property
    def start_params(self) -> np.ndarray:
        return np.array([0.5, 1.0])          # sigma_eta, sigma_eps

    @property
    def param_names(self) -> list[str]:
        return ["sigma_eta", "sigma_eps"]


model = DampedLocalLevel(endog=y, phi=0.97)
result = model.fit()
print(result.params)
```

For time-varying matrices, return a 3-D array with shape `(T, m, m)`.
See [Building Custom Models](../user-guide/kalman/custom-models.md) for a
full walkthrough.

---

## How do I implement parameter constraints?

Use `param_bounds` (box constraints) or `param_transforms` (reparametrisation)
on your `StateSpaceModel` subclass.

**Box constraints via `param_bounds`**

```python
class MyModel(StateSpaceModel):
    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        # sigma_eta > 0, sigma_eps > 0, rho in (-1, 1)
        return [(1e-6, None), (1e-6, None), (-0.999, 0.999)]
```

Bounds are forwarded to `scipy.optimize.minimize` as `bounds`.

**Reparametrisation via `param_transforms`**

For unconstrained optimisation (e.g., L-BFGS-B without bounds), use a
log/logistic transform to keep parameters in their natural domain:

```python
from kalmanbox.transforms import log_transform, logistic_transform

class MyModel(StateSpaceModel):
    @property
    def param_transforms(self):
        return [log_transform, log_transform, logistic_transform]
```

The transforms handle the Jacobian correction for the log-likelihood
automatically.

**Hard equality constraints**

Fix a parameter to a known value by excluding it from `start_params` and
hard-coding it in `build_system_matrices`:

```python
class LocalLevelFixedNoise(StateSpaceModel):
    def __init__(self, endog, sigma_eps_fixed=1.0):
        super().__init__(endog=endog, k_states=1)
        self._H = np.array([[sigma_eps_fixed**2]])

    def build_system_matrices(self, params):
        (sigma_eta,) = params
        return dict(T=[[1.0]], Z=[[1.0]], H=self._H, Q=[[sigma_eta**2]], R=[[1.0]])

    @property
    def start_params(self):
        return np.array([0.5])     # only sigma_eta is free
```

---

## How do I handle very long series?

For series with $T > 100\,000$ observations there are three main strategies.

**1. Enable the Numba JIT backend**

The Numba backend compiles the filter loop to native code.
Enable it at model construction or globally:

```python
model = LocalLevelModel(endog=y, backend="numba")
# or globally:
import kalmanbox
kalmanbox.config.set(backend="numba")
```

This typically yields a 5–8× speedup over pure NumPy on long series.

**2. Chunked filtering with warm start**

If memory is the bottleneck, filter the series in chunks:

```python
from kalmanbox.utils import chunked_filter

kf = KalmanFilter(T=T_mat, Z=Z_mat, H=H_mat, Q=Q_mat)
kf.initialize_diffuse()

results = chunked_filter(kf, y, chunk_size=10_000)
# Each chunk picks up where the previous left off (a_t, P_t as warm start)
```

**3. Reduce state dimension**

Unnecessary state dimensions dominate the $O(T m^3)$ complexity. For DFM
with many series, use the `max_factors` argument to keep only the leading
factors. For AR components, use a companion-form representation with the
minimum lag order.

---

## How do I use Numba for acceleration?

Install the optional Numba dependency:

```bash
pip install "kalmanbox[numba]"
# or
pip install kalmanbox numba
```

Then either pass `backend="numba"` to any model or filter class, or set it
globally before your script starts:

```python
import kalmanbox
kalmanbox.config.set(backend="numba")
```

On the first call, Numba will JIT-compile the filter kernels (typically
1–5 seconds). Subsequent calls reuse the compiled code. To pre-warm the
cache:

```python
kalmanbox.warmup()   # compiles all kernels for a 1-state, 1-obs model
```

The Numba backend is transparent — it does not change the API or results,
only execution speed.

!!! note "Apple Silicon"
    Numba supports Apple Silicon (M-series) from version 0.57 onward.
    Install via `pip install numba` (not Conda) for best compatibility on
    macOS ARM64.

---

## How do I integrate with pandas or xarray?

kalmanbox accepts `numpy.ndarray` natively. For pandas `Series` or
`DataFrame`, pass `.to_numpy()` or use the built-in pandas bridge:

```python
import pandas as pd
from kalmanbox import LocalLevelModel

gdp = pd.read_csv("gdp.csv", index_col=0, parse_dates=True)["GDP"]

model = LocalLevelModel(endog=gdp.to_numpy())
result = model.fit()

# Convert outputs back to pandas
filtered = pd.Series(result.filtered_state[0], index=gdp.index, name="trend")
```

The `KalmanFilterResult.to_dataframe()` method returns a tidy DataFrame with
columns for filtered/smoothed states, variances, and innovations:

```python
df = result.to_dataframe()
# columns: state_0, state_0_var, innovation, innovation_var, …
```

**xarray support**

Install `kalmanbox[xarray]` and use `KalmanFilterResult.to_dataset()` for
labelled multi-dimensional arrays, useful for DFM or multivariate models:

```python
ds = result.to_dataset()   # xarray.Dataset with time + state coordinates
```

---

## Can I use kalmanbox with JAX or PyTorch?

kalmanbox's core is NumPy/SciPy-based and does not directly use JAX or
PyTorch. However, there are two integration patterns:

**Pattern 1 — kalmanbox as a fixed pre-processor**

Estimate the state-space model in kalmanbox and pass the filtered states
as fixed features into a JAX/PyTorch neural network:

```python
result  = model.fit()
states  = result.smoothed_state.T          # shape (T, m)
import torch
features = torch.from_numpy(states).float()
```

**Pattern 2 — Wrap kalmanbox matrices in a JAX `lax.scan`**

Extract the system matrices from a kalmanbox model and re-implement the
filter loop in JAX for autodiff or GPU execution:

```python
import jax, jax.numpy as jnp

mats   = model.build_system_matrices(params)
T, Z   = jnp.array(mats["T"]), jnp.array(mats["Z"])

def kf_step(carry, y_t):
    a, P = carry
    # ... Kalman update equations ...
    return (a_new, P_new), (a_new, P_new)

(a_T, P_T), (a_all, P_all) = jax.lax.scan(kf_step, (a0, P0), Y)
```

Native JAX/PyTorch backends are on the roadmap for a future minor release.

---

## How do I serialize and save an estimated model?

kalmanbox `FitResult` objects support three serialisation formats:

**Pickle (fastest, Python-only)**

```python
import pickle

result = model.fit()
with open("my_model.pkl", "wb") as f:
    pickle.dump(result, f)

# later:
with open("my_model.pkl", "rb") as f:
    result = pickle.load(f)
```

**JSON (portable, human-readable)**

```python
result.save("my_model.json")           # saves params + metadata
result2 = type(model).load("my_model.json", endog=y_new)
```

The JSON file stores parameter names and values, model class name, and
`fit_options`. It does **not** store the original `endog` data.

**joblib (recommended for large NumPy arrays)**

```python
from joblib import dump, load
dump(result, "my_model.joblib", compress=3)
result = load("my_model.joblib")
```

!!! warning "Pickle security"
    Never load a pickle file from an untrusted source. For cross-environment
    deployments, use the JSON format and reconstruct the model from parameters.

---

## How do I use kalmanbox in an ML pipeline?

kalmanbox models implement a scikit-learn-compatible interface with `fit` and
`predict`/`transform` methods, so they can be placed in `sklearn.Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from kalmanbox.sklearn import KalmanFilterTransformer

pipe = Pipeline([
    ("scaler",   StandardScaler()),
    ("kalman",   KalmanFilterTransformer(model="local_level", backend="numba")),
    # downstream models receive smoothed states as features
])

X_transformed = pipe.fit_transform(X_train)
```

`KalmanFilterTransformer` exposes:

- `fit(X)` — estimates parameters by MLE on `X`
- `transform(X)` — returns smoothed states as a feature matrix of shape
  `(n_samples, k_states)`
- `fit_transform(X)` — combined

For cross-validation of time-series pipelines, use forecastbox's
`TimeSeriesSplit`-aware CV, which respects the temporal ordering:

```python
from forecastbox.cv import expanding_cv
scores = expanding_cv(pipe, X, metric="rmse", min_train=200)
```

---

## How do I set reproducible random seeds?

Pass `random_state` (an integer or `numpy.random.Generator`) to any method
that uses randomness (EM initialisation, Gibbs sampler, EnKF):

```python
result = model.fit(method="em", random_state=42)
sampler = GibbsSampler(model=model, n_iter=2000, random_state=np.random.default_rng(0))
```

For full end-to-end reproducibility in scripts, also seed NumPy and Python:

```python
import numpy as np, random
np.random.seed(42)
random.seed(42)
```
