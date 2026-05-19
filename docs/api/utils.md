# Utilities API

`kalmanbox.utils` contains low-level helpers used throughout the
library. They are part of the public API and may be useful when
implementing custom models or extending kalmanbox.

## matrix_ops

Numerically stable matrix routines used by the Kalman recursions:
symmetric positive-definite solvers, Cholesky updates/downdates,
Woodbury identity, and vectorisation helpers for batched covariance
matrices.

```python
from kalmanbox.utils.matrix_ops import (
    solve_pd,       # solve A x = b with A symmetric PD
    chol_update,    # rank-1 Cholesky update
    logdet_pd,      # log-determinant via Cholesky
)
```

::: kalmanbox.utils.matrix_ops

## transforms

Statistical and time-series preprocessing transforms: differencing,
log, Box-Cox, standardisation, and their inverses. Used internally
by models that support `transform="log"` arguments and available
standalone for custom pipelines.

```python
from kalmanbox.utils.transforms import BoxCox, Differencer

bc = BoxCox(lmbda=0.0)          # log transform
y_t = bc.fit_transform(y)
y_inv = bc.inverse_transform(y_t)
```

::: kalmanbox.utils.transforms

## numba_core

JIT-compiled inner loops for the Kalman filter predict/update cycle and
the RTS backward pass. These kernels are used automatically when
`numba` is installed; the pure-NumPy fallbacks are always available.

!!! note "Optional dependency"

    Numba acceleration is **opt-in**:

    ```bash
    pip install "kalmanbox[numba]"
    ```

    Without numba, `kalmanbox` falls back to NumPy implementations with
    no loss of correctness. For large $n$ (> 10 000) or repeated filter
    evaluations (MLE iterations), the JIT versions deliver 5–15×
    speedups.

::: kalmanbox.utils.numba_core
