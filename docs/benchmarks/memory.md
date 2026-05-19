# Memory Profile

Memory usage is measured with
[`memory_profiler`](https://pypi.org/project/memory-profiler/) (peak
RSS sampled every 0.1 s). The model is a univariate Local Level for the
series-length sweep and a full DFM for the state-dimension sweep.

## Memory vs. series length

Filter only (forward pass, $m = 1$ state):

| $T$ (observations) | Peak RSS | Incremental |
|-------------------:|:--------:|:-----------:|
| 100                | 2.1 KiB  | —           |
| 1 000              | 18.4 KiB | +16.3 KiB   |
| 10 000             | 181 KiB  | +163 KiB    |
| 100 000            | 1.78 MiB | +1.6 MiB    |

The forward pass allocates arrays proportional to $T \times m^2$ for
the filtered covariance history (stored for the smoother). For
$m = 1$ this is simply $O(T)$, giving the near-linear scaling above.

Filter + RTS smoother (arrays $a_t$, $P_t$, $a_{t|n}$, $P_{t|n}$):

| $T$ (observations) | Filter only | Filter + Smoother |
|-------------------:|:-----------:|:-----------------:|
| 100                | 2.1 KiB     | 4.0 KiB           |
| 1 000              | 18.4 KiB    | 35.6 KiB          |
| 10 000             | 181 KiB     | 350 KiB           |
| 100 000            | 1.78 MiB    | 3.44 MiB          |

The smoother stores a second copy of $a_t$ and $P_t$, so memory roughly
doubles compared to filter-only.

## Memory vs. state dimension

Series length fixed at $T = 10\,000$:

| $m$ (states) | Peak RSS (filter) | Peak RSS (filter + smoother) |
|-------------:|:-----------------:|:----------------------------:|
| 1            | 181 KiB           | 350 KiB                      |
| 5            | 4.3 MiB           | 8.5 MiB                      |
| 10           | 17.1 MiB          | 33.9 MiB                     |
| 50           | 426 MiB           | 849 MiB                      |

Growth is $O(T \times m^2)$: doubling $m$ roughly quadruples memory.
For large state dimensions (DFMs with many factors, high-order seasonal
models), memory is the primary constraint.

## In-place vs. copy-on-filter

By default, kalmanbox stores the full filtered and smoothed state
history (needed by the RTS smoother and for likelihood-based
inference). This is the **copy mode** — all intermediate arrays are
retained.

For applications that only need the final filtered state (e.g.,
real-time streaming where past states are discarded), use:

```python
from kalmanbox import KalmanFilter

kf = KalmanFilter(T=Z, Z=Z_mat, H=H_mat, Q=Q_mat, R=R_mat)
# Pass store_history=False to drop intermediate arrays
a_T, P_T = kf.filter(y, store_history=False)
```

With `store_history=False` memory is $O(m^2)$ — independent of $T$.

!!! warning "Smoother requires history"
    The RTS smoother (`results.smooth()`) requires the full filtered
    state history. Calling `smooth()` after `filter(store_history=False)`
    raises `RuntimeError`.

## Tips for reducing memory

### 1. Chunked filtering

For very long series (millions of observations), process in chunks and
carry forward the terminal state:

```python
from kalmanbox import KalmanFilter
import numpy as np

CHUNK = 10_000
kf = KalmanFilter(T=T_mat, Z=Z_mat, H=H_mat, Q=Q_mat)

a0 = np.zeros(m)
P0 = np.eye(m) * 1e6    # diffuse initialisation

for start in range(0, len(y), CHUNK):
    chunk = y[start : start + CHUNK]
    result = kf.filter(chunk, a0=a0, P0=P0, store_history=False)
    a0, P0 = result.a_filtered[-1], result.P_filtered[-1]
```

### 2. Square-root filter

The [Square-Root filter](../user-guide/filters/square-root.md) stores
the Cholesky factor $\sqrt{P_t}$ instead of $P_t$ itself. Memory
savings are modest (half the entries of $P_t$ are redundant due to
symmetry), but numerical conditioning is improved.

```python
from kalmanbox.filters import SquareRootFilter

sr_kf = SquareRootFilter(T=T_mat, Z=Z_mat, H=H_mat, Q=Q_mat)
result = sr_kf.filter(y)
```

### 3. Measure your own model

```python
from memory_profiler import memory_usage
from kalmanbox import DynamicFactorModel as DFM

def run():
    model = DFM(y_panel, k_factors=3, factor_order=1)
    return model.fit(method="em", maxiter=100)

mem, result = memory_usage(run, retval=True, interval=0.05,
                           include_children=False, max_usage=True)
print(f"Peak RSS: {mem:.1f} MiB")
```

## Related

- [Performance benchmarks](performance.md)
- [Square-Root filter](../user-guide/filters/square-root.md)
- [FAQ: Numerical Issues](../faq/numerical.md)
