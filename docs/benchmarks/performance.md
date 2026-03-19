# Performance Benchmarks

## Hardware

Benchmarks run on: (to be filled at release time)

## Kalman Filter (forward pass)

| Scenario | T | k_states | Backend | Time |
|:---------|--:|---------:|:--------|-----:|
| LocalLevel | 1,000 | 1 | Python | < 50ms |
| LocalLevel | 1,000 | 1 | Numba | < 5ms |
| LocalLevel | 10,000 | 1 | Numba | < 50ms |
| BSM(12) | 1,000 | 13 | Numba | < 20ms |
| DFM(3,10) | 500 | 3 | Numba | < 100ms |

## Scaling behavior

KalmanBox's computational complexity is $O(T \cdot k^3)$ where $T$ is the
number of observations and $k$ is the number of states.

## Numba acceleration

Install Numba for automatic acceleration of core loops:

```bash
pip install numba
```

Expected speedup: **5x-20x** over pure Python, depending on model size.

```python
from kalmanbox.utils.numba_core import get_backend_info
print(get_backend_info())
```
