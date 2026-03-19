# Installation

## Requirements

- Python >= 3.11
- NumPy >= 1.24
- SciPy >= 1.10
- pandas >= 2.0

## Install from PyPI

```bash
pip install kalmanbox
```

## Install with optional dependencies

```bash
# Development tools
pip install kalmanbox[dev]

# Numba acceleration (recommended for large datasets)
pip install kalmanbox[numba]

# Documentation building
pip install kalmanbox[docs]
```

## Install from source

```bash
git clone https://github.com/nodesecon/kalmanbox.git
cd kalmanbox
pip install -e ".[dev]"
```

## Verify installation

```python
import kalmanbox
print(kalmanbox.__version__)
```

## Optional: Numba acceleration

KalmanBox can use [Numba](https://numba.pydata.org/) to JIT-compile the core
Kalman filter loops for significant speedup (5x+):

```bash
pip install numba
```

Check if Numba is detected:

```python
from kalmanbox.utils.numba_core import get_backend_info
print(get_backend_info())
# {'numba_available': True, 'backend': 'numba', 'numba_version': '0.59.0'}
```
