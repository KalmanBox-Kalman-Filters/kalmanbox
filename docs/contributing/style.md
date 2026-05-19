# Code Style

Consistent style makes the codebase easier to read, review, and
maintain. All rules here are enforced automatically by ruff and pyright
in CI; the pre-commit hooks catch most issues before you push.

## Formatter and linter

kalmanbox uses **ruff** for both formatting and linting (replacing
black, isort, and flake8 in a single tool).

```bash
ruff format kalmanbox/         # format
ruff check kalmanbox/ --fix    # lint + auto-fix
```

ruff lint rules enabled (see `pyproject.toml`):

| Code group | Covers                                          |
|------------|-------------------------------------------------|
| `E`, `W`   | pycodestyle errors and warnings                 |
| `F`        | pyflakes (undefined names, unused imports, …)   |
| `I`        | isort import ordering                           |
| `N`        | PEP-8 naming conventions                        |
| `UP`       | pyupgrade — modernise syntax for Python 3.11+   |
| `B`        | flake8-bugbear — common gotchas                 |
| `SIM`      | flake8-simplify — simplifiable constructs       |

## Line length

**100 characters**. The formatter wraps at this limit. Longer lines
in docstrings or comments are acceptable when breaking would hurt
readability.

## Naming conventions

### Matrix and vector variables

State-space notation follows Durbin & Koopman (2012). Use these names
consistently:

| Symbol | Type    | Meaning                          |
|--------|---------|----------------------------------|
| `T`    | matrix  | State transition matrix          |
| `Z`    | matrix  | Observation (design) matrix      |
| `H`    | matrix  | Observation noise covariance     |
| `Q`    | matrix  | State noise covariance           |
| `R`    | matrix  | Selection matrix for state noise |
| `a`    | vector  | Filtered/predicted state mean    |
| `v`    | vector  | Innovation (observation residual)|
| `K`    | matrix  | Kalman gain                      |
| `P`    | matrix  | State covariance matrix          |
| `F`    | matrix  | Innovation covariance            |

Uppercase for matrices, lowercase for vectors. Do not use `M`, `S`,
`C`, or other letters that conflict with this convention in the same
scope.

### Dimension names

Spell out dimension names in full — no single-letter or ambiguous
abbreviations:

```python
# Good
n_obs: int       # number of observations
n_states: int    # state vector dimension
n_series: int    # number of observed series (multivariate)
n_factors: int   # number of latent factors in DFM

# Bad — do not use
n: int
m: int
p: int
k: int
```

Using full names avoids ambiguity (is `k` factors or states?) and
makes the code self-documenting.

## Type hints

Type hints are **required** for all public API functions, methods, and
class attributes. Internal helpers should have hints where the types
are non-obvious.

```python
import numpy as np
from numpy.typing import NDArray

def filter(
    self,
    y: NDArray[np.float64],
    a0: NDArray[np.float64] | None = None,
    P0: NDArray[np.float64] | None = None,
    *,
    store_history: bool = True,
) -> FilterResult:
    ...
```

Run pyright in strict mode to catch type errors:

```bash
pyright kalmanbox/
```

`pyright` is configured in `pyproject.toml` under
`[tool.pyright]` with `typeCheckingMode = "strict"`. New code
must pass without errors.

## Docstrings

Docstrings are **required** for all public classes, methods, and
module-level functions. Use **NumPy style**:

```python
def filter(
    self,
    y: NDArray[np.float64],
    store_history: bool = True,
) -> FilterResult:
    """Run the forward Kalman filter.

    Parameters
    ----------
    y : NDArray[np.float64]
        Observation array of shape ``(n_obs,)`` or ``(n_obs, n_series)``.
        ``np.nan`` entries are treated as missing observations.
    store_history : bool, optional
        If True (default), all filtered states and covariances are
        stored in the returned ``FilterResult``. Set to False for
        streaming / memory-constrained use.

    Returns
    -------
    FilterResult
        Named tuple containing ``a_filtered``, ``P_filtered``,
        ``innovations``, and ``log_likelihood``.

    Raises
    ------
    ValueError
        If ``y`` has more than 2 dimensions or is empty.

    Examples
    --------
    >>> result = kf.filter(y)
    >>> print(result.log_likelihood)
    -632.55
    """
```

Interrogate enforces a docstring coverage threshold of 90 %; CI
will fail below this threshold.

## Imports

Group imports in this order (enforced by ruff `I`):

1. Standard library
2. Third-party packages
3. Local (`kalmanbox`) modules

```python
# 1. stdlib
from __future__ import annotations
import warnings
from typing import TYPE_CHECKING

# 2. third-party
import numpy as np
from scipy.optimize import minimize

# 3. local
from kalmanbox.core import FilterResult
from kalmanbox._typing import FloatArray
```

No star imports (`from module import *`) in library code.
