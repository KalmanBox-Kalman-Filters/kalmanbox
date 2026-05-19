# Testing

kalmanbox uses **pytest** as the test runner. All new public methods
and bug fixes must be accompanied by tests.

## Test layout

```
tests/
├── unit/                   # isolated unit tests per module
│   ├── test_filters.py
│   ├── test_models.py
│   ├── test_estimation.py
│   └── test_utils.py
├── integration/            # end-to-end workflows
│   ├── test_nile_pipeline.py
│   ├── test_dfm_pipeline.py
│   └── test_tvp_pipeline.py
├── property/               # hypothesis property-based tests
│   ├── test_kf_properties.py
│   └── test_smoother_properties.py
└── conftest.py             # shared fixtures
```

Unit tests focus on a **single function or class method** with mocked
or minimal dependencies. Integration tests exercise the full pipeline
(`load → fit → smooth → forecast`) and check outputs against
known-good reference values stored in `tests/fixtures/`.

## Running tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/unit/test_filters.py -v

# Run with coverage report
pytest --cov=kalmanbox --cov-report=term-missing tests/

# Run only fast tests (exclude slow integration tests)
pytest tests/ -m "not slow"
```

Slow tests are marked with `@pytest.mark.slow` and are excluded from
the default local run but always run in CI.

## Numerical tolerance

State-space calculations involve floating-point accumulation. Use
`np.testing.assert_allclose` with explicit tolerances rather than `==`:

```python
import numpy as np

# Good — explicit relative tolerance
np.testing.assert_allclose(
    result.log_likelihood,
    expected_ll,
    rtol=1e-6,
)

# For filtered state means (less tight — smoothing has rounding)
np.testing.assert_allclose(
    result.a_filtered,
    reference_a_filtered,
    rtol=1e-5,
    atol=1e-8,
)

# Bad — do not use
assert result.log_likelihood == expected_ll
assert np.allclose(result.a_filtered, reference_a_filtered)   # default tol too loose
```

The tolerance `rtol=1e-6` is the project standard for filter outputs.
Smoother outputs may require `rtol=1e-5` due to the backward pass.

## Property-based testing with Hypothesis

For mathematical properties that should hold for **any valid input**,
use [Hypothesis](https://hypothesis.readthedocs.io/):

```python
from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np
from kalmanbox import KalmanFilter

@given(
    n_obs=st.integers(min_value=10, max_value=500),
    sigma_eta=st.floats(min_value=1e-4, max_value=100.0),
    sigma_eps=st.floats(min_value=1e-4, max_value=100.0),
)
@settings(max_examples=200)
def test_P_is_spsd_throughout_filter(
    n_obs: int, sigma_eta: float, sigma_eps: float
) -> None:
    """Filtered covariance must be symmetric positive semi-definite at every step."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=n_obs)
    kf = KalmanFilter(
        T=np.array([[1.0]]),
        Z=np.array([[1.0]]),
        H=np.array([[sigma_eps**2]]),
        Q=np.array([[sigma_eta**2]]),
    )
    result = kf.filter(y)
    for P in result.P_filtered:
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals >= -1e-10), f"P not SPSD: min eigenvalue {eigvals.min()}"
```

Good properties to test with Hypothesis:

- Symmetry of $P_t$ throughout the filter.
- Log-likelihood is finite for all valid (positive-definite $H$, $Q$) inputs.
- Smoothed state variance $\le$ filtered state variance at every $t$.
- RTS smoother output matches the Kalman filter in the final period.
- EM updates never decrease the log-likelihood (monotone property).

## Mutation testing

[`mutmut`](https://mutmut.readthedocs.io/) checks whether the test
suite catches small code mutations (flipped signs, off-by-one errors):

```bash
mutmut run --paths-to-mutate kalmanbox/filters/kalman.py
mutmut results
```

A mutation score ≥ 80 % is the project target for core filter code.
Mutation testing is not required for every PR but is run periodically
on the CI schedule.

## When to add tests

Add tests in these situations — no exceptions:

- **New public method**: add a unit test that covers the happy path and
  at least one error case.
- **Bug fix**: add a regression test that reproduces the bug (fails
  before the fix, passes after).
- **Performance-critical code path**: add a benchmark test that asserts
  an upper bound on execution time (use `pytest-benchmark` for this).
- **New model**: add an integration test that fits the model to a
  small synthetic dataset and checks the log-likelihood against a
  reference value computed from a trusted implementation.

## Fixtures

Shared fixtures (data generators, model instances) belong in
`tests/conftest.py`:

```python
# tests/conftest.py
import numpy as np
import pytest

@pytest.fixture(scope="session")
def nile_data() -> np.ndarray:
    """Nile flow series as a 1-D float64 array."""
    from kalmanbox.datasets import load_dataset
    return load_dataset("nile")["volume"].to_numpy(dtype=np.float64)
```

Use `scope="session"` for expensive fixtures that can be shared across
the entire test session.
