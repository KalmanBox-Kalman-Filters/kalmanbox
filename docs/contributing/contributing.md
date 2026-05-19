# Contributing Guide

Thank you for considering a contribution to **kalmanbox** — the foundational
Kalman Filter and State-Space Model library of the
[NodesEcon](https://github.com/nodesecon) ecosystem.

This guide walks you through every stage of contribution, from opening your
first issue to getting a pull request merged.

---

## Ways to contribute

kalmanbox welcomes contributions at all levels of experience:

<div class="grid cards" markdown>

-   :material-bug:{ .lg .middle } **Bug Reports**

    ---

    Found something that behaves unexpectedly? Open a GitHub Issue with a
    minimal reproducible example. Good bug reports are some of the most
    valuable contributions.

    [:octicons-arrow-right-24: Open an issue](https://github.com/nodesecon/kalmanbox/issues/new/choose)

-   :material-lightbulb-outline:{ .lg .middle } **Feature Requests**

    ---

    Have an idea for a new model, estimator, or diagnostic? Start a GitHub
    Discussion so the community can shape the design before implementation
    begins.

    [:octicons-arrow-right-24: Start a discussion](https://github.com/nodesecon/kalmanbox/discussions)

-   :material-code-braces:{ .lg .middle } **Code Contributions**

    ---

    Bug fixes, new models, performance improvements, and refactors.
    Follow the workflow below to open a pull request.

    [:octicons-arrow-right-24: Development setup](setup.md)

-   :material-book-open-outline:{ .lg .middle } **Documentation**

    ---

    Fix typos, improve explanations, add examples, or write tutorials.
    Documentation contributions are always welcome and have a low barrier
    to entry.

    [:octicons-arrow-right-24: Docs guide](documentation.md)

-   :material-test-tube:{ .lg .middle } **Tests**

    ---

    Add tests for untested edge cases, improve coverage, or write property-based
    tests with Hypothesis for numerical routines.

    [:octicons-arrow-right-24: Testing guide](testing.md)

</div>

---

## Before you start

### Check existing issues

Search [open issues](https://github.com/nodesecon/kalmanbox/issues) and
[pull requests](https://github.com/nodesecon/kalmanbox/pulls) before starting
work. Someone might already be working on the same thing.

### For large changes — open a discussion first

If you plan to add a new model class, change a public API, or make a
structural refactor, open a
[GitHub Discussion](https://github.com/nodesecon/kalmanbox/discussions) first.
This avoids the pain of building something that does not align with the
project's direction.

### Sign the DCO

kalmanbox uses the
[Developer Certificate of Origin](https://developercertificate.org/) (DCO).
Every commit must be signed off:

```bash
git commit -s -m "feat: add streaming Kalman Filter"
```

The `-s` flag appends `Signed-off-by: Your Name <your@email.com>` to the
commit message, certifying that you wrote the code and have the right to
submit it under the MIT License.

---

## Development environment setup

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11 or later |
| git | 2.40 or later |
| (optional) Numba | latest stable, for JIT benchmarks |

### Step 1 — Fork and clone

```bash
# 1. Fork on GitHub, then clone your fork
git clone https://github.com/<your-username>/kalmanbox.git
cd kalmanbox

# 2. Add the upstream remote so you can pull future changes
git remote add upstream https://github.com/nodesecon/kalmanbox.git
```

### Step 2 — Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python --version                 # confirm: Python 3.11.x
```

### Step 3 — Install in editable mode

```bash
pip install --upgrade pip
pip install -e ".[dev,docs]"
```

The `dev` extra installs all tools needed for developing and testing:

| Tool | Purpose |
|------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reports |
| `hypothesis` | Property-based testing |
| `ruff` | Linting and formatting |
| `pyright` | Static type checking |
| `pre-commit` | Git hook management |
| `bandit` | Security linting |
| `interrogate` | Docstring coverage |

### Step 4 — Install pre-commit hooks

```bash
pre-commit install
```

Hooks run automatically on `git commit`. To run them manually across all files:

```bash
pre-commit run --all-files
```

### Step 5 — Verify the setup

```bash
pytest tests/ -x -q
```

All tests should pass before you make any changes. If they do not, check the
[troubleshooting FAQ](../faq/troubleshooting.md) or open an issue.

---

## Contribution workflow

### 1. Sync with upstream

Keep your fork current with the upstream `main` branch:

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

### 2. Create a feature branch

```bash
git checkout -b feat/rts-smoother-gpu     # new feature
git checkout -b fix/covariance-symmetry   # bug fix
git checkout -b docs/tvp-tutorial         # documentation
git checkout -b test/dfm-edge-cases       # tests only
```

Use a descriptive prefix:

| Prefix | Use case |
|--------|----------|
| `feat/` | New feature or model |
| `fix/` | Bug fix |
| `docs/` | Documentation only |
| `test/` | Test additions without code changes |
| `perf/` | Performance improvement |
| `refactor/` | Code restructuring |
| `chore/` | Maintenance (deps, CI) |

### 3. Write code

Follow the project's style and conventions (see [Code Style](style.md)).
Key points:

- All public functions and classes must have **NumPy-style docstrings**.
- Add **type hints** to all function signatures.
- Matrix and vector variables must follow the **Durbin & Koopman (2012)**
  naming convention (`Z`, `T`, `H`, `Q`, `R`, `P`, `a`).
- Target **90 % or higher** branch coverage for new code.
- New numerical algorithms must include a **reference** (paper, textbook,
  or URL) in the docstring.

### 4. Write tests

Every change that touches source code needs tests. See the
[Testing guide](testing.md) for conventions. At minimum:

- Unit test the happy path.
- Test boundary conditions (empty arrays, single observations, `NaN` inputs).
- For numerical routines, include a property-based test with Hypothesis.

Run the test suite frequently:

```bash
pytest tests/ -x --tb=short                   # stop on first failure
pytest tests/ --cov=kalmanbox --cov-report=term-missing  # with coverage
```

### 5. Update documentation

If your change adds or modifies a public API:

- Update the relevant page in `docs/api/` or `docs/user-guide/`.
- Add or update the docstring in the source code.
- If the feature warrants a tutorial or new concept page, add it (see
  [Documentation guide](documentation.md)).

### 6. Update the changelog

Add an entry to the `[Unreleased]` section of
[`CHANGELOG.md`](changelog.md):

```markdown
## [Unreleased]

### Added
- `KalmanFilter.stream()`: online update without re-fitting (#123).
```

Use the Keep a Changelog categories: `Added`, `Changed`, `Deprecated`,
`Removed`, `Fixed`, `Security`.

### 7. Commit your changes

Write clear, atomic commit messages following the
[Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <short summary>

[Optional body: what and why, not how]

Signed-off-by: Your Name <your@email.com>
```

Examples:

```bash
git commit -s -m "feat(filters): add Square-Root UKF variant"
git commit -s -m "fix(mle): correct log-determinant sign for diffuse init"
git commit -s -m "docs(tutorial): add TVP CAPM walkthrough"
git commit -s -m "test(dfm): cover rank-deficient observation matrix"
```

Keep commits atomic: one logical change per commit. Squash fixup commits
before opening the PR.

### 8. Push and open a pull request

```bash
git push origin feat/my-feature
```

Then go to GitHub and open a pull request against `main`. Fill in the PR
template:

- **What** this PR changes and **why**.
- Link to the related issue (e.g., `Closes #123`).
- Steps to test manually, if relevant.
- Checklist of acceptance criteria.

---

## Code conventions

### Type hints

All public function signatures must be fully type-annotated:

```python
import numpy as np
from numpy.typing import NDArray

def predict(
    a: NDArray[np.float64],
    P: NDArray[np.float64],
    T: NDArray[np.float64],
    Q: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Predict step of the Kalman filter."""
    a_pred = T @ a
    P_pred = T @ P @ T.T + Q
    return a_pred, P_pred
```

### Docstrings

Use the **NumPy docstring format**. Every public symbol must document its
parameters, return values, and exceptions:

```python
def log_likelihood(
    y: NDArray[np.float64],
    *,
    diffuse: bool = False,
) -> float:
    """
    Compute the Gaussian log-likelihood of the observations.

    Parameters
    ----------
    y : ndarray of shape (T, n)
        Observation matrix. `NaN` values indicate missing observations.
    diffuse : bool, default False
        If ``True``, use the diffuse log-likelihood (Durbin & Koopman, 2012,
        Section 7.2), omitting the first ``d`` time steps.

    Returns
    -------
    float
        Sum of log-likelihood contributions over non-diffuse time steps.

    References
    ----------
    Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space
    Methods*, 2nd ed. Oxford University Press. Section 7.2.
    """
```

### PEP 8 and ruff

All code is formatted and linted with **ruff**:

```bash
ruff format kalmanbox/          # format (replaces black)
ruff check kalmanbox/ --fix     # lint + auto-fix
```

Line length: **100 characters**.

### Matrix naming

Follow Durbin & Koopman (2012) notation consistently:

| Symbol | Meaning |
|--------|---------|
| `Z` | Observation matrix |
| `H` | Observation noise covariance |
| `T` | State transition matrix |
| `R` | Selection matrix |
| `Q` | State noise covariance |
| `P` | State covariance |
| `a` | State vector (predicted) |
| `alpha` | State vector (smoothed) |
| `v` | Innovation vector |
| `F` | Innovation covariance |
| `K` | Kalman gain |

### No magic numbers

Named constants or parameters instead of inline literals:

```python
# Bad
if np.linalg.cond(P) > 1e12:
    ...

# Good
_COND_THRESHOLD = 1e12
if np.linalg.cond(P) > _COND_THRESHOLD:
    ...
```

---

## Testing requirements

### Coverage target

New code must achieve **90 % branch coverage** or higher. Check:

```bash
pytest tests/ --cov=kalmanbox --cov-report=html
open htmlcov/index.html
```

### Numerical tolerances

State-space computations accumulate floating-point error. Use relative
tolerances with `numpy.testing` rather than exact equality:

```python
import numpy.testing as npt

npt.assert_allclose(result, expected, rtol=1e-10, atol=0)
```

For covariance matrices, verify symmetry and positive definiteness:

```python
assert np.allclose(P, P.T), "Covariance not symmetric"
eigenvalues = np.linalg.eigvalsh(P)
assert np.all(eigenvalues >= 0), "Covariance not positive semi-definite"
```

### Property-based tests

Numerical invariants should be expressed as property-based tests:

```python
from hypothesis import given, settings
from hypothesis import strategies as st

@given(st.integers(min_value=10, max_value=500))
def test_filter_invariant_covariance_psd(n_obs: int) -> None:
    """Filtered covariance must remain positive semi-definite."""
    y = np.random.randn(n_obs)
    kf = KalmanFilter.local_level(sigma2_obs=1.0, sigma2_level=0.1)
    result = kf.filter(y)
    for P in result.P_filtered:
        assert np.all(np.linalg.eigvalsh(P) >= -1e-10)
```

---

## Documentation contributions

Documentation lives in `docs/` and is built with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

### Local preview

```bash
mkdocs serve          # serves at http://localhost:8000
```

The server reloads automatically on file changes.

### Adding a new page

1. Create the Markdown file in the appropriate `docs/` subdirectory.
2. Add the page to the `nav:` section of `mkdocs.yml`.
3. Link to it from the relevant section index page.

### Math

Inline math uses `$...$` and display math uses `$$...$$`:

```markdown
The innovation is $v_t = y_t - Z_t a_t$, with covariance

$$F_t = Z_t P_t Z_t^\top + H_t.$$
```

### Code examples

All code examples must be runnable. Prefer examples that demonstrate
a real use case over abstract snippets:

```python
import numpy as np
from kalmanbox import KalmanFilter

# Nile River data (annual flow, m³/s, 1871–1970)
nile = np.array([1120, 1160, 963, ...])

kf = KalmanFilter.local_level(sigma2_obs=15099.0, sigma2_level=1469.1)
result = kf.filter(nile)
print(f"Log-likelihood: {result.loglikelihood:.2f}")
```

---

## Pull request review process

### What reviewers look for

- **Correctness**: Does the algorithm match the cited reference?
- **Tests**: Is coverage adequate? Are edge cases tested?
- **Style**: Does the code follow project conventions?
- **Documentation**: Are docstrings complete? Is the user guide updated?
- **Performance**: Does the change introduce regressions?

### Review timeline

Maintainers aim to provide initial feedback within **5 business days**.
If you have not heard back in 7 days, ping the PR with a comment.

### CI requirements

All CI checks must pass before merging:

| Check | Tool |
|-------|------|
| Tests (Python 3.11, 3.12) | pytest |
| Coverage ≥ 90 % | pytest-cov |
| Linting | ruff check |
| Formatting | ruff format |
| Type checking | pyright |
| Security | bandit |
| Docstring coverage | interrogate |
| Pre-commit hooks | pre-commit |

### Merging strategy

- Feature branches are merged with a **squash merge** to keep `main`
  history linear.
- Hotfixes may use a merge commit when the branch history is important
  for traceability.

---

## Ecosystem context

kalmanbox is the **foundation** of the NodesEcon ecosystem. Changes to
public APIs may break dependent libraries:

| Library | Dependency on kalmanbox |
|---------|------------------------|
| [chronobox](https://github.com/nodesecon/chronobox) | Hierarchical time-series models |
| [forecastbox](https://github.com/nodesecon/forecastbox) | Ensemble forecasting |
| [particlefilterbox](https://github.com/nodesecon/particlefilterbox) | Sequential Monte Carlo |

Before changing any public API (class names, method signatures, default
parameter values), check whether the change would break downstream libraries.
If it would, follow the **deprecation process**:

1. Add a `DeprecationWarning` in the current version.
2. Document the migration path in the changelog.
3. Remove the deprecated API in the next **major** version.

---

## Getting help

| Channel | Purpose |
|---------|---------|
| [GitHub Discussions](https://github.com/nodesecon/kalmanbox/discussions) | Design discussions, questions |
| [GitHub Issues](https://github.com/nodesecon/kalmanbox/issues) | Bug reports, feature requests |
| [Pull Requests](https://github.com/nodesecon/kalmanbox/pulls) | Code review |

!!! note "Code of Conduct"
    All community spaces are governed by the
    [Code of Conduct](code-of-conduct.md). Be respectful and constructive.
