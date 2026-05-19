---
title: Installation
description: >-
  Install kalmanbox from PyPI, configure optional extras for acceleration
  and documentation, and set up a development environment.
---

# Installation

`kalmanbox` is available on [PyPI](https://pypi.org/project/kalmanbox/) and
installs with a single `pip` command. This page covers every installation
scenario, from a standard user install to a full development environment.

---

## Requirements

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| Python  | 3.11 | Runtime |
| NumPy   | 1.24 | Array operations, linear algebra |
| SciPy   | 1.10 | Matrix decompositions, optimisers |
| pandas  | 2.0  | Time-indexed data, `DatetimeIndex` support |

These three dependencies are installed automatically when you run any of the
`pip install kalmanbox` commands below.

---

## Standard install

```bash
pip install kalmanbox
```

This gives you the full public API:

- `KalmanFilter`, `RTSSmoother` — core recursion engine
- `LocalLevel`, `LocalLinearTrend`, `BSM`, `UCM` — structural models
- `DFM`, `TVP` — advanced models
- `EKF`, `UKF`, `SquareRootFilter`, `InformationFilter`, `EnKF` — alternative filters
- `GibbsSampler`, `FFBS` — Bayesian estimators
- `MLEstimator`, `EMEstimator` — classical estimators
- `kalmanbox` CLI

---

## Optional extras

### Numba acceleration

```bash
pip install kalmanbox[numba]
```

Installs [Numba](https://numba.pydata.org/) (>= 0.58), which JIT-compiles the
inner Kalman recursion loops. On large datasets (T > 5 000) this typically
delivers a **5–10× speed-up** with no change to the API.

!!! tip "First-run compilation"
    The first call after installing Numba triggers JIT compilation, which
    takes a few seconds. Subsequent calls in the same Python session (and
    all future sessions) use the cached compiled functions.

Verify the backend after installation:

```python
from kalmanbox.utils.numba_core import get_backend_info

info = get_backend_info()
print(info)
# {'numba_available': True, 'backend': 'numba', 'numba_version': '0.59.0'}
```

If `numba_available` is `False`, the library falls back silently to the
pure-NumPy backend without raising an error.

### Documentation (local build)

```bash
pip install kalmanbox[docs]
```

Installs MkDocs Material and all plugins needed to build this documentation
site locally. Use this only if you intend to contribute to the docs or render
them offline.

```bash
mkdocs serve   # live-reload preview at http://localhost:8000
mkdocs build   # static site in site/
```

---

## Development install

Clone the repository and install in editable mode with all development
and documentation tools:

```bash
git clone https://github.com/nodesecon/kalmanbox.git
cd kalmanbox
pip install -e ".[dev,docs]"
```

The `dev` extra adds:

| Tool | Purpose |
|------|---------|
| `pytest` + `pytest-cov` | Test runner and coverage |
| `ruff` | Linter and formatter |
| `pyright` | Static type checker |
| `hypothesis` | Property-based testing |
| `pre-commit` | Git hook manager |
| `bandit` | Security static analysis |
| `mutmut` | Mutation testing |
| `structlog` | Structured logging in tests |

Set up pre-commit hooks after the editable install:

```bash
pre-commit install
```

!!! note "Hatchling build backend"
    `kalmanbox` uses [Hatchling](https://hatch.pypa.io/) as its build
    backend. You do not need to install it separately — `pip` handles
    it automatically via `build-system` in `pyproject.toml`.

---

## Optional standalone packages

The following packages are not declared as extras but integrate seamlessly
with `kalmanbox`:

| Package | Install | When you need it |
|---------|---------|-----------------|
| `matplotlib` | `pip install matplotlib` | `kalmanbox.visualization` plotting functions |
| `arviz` | `pip install arviz` | Bayesian posterior diagnostics, trace plots, HDI |
| `numba` | `pip install numba` | Filter loop acceleration (same as `[numba]` extra) |

None of these are required at import time. `kalmanbox` raises an informative
`ImportError` only when you call a function that specifically needs the missing
package.

---

## Verify the installation

Run this one-liner to confirm `kalmanbox` is installed correctly:

```bash
python -c "import kalmanbox; print(kalmanbox.__version__)"
```

Expected output (version number may differ):

```
0.1.0
```

For a more thorough check, run the built-in diagnostic:

```python
import kalmanbox

kalmanbox.show_config()
# kalmanbox 0.1.0
# Python      3.12.3
# NumPy       1.26.4
# SciPy       1.13.0
# pandas      2.2.1
# Numba       0.59.0 (available)
# matplotlib  3.8.4  (available)
# arviz       0.18.0 (available)
```

---

## Upgrading

```bash
pip install --upgrade kalmanbox
```

Check the [Changelog](../contributing/changelog.md) for breaking changes
before upgrading major versions.

---

## Troubleshooting

### `No module named 'kalmanbox'`

Ensure you are running Python from the same environment where you installed
the package:

```bash
which python          # shows active interpreter
pip show kalmanbox    # confirms install and location
```

If they point to different locations, activate your virtual environment
before running:

```bash
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows PowerShell
```

### NumPy / SciPy version conflicts

If another package in your environment pins an older NumPy, you may see:

```
ERROR: pip's dependency resolver does not currently take into account
all the packages that are installed.
```

Create a fresh virtual environment and install `kalmanbox` first:

```bash
python -m venv .venv && source .venv/bin/activate
pip install kalmanbox
```

### Numba JIT fails on Apple Silicon (M-series)

On `arm64` Macs, some Numba versions require Rosetta or a native arm64 build.
If you see a `CompilationError`, try:

```bash
pip install "numba>=0.59"
```

Or disable Numba entirely for the session:

```python
import os
os.environ["KALMANBOX_DISABLE_NUMBA"] = "1"
import kalmanbox
```

### `pip install` is slow

Use `uv` for faster dependency resolution:

```bash
pip install uv
uv pip install kalmanbox
```

---

## Next steps

- [Quickstart](quickstart.md) — fit your first model in under 10 minutes
- [Key Concepts](key-concepts.md) — understand the state-space framework
- [Ecosystem](ecosystem.md) — how `kalmanbox` powers the NodesEcon stack
