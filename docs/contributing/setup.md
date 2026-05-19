# Development Setup

## Prerequisites

- Python 3.11 or later
- `git`
- (Optional) [Numba](https://numba.readthedocs.io/) for JIT benchmarks

## Step 1 — Fork and clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-username>/kalmanbox.git
cd kalmanbox
git remote add upstream https://github.com/nodesecon/kalmanbox.git
```

## Step 2 — Create a virtual environment

Use Python 3.11+. `venv` is recommended to keep your system Python
clean:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python --version                   # confirm: Python 3.11.x
```

## Step 3 — Install in editable mode with dev extras

```bash
pip install --upgrade pip
pip install -e ".[dev,docs]"
```

The `dev` extra installs:

| Tool          | Purpose                            |
|---------------|------------------------------------|
| `pytest`      | Test runner                        |
| `pytest-cov`  | Coverage reports                   |
| `hypothesis`  | Property-based testing             |
| `mutmut`      | Mutation testing                   |
| `ruff`        | Linting and formatting             |
| `pyright`     | Static type checking               |
| `bandit`      | Security linting                   |
| `interrogate` | Docstring coverage                 |
| `memory_profiler` | Memory profiling              |
| `pre-commit`  | Git hook management                |

The `docs` extra adds MkDocs, Material theme, and mkdocstrings.

## Step 4 — Install pre-commit hooks

```bash
pre-commit install
```

The hooks run automatically on `git commit`. To run them manually
against all files:

```bash
pre-commit run --all-files
```

Hooks configured in `.pre-commit-config.yaml`:

- `ruff` — lint + auto-fix
- `ruff-format` — code formatting
- `pyright` — type checks (non-blocking warning mode)
- `interrogate` — enforce docstring coverage ≥ 90 %
- `bandit` — security checks

## Step 5 — Run tests to verify setup

```bash
pytest tests/ -q
```

All tests should pass. If you see import errors, confirm that the
editable install succeeded (`pip show kalmanbox` should point to your
local clone).

For a coverage report:

```bash
pytest --cov=kalmanbox --cov-report=term-missing tests/
```

Target: ≥ 90 % line coverage. New code should not reduce coverage.

## Step 6 — Start the documentation server

```bash
mkdocs serve
```

The docs server starts at `http://127.0.0.1:8000` with **live reload**.
Edits to any `.md` file or the `mkdocs.yml` configuration are reflected
immediately.

!!! tip "Keeping your fork up to date"
    Before starting a new branch, sync with upstream:
    ```bash
    git fetch upstream
    git merge upstream/main
    ```
