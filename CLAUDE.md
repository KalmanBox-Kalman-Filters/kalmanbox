# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

### Installation & Setup
```bash
pip install -e ".[dev]"        # Install in editable mode with dev dependencies
pre-commit install             # Set up git hooks
```

### Development Workflow
```bash
# Lint and format
ruff check kalmanbox/ tests/
ruff format kalmanbox/ tests/

# Type checking
pyright kalmanbox/

# Run tests
pytest                         # Run all tests
pytest tests/filters/          # Run specific test module
pytest -k test_kalman          # Run tests matching a name
pytest --cov=kalmanbox         # Run with coverage report

# Code quality checks
bandit -r kalmanbox/ -ll       # Security scan
radon cc kalmanbox/ -a -nc     # Complexity analysis
interrogate kalmanbox/ -v      # Docstring coverage

# Build documentation
mkdocs build                   # Build docs locally
mkdocs serve                   # Serve docs at http://localhost:8000
```

### Pre-commit Workflow
Pre-commit hooks run automatically on `git commit`. They check:
- Code formatting (ruff)
- Type checking (pyright)
- YAML/TOML syntax, trailing whitespace, file endings
- Large files (>500KB)

To run hooks manually: `pre-commit run --all-files`

### Quality Requirements (enforced in CI)
- Test coverage: ≥90%
- Cyclomatic complexity: ≤10 per function
- Docstring coverage: ≥95% (excluding `__init__` methods)
- Type checking: strict mode
- No security issues (bandit scan)

## Architecture

### Core Design
The library centers on **state-space models** for time series analysis. The architecture follows a three-level abstraction:

1. **Core abstractions** (`kalmanbox/core/`)
   - `StateSpaceModel`: Abstract base class for all models. Defines the interface for parameter handling, fitting, filtering, and smoothing.
   - `StateSpaceRepresentation`: Container for state-space matrices (T, Z, R, Q, H, c, d). Used internally by filters/smoothers.
   - `StateSpaceResults`: Holds estimation results and provides summary/diagnostic methods.

2. **Concrete models** (`kalmanbox/models/`)
   - All models inherit from `StateSpaceModel` and implement `_build_ssm()` to construct their `StateSpaceRepresentation` from parameters.
   - Examples: `LocalLevel`, `LocalLinearTrend`, `BasicStructuralModel` (BSM), `UnobservedComponents` (UCM), `ARIMA_SSM`, `DynamicFactorModel`, `TimeVaryingParameters`, `CustomStateSpace`.
   - Each model handles parameter constraints, transformations, and default settings.

3. **Inference engine** (filters + smoothers)
   - **Filters** (`kalmanbox/filters/`): Implement forward-pass algorithms
     - `KalmanFilter`: Standard linear Gaussian filter
     - `ExtendedKalmanFilter` (EKF), `UnscentedKalmanFilter` (UKF): For nonlinear systems
     - `EnsembleKalmanFilter` (EnKF): Ensemble-based approximation
     - `SquareRootKalmanFilter`: Numerically stable variant
     - `InformationFilter`: Information form (inverse covariance)
   - **Smoothers** (`kalmanbox/smoothers/`): Implement backward-pass/fixed-lag algorithms
     - `RTSSmoother`: Rauch-Tung-Striebel (standard two-pass smoother)
     - `FixedIntervalSmoother`, `FixedLagSmoother`: Variants with memory constraints
     - `DisturbanceSmoother`: Estimates state disturbances

### Model Fitting & Estimation
- `StateSpaceModel.fit()` performs Maximum Likelihood Estimation via `kalmanbox/estimation/mle.py`
- MLE uses the Kalman filter likelihood to optimize parameters
- Results returned in `StateSpaceResults` with standard errors, t-stats, and convergence info

### Supporting Modules
- **Diagnostics** (`kalmanbox/diagnostics/`): Residual analysis, stability checks, diagnostic tests
- **Visualization** (`kalmanbox/visualization/`): Plotting filters, smoothers, states, residuals
- **Reports** (`kalmanbox/reports/`): HTML/template-based reporting
- **Datasets** (`kalmanbox/datasets/`): Built-in example datasets (Nile flow, US macroeconomic data, etc.)
- **CLI** (`kalmanbox/cli/`): Command-line interface to kalmanbox commands
- **Simulation** (`kalmanbox/simulation/`): Generate synthetic state-space data
- **Utils** (`kalmanbox/utils/`): Matrix operations, Numba-accelerated cores, helpers

### Data Flow
```
User Code
  ↓
Model.__init__(endog) → StateSpaceModel with Kalman filter + RTS smoother
  ↓
Model.fit() → MLE optimization → StateSpaceResults
  ↓
Results.summary() / diagnostic_tests() / etc.
```

Within MLE, each iteration:
1. Build `StateSpaceRepresentation` from current parameters
2. Pass to Kalman filter for log-likelihood
3. Update parameters via optimizer
4. Repeat until convergence

## Convention Notes

### Parameter Naming in Filters/Smoothers
- Files in `filters/`, `smoothers/`, `models/`, and `estimation/` use mathematical variable names (T, Z, R, Q, H, c, d, α, η, ε) to match textbooks and papers.
- Per `pyproject.toml`, these files have linting exceptions for variable name style (N803, N806).

### Matrix Conventions
- State equation: `α_{t+1} = T @ α_t + R @ η_t + c` with `η_t ~ N(0, Q)`
- Observation: `y_t = Z @ α_t + ε_t + d` with `ε_t ~ N(0, H)`
- See `kalmanbox/core/representation.py` for the full system definition

## Testing & Coverage

- Test structure mirrors source: `tests/filters/`, `tests/models/`, `tests/estimation/`, etc.
- Use `pytest` with hypothesis for property-based testing (random parameter generation).
- Coverage threshold 90% enforced in CI.
- Integration tests should use realistic datasets from `kalmanbox.datasets`.

## Documentation

- User guides in `docs/user-guide/` (LocalLevel, BSM, UCM, DynamicFactor, TVP, ARIMA_SSM, Custom)
- API reference in `docs/api/`
- Theory background in `docs/theory/` (state-space basics, Kalman filter derivation)
- Examples in `examples/` (numbered 01–09, each with a README explaining the scenario)
- Built with MkDocs Material, strict mode enforced in CI

## Key Files to Know

- `kalmanbox/__init__.py`: Main exports (all public models and core classes)
- `kalmanbox/core/model.py`: `StateSpaceModel` abstract class—the foundation for all models
- `kalmanbox/core/representation.py`: `StateSpaceRepresentation` matrix container
- `kalmanbox/filters/kalman.py`: Core Kalman filter implementation
- `kalmanbox/smoothers/rts.py`: Standard RTS smoothing (backward pass)
- `kalmanbox/models/local_level.py`: Simplest concrete model example—good reference for implementing new models
- `kalmanbox/estimation/mle.py`: MLE optimization logic
- `pyproject.toml`: Project metadata, dependencies, linting/type-checking/test/doc build config
- `.pre-commit-config.yaml`: Git hooks (ruff format, ruff check, pyright)
- `.github/workflows/ci.yml`: CI test matrix (Python 3.11, 3.12), quality gates, docstring coverage
