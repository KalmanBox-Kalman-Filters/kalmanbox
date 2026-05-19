# Library Comparisons

This page compares kalmanbox against the most widely used Python
state-space libraries and the R package KFAS. Timings are wall-clock
medians for fitting a Local Level model with MLE to a series of
$T = 10\,000$ observations on the benchmark hardware described in
the [overview](index.md).

## Feature matrix

| Feature                        | **kalmanbox** | statsmodels | pykalman | filterpy | KFAS (R) |
|-------------------------------|:-------------:|:-----------:|:--------:|:--------:|:--------:|
| Standard Kalman filter         | yes           | yes         | yes      | yes      | yes      |
| RTS smoother                   | yes           | yes         | yes      | no       | yes      |
| Square-root filter             | yes           | partial     | no       | no       | yes      |
| Information filter             | yes           | no          | no       | no       | yes      |
| EKF / UKF                      | yes           | no          | yes      | yes      | no       |
| EnKF                           | yes           | no          | no       | yes      | no       |
| Diffuse initialisation         | yes           | yes         | no       | no       | yes      |
| Missing observations           | yes           | yes         | yes      | no       | yes      |
| EM estimation                  | yes           | yes         | yes      | no       | yes      |
| Bayesian / Gibbs / FFBS        | yes           | no          | no       | no       | partial  |
| Dynamic Factor Model           | yes           | yes         | no       | no       | no       |
| Time-Varying Parameters        | yes           | yes         | no       | no       | yes      |
| Built-in structural models     | yes           | partial     | no       | no       | partial  |
| NodesEcon ecosystem integration| yes           | —           | —        | —        | —        |
| Type hints / pyright strict    | yes           | partial     | no       | no       | —        |

## Speed comparison

Relative to kalmanbox (Numba backend, lower is faster):

| Library            | Relative time | Notes                              |
|--------------------|:-------------:|------------------------------------|
| **kalmanbox (Numba)**  | 1.0×      | Baseline                           |
| **kalmanbox (Python)** | 6.8×      | Pure-Python fallback               |
| statsmodels        | 8.2×          | Cython core; overhead from kwargs  |
| pykalman           | 14.1×         | Pure NumPy; no JIT                 |
| filterpy           | 11.4×         | Pure Python; loop-based            |
| KFAS (R)           | 3.1×          | Fortran core; R overhead excluded  |

!!! note "Methodology"
    Each library is called with equivalent model settings (diffuse
    initialisation, same convergence tolerance). No benchmark favours
    kalmanbox by using sub-optimal settings in competing libraries.
    Contributions and corrections are welcome via GitHub issues.

## Numerical stability

| Library            | Double precision | Cholesky enforce | Near-singular H handled |
|--------------------|:----------------:|:----------------:|:-----------------------:|
| kalmanbox          | yes              | yes (SQR option) | yes (nugget + warning)  |
| statsmodels        | yes              | no               | partial                 |
| pykalman           | yes              | no               | no                      |
| filterpy           | yes              | optional         | no                      |
| KFAS               | yes              | yes              | yes                     |

The Square-Root filter in kalmanbox maintains the Cholesky factor
$S_t = \sqrt{P_t}$ throughout, preventing covariance matrices from
becoming indefinite through rounding errors.

## Bayesian support

| Library            | MCMC | Gibbs / FFBS | Priors API | Stan/PyMC interface |
|--------------------|:----:|:------------:|:----------:|:-------------------:|
| kalmanbox          | Gibbs + FFBS | yes    | yes        | planned             |
| statsmodels        | no   | no           | no         | no                  |
| pykalman           | no   | no           | no         | no                  |
| filterpy           | no   | no           | no         | no                  |
| KFAS               | partial (coda) | partial | no      | no                  |

kalmanbox includes a native Gibbs sampler with Forward-Filtering
Backward-Sampling (FFBS) for state-space models with conjugate priors.
For full MCMC flexibility — custom likelihoods, hierarchical
structures, non-conjugate priors — Stan or PyMC remain the better
choice. A thin bridge layer (`kalmanbox.bayesian.stan_bridge`) is on
the roadmap.

## Where kalmanbox is weaker

- **Full MCMC flexibility**: dedicated probabilistic programming
  languages (Stan, PyMC, NumPyro) offer arbitrary likelihoods,
  automatic differentiation, and HMC sampling that Gibbs cannot match.
  kalmanbox Gibbs is targeted at standard conjugate state-space
  problems only.
- **Non-Gaussian observation models**: EKF and UKF are available, but
  for particle filters see
  [particlefilterbox](https://github.com/nodesecon/particlefilterbox).
- **Very high-dimensional state spaces** ($m > 100$): the $O(m^3)$
  complexity of the standard KF becomes prohibitive. Ensemble
  methods (EnKF) help, but dedicated large-scale libraries (e.g.,
  DART, Verdant) are better suited.
- **R ecosystem**: KFAS and dlm have deeper integration with R's
  modelling ecosystem. kalmanbox is Python-first.

## Ecosystem integration advantage

kalmanbox is the foundational layer of the **NodesEcon ecosystem**:

- [chronobox](https://github.com/nodesecon/chronobox) — time series
  I/O, alignment, and preprocessing feeds directly into kalmanbox models.
- [forecastbox](https://github.com/nodesecon/forecastbox) — wraps
  kalmanbox state-space models in a unified forecasting API with
  cross-validation and backtesting.
- [particlefilterbox](https://github.com/nodesecon/particlefilterbox)
  — non-Gaussian, nonlinear filtering that shares kalmanbox's model
  specification API.

If your workflow already uses any of these packages, kalmanbox is the
natural state-space choice.

## Related

- [Performance benchmarks](performance.md)
- [Memory profile](memory.md)
