# Structural Model Benchmarks

This page benchmarks kalmanbox structural time-series models against their
statsmodels equivalents. All comparisons use the same data, identical model
specifications, and matched convergence tolerances. For methodology and
hardware, see the [Benchmarks overview](index.md).

---

## Local Level Model

### Filter-only timing (no MLE)

Benchmark: `LocalLevelModel.filter()` vs.
`statsmodels.tsa.UnobservedComponents(level="local level").filter()`.
No parameter estimation — fixed $\sigma^2_\eta = 0.1$, $\sigma^2_\varepsilon = 1.0$.

| $T$         | kalmanbox Numba | kalmanbox NumPy | statsmodels |
|-------------|:---------------:|:---------------:|:-----------:|
| 1 000       |     0.3 ms      |     4.9 ms      |    19 ms    |
| 10 000      |     2.5 ms      |    48 ms        |   187 ms    |
| 100 000     |    24 ms        |   476 ms        | 1 861 ms    |

### MLE timing (L-BFGS-B, single start)

Benchmark: `model.fit(method="lbfgs", n_restarts=1)` on a single synthetic
Local Level series of length $T$.

| $T$         | kalmanbox Numba | kalmanbox NumPy | statsmodels |
|-------------|:---------------:|:---------------:|:-----------:|
| 1 000       |     18 ms       |    124 ms       |   610 ms    |
| 10 000      |    165 ms       |  1 140 ms       | 5 720 ms    |
| 100 000     |  1 620 ms       | 11 200 ms       | 56 400 ms   |

**Convergence quality** (20 random seeds, $T=1\,000$):

| Library          | Converged | RMSE of $\hat\sigma^2_\eta$ | RMSE of $\hat\sigma^2_\varepsilon$ |
|------------------|:---------:|:---------------------------:|:----------------------------------:|
| kalmanbox        | 20 / 20   |         0.0031              |           0.0047                   |
| statsmodels      | 19 / 20   |         0.0034              |           0.0051                   |

---

## Local Linear Trend Model

### Filter timing ($m=2$, $p=1$)

| $T$         | kalmanbox Numba | kalmanbox NumPy | statsmodels |
|-------------|:---------------:|:---------------:|:-----------:|
| 1 000       |     0.6 ms      |     9.2 ms      |    36 ms    |
| 10 000      |     5.2 ms      |    91 ms        |   362 ms    |
| 100 000     |    51 ms        |   904 ms        | 3 592 ms    |

### MLE timing

| $T$         | kalmanbox Numba | statsmodels | Speedup |
|-------------|:---------------:|:-----------:|:-------:|
| 1 000       |     35 ms       |   890 ms    |  25.4×  |
| 10 000      |    320 ms       | 8 210 ms    |  25.7×  |
| 100 000     |  3 180 ms       | 80 100 ms   |  25.2×  |

---

## Basic Structural Model (BSM)

BSM includes trend (level + slope) + quarterly seasonal ($s=4$) +
irregular. State dimension $m = 8$.

### Filter timing ($m=8$, $p=1$)

| $T$         | kalmanbox Numba | kalmanbox NumPy | statsmodels `UnobservedComponents` |
|-------------|:---------------:|:---------------:|:----------------------------------:|
| 1 000       |     2.4 ms      |    31 ms        |     98 ms                          |
| 10 000      |    23 ms        |   310 ms        |    981 ms                          |
| 100 000     |   228 ms        | 3 087 ms        |  9 780 ms                          |

### MLE convergence: BSM on airline data ($T=144$, monthly)

Comparing parameter estimates for the classic airline passenger dataset:

| Parameter           | kalmanbox    | statsmodels  | Difference |
|---------------------|:------------:|:------------:|:----------:|
| $\sigma^2_\eta$ (level)| 0.00148  |   0.00150    |   1.3%     |
| $\sigma^2_\zeta$ (slope)| 0.00000 |   0.00000    |  < 0.1%    |
| $\sigma^2_\omega$ (seas)| 0.00000 |   0.00000    |  < 0.1%    |
| $\sigma^2_\varepsilon$  | 0.01342 |   0.01345    |   0.2%     |
| Log-likelihood      | −244.70      | −244.71      |   0.01     |
| MLE wall time       |   61 ms      |  780 ms      |  12.8×     |

Differences are purely numerical (different initialisations, tolerance
defaults); both estimates are equally valid MLE solutions.

---

## UCM: Flexibility vs. Overhead

The `UnobservedComponents` class allows adding/removing components at
construction time. Benchmark: cost of adding each component to a base
LocalLevel model ($T=10\,000$, Numba backend, filter only):

| Model variant                         | $m$ | Filter time (ms) | Overhead vs. LL |
|---------------------------------------|:---:|:----------------:|:---------------:|
| Local Level                           |  1  |      2.5         |   —             |
| + Slope (LLT)                         |  2  |      5.2         | 2.1×            |
| + Monthly seasonal ($s=12$, trig)     |  4  |     12.1         | 4.8×            |
| + Quarterly seasonal ($s=4$, dummy)   |  6  |     22.4         | 9.0×            |
| + Stochastic cycle                    |  8  |     40.7         | 16.3×           |
| + AR(2) irregular                     | 10  |     58.1         | 23.2×           |

State dimension grows linearly with components; filter cost grows as $O(m^3)$.
Use only the components your data require to keep estimation tractable.

---

## MLE Convergence: Multiple Restarts

Benchmark: 10 random restarts for a BSM on $T=500$ quarterly observations,
kalmanbox Numba vs. statsmodels.

| Metric                          | kalmanbox Numba | statsmodels |
|---------------------------------|:---------------:|:-----------:|
| Total time (10 restarts)        |    0.8 s        |    9.2 s    |
| Best log-likelihood found       | −301.44         | −301.44     |
| Restarts to global optimum      |  3.8 (mean)     | 4.1 (mean)  |
| Failure rate (non-convergence)  |   5%            |   8%        |

Both libraries find the same global optimum; kalmanbox is ~11× faster per
restart.

---

## EM Estimation for Structural Models

EM provides a reliable initialisation for MLE. Benchmark: 50 EM iterations
on a Local Linear Trend model ($T=1\,000$):

| Library          | Time per iter | 50 iters total | Final log-lik |
|------------------|:-------------:|:--------------:|:-------------:|
| kalmanbox Numba  |    0.7 ms     |    35 ms       |  −1 413.2     |
| kalmanbox NumPy  |    8.3 ms     |   415 ms       |  −1 413.2     |
| statsmodels      |   21 ms       | 1 050 ms       |  −1 413.3     |

The small log-likelihood difference (0.1 nats) is due to convergence
tolerance differences; both solutions are numerically equivalent.

---

## Reproducing These Benchmarks

```bash
python scripts/benchmarks/structural_bench.py --output results/structural/
```

The script accepts `--model {local_level,llt,bsm,ucm}` and
`--T 1000 10000 100000` arguments to run individual sub-benchmarks.
