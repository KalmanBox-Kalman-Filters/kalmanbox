# Benchmarks

This section presents systematic performance measurements of kalmanbox across
the dimensions that matter most in practice: observation count, state
dimension, filter variant, and memory footprint. All benchmarks follow a
rigorous methodology to ensure fair and reproducible comparisons.

<div class="grid cards" markdown>

-   :material-speedometer:{ .lg .middle } **Kalman Filter**

    ---

    Wall-clock timing for the KalmanFilter forward pass and RTS smoother
    versus statsmodels, pykalman, and filterpy across series lengths,
    state dimensions, and observation counts.

    [:octicons-arrow-right-24: Kalman benchmarks](kalman.md)

-   :material-chart-line:{ .lg .middle } **Structural Models**

    ---

    BSM, UCM, and LocalLevel comparison with statsmodels
    `UnobservedComponents`. MLE convergence speed and accuracy.

    [:octicons-arrow-right-24: Structural benchmarks](structural.md)

-   :material-graph:{ .lg .middle } **Dynamic Factor Model**

    ---

    DFM scalability across number of series and factors. EM vs. two-step
    estimation. Comparison with statsmodels `DynamicFactorMQ`.

    [:octicons-arrow-right-24: DFM benchmarks](dfm.md)

-   :material-scale-balance:{ .lg .middle } **Library Comparison**

    ---

    Summary table: features, speed, memory, usability, and ecosystem
    integration across kalmanbox, statsmodels, pykalman, and filterpy.

    [:octicons-arrow-right-24: Full comparison](comparison.md)

</div>

---

## Benchmarking Methodology

### Timing measurement

All wall-clock timings are measured using Python's `time.perf_counter()`
with the following protocol:

1. **Warm-up run** — one untimed pass to trigger Numba JIT compilation and
   OS page-in of data. Ensures subsequent runs reflect steady-state performance.
2. **Repeated runs** — each benchmark executes **20 independent trials**
   with a fresh random seed offset per trial.
3. **Reported statistic** — the **median** of the 20 trials. The median is
   more robust than the mean when occasional OS interruptions inflate individual
   runs. P5 and P95 are also recorded to characterise variance.
4. **Isolation** — each benchmark runs in a separate process with
   `subprocess.run` to prevent cross-contamination from Python's garbage
   collector or JIT state.

### Memory measurement

Peak resident set size (RSS) is sampled via `tracemalloc` and
`resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` at 10 ms intervals
during each benchmark run. The reported figure is the maximum RSS during
the filter pass, excluding model initialisation.

### Accuracy metrics

Numerical accuracy is assessed relative to a high-precision reference
computed in Python's `mpmath` at 50 decimal digits. Two metrics are reported:

- **RMSE of filtered states** — root mean squared error of filtered state
  means relative to the reference.
- **Log-likelihood error** — absolute difference in total log-likelihood.

### Data generation

All benchmarks use synthetic data generated from the true model to guarantee
a known ground truth. Data is generated with a fixed seed (42) for
reproducibility.

---

## Test Hardware and Software

| Component      | Specification                                    |
|----------------|--------------------------------------------------|
| **CPU**        | Intel Core i7-1365U (12th gen, 10 cores, 5.2 GHz boost) |
| **RAM**        | 32 GB DDR5-4800                                  |
| **OS**         | Ubuntu 22.04 LTS (kernel 5.15)                   |
| **Python**     | 3.11.9                                           |
| **NumPy**      | 1.26.4 (OpenBLAS 0.3.26)                        |
| **SciPy**      | 1.12.0                                           |
| **Numba**      | 0.59.1 (LLVM 14.0)                              |
| **kalmanbox**  | 0.4.0                                            |
| **statsmodels**| 0.14.2                                           |
| **pykalman**   | 0.9.7                                            |
| **filterpy**   | 1.4.5                                            |

!!! note "Reproducibility"
    All benchmark scripts are in `scripts/benchmarks/` of the repository.
    Run `python scripts/benchmarks/run_all.py --output results/` to reproduce
    on your hardware. Results are written as CSV and rendered with
    `scripts/benchmarks/plot_results.py`.

---

## Benchmark Dimensions

| Dimension              | Values tested                          |
|------------------------|----------------------------------------|
| Series length $T$      | 100; 1 000; 10 000; 100 000            |
| State count $m$        | 1; 5; 10; 50                           |
| Observation count $p$  | 1; 3; 10                               |
| Filter variant         | Standard KF; Square-Root; Information  |
| Estimation method      | Filter only; Filter + Smooth; MLE      |
| Backend                | NumPy (pure Python); Numba JIT         |

---

## Metrics Summary

| Metric             | Symbol          | Interpretation                                        |
|--------------------|-----------------|-------------------------------------------------------|
| Median wall time   | $\tilde{t}$     | Typical execution time in milliseconds                |
| P5–P95 range       | $[t_{5}, t_{95}]$ | Timing variability across 20 trials               |
| Peak RSS           | $M_{\max}$      | Maximum resident memory in MB during the filter pass  |
| State RMSE         | $\text{RMSE}(\hat{\alpha})$ | Accuracy vs. high-precision reference    |
| Log-lik error      | $|\ell - \ell^*|$ | Log-likelihood discrepancy from reference          |

---

## Quick Reference: Speed Winners

| Scenario                       | Fastest choice                       | Speedup vs. kalmanbox NumPy |
|--------------------------------|--------------------------------------|-----------------------------|
| $T=10\,000$, $m=1$, filter     | kalmanbox Numba                      | 7.4×                        |
| $T=10\,000$, $m=5$, filter     | kalmanbox Numba                      | 6.8×                        |
| $T=10\,000$, MLE (10 restarts) | kalmanbox Numba                      | 8.1×                        |
| $T=100\,000$, $m=1$, filter    | kalmanbox Numba                      | 7.9×                        |
| $m=50$, any $T$                | kalmanbox InformationFilter (Numba)  | 2.1× over standard KF Numba |
| DFM $r=5$, $N=20$ series       | kalmanbox EM (Numba)                 | 4.3× over statsmodels       |
