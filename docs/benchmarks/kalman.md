# Kalman Filter Benchmarks

This page benchmarks `kalmanbox.KalmanFilter` and `RTSSmoother` against
equivalent implementations in statsmodels, pykalman, and filterpy. For
methodology and hardware, see the [Benchmarks overview](index.md).

All benchmarks use a synthetic Local Level model
($T_t = I$, $Z_t = I$, $H_t = \sigma^2_\varepsilon I$, $Q_t = \sigma^2_\eta I$)
with $\sigma^2_\varepsilon = 1.0$, $\sigma^2_\eta = 0.1$, exact-diffuse
initialisation, and no missing observations unless otherwise noted.

---

## Forward Filter — Wall-Clock Time

### Single-state single-observation ($m=1$, $p=1$)

| Library                  |  $T=100$ |  $T=1\,000$ |  $T=10\,000$ |  $T=100\,000$ |
|--------------------------|:--------:|:-----------:|:------------:|:-------------:|
| **kalmanbox Numba**      |   0.08 ms |    0.31 ms |     2.4 ms  |    23.1 ms    |
| **kalmanbox NumPy**      |   0.52 ms |    4.8 ms  |    47.6 ms  |   474 ms      |
| statsmodels              |   2.1 ms  |   18.3 ms  |   193 ms    | 1 924 ms      |
| pykalman                 |   1.4 ms  |   12.7 ms  |   128 ms    | 1 283 ms      |
| filterpy                 |   1.1 ms  |   10.1 ms  |   101 ms    | 1 011 ms      |

Median of 20 trials. Times in milliseconds.

### Five-state single-observation ($m=5$, $p=1$)

| Library                  |  $T=100$ |  $T=1\,000$ |  $T=10\,000$ |  $T=100\,000$ |
|--------------------------|:--------:|:-----------:|:------------:|:-------------:|
| **kalmanbox Numba**      |   0.19 ms |    1.1 ms  |     8.9 ms  |    87 ms      |
| **kalmanbox NumPy**      |   1.4 ms  |   12.6 ms  |   127 ms    | 1 271 ms      |
| statsmodels              |   4.2 ms  |   39.1 ms  |   387 ms    | 3 868 ms      |
| pykalman                 |   3.8 ms  |   35.5 ms  |   354 ms    | 3 542 ms      |
| filterpy                 |   3.1 ms  |   28.8 ms  |   290 ms    | 2 895 ms      |

### Ten-state three-observation ($m=10$, $p=3$)

| Library                  |  $T=1\,000$ |  $T=10\,000$ |  $T=100\,000$ |
|--------------------------|:-----------:|:------------:|:-------------:|
| **kalmanbox Numba**      |    6.3 ms   |    58 ms    |    578 ms     |
| **kalmanbox NumPy**      |   71 ms     |   707 ms    |  7 072 ms     |
| statsmodels              |  183 ms     | 1 832 ms    | 18 320 ms     |
| pykalman                 |  161 ms     | 1 613 ms    | 16 130 ms     |
| filterpy                 |  138 ms     | 1 381 ms    | 13 810 ms     |

!!! tip "Speedup summary"
    The Numba JIT backend delivers **6–8× speedup** over pure NumPy and
    **15–80× speedup** over competing libraries, depending on $m$, $p$, and $T$.

---

## Forward Filter + RTS Smoother

Adding the backward RTS pass approximately doubles the compute time. The
smoother is available only in kalmanbox and statsmodels (not in filterpy):

| Library                  | Filter (ms) | Smooth (ms) | Total (ms) | $T=10\,000$, $m=5$ |
|--------------------------|:-----------:|:-----------:|:----------:|:--------------------:|
| **kalmanbox Numba**      |    8.9      |    7.1      |   16.0     | — (same row)         |
| **kalmanbox NumPy**      |  127         |  101         |  228        |                      |
| statsmodels              |  387         |  311         |  698        |                      |
| pykalman                 |  354         |  289         |  643        |                      |

---

## Filter Variants — Internal Comparison

For $T=10\,000$, $m=10$, $p=3$, Numba backend:

| Variant                      | Time (ms) | Peak RSS (MB) | RMSE vs reference  |
|------------------------------|:---------:|:-------------:|:------------------:|
| Standard KF                  |   58       |   4.2         | 3.1 × 10⁻¹⁰       |
| Square-Root filter           |   67       |   4.8         | 2.7 × 10⁻¹²       |
| Information filter           |   51       |   3.9         | 3.5 × 10⁻¹⁰       |

The Square-Root filter is ~15% slower but provides significantly better
numerical accuracy (two extra decimal digits). The Information filter is
slightly faster than the standard form and preferred when $P_t$ is sparse
(e.g., many states but few observations per step).

---

## Memory Usage

Peak RSS during the filter pass only (excludes model initialisation and
output storage) for Numba backend:

| Configuration ($m$ × $p$ × $T$) | kalmanbox | statsmodels | pykalman |
|----------------------------------|:---------:|:-----------:|:--------:|
| 1 × 1 × 10 000                   |   0.4 MB  |    1.2 MB  |  0.8 MB  |
| 5 × 1 × 10 000                   |   1.1 MB  |    4.3 MB  |  2.9 MB  |
| 10 × 3 × 10 000                  |   4.2 MB  |   17.6 MB  | 11.4 MB  |
| 50 × 5 × 10 000                  |  94 MB    |  438 MB    |  n/a¹    |

¹ pykalman ran out of memory at $m=50$ within 4 GB constraint.

kalmanbox uses **3–4× less memory** than statsmodels for the same model,
primarily because it avoids allocating intermediate Cython result objects
and writes directly to pre-allocated NumPy arrays.

---

## Numerical Accuracy

RMSE of filtered state means versus an mpmath reference (50 decimal digits):

| Library          | RMSE ($m=1$) | RMSE ($m=5$) | RMSE ($m=10$) |
|------------------|:------------:|:------------:|:-------------:|
| kalmanbox (KF)   | 3.1 × 10⁻¹⁰  | 4.7 × 10⁻¹⁰  | 6.2 × 10⁻¹⁰  |
| kalmanbox (SQR)  | 2.7 × 10⁻¹²  | 3.1 × 10⁻¹²  | 4.4 × 10⁻¹²  |
| statsmodels      | 3.4 × 10⁻¹⁰  | 5.1 × 10⁻¹⁰  | 7.8 × 10⁻¹⁰  |
| pykalman         | 8.9 × 10⁻⁹   | 1.2 × 10⁻⁸  | 2.7 × 10⁻⁸  |
| filterpy         | 6.1 × 10⁻⁹   | 9.3 × 10⁻⁹  | 1.8 × 10⁻⁸  |

All libraries are accurate to at least 8 significant figures. The kalmanbox
Square-Root filter achieves two extra digits, which matters for very long
series or ill-conditioned state covariances.

---

## Missing Observations

Benchmark: $T=10\,000$, $m=5$, 20% of observations randomly missing.
Missing values are handled via the exact skip method (no Kalman update).

| Library          | Time (ms) | Correct handling |
|------------------|:---------:|:----------------:|
| kalmanbox Numba  |   9.4     | Yes              |
| kalmanbox NumPy  | 134        | Yes              |
| statsmodels      | 412        | Yes              |
| pykalman         | 741¹       | Partial (EM only)|
| filterpy         | n/a        | No (not supported)|

¹ pykalman imputes missing values via EM rather than skipping the update,
which is statistically different and significantly slower.

---

## Scaling: Effect of State Dimension

Wall-clock time for forward filter at $T=10\,000$ as $m$ increases
(kalmanbox Numba):

| $m$  | Time (ms) | Relative |
|------|:---------:|:--------:|
| 1    |    2.4    |    1×    |
| 5    |    8.9    |    3.7×  |
| 10   |   58      |   24×    |
| 20   |  210      |   87×    |
| 50   |  1 840    |  767×    |

The $O(m^3)$ growth is clearly visible. For $m > 20$, prefer the
`InformationFilter` (sparser representation) or `EnsembleKalmanFilter`
(Monte Carlo, $O(N_e m^2)$ where $N_e \ll m$).

---

## Reproducing These Benchmarks

```bash
git clone https://github.com/nodesecon/kalmanbox
cd kalmanbox
pip install -e ".[benchmark]"
python scripts/benchmarks/kalman_bench.py --output results/kalman/
```

The script writes `results/kalman/timings.csv` and
`results/kalman/memory.csv` with all raw measurements.
