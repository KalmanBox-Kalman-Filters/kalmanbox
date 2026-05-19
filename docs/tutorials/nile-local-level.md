# Tutorial — Nile river with Local Level

The classic Nile dataset (Cobb 1978; Durbin & Koopman 2012) records the
annual volume of the Nile at Aswan from 1871 to 1970. The series shows
a clear **structural break** around 1898 (construction of the Aswan
dam) followed by a lower mean level — making it the perfect testbed
for a Local Level model.

## 1. Load

```python
import matplotlib.pyplot as plt
from kalmanbox import LocalLevel
from kalmanbox.datasets import load_dataset

nile = load_dataset("nile")          # pandas DataFrame
y = nile["volume"]
y.plot(figsize=(9, 3), title="Nile annual flow at Aswan")
```

## 2. Fit

```python
model = LocalLevel(y)
results = model.fit()
print(results.summary())
```

Typical output:

```
                    Local Level Results
==============================================================
Log-likelihood              -632.55
AIC                         1269.10
BIC                         1274.31
sigma2_eta                   1469.1   (level innovation)
sigma2_eps                  15099.1   (measurement noise)
==============================================================
```

## 3. Smooth

```python
sm = results.smooth()
mu = sm.a_smoothed[:, 0]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(y.index, y.values, "k.", label="observed")
ax.plot(y.index, mu, "C0",   label=r"$\hat\mu_{t|n}$")
ax.fill_between(y.index,
                mu - 1.96 * sm.P_smoothed[:, 0, 0]**0.5,
                mu + 1.96 * sm.P_smoothed[:, 0, 0]**0.5,
                alpha=0.2, color="C0")
ax.legend(); ax.set_title("Smoothed level — Nile")
```

You should see a smooth descent from ~1100 to ~850 between 1898 and
1905, exactly tracking the construction of the dam.

## 4. Forecast

```python
fc = results.forecast(steps=20)

ax = y.plot(figsize=(9, 4), label="observed")
ax.plot(fc.index, fc["mean"], "C1", label="forecast")
ax.fill_between(fc.index, fc["lower_95"], fc["upper_95"],
                color="C1", alpha=0.2)
ax.legend()
```

The forecast is a flat line — Local Level has no trend or seasonal —
with bands that **fan out** as $\sigma_\eta^2$ accumulates.

## 5. Residual diagnostics

```python
from kalmanbox.diagnostics import residual_diagnostics
print(residual_diagnostics(results))
```

Ljung–Box should comfortably pass; Jarque–Bera has a hint of leptokurtosis
caused by the structural break.

## What we learned

- Local Level captures the **slowly drifting mean** without imposing a
  trend.
- The smoother resolves the 1898 break into a smooth transition because
  Local Level cannot represent abrupt jumps; for that, you'd need a
  Markov-switching extension.
- AIC/BIC for this model give a baseline against which you can compare
  the next tutorial's models.

## Next

- [Airline passengers — BSM](airline-bsm.md)
- [User guide: Local Level](../user-guide/structural/local-level.md)
