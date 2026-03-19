# Quickstart

This guide walks you through fitting your first state-space model with KalmanBox.

## 1. Load data

KalmanBox includes 19 built-in datasets:

```python
from kalmanbox.datasets import load_dataset, list_datasets

# See available datasets
print(list_datasets())

# Load the classic Nile dataset
nile = load_dataset('nile')
print(nile.head())
```

## 2. Fit a model

```python
from kalmanbox import LocalLevel

model = LocalLevel(nile['volume'])
results = model.fit()
```

## 3. View results

```python
print(results.summary())
print(f"Log-likelihood: {results.loglike:.2f}")
print(f"AIC: {results.aic:.2f}")
```

## 4. Forecast

```python
forecast = results.forecast(steps=10)
print(f"Forecast mean: {forecast['mean'][:5]}")
print(f"95% CI: [{forecast['lower_95'][0]:.1f}, {forecast['upper_95'][0]:.1f}]")
```

## 5. Compare models

```python
from kalmanbox.experiment import KalmanExperiment

exp = KalmanExperiment(nile['volume'])
exp.fit_all_models([
    ('LocalLevel', {}),
    ('LocalLinearTrend', {}),
])
comparison = exp.compare_models()
print(comparison.ranking())
print(f"Best model: {comparison.best_model()}")
```

## 6. CLI

```bash
# Estimate from command line
kalmanbox estimate --model local_level --data nile.csv --output results.json

# Get model info
kalmanbox info --model local_level

# Forecast
kalmanbox forecast --model local_level --data nile.csv --steps 10
```

## Next steps

- [Key Concepts](key-concepts.md) - Understand state-space models
- [Local Level Guide](../user-guide/local-level/overview.md) - Deep dive into the simplest model
- [BSM Guide](../user-guide/bsm/overview.md) - Seasonal models
