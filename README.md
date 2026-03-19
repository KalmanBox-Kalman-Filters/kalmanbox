# kalmanbox

State-space models and Kalman filtering for time series analysis.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from kalmanbox import LocalLevel
from kalmanbox.datasets import load_dataset

nile = load_dataset('nile')
model = LocalLevel(nile['volume'])
results = model.fit()
print(results.summary())
```

## License

MIT
