# ARIMA via State Space

Represent ARIMA models in state-space form for unified estimation.

## Usage

```python
from kalmanbox.models.arima_ssm import ARIMASSM

model = ARIMASSM(y, order=(1, 1, 1))
results = model.fit()
```

## Advantages over traditional ARIMA

- Unified framework with other state-space models
- Natural handling of missing data
- Easy extension to seasonal ARIMA
