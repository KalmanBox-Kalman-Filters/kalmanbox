# CLI Reference

## Usage

```bash
kalmanbox [command] [options]
```

## Commands

### estimate

Estimate a state-space model from CSV data.

```bash
kalmanbox estimate --model MODEL --data FILE [--output FILE] [--seasonal-period N]
```

| Option | Required | Description |
|:-------|:---------|:------------|
| `--model` | Yes | Model type: local_level, local_linear_trend, bsm, ucm, arima, tvp |
| `--data` | Yes | Path to CSV data file |
| `--output` | No | Output JSON file (default: results.json) |
| `--seasonal-period` | No | Seasonal period (for bsm, ucm) |
| `--order` | No | ARIMA order as p,d,q (for arima) |
| `--exog` | No | Path to exogenous CSV (for tvp) |

### info

Show model information without fitting.

```bash
kalmanbox info --model MODEL [--seasonal-period N]
```

### forecast

Fit model and produce forecasts.

```bash
kalmanbox forecast --model MODEL --data FILE --steps N [--output FILE]
```
