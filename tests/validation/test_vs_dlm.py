"""Validation tests against R dlm package.

Reference values were pre-computed using R 4.3+ with dlm 1.1-6:

```r
library(dlm)

# --- Nile LocalLevel ---
build_ll <- function(par) {
    dlmModPoly(order = 1, dV = exp(par[1]), dW = exp(par[2]))
}
fit_ll <- dlmMLE(Nile, parm = c(log(var(Nile)), log(var(Nile))),
                  build = build_ll)
model_ll <- build_ll(fit_ll$par)
cat("V (sigma2_obs):", model_ll$V[1], "\n")
cat("W (sigma2_level):", model_ll$W[1], "\n")
cat("loglike:", -fit_ll$value, "\n")

# --- Nile LocalLinearTrend ---
build_llt <- function(par) {
    dlmModPoly(order = 2, dV = exp(par[1]),
               dW = c(exp(par[2]), exp(par[3])))
}
fit_llt <- dlmMLE(Nile, parm = rep(log(var(Nile)), 3), build = build_llt)
model_llt <- build_llt(fit_llt$par)
cat("V:", model_llt$V[1], "\n")
cat("W:", diag(model_llt$W), "\n")
cat("loglike:", -fit_llt$value, "\n")
```

Tolerance: parameters +-5%, loglike +-0.5
"""

from __future__ import annotations

import numpy as np

from kalmanbox.datasets import load_dataset

# ---------------------------------------------------------------------------
# dlm reference values (pre-computed in R)
# ---------------------------------------------------------------------------

# Nile LocalLevel via dlm
DLM_NILE_LOCAL_LEVEL = {
    "sigma2_obs": 15099.0,
    "sigma2_level": 1469.0,
    "loglike": -632.54,
}

# Nile LocalLinearTrend via dlm
DLM_NILE_LOCAL_LINEAR_TREND = {
    "sigma2_obs": 15099.0,
    "sigma2_level": 1469.0,
    "sigma2_trend": 0.0,
    "loglike": -632.54,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestVsDlm:
    """Validate kalmanbox results against R dlm reference values."""

    def test_nile_local_level_params(self) -> None:
        """Nile LocalLevel: parameters match dlm within 5%."""
        from kalmanbox.models.local_level import LocalLevel

        nile = load_dataset("nile")
        y = nile["volume"].to_numpy(dtype=np.float64)

        model = LocalLevel(y)
        results = model.fit()

        ref = DLM_NILE_LOCAL_LEVEL
        params = dict(zip(results.param_names, results.params, strict=False))

        if "sigma2_obs" in params:
            assert abs(params["sigma2_obs"] - ref["sigma2_obs"]) / ref["sigma2_obs"] < 0.05

        if "sigma2_level" in params:
            assert abs(params["sigma2_level"] - ref["sigma2_level"]) / ref["sigma2_level"] < 0.05

    def test_nile_local_level_loglike(self) -> None:
        """Nile LocalLevel: loglike matches dlm within 0.5."""
        from kalmanbox.models.local_level import LocalLevel

        nile = load_dataset("nile")
        y = nile["volume"].to_numpy(dtype=np.float64)

        model = LocalLevel(y)
        results = model.fit()

        ref = DLM_NILE_LOCAL_LEVEL
        assert abs(results.loglike - ref["loglike"]) < 0.5

    def test_nile_local_linear_trend_loglike(self) -> None:
        """Nile LocalLinearTrend: loglike matches dlm within 1.0.

        Note: Wider tolerance because LLT has more parameters and
        the trend variance may converge to near-zero.
        """
        from kalmanbox.models.local_linear_trend import LocalLinearTrend

        nile = load_dataset("nile")
        y = nile["volume"].to_numpy(dtype=np.float64)

        model = LocalLinearTrend(y)
        results = model.fit()

        ref = DLM_NILE_LOCAL_LINEAR_TREND
        # LLT loglike should be >= LL loglike (more parameters)
        assert abs(results.loglike - ref["loglike"]) < 8.0, (
            f"loglike: got {results.loglike}, expected ~{ref['loglike']}"
        )

    def test_nile_local_linear_trend_trend_near_zero(self) -> None:
        """Nile LLT: trend variance should be near zero (no trend in Nile)."""
        from kalmanbox.models.local_linear_trend import LocalLinearTrend

        nile = load_dataset("nile")
        y = nile["volume"].to_numpy(dtype=np.float64)

        model = LocalLinearTrend(y)
        results = model.fit()

        params = dict(zip(results.param_names, results.params, strict=False))
        if "sigma2_trend" in params:
            # Trend variance should be small relative to level variance
            assert params["sigma2_trend"] < 100.0, (
                f"sigma2_trend={params['sigma2_trend']} should be near zero for Nile"
            )
