"""Unobserved Components Model (UCM)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from kalmanbox.core.config import config
from kalmanbox.core.model import StateSpaceModel
from kalmanbox.core.representation import StateSpaceRepresentation


class UnobservedComponents(StateSpaceModel):
    """Unobserved Components Model (UCM).

    A configurable structural time series model that combines optional
    components: level, trend, seasonal, cycle, autoregressive, and
    exogenous regressors.

    Parameters
    ----------
    endog : NDArray
        Observed time series data.
    level : bool
        Include stochastic level component. Default True.
    trend : str
        Trend specification:
        - 'none': no trend
        - 'fixed': deterministic trend (sigma2_trend = 0)
        - 'stochastic': stochastic trend (default)
        - 'damped': damped stochastic trend
    seasonal : str
        Seasonal specification: 'none', 'dummy', or 'trigonometric'.
    seasonal_period : int | None
        Seasonal period (e.g., 12 for monthly, 4 for quarterly).
        Required if seasonal != 'none'.
    cycle : bool
        Include stochastic cycle. Default False.
    autoregressive : int
        AR order for the irregular component. Default 0.
    exog : NDArray | None
        Exogenous regressors, shape (nobs, k). Default None.
    """

    def __init__(
        self,
        endog: NDArray[np.float64],
        *,
        level: bool = True,
        trend: Literal["none", "fixed", "stochastic", "damped"] = "stochastic",
        seasonal: Literal["none", "dummy", "trigonometric"] = "none",
        seasonal_period: int | None = None,
        cycle: bool = False,
        autoregressive: int = 0,
        exog: NDArray[np.float64] | None = None,
    ) -> None:
        self._level = level
        self._trend = trend
        self._seasonal = seasonal
        self._seasonal_period = seasonal_period
        self._cycle = cycle
        self._ar_order = autoregressive
        self._exog = np.asarray(exog, dtype=np.float64) if exog is not None else None

        if seasonal != "none" and seasonal_period is None:
            msg = "seasonal_period is required when seasonal != 'none'"
            raise ValueError(msg)

        if not level and trend != "none":
            msg = "trend requires level=True"
            raise ValueError(msg)

        # Compute component layout (state indices)
        self._layout = self._compute_layout()

        super().__init__(endog)

    def _compute_layout(self) -> dict[str, dict[str, int]]:
        """Compute state vector layout: start index and size for each component."""
        layout: dict[str, dict[str, int]] = {}
        idx = 0

        # Level
        if self._level:
            layout["level"] = {"start": idx, "size": 1}
            idx += 1

        # Trend (slope)
        if self._trend in ("fixed", "stochastic", "damped"):
            layout["trend"] = {"start": idx, "size": 1}
            idx += 1

        # Seasonal
        if self._seasonal != "none":
            assert self._seasonal_period is not None
            s = self._seasonal_period
            if self._seasonal == "dummy":
                n_states = s - 1
            else:  # trigonometric
                n_harmonics = s // 2
                n_states = 2 * n_harmonics
                if s % 2 == 0:
                    n_states -= 1
            layout["seasonal"] = {"start": idx, "size": n_states}
            idx += n_states

        # Cycle
        if self._cycle:
            layout["cycle"] = {"start": idx, "size": 2}
            idx += 2

        # AR
        if self._ar_order > 0:
            layout["ar"] = {"start": idx, "size": self._ar_order}
            idx += self._ar_order

        # Exogenous (fixed coefficients)
        if self._exog is not None:
            k = self._exog.shape[1]
            layout["exog"] = {"start": idx, "size": k}
            idx += k

        layout["_total"] = {"start": 0, "size": idx}
        return layout

    @property
    def _k_states(self) -> int:
        return self._layout["_total"]["size"]

    @property
    def _n_noise_params(self) -> int:
        """Number of noise variance parameters."""
        n = 1  # sigma2_obs always
        if self._level:
            n += 1  # sigma2_level
        if self._trend in ("stochastic", "damped"):
            n += 1  # sigma2_trend
        if self._seasonal != "none":
            n += 1  # sigma2_seasonal
        if self._cycle:
            n += 3  # rho, lambda_c, sigma2_cycle
        if self._ar_order > 0:
            n += self._ar_order + 1  # AR coefficients + sigma2_ar
        return n

    @property
    def start_params(self) -> NDArray[np.float64]:
        """Initial guesses."""
        var = float(np.nanvar(self.endog))
        params: list[float] = []

        # sigma2_obs
        params.append(var / 4.0)

        # Level
        if self._level:
            params.append(var / 4.0)  # sigma2_level

        # Trend
        if self._trend in ("stochastic", "damped"):
            params.append(var / 8.0)  # sigma2_trend

        # Damped trend: phi_trend
        if self._trend == "damped":
            params.append(0.95)  # phi_trend

        # Seasonal
        if self._seasonal != "none":
            params.append(var / 8.0)  # sigma2_seasonal

        # Cycle
        if self._cycle:
            params.append(0.9)  # rho
            params.append(np.pi / 20)  # lambda_c
            params.append(var / 4.0)  # sigma2_cycle

        # AR
        if self._ar_order > 0:
            for i in range(self._ar_order):
                params.append(0.5 if i == 0 else 0.0)
            params.append(var / 4.0)  # sigma2_ar

        return np.array(params)

    @property
    def param_names(self) -> list[str]:
        names: list[str] = ["sigma2_obs"]

        if self._level:
            names.append("sigma2_level")

        if self._trend in ("stochastic", "damped"):
            names.append("sigma2_trend")

        if self._trend == "damped":
            names.append("phi_trend")

        if self._seasonal != "none":
            names.append("sigma2_seasonal")

        if self._cycle:
            names.extend(["rho", "lambda_c", "sigma2_cycle"])

        if self._ar_order > 0:
            for i in range(1, self._ar_order + 1):
                names.append(f"phi_ar_{i}")
            names.append("sigma2_ar")

        return names

    def transform_params(self, unconstrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform to constrained space."""
        constrained = unconstrained.copy()
        idx = 0

        # sigma2_obs: exp
        constrained[idx] = np.exp(unconstrained[idx])
        idx += 1

        # sigma2_level: exp
        if self._level:
            constrained[idx] = np.exp(unconstrained[idx])
            idx += 1

        # sigma2_trend: exp
        if self._trend in ("stochastic", "damped"):
            constrained[idx] = np.exp(unconstrained[idx])
            idx += 1

        # phi_trend: tanh -> (-1, 1), then shift to (0, 1)
        if self._trend == "damped":
            constrained[idx] = (np.tanh(unconstrained[idx]) + 1) / 2
            idx += 1

        # sigma2_seasonal: exp
        if self._seasonal != "none":
            constrained[idx] = np.exp(unconstrained[idx])
            idx += 1

        # Cycle: rho, lambda_c, sigma2_cycle
        if self._cycle:
            constrained[idx] = (np.tanh(unconstrained[idx]) + 1) / 2  # rho
            idx += 1
            # lambda_c: sigmoid * pi
            z = unconstrained[idx]
            sig = 1 / (1 + np.exp(-z)) if z >= 0 else np.exp(z) / (1 + np.exp(z))
            constrained[idx] = np.pi * sig
            idx += 1
            constrained[idx] = np.exp(unconstrained[idx])  # sigma2_cycle
            idx += 1

        # AR: phi coefficients (identity), sigma2_ar (exp)
        if self._ar_order > 0:
            idx += self._ar_order  # phi_ar: identity (no constraint)
            constrained[idx] = np.exp(unconstrained[idx])  # sigma2_ar
            idx += 1

        return constrained

    def untransform_params(self, constrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform to unconstrained space."""
        unconstrained = constrained.copy()
        idx = 0

        # sigma2_obs
        unconstrained[idx] = np.log(constrained[idx])
        idx += 1

        if self._level:
            unconstrained[idx] = np.log(constrained[idx])
            idx += 1

        if self._trend in ("stochastic", "damped"):
            unconstrained[idx] = np.log(constrained[idx])
            idx += 1

        if self._trend == "damped":
            unconstrained[idx] = np.arctanh(2 * constrained[idx] - 1)
            idx += 1

        if self._seasonal != "none":
            unconstrained[idx] = np.log(constrained[idx])
            idx += 1

        if self._cycle:
            unconstrained[idx] = np.arctanh(2 * constrained[idx] - 1)  # rho
            idx += 1
            p = constrained[idx] / np.pi
            unconstrained[idx] = np.log(p / (1 - p))  # lambda_c
            idx += 1
            unconstrained[idx] = np.log(constrained[idx])  # sigma2_cycle
            idx += 1

        if self._ar_order > 0:
            idx += self._ar_order  # phi_ar: identity
            unconstrained[idx] = np.log(constrained[idx])  # sigma2_ar
            idx += 1

        return unconstrained

    def _parse_params(self, params: NDArray[np.float64]) -> dict[str, float | NDArray[np.float64]]:
        """Parse the parameter vector into named components."""
        idx = 0
        parsed: dict[str, float | NDArray[np.float64]] = {
            "sigma2_obs": params[idx],
            "sigma2_level": 0.0,
            "sigma2_trend": 0.0,
            "phi_trend": 1.0,
            "sigma2_seasonal": 0.0,
            "rho": 0.0,
            "lambda_c": 0.0,
            "sigma2_cycle": 0.0,
            "phi_ar": np.array([]),
            "sigma2_ar": 0.0,
        }
        idx += 1

        if self._level:
            parsed["sigma2_level"] = params[idx]
            idx += 1

        if self._trend in ("stochastic", "damped"):
            parsed["sigma2_trend"] = params[idx]
            idx += 1

        if self._trend == "damped":
            parsed["phi_trend"] = params[idx]
            idx += 1

        if self._seasonal != "none":
            parsed["sigma2_seasonal"] = params[idx]
            idx += 1

        if self._cycle:
            parsed["rho"] = params[idx]
            parsed["lambda_c"] = params[idx + 1]
            parsed["sigma2_cycle"] = params[idx + 2]
            idx += 3

        if self._ar_order > 0:
            parsed["phi_ar"] = params[idx : idx + self._ar_order]
            idx += self._ar_order
            parsed["sigma2_ar"] = params[idx]
            idx += 1

        return parsed

    def _build_noise_list(
        self, parsed: dict[str, float | NDArray[np.float64]]
    ) -> list[tuple[int, float]]:
        """Build the list of (state_index, variance) noise entries."""
        layout = self._layout
        noise_list: list[tuple[int, float]] = []

        if self._level:
            noise_list.append((layout["level"]["start"], float(parsed["sigma2_level"])))

        if self._trend in ("stochastic", "damped"):
            noise_list.append((layout["trend"]["start"], float(parsed["sigma2_trend"])))

        if self._seasonal != "none":
            noise_list.append((layout["seasonal"]["start"], float(parsed["sigma2_seasonal"])))

        if self._cycle:
            sigma2_cycle = float(parsed["sigma2_cycle"])
            noise_list.append((layout["cycle"]["start"], sigma2_cycle))
            noise_list.append((layout["cycle"]["start"] + 1, sigma2_cycle))

        if self._ar_order > 0:
            noise_list.append((layout["ar"]["start"], float(parsed["sigma2_ar"])))

        return noise_list

    def _fill_level_trend(
        self, T: NDArray[np.float64], Z: NDArray[np.float64], phi_trend: float
    ) -> None:
        """Fill level and trend blocks of T and Z."""
        layout = self._layout
        if self._level:
            s = layout["level"]["start"]
            T[s, s] = 1.0
            Z[0, s] = 1.0
            if "trend" in layout:
                T[s, layout["trend"]["start"]] = 1.0  # mu_t gets nu_{t-1}

        if "trend" in layout:
            ts = layout["trend"]["start"]
            T[ts, ts] = phi_trend if self._trend == "damped" else 1.0

    def _fill_seasonal(self, T: NDArray[np.float64], Z: NDArray[np.float64]) -> None:
        """Fill the seasonal block of T and Z."""
        if "seasonal" not in self._layout:
            return
        assert self._seasonal_period is not None
        ss = self._layout["seasonal"]["start"]
        Z[0, ss] = 1.0
        s_period = self._seasonal_period

        if self._seasonal == "dummy":
            for j in range(s_period - 1):
                T[ss, ss + j] = -1.0
            for j in range(s_period - 2):
                T[ss + 1 + j, ss + j] = 1.0
            return

        # Trigonometric
        n_harmonics = s_period // 2
        si = ss
        for j in range(1, n_harmonics + 1):
            lam = 2.0 * np.pi * j / s_period
            if j < n_harmonics or s_period % 2 != 0:
                T[si, si] = np.cos(lam)
                T[si, si + 1] = np.sin(lam)
                T[si + 1, si] = -np.sin(lam)
                T[si + 1, si + 1] = np.cos(lam)
                si += 2
            else:
                T[si, si] = -1.0
                si += 1

    def _fill_cycle(
        self, T: NDArray[np.float64], Z: NDArray[np.float64], rho: float, lambda_c: float
    ) -> None:
        """Fill the cycle block of T and Z."""
        if not (self._cycle and "cycle" in self._layout):
            return
        cs = self._layout["cycle"]["start"]
        cos_l = np.cos(lambda_c)
        sin_l = np.sin(lambda_c)
        T[cs, cs] = rho * cos_l
        T[cs, cs + 1] = rho * sin_l
        T[cs + 1, cs] = -rho * sin_l
        T[cs + 1, cs + 1] = rho * cos_l
        Z[0, cs] = 1.0

    def _fill_ar(
        self, T: NDArray[np.float64], Z: NDArray[np.float64], phi_ar: NDArray[np.float64]
    ) -> None:
        """Fill the autoregressive block of T and Z."""
        if not (self._ar_order > 0 and "ar" in self._layout):
            return
        ars = self._layout["ar"]["start"]
        for i in range(self._ar_order):
            T[ars, ars + i] = phi_ar[i] if i < len(phi_ar) else 0.0
        for i in range(1, self._ar_order):
            T[ars + i, ars + i - 1] = 1.0
        Z[0, ars] = 1.0

    def _fill_exog(self, T: NDArray[np.float64], Z: NDArray[np.float64]) -> None:
        """Fill the exogenous (fixed coefficient) block of T and Z."""
        if not (self._exog is not None and "exog" in self._layout):
            return
        es = self._layout["exog"]["start"]
        k = self._layout["exog"]["size"]
        for i in range(k):
            T[es + i, es + i] = 1.0  # Identity (fixed)
        # Z for exog is X_t (time-varying) - use mean as approximation
        Z[0, es : es + k] = np.mean(self._exog, axis=0)

    def _build_ssm(self, params: NDArray[np.float64]) -> StateSpaceRepresentation:
        """Build the composite state-space representation."""
        k_states = self._k_states
        parsed = self._parse_params(params)
        noise_list = self._build_noise_list(parsed)

        k_posdef = max(len(noise_list), 1)
        ssm = StateSpaceRepresentation(k_states=k_states, k_endog=1, k_posdef=k_posdef)

        T = np.zeros((k_states, k_states))
        Z = np.zeros((1, k_states))
        R = np.zeros((k_states, k_posdef))
        Q = np.zeros((k_posdef, k_posdef))

        self._fill_level_trend(T, Z, float(parsed["phi_trend"]))
        self._fill_seasonal(T, Z)
        self._fill_cycle(T, Z, float(parsed["rho"]), float(parsed["lambda_c"]))
        phi_ar = parsed["phi_ar"]
        assert isinstance(phi_ar, np.ndarray)
        self._fill_ar(T, Z, phi_ar)
        self._fill_exog(T, Z)

        # Selection matrix R and covariance Q
        for ni, (state_idx, variance) in enumerate(noise_list):
            R[state_idx, ni] = 1.0
            Q[ni, ni] = variance

        ssm.T = T
        ssm.Z = Z
        ssm.R = R
        ssm.Q = Q
        ssm.H = np.array([[float(parsed["sigma2_obs"])]])
        ssm.a1 = np.zeros(k_states)
        ssm.P1 = np.eye(k_states) * config.diffuse_initial_variance

        return ssm
