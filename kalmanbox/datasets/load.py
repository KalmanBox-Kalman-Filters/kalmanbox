"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"

_DATASETS: dict[str, dict[str, Any]] = {
    # --- Classic (5) ---
    "nile": {
        "path": "classic/nile.csv",
        "description": "Annual flow of the Nile river at Aswan, 1871-1970 (Durbin & Koopman 2012)",
        "target": "volume",
    },
    "airline": {
        "path": "classic/airline.csv",
        "description": "Monthly international airline passengers, 1949-1960 (Box & Jenkins)",
        "target": "passengers",
    },
    "ukdrivers": {
        "path": "classic/ukdrivers.csv",
        "description": "Monthly UK car driver deaths, 1969-1984 (Harvey & Durbin 1986)",
        "target": "deaths",
    },
    "lynx": {
        "path": "classic/lynx.csv",
        "description": "Annual Canadian lynx trappings, 1821-1934",
        "target": "trappings",
    },
    "sunspots": {
        "path": "classic/sunspots.csv",
        "description": "Monthly mean sunspot numbers, 1749-1983",
        "target": "sunspots",
    },
    # --- Macro international (7) ---
    "us_gdp": {
        "path": "macro/us_gdp.csv",
        "description": "US quarterly real GDP growth, 1947-2023",
        "target": "gdp_growth",
    },
    "us_inflation": {
        "path": "macro/us_inflation.csv",
        "description": "US monthly CPI inflation, 1947-2023",
        "target": "inflation",
    },
    "us_unemployment": {
        "path": "macro/us_unemployment.csv",
        "description": "US monthly unemployment rate, 1948-2023",
        "target": "unemployment",
    },
    "uk_gas": {
        "path": "macro/uk_gas.csv",
        "description": "UK quarterly gas consumption, 1960-1986",
        "target": "gas",
    },
    "global_temp": {
        "path": "macro/global_temp.csv",
        "description": "Global monthly temperature anomalies, 1880-2023",
        "target": "temperature",
    },
    "co2": {
        "path": "macro/co2.csv",
        "description": "Monthly atmospheric CO2 at Mauna Loa, 1958-2023",
        "target": "co2",
    },
    "exchange_rates": {
        "path": "macro/exchange_rates.csv",
        "description": "Daily USD/EUR exchange rates, 1999-2023",
        "target": "rate",
    },
    # --- Brazil (7) ---
    "brazil_ipca": {
        "path": "brazil/ipca.csv",
        "description": "Brazil monthly IPCA inflation, 1980-2023 (BCB SGS 433)",
        "target": "ipca",
    },
    "brazil_selic": {
        "path": "brazil/selic.csv",
        "description": "Brazil monthly SELIC rate, 1986-2023 (BCB SGS 4189)",
        "target": "selic",
    },
    "brazil_pib": {
        "path": "brazil/pib.csv",
        "description": "Brazil quarterly GDP index, 1996-2023 (BCB SGS 22099)",
        "target": "pib",
    },
    "brazil_industrial": {
        "path": "brazil/industrial.csv",
        "description": "Brazil monthly industrial production, 2002-2023 (BCB SGS 21859)",
        "target": "production",
    },
    "brazil_cambio": {
        "path": "brazil/cambio.csv",
        "description": "Brazil daily BRL/USD exchange rate, 2000-2023 (BCB SGS 1)",
        "target": "cambio",
    },
    "brazil_m1": {
        "path": "brazil/m1.csv",
        "description": "Brazil monthly M1 money supply, 1988-2023 (BCB SGS 27788)",
        "target": "m1",
    },
    "brazil_desemprego": {
        "path": "brazil/desemprego.csv",
        "description": "Brazil monthly unemployment rate, 2012-2023 (BCB SGS 24369)",
        "target": "desemprego",
    },
}


def load_dataset(name: str) -> pd.DataFrame:
    """Load a built-in dataset by name.

    Parameters
    ----------
    name : str
        Dataset name. Use list_datasets() to see available names.

    Returns
    -------
    pd.DataFrame
        DataFrame with the dataset.

    Raises
    ------
    ValueError
        If dataset name is not found.
    """
    if name not in _DATASETS:
        available = ", ".join(sorted(_DATASETS.keys()))
        msg = f"Unknown dataset '{name}'. Available: {available}"
        raise ValueError(msg)

    info = _DATASETS[name]
    path = _DATA_DIR / info["path"]

    if not path.exists():
        msg = (
            f"Dataset file not found: {path}. "
            f"For Brazilian data, run: python scripts/download_brazil_data.py"
        )
        raise FileNotFoundError(msg)

    return pd.read_csv(path)


def list_datasets() -> list[str]:
    """List available dataset names.

    Returns
    -------
    list[str]
        Sorted list of dataset names.
    """
    return sorted(_DATASETS.keys())


def dataset_info(name: str) -> dict[str, Any]:
    """Get metadata for a dataset.

    Parameters
    ----------
    name : str
        Dataset name.

    Returns
    -------
    dict
        Dictionary with path, description, target column.
    """
    if name not in _DATASETS:
        available = ", ".join(sorted(_DATASETS.keys()))
        msg = f"Unknown dataset '{name}'. Available: {available}"
        raise ValueError(msg)
    return _DATASETS[name].copy()
