"""
config/cities.py — Per-city configuration seam for the multi-city framework.

Every city-specific fact that used to be a scattered hardcoded constant
(coordinates, skew-normal calibration table, Gamma market slug/title
pattern, bracket temperature labels, settlement source) lives here as a
CityConfig entry. Core trading modules (core/model.py, core/discovery.py,
core/settlement.py) take a CityConfig instead of reading module-level
globals, so adding a new city is a config-only change.

Only WSSS (Singapore) is configured today — Polymarket doesn't yet list
confirmed weather-bracket markets for other cities. Add a new CITIES
entry once a matching market exists; no core code changes required.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("hermes.config.cities")


@dataclass
class CityConfig:
    icao:                str
    display_name:        str
    lat:                 float
    lon:                 float
    timezone:            str
    gamma_slug_template: str                       # "highest-temperature-in-{slug}-on-{month}-{day}-{year}"
    title_keywords:      List[str]                  # browse-fallback filter terms
    bracket_labels:      List[str]                   # e.g. ["29°C", ...]
    bracket_bounds:      Dict[str, Tuple[float, float]]  # label -> (lo, hi), upper exclusive
    skew_alpha_by_month: Dict[int, float]
    default_skew_alpha:  float
    vault_allocation_pct: float                      # fraction of total vault this city may use
    hard_prior_mu:       float                       # fallback forecast mean if all upstream fails
    hard_prior_sigma:    float
    official_station_fetcher: Optional[Callable[[str, int], Optional[float]]] = None


# ── WSSS: NEA Changi (S24) official-station override ─────────────────────────
# Optional per-city override for SettlementEngine — used only if all
# generic Open-Meteo archive attempts fail. Other cities default to None
# (Open-Meteo archive only, per framework design).
NEA_READINGS_URL = "https://api.data.gov.sg/v1/environment/air-temperature?date={date}"
CHANGI_STATION_ID = "S24"


def fetch_nea_changi(date: str, timeout: int = 15) -> Optional[float]:
    """NEA data.gov.sg air-temperature reading for Changi station S24."""
    try:
        resp = requests.get(NEA_READINGS_URL.format(date=date), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        readings = data.get("items", [])
        changi_max = None
        for item in readings:
            for reading in item.get("readings", []):
                if reading.get("station_id") == CHANGI_STATION_ID:
                    val = reading.get("value")
                    if val is not None:
                        changi_max = max(changi_max or 0.0, float(val))

        if changi_max is not None:
            logger.info(f"[CITIES] NEA Changi actual max = {changi_max:.2f}°C")
        return changi_max

    except Exception as e:
        logger.error(f"[CITIES] NEA Changi fetch failed: {e}")
        return None


CITIES: Dict[str, CityConfig] = {
    "WSSS": CityConfig(
        icao="WSSS",
        display_name="Singapore",
        lat=1.3644,
        lon=103.9915,
        timezone="Asia/Singapore",
        gamma_slug_template="highest-temperature-in-singapore-on-{month}-{day}-{year}",
        title_keywords=["singapore", "temperature"],
        bracket_labels=["29°C", "30°C", "31°C", "32°C", "33°C"],
        bracket_bounds={
            "29°C": (29.0, 30.0),
            "30°C": (30.0, 31.0),
            "31°C": (31.0, 32.0),
            "32°C": (32.0, 33.0),
            "33°C": (33.0, 34.0),
        },
        # Negative = left skew (colder tail heavier). SW monsoon months
        # (May-Sep): stronger left skew. Moved verbatim from core/model.py's
        # old SKEW_ALPHA_TABLE[("WSSS", month)].
        skew_alpha_by_month={
            1: -1.0, 2: -1.0,
            3: -1.2, 4: -1.8,
            5: -2.0, 6: -2.2,
            7: -2.2, 8: -2.0,
            9: -1.8, 10: -1.3,
            11: -1.0, 12: -1.0,
        },
        default_skew_alpha=-1.5,
        vault_allocation_pct=1.0,
        hard_prior_mu=31.5,
        hard_prior_sigma=1.0,
        official_station_fetcher=fetch_nea_changi,
    ),
}


def get_city(icao: str) -> CityConfig:
    config = CITIES.get(icao.upper())
    if config is None:
        raise KeyError(f"No CityConfig registered for ICAO '{icao}'. Known: {list(CITIES)}")
    return config
