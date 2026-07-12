"""
config/cities.py — Per-city configuration seam for the multi-city framework.

Every city-specific fact that used to be a scattered hardcoded constant
(coordinates, skew-normal calibration table, Gamma market slug/title
pattern, bracket temperature labels, settlement source) lives here as a
CityConfig entry. Core trading modules (core/model.py, core/discovery.py,
core/settlement.py) take a CityConfig instead of reading module-level
globals, so adding a new city is a config-only change.

WSSS (Singapore) and WMKK (Kuala Lumpur) are configured. WMKK's slug
template, bracket range, and skew table are best-effort estimates (mirrored
from WSSS's tropical-climate pattern) since no confirmed Polymarket
"highest temperature in Kuala Lumpur" market has been observed yet —
verify/adjust gamma_slug_template and title_keywords against the live
market before relying on discovery to find it. Add further CITIES entries
the same way; no core code changes required.
"""

import os
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
    # Each city has its OWN vault, sized independently — not a percentage
    # split of one shared pool. default_vault_usd is the fallback used when
    # no VAULT_USD_<ICAO> env var is set (see resolve_vault_usd() below).
    default_vault_usd:  float
    hard_prior_mu:       float                       # fallback forecast mean if all upstream fails
    hard_prior_sigma:    float
    official_station_fetcher: Optional[Callable[[str, int], Optional[float]]] = None


def resolve_vault_usd(config: "CityConfig") -> float:
    """
    Resolve this city's own vault size, dynamically configurable per-city
    via env var VAULT_USD_<ICAO> (e.g. VAULT_USD_WSSS=200, VAULT_USD_WMKK=100),
    falling back to CityConfig.default_vault_usd if unset.

    Legacy fallback: WSSS alone also honors the original MAX_VAULT_ALLOCATION
    env var (pre-multi-city name) if VAULT_USD_WSSS isn't set, so existing
    single-city .env deployments keep their configured vault size unchanged.
    """
    icao = config.icao.upper()
    env_key = f"VAULT_USD_{icao}"
    raw = os.getenv(env_key)
    if raw:
        return float(raw)

    if icao == "WSSS":
        legacy = os.getenv("MAX_VAULT_ALLOCATION")
        if legacy:
            return float(legacy)

    return config.default_vault_usd


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
        # Own vault, independent of WMKK's — override via VAULT_USD_WSSS
        # (or the legacy MAX_VAULT_ALLOCATION env var) without touching code.
        default_vault_usd=200.0,
        hard_prior_mu=31.5,
        hard_prior_sigma=1.0,
        official_station_fetcher=fetch_nea_changi,
    ),
    "WMKK": CityConfig(
        icao="WMKK",
        display_name="Kuala Lumpur",
        # Kuala Lumpur International Airport (Sepang) — the standard ICAO
        # station for "Kuala Lumpur" in aviation/weather datasets.
        lat=2.7456,
        lon=101.7099,
        timezone="Asia/Kuala_Lumpur",
        # Best-effort guess mirroring WSSS's confirmed slug pattern — verify
        # against a live Polymarket "highest temperature in Kuala Lumpur"
        # event before relying on this for discovery.
        gamma_slug_template="highest-temperature-in-kuala-lumpur-on-{month}-{day}-{year}",
        title_keywords=["kuala lumpur", "temperature"],
        # KL's equatorial climate runs slightly hotter than Singapore's —
        # shifted one degree band up. Also a best-effort guess pending a
        # real market to calibrate bracket definitions against.
        bracket_labels=["30°C", "31°C", "32°C", "33°C", "34°C"],
        bracket_bounds={
            "30°C": (30.0, 31.0),
            "31°C": (31.0, 32.0),
            "32°C": (32.0, 33.0),
            "33°C": (33.0, 34.0),
            "34°C": (34.0, 35.0),
        },
        # KL's NE monsoon (wet season, Nov-Mar) is milder than Singapore's SW
        # monsoon skew; dry inter-monsoon months (Jun-Sep, haze-prone) trend
        # hotter with a heavier left tail. Estimated, not yet calibrated
        # against real settlement data (see core/settlement.py trailing bias).
        skew_alpha_by_month={
            1: -0.8, 2: -0.8,
            3: -1.0, 4: -1.3,
            5: -1.5, 6: -1.8,
            7: -1.8, 8: -1.6,
            9: -1.4, 10: -1.0,
            11: -0.8, 12: -0.8,
        },
        default_skew_alpha=-1.3,
        # Own vault, independent of WSSS's — override via VAULT_USD_WMKK.
        default_vault_usd=100.0,
        hard_prior_mu=33.0,
        hard_prior_sigma=1.2,
        # No official-station override wired yet (Malaysia's Meteorological
        # Department has no equivalent of NEA's public data.gov.sg API
        # verified in this codebase) — falls back to Open-Meteo archive only.
        official_station_fetcher=None,
    ),
}


def get_city(icao: str) -> CityConfig:
    config = CITIES.get(icao.upper())
    if config is None:
        raise KeyError(f"No CityConfig registered for ICAO '{icao}'. Known: {list(CITIES)}")
    return config
