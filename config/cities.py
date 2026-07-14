"""
config/cities.py — Per-city configuration seam for the multi-city framework.

Every city-specific fact that used to be a scattered hardcoded constant
(coordinates, skew-normal calibration table, Gamma market slug/title
pattern, bracket temperature labels, settlement source) lives here as a
CityConfig entry. Core trading modules (core/model.py, core/discovery.py,
core/settlement.py) take a CityConfig instead of reading module-level
globals, so adding a new city is a config-only change.

WSSS (Singapore) and WMKK (Kuala Lumpur) are configured.

Bracket ranges (2026-07-12 verification, corrected same day): live
Polymarket "highest temperature" events use 11 outcome brackets per day,
not 5, and BOTH cities use the identical 26-36°C range — confirmed
directly via a live market slug,
"highest-temperature-in-kuala-lumpur-on-july-6-2026-26corbelow" i.e.
"26°C or below" — so WMKK is NOT shifted +1°C from WSSS as an earlier
pass here assumed. The two tail brackets are open-ended catch-alls
("26°C or below" / "36°C or above"), not narrow 1-degree bins like the 9
middle brackets — bracket_bounds reflects that with -inf/+inf edges.
"""

import os
import logging
import functools
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
    # When True, core/execution.py and core/position_monitor.py still fetch
    # live order books (for realistic fill prices) but skip real order
    # placement entirely — no CLOB order is ever created or posted, so no
    # real capital is at risk. Positions/exits are recorded normally with
    # is_paper=1 so the dashboard can show simulated results without mixing
    # them into any city's real vault P&L (see dashboard_api.py's _is_paper).
    paper_trading:       bool = False


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


# ── ASOS/METAR station archive — primary official-station source ────────────
# Polymarket resolves these markets from Wunderground's airport-station
# history (confirmed 2026-07-12: Wunderground's own market rules text names
# "Singapore Changi Airport Station" / "Kuala Lumpur Intl Airport Station").
# Wunderground's airport pages are themselves sourced from the METAR feed for
# that station's ICAO code — and city_config.icao already *is* that code
# (WSSS, WMKK are ICAO airport identifiers). The Iowa Environmental Mesonet
# (mesonet.agron.iastate.edu) mirrors the same global ASOS/METAR network for
# free with no API key, keyed by ICAO code, so it's usable generically for
# any city here without a per-country integration.
#
# Verified by cross-checking three live Polymarket resolutions against this
# feed's computed daily max — all matched exactly: WSSS Jul 3 -> 31°C,
# WSSS Jul 6 -> 32°C, WMKK Jul 8 -> 33°C. This is a closer match to the
# actual settlement source than Open-Meteo's archive (ERA5 reanalysis grid
# output, not a station observation) or NEA's data.gov.sg feed (a different
# station network than the airport METAR Wunderground/Polymarket use).
ASOS_HOURLY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def fetch_asos_daily_max(icao: str, timezone: str, date: str, timeout: int = 15) -> Optional[float]:
    """Daily max temperature (°C) for an ICAO station's local calendar date, from Iowa Mesonet ASOS."""
    params = {
        "station": icao,
        "data": "tmpf",
        "year1": date[0:4], "month1": date[5:7], "day1": date[8:10],
        "year2": date[0:4], "month2": date[5:7], "day2": date[8:10],
        "tz": timezone,
        "format": "onlycomma",
        "latlon": "no", "elev": "no",
        "missing": "M", "trace": "T", "direct": "no",
    }
    try:
        resp = requests.get(ASOS_HOURLY_URL, params=params, timeout=timeout)
        resp.raise_for_status()

        max_f = None
        lines = resp.text.strip().splitlines()
        for line in lines[1:]:  # skip "station,valid,tmpf" header
            parts = line.strip().split(",")
            if len(parts) < 3 or parts[2] in ("M", ""):
                continue
            try:
                f_val = float(parts[2])
            except ValueError:
                continue
            max_f = f_val if max_f is None else max(max_f, f_val)

        if max_f is None:
            return None

        celsius = (max_f - 32.0) * 5.0 / 9.0
        logger.info(f"[CITIES] ASOS {icao} actual max = {celsius:.2f}°C ({max_f:.1f}°F)")
        return celsius

    except Exception as e:
        logger.error(f"[CITIES] ASOS {icao} fetch failed: {e}")
        return None


# ── WSSS: NEA Changi (S24) — secondary fallback behind ASOS ──────────────────
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


def _wsss_official_fetcher(date: str, timeout: int = 15) -> Optional[float]:
    """ASOS (WSSS/Changi) primary, NEA Changi S24 secondary if ASOS has no reading for the date."""
    asos = fetch_asos_daily_max("WSSS", "Asia/Singapore", date, timeout)
    if asos is not None:
        return asos
    return fetch_nea_changi(date, timeout)


CITIES: Dict[str, CityConfig] = {
    "WSSS": CityConfig(
        icao="WSSS",
        display_name="Singapore",
        lat=1.3644,
        lon=103.9915,
        timezone="Asia/Singapore",
        gamma_slug_template="highest-temperature-in-singapore-on-{month}-{day}-{year}",
        title_keywords=["singapore", "temperature"],
        # Full 11-outcome range confirmed live (2026-07-12) — Polymarket's
        # WSSS event lists 26°C through 36°C, not just the 29-33°C middle
        # band this config used to define. Tails are open-ended catch-alls
        # ("26°C or below" / "36°C or above"), not narrow 1-degree bins.
        bracket_labels=["26°C", "27°C", "28°C", "29°C", "30°C", "31°C", "32°C", "33°C", "34°C", "35°C", "36°C"],
        bracket_bounds={
            "26°C": (float("-inf"), 27.0),
            "27°C": (27.0, 28.0),
            "28°C": (28.0, 29.0),
            "29°C": (29.0, 30.0),
            "30°C": (30.0, 31.0),
            "31°C": (31.0, 32.0),
            "32°C": (32.0, 33.0),
            "33°C": (33.0, 34.0),
            "34°C": (34.0, 35.0),
            "35°C": (35.0, 36.0),
            "36°C": (36.0, float("inf")),
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
        official_station_fetcher=_wsss_official_fetcher,
        # Paper trading, same as WMKK (see CityConfig.paper_trading) — its
        # bracket range/settlement source were both corrected this session
        # (5->11 brackets, ASOS/METAR settlement), so re-validate against
        # live prices before resuming real capital. Flip to False once satisfied.
        paper_trading=True,
    ),
    "WMKK": CityConfig(
        icao="WMKK",
        display_name="Kuala Lumpur",
        # Kuala Lumpur International Airport (Sepang) — the standard ICAO
        # station for "Kuala Lumpur" in aviation/weather datasets.
        lat=2.7456,
        lon=101.7099,
        timezone="Asia/Kuala_Lumpur",
        # Confirmed against live Polymarket events (2026-07-12): matches
        # highest-temperature-in-kuala-lumpur-on-july-{d}-{yyyy} exactly.
        gamma_slug_template="highest-temperature-in-kuala-lumpur-on-{month}-{day}-{year}",
        title_keywords=["kuala lumpur", "temperature"],
        # NOT shifted from WSSS — confirmed live (2026-07-12) via a real
        # market slug, "...-on-july-6-2026-26corbelow" ("26°C or below"),
        # that WMKK uses the SAME 26-36°C range as WSSS despite KL's higher
        # average climate. An earlier pass here guessed a +1°C shift
        # (27-37°C); that was wrong and has been corrected. Tails are
        # open-ended catch-alls, same as WSSS.
        bracket_labels=["26°C", "27°C", "28°C", "29°C", "30°C", "31°C", "32°C", "33°C", "34°C", "35°C", "36°C"],
        bracket_bounds={
            "26°C": (float("-inf"), 27.0),
            "27°C": (27.0, 28.0),
            "28°C": (28.0, 29.0),
            "29°C": (29.0, 30.0),
            "30°C": (30.0, 31.0),
            "31°C": (31.0, 32.0),
            "32°C": (32.0, 33.0),
            "33°C": (33.0, 34.0),
            "34°C": (34.0, 35.0),
            "35°C": (35.0, 36.0),
            "36°C": (36.0, float("inf")),
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
        # ASOS/METAR archive for WMKK (KL Intl Airport) — confirmed exact
        # match against a live Polymarket resolution (Jul 8, 2026 -> 33°C).
        # No Malaysian government equivalent of NEA's data.gov.sg API is
        # wired, so this is the only official-station source for WMKK.
        official_station_fetcher=functools.partial(fetch_asos_daily_max, "WMKK", "Asia/Kuala_Lumpur"),
        # WMKK is new/unproven (bracket ranges and skew table only recently
        # verified, no live trading history) — paper trading first to prove
        # the full pipeline against real market prices before risking capital.
        # Flip to False once satisfied, or set VAULT_USD_WMKK and this flag
        # together when going live.
        paper_trading=True,
    ),
}


def get_city(icao: str) -> CityConfig:
    config = CITIES.get(icao.upper())
    if config is None:
        raise KeyError(f"No CityConfig registered for ICAO '{icao}'. Known: {list(CITIES)}")
    return config
