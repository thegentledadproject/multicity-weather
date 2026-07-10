"""
core/model.py — H1: Skewnorm bracket probability model
v4.5: Dual forecast source — GFS ensemble + ECMWF ensemble, blended.
v5.0: City-agnostic — lat/lon, skew table, and hard-prior values now come
      from a config.cities.CityConfig instead of module-level WSSS constants.

FORECAST PIPELINE:
  Source 1 — GFS 31-member ensemble (Open-Meteo ensemble API)
    Provides mu_gfs and sigma_gfs from member spread.

  Source 2 — ECMWF 51-member ensemble (Open-Meteo ensemble API)
    Provides mu_ecmwf and sigma_ecmwf from member spread.
    ECMWF is the globally superior model (lower RMSE on tropical stations).
    Weighted more heavily in the blend: 60% ECMWF / 40% GFS.

  Blend: mu = 0.6 * mu_ecmwf + 0.4 * mu_gfs
         sigma = sqrt(0.6*sigma_ecmwf² + 0.4*sigma_gfs² + 0.6*0.4*(mu_ecmwf-mu_gfs)²)
         The third term captures inter-model spread — when GFS and ECMWF
         disagree on mu, total uncertainty is wider than either alone.
         source = "ensemble_blend"

  If only one source available: use that source alone (log WARNING).
  If neither: fallback to standard GFS forecast API (diurnal proxy sigma).
  If that fails: hard prior (from CityConfig.hard_prior_mu/sigma) — NO TRADING.

Bracket probabilities:
  P(bracket) = CDF(upper_bound) - CDF(lower_bound)
  using skewnorm(alpha, loc=calibrated_mu, scale=blended_sigma)

This is correct for Polymarket bracket markets. Do NOT use CDF values
directly (that gives P(temp > threshold), not P(temp in bracket)).
"""

import logging
import datetime
import requests
import numpy as np
from scipy.stats import skewnorm
from typing import Dict, Optional, Tuple

logger = logging.getLogger("hermes.model")

# ── Open-Meteo ensemble API ─────────────────────────────────────────────────
# GFS: 31 members | ECMWF: 51 members
# Both available free, no API key required. Timezone left as Singapore for
# the shared template's default — callers that need a different tz can
# still get correct UTC-anchored peak-hour data since hourly timestamps are
# ISO and peak-hour extraction (06:00-21:00) is a local-time convention
# specific to WSSS's usage; other cities should verify this window applies.
_ENSEMBLE_BASE = (
    "https://ensemble-api.open-meteo.com/v1/ensemble"
    "?latitude={lat}&longitude={lon}"
    "&hourly=temperature_2m"
    "&models={model}"
    "&timezone=Asia%2FSingapore"
    "&forecast_days=1"
)
GFS_ENSEMBLE_URL    = _ENSEMBLE_BASE.replace("{model}", "gfs_seamless")
ECMWF_ENSEMBLE_URL  = _ENSEMBLE_BASE.replace("{model}", "ecmwf_ifs04")

# ── Open-Meteo standard forecast (fallback if ensemble unavailable) ──────────
OPEN_METEO_FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&daily=temperature_2m_max,temperature_2m_min"
    "&timezone=Asia%2FSingapore"
    "&forecast_days=1"
    "&models=gfs_seamless"
)

# ── Blend weights ────────────────────────────────────────────────────────────
ECMWF_WEIGHT = 0.60   # ECMWF weighted higher — lower tropical RMSE
GFS_WEIGHT   = 1.0 - ECMWF_WEIGHT

# Bracket boundaries use TRUNCATION, not rounding — see CityConfig.bracket_bounds
# for the per-city definition (moved out of this module in v5.0).


class ForecastResult:
    """Container for blended forecast output."""
    def __init__(
        self,
        mu:          float,
        sigma:       float,
        source:      str,
        mu_gfs:      Optional[float] = None,
        mu_ecmwf:    Optional[float] = None,
        sigma_gfs:   Optional[float] = None,
        sigma_ecmwf: Optional[float] = None,
    ):
        self.mu          = mu
        self.sigma       = sigma
        self.source      = source     # "ensemble_blend"|"gfs_only"|"ecmwf_only"|"forecast_spread"|"fallback"
        self.mu_gfs      = mu_gfs
        self.mu_ecmwf    = mu_ecmwf
        self.sigma_gfs   = sigma_gfs
        self.sigma_ecmwf = sigma_ecmwf

    def __repr__(self):
        parts = [f"μ={self.mu:.2f}°C", f"σ={self.sigma:.2f}°C", f"src={self.source}"]
        if self.mu_gfs and self.mu_ecmwf:
            parts.append(f"gfs={self.mu_gfs:.2f} ecmwf={self.mu_ecmwf:.2f}")
        return f"ForecastResult({', '.join(parts)})"


def _fetch_ensemble_members(
    url: str, lat: float, lon: float, timeout: int, model_name: str,
) -> Optional[Tuple[float, float, int]]:
    """
    Fetch ensemble members from Open-Meteo and compute (mu, sigma, n_members).
    Returns None on any failure.

    Peak heating window: hours 06:00–21:00 (local, per URL timezone). Daily
    max per member = max temperature across this window.
    mu    = mean of member daily maxes
    sigma = std dev of member daily maxes (ddof=1), clipped to [0.30, 2.00]
    """
    try:
        resp = requests.get(url.format(lat=lat, lon=lon), timeout=timeout)
        resp.raise_for_status()
        data   = resp.json()
        hourly = data.get("hourly", {})

        member_keys = [k for k in hourly if k.startswith("temperature_2m_member")]
        if len(member_keys) < 3:
            logger.warning(f"[MODEL] {model_name}: only {len(member_keys)} members returned")
            return None

        times    = hourly.get("time", [])
        # Open-Meteo returns ISO timestamps like "2026-07-04T06:00".
        # t[11:] strips the date+T and yields "06:00" — so the comparison must be
        # against "06:00".."21:00", NOT "T06".."T21". The old "T06" <= t[11:] test
        # never matched (t[11:] has no leading 'T'), so peak_idx was always empty,
        # every ensemble fetch returned None, and the forecast fell back to the
        # hard prior on EVERY scan — the root cause of model_mu=31.5 in the logs.
        peak_idx = [i for i, t in enumerate(times) if len(t) >= 16 and "06:00" <= t[11:16] <= "21:00"]
        if not peak_idx:
            logger.warning(f"[MODEL] {model_name}: no peak-hour indices found")
            return None

        member_maxes = []
        for key in member_keys:
            vals      = hourly[key]
            peak_vals = [vals[i] for i in peak_idx if i < len(vals) and vals[i] is not None]
            if peak_vals:
                member_maxes.append(max(peak_vals))

        if len(member_maxes) < 3:
            return None

        mu    = float(np.mean(member_maxes))
        sigma = float(np.clip(np.std(member_maxes, ddof=1), 0.30, 2.00))
        n     = len(member_maxes)
        logger.info(f"[MODEL] {model_name}: μ={mu:.2f}°C σ={sigma:.2f}°C ({n} members)")
        return mu, sigma, n

    except Exception as e:
        logger.warning(f"[MODEL] {model_name} fetch failed: {e}")
        return None


def _blend(
    gfs:   Optional[Tuple[float, float, int]],
    ecmwf: Optional[Tuple[float, float, int]],
) -> ForecastResult:
    """
    Blend GFS and ECMWF ensemble results into a single ForecastResult.

    When both available:
      mu    = w_e * mu_ecmwf + w_g * mu_gfs
      sigma = sqrt(w_e*sigma_e² + w_g*sigma_g² + w_e*w_g*(mu_e - mu_g)²)

      The third term (inter-model disagreement) inflates sigma when the two
      models diverge — correctly expressing higher uncertainty on contentious
      forecast days. On days where GFS and ECMWF agree closely, the blend
      sigma ≈ weighted average of the two sigmas.

    When only one available: use it with a WARNING.
    When neither: return fallback (caller handles).
    """
    if gfs and ecmwf:
        mu_g, sig_g, _ = gfs
        mu_e, sig_e, _ = ecmwf

        mu    = ECMWF_WEIGHT * mu_e + GFS_WEIGHT * mu_g
        sigma = float(np.sqrt(
            ECMWF_WEIGHT * sig_e**2
            + GFS_WEIGHT  * sig_g**2
            + ECMWF_WEIGHT * GFS_WEIGHT * (mu_e - mu_g)**2
        ))
        sigma = float(np.clip(sigma, 0.30, 2.00))

        divergence = abs(mu_e - mu_g)
        logger.info(
            f"[MODEL] Blend: μ={mu:.2f}°C σ={sigma:.2f}°C "
            f"(gfs={mu_g:.2f} ecmwf={mu_e:.2f} divergence={divergence:.2f}°C)"
        )
        if divergence > 1.0:
            logger.warning(
                f"[MODEL] ⚠️  GFS/ECMWF divergence = {divergence:.2f}°C — "
                f"sigma inflated to {sigma:.2f}°C. Edge signals may be suppressed."
            )

        return ForecastResult(
            mu=round(mu, 3), sigma=round(sigma, 3),
            source="ensemble_blend",
            mu_gfs=round(mu_g, 3), mu_ecmwf=round(mu_e, 3),
            sigma_gfs=round(sig_g, 3), sigma_ecmwf=round(sig_e, 3),
        )

    if ecmwf:
        mu_e, sig_e, _ = ecmwf
        logger.warning("[MODEL] GFS unavailable — using ECMWF only")
        return ForecastResult(mu=round(mu_e, 3), sigma=round(sig_e, 3),
                              source="ecmwf_only", mu_ecmwf=round(mu_e, 3))

    if gfs:
        mu_g, sig_g, _ = gfs
        logger.warning("[MODEL] ECMWF unavailable — using GFS only")
        return ForecastResult(mu=round(mu_g, 3), sigma=round(sig_g, 3),
                              source="gfs_only", mu_gfs=round(mu_g, 3))

    return ForecastResult(mu=0.0, sigma=0.0, source="none")


def fetch_gfs_forecast(
    lat: float,
    lon: float,
    hard_prior_mu: float = 31.5,
    hard_prior_sigma: float = 1.0,
    timeout: int = 15,
) -> ForecastResult:
    """
    Fetch dual-source forecast for the given coordinates and return a
    blended ForecastResult.

    Pipeline:
      1. Fetch GFS 31-member ensemble (parallel-capable but sequential here)
      2. Fetch ECMWF 51-member ensemble
      3. Blend → ForecastResult(source="ensemble_blend")
      4. If both fail → standard GFS forecast API (diurnal sigma proxy)
      5. If that fails → hard prior (hard_prior_mu/sigma), BLOCK TRADING
    """

    # ── Fetch both ensemble sources ───────────────────────────────────────────
    gfs_result   = _fetch_ensemble_members(GFS_ENSEMBLE_URL,   lat, lon, timeout, "GFS")
    ecmwf_result = _fetch_ensemble_members(ECMWF_ENSEMBLE_URL, lat, lon, timeout, "ECMWF")

    blended = _blend(gfs_result, ecmwf_result)
    if blended.source != "none":
        return blended

    # ── Fallback 1: standard forecast API (GFS deterministic) ────────────────
    logger.warning("[MODEL] Both ensemble sources failed — trying standard GFS forecast API")
    try:
        url  = OPEN_METEO_FORECAST_URL.format(lat=lat, lon=lon)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data  = resp.json()
        daily = data.get("daily", {})
        t_max = daily.get("temperature_2m_max", [None])[0]
        t_min = daily.get("temperature_2m_min", [None])[0]

        if t_max is not None:
            mu = float(t_max)
            # Use explicit None check, not truthiness: `t_min or (t_max-4)` would
            # wrongly discard a valid t_min of 0.0.
            t_min_eff = t_min if t_min is not None else (t_max - 4)
            sigma = float(np.clip((t_max - t_min_eff) / 4.0, 0.50, 1.50))
            logger.warning(
                f"[MODEL] Forecast fallback: μ={mu:.2f}°C σ={sigma:.2f}°C "
                f"(diurnal proxy — less accurate)"
            )
            return ForecastResult(mu=mu, sigma=sigma, source="forecast_spread")

    except Exception as e:
        logger.error(f"[MODEL] Standard forecast API failed: {e}")

    # ── Fallback 2: hard prior — block trading ────────────────────────────────
    logger.error(
        f"[MODEL] All forecast sources failed. Using prior "
        f"μ={hard_prior_mu:.1f}°C σ={hard_prior_sigma:.1f}°C. NO TRADING."
    )
    return ForecastResult(mu=hard_prior_mu, sigma=hard_prior_sigma, source="fallback")


class BracketModel:
    """
    Computes bracket probabilities for a city's temperature markets.
    Applies trailing bias correction from the calibration ledger.
    Uses blended GFS+ECMWF forecast when available.
    """

    def __init__(self, city_config, trailing_bias: float = 0.0):
        self.city_config   = city_config
        self.icao          = city_config.icao
        self.trailing_bias = trailing_bias

    def compute(
        self,
        forecast: ForecastResult,
        month: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Returns {bracket_label: probability} for all defined brackets.
        Probabilities sum to < 1.0 (remainder = tails outside the bracket range).
        Returns {} if forecast source is "fallback" or "none" — no trading.
        """
        if forecast.source in ("fallback", "none"):
            logger.error(
                f"[MODEL] Refusing to compute on source='{forecast.source}'. Return empty."
            )
            return {}

        if month is None:
            sg_now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
            month  = sg_now.month

        config = self.city_config
        alpha  = config.skew_alpha_by_month.get(month, config.default_skew_alpha)
        cal_mu = forecast.mu + self.trailing_bias

        logger.info(
            f"[MODEL] {self.icao} μ_blend={forecast.mu:.3f} bias={self.trailing_bias:+.3f} "
            f"→ μ_cal={cal_mu:.3f} σ={forecast.sigma:.3f} α={alpha} "
            f"src={forecast.source}"
        )

        probs: Dict[str, float] = {}
        for label, (lo, hi) in config.bracket_bounds.items():
            p = (
                skewnorm.cdf(hi, alpha, loc=cal_mu, scale=forecast.sigma)
                - skewnorm.cdf(lo, alpha, loc=cal_mu, scale=forecast.sigma)
            )
            probs[label] = max(0.001, float(p))

        return probs
