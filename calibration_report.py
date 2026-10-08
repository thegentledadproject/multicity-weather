"""
calibration_report.py — P4: out-of-sample calibration of the bracket model.

Two independent scores, because the live model can't be replayed historically
(its sigma is ensemble spread, and Open-Meteo does not archive ensembles):

  A. PROXY REPLAY (2024-06 onward). Point-in-time deterministic runs from
     Open-Meteo's previous-runs archive: temperature_2m_previous_day{h} for
     GFS and ECMWF, i.e. issued ~h days before each hour. Live blend weights,
     the live skew-normal bracket maths (BracketModel), the live trailing
     bias rule (mean of the last 10 residuals) and, standing in for ensemble
     spread, sigma = std of the last 30 residuals. A pass here says the
     model *family* is calibrated; it does not validate the live sigma.

  B. LIVE FORWARD. The exact live model, read from the bot's own database:
     the scan_snapshots row nearest before 06:00 local on each market date,
     with the market mids its signal_log rows recorded. Needs the VPS DB
     (--db) and ~30 settled days per verdict.

Point-in-time rules (both): decisions at 06:00 local on D (same-day, h=1)
and on D-1 (day-ahead, h=2); a day's outcome only informs decisions after
the day ended. Outcomes are the settlement station's METAR high (Iowa
Mesonet ASOS), which matched every Polymarket resolution checked.

Baselines on the same days:
  - climatology: the station's bracket frequency for the calendar month over
    2016-2023, strictly before every scored day
  - market: Polymarket prices at the decision time (A: price history API on
    days with an event; B: the mids the bot logged), normalised
  - market blend: (1-w)*market + w*model, w chosen leave-one-date-out; a
    blend beating the market means the model adds information, even when
    it loses to the market head-to-head

Raw downloads are cached under .cache/calibration/ so reruns are offline.

Usage:
    python calibration_report.py                       # proxy replay, both cities
    python calibration_report.py --db /path/hermes.db  # plus live forward score
    python calibration_report.py --icao WSSS --start 2025-01-01 --no-market
"""

import argparse
import dataclasses
import datetime
import json
import logging
import math
import pathlib
import random
import sqlite3
import time
from typing import Dict, List, Optional

import pytz
import requests

from config.cities import CITIES, _bracket_range
from core.discovery import MarketDiscovery, bracket_from_title
from core.model import ECMWF_WEIGHT, GFS_WEIGHT, BracketModel, ForecastResult

CACHE = pathlib.Path(".cache/calibration")
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
PRICES_URL = "https://clob.polymarket.com/prices-history"

DEFAULT_START = "2024-06-01"   # previous-runs archive verified back to here
FIRST_EVENTS = "2026-06-01"    # no Polymarket temperature events looked up before this
HORIZONS = (1, 2)              # decision 06:00 local on D-(h-1), forecast previous_day{h}
DECISION_HOUR = 6
PEAK_HOURS = ("06:00", "21:00")  # same daily-max window as core/model.py
BIAS_WINDOW = 10               # matches Ledger.fetch_trailing_bias
SIGMA_WINDOW = 30              # proxy sigma: std of this many past residuals
MIN_HISTORY = 10               # proxy rows start once this many residuals exist
SIGMA_BOUNDS = (0.30, 3.00)
CLIMATOLOGY_YEARS = (2016, 2023)  # strictly before DEFAULT_START
SCORING_BRACKETS = _bracket_range(26, 38)  # fixed partition for model-vs-climatology
MAX_PRICE_AGE_S = 3 * 3600
PRICE_REQUEST_GAP_S = 2.0      # Polymarket blocks this IP on parallel/bursty requests
MIN_VERDICT_DAYS = 30

logger = logging.getLogger("hermes.calibration")


# ── Cached fetching ──────────────────────────────────────────────────────────

def _get(url: str, params: dict, as_json: bool = True):
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=(10, 180))
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json() if as_json else resp.text
        except (requests.RequestException, ValueError) as e:  # Gamma intermittently sends empty bodies
            if attempt == 4:
                raise
            logger.warning(f"retry {attempt + 1} {url}: {e}")
            time.sleep(5 * 2 ** attempt)


def cached(name: str, fetch):
    """JSON cache keyed by name. Only closed history is cached, so entries never go stale."""
    path = CACHE / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    value = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value))
    tmp.replace(path)  # atomic: an interrupted run never leaves a half-written file
    return value


def fetch_previous_runs(cfg, start: str, end: str) -> dict:
    variables = ",".join(f"temperature_2m_previous_day{h}" for h in HORIZONS)

    def fetch():
        data = _get(PREVIOUS_RUNS_URL, {
            "latitude": cfg.lat, "longitude": cfg.lon, "hourly": variables,
            "models": "gfs_seamless,ecmwf_ifs025", "timezone": cfg.timezone,
            "start_date": start, "end_date": end,
        })
        if not data or "hourly" not in data:
            raise RuntimeError(f"Open-Meteo previous-runs returned nothing for {cfg.icao} {start}..{end}")
        return data["hourly"]
    return cached(f"prev_{cfg.icao}_{start}_{end}", fetch)


def fetch_station_highs(cfg, start: str, end: str) -> Dict[str, int]:
    """Daily max (whole °C, as the METAR reports it) per local date, from Iowa Mesonet ASOS."""
    s, e = datetime.date.fromisoformat(start), datetime.date.fromisoformat(end)
    text = cached(f"asos_{cfg.icao}_{start}_{end}", lambda: _get(ASOS_URL, {
        "station": cfg.icao, "data": "tmpf", "tz": cfg.timezone, "format": "onlycomma",
        "year1": s.year, "month1": s.month, "day1": s.day, "year2": e.year, "month2": e.month, "day2": e.day,
        "latlon": "no", "elev": "no", "missing": "M", "trace": "T", "direct": "no",
    }, as_json=False))
    highs: Dict[str, float] = {}
    for line in text.strip().splitlines()[1:]:
        _, valid, tmpf = line.split(",")
        if tmpf not in ("M", "T", ""):
            highs[valid[:10]] = max(highs.get(valid[:10], -99.0), (float(tmpf) - 32) * 5 / 9)
    return {day: round(c) for day, c in highs.items()}


# Polymarket blocks bursts from one IP. Once it stops answering, remaining
# uncached events/prices are skipped (not cached) and a rerun fills them in.
# Single failures are common (the connection here is flaky), so only
# MAX_CONSECUTIVE_FAILURES in a row stop the run's downloads.
MAX_CONSECUTIVE_FAILURES = 3
_polymarket = {"blocked": False, "missing": 0, "failures": 0}


def _polymarket_cached(name: str, fetch):
    if not (CACHE / f"{name}.json").exists():
        if _polymarket["blocked"]:
            _polymarket["missing"] += 1
            return None
        time.sleep(PRICE_REQUEST_GAP_S)
    try:
        value = cached(name, fetch)
        _polymarket["failures"] = 0
        return value
    except requests.RequestException as e:
        _polymarket["missing"] += 1
        _polymarket["failures"] += 1
        if _polymarket["failures"] >= MAX_CONSECUTIVE_FAILURES:
            logger.error(f"Polymarket unavailable, skipping uncached downloads this run: {e}")
            _polymarket["blocked"] = True
        else:
            logger.warning(f"Polymarket download failed, skipping {name}: {e}")
        return None


def fetch_event(cfg, day: str) -> Optional[dict]:
    """Closed event for the city's day: bracket bounds, YES tokens and the winner. None if unusable."""
    def fetch():
        for slug in MarketDiscovery(None, cfg)._build_slugs(day):
            events = _get(GAMMA_EVENTS_URL, {"slug": slug}) or []
            if events:
                return events[0]
        return None

    event = _polymarket_cached(f"event_{cfg.icao}_{day}", fetch)
    if not event or not event.get("closed"):
        return None
    brackets = []
    for m in event.get("markets", []):
        parsed = bracket_from_title(m.get("groupItemTitle") or "")
        if parsed is None:
            return None
        prices = json.loads(m["outcomePrices"]) if isinstance(m["outcomePrices"], str) else m["outcomePrices"]
        tokens = json.loads(m["clobTokenIds"]) if isinstance(m["clobTokenIds"], str) else m["clobTokenIds"]
        brackets.append({"label": parsed[0], "bounds": parsed[1], "token": tokens[0],
                         "won": float(prices[0]) > 0.99})
    if sum(b["won"] for b in brackets) != 1:
        return None
    brackets.sort(key=lambda b: b["bounds"][0])
    return {"brackets": brackets}


def fetch_price_history(token: str, day: str) -> list:
    d = datetime.date.fromisoformat(day)
    start = int(datetime.datetime(d.year, d.month, d.day, tzinfo=datetime.timezone.utc).timestamp()) - 3 * 86400
    history = _polymarket_cached(f"px_{token}", lambda: (_get(PRICES_URL, {
        "market": token, "startTs": start, "endTs": start + 4 * 86400, "fidelity": 60,
    }) or {}).get("history", []))
    return history or []


# ── Scoring helpers ──────────────────────────────────────────────────────────

def normalise(p: List[float]) -> List[float]:
    total = sum(p)
    return [x / total for x in p]


def brier(p: List[float], outcome: List[int]) -> float:
    """Multi-category Brier: 0 is perfect, 2 is certain and wrong."""
    return sum((pi - oi) ** 2 for pi, oi in zip(p, outcome))


def outcome_vector(high: int, bounds) -> List[int]:
    return [int(lo <= high < hi) for lo, hi in bounds]


def climatology_probs(highs_by_month: Dict[int, List[int]], month: int, bounds) -> List[float]:
    """Bracket frequencies with add-one smoothing so no bracket scores as impossible."""
    highs = highs_by_month.get(month, [])
    return normalise([sum(lo <= h < hi for h in highs) + 1 for lo, hi in bounds])


def bootstrap_ci(diffs: List[float], dates: List[str], n: int = 2000, seed: int = 0):
    """
    95% CI of the mean diff, resampling whole dates: both cities on one date
    share a weather regime, so treating them as independent would overstate
    confidence.
    """
    by_date: Dict[str, List[float]] = {}
    for d, x in zip(dates, diffs):
        by_date.setdefault(d, []).append(x)
    clusters = list(by_date.values())
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [x for _ in clusters for x in rng.choice(clusters)]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def decision_ts(cfg, day: str, h: int) -> float:
    d = datetime.date.fromisoformat(day) - datetime.timedelta(days=h - 1)
    return pytz.timezone(cfg.timezone).localize(
        datetime.datetime.combine(d, datetime.time(DECISION_HOUR))).timestamp()


CLIMATOLOGY_SOURCE = ("2016-01-01", "2025-12-31")  # one cached download, filtered below


def fetch_climatology(cfg) -> Dict[int, List[int]]:
    """Station highs by calendar month over CLIMATOLOGY_YEARS only (the download spans
    further; years inside the scoring window must not leak into the baseline)."""
    y0, y1 = CLIMATOLOGY_YEARS
    highs = fetch_station_highs(cfg, *CLIMATOLOGY_SOURCE)
    return by_month({d: h for d, h in highs.items() if y0 <= int(d[:4]) <= y1})


def by_month(highs: Dict[str, int]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for day, h in highs.items():
        out.setdefault(int(day[5:7]), []).append(h)
    return out


def bracket_probs(cfg, bounds: Dict[str, tuple], mu: float, sigma: float, bias: float, month: int):
    """Live bracket maths (skew-normal, monthly alpha, trailing bias) on arbitrary bracket bounds."""
    model = BracketModel(dataclasses.replace(cfg, bracket_bounds=bounds), trailing_bias=bias)
    return model.compute(ForecastResult(mu=mu, sigma=sigma, source="proxy_replay"), month)


# ── A. Proxy replay ──────────────────────────────────────────────────────────

def daily_mu(hourly: dict, h: int) -> Dict[str, float]:
    """Blended day-ahead mu per local date: live weights over each model's 06:00-21:00 max."""
    peak: Dict[str, Dict[str, float]] = {}
    for model in ("gfs_seamless", "ecmwf_ifs025"):
        values = hourly.get(f"temperature_2m_previous_day{h}_{model}", [])
        for t, v in zip(hourly["time"], values):
            if v is not None and PEAK_HOURS[0] <= t[11:16] <= PEAK_HOURS[1]:
                day = peak.setdefault(t[:10], {})
                day[model] = max(day.get(model, -99.0), v)
    return {d: ECMWF_WEIGHT * m["ecmwf_ifs025"] + GFS_WEIGHT * m["gfs_seamless"]
            for d, m in peak.items() if len(m) == 2}


def replay(cfg, start: str, end: str, with_market: bool = True) -> List[dict]:
    days = [(datetime.date.fromisoformat(start) + datetime.timedelta(n)).isoformat()
            for n in range((datetime.date.fromisoformat(end) - datetime.date.fromisoformat(start)).days + 1)]
    hourly = fetch_previous_runs(cfg, start, end)
    mus = {h: daily_mu(hourly, h) for h in HORIZONS}
    highs = fetch_station_highs(cfg, start, end)
    clim = fetch_climatology(cfg)
    fixed = list(SCORING_BRACKETS.values())

    residuals = {h: [] for h in HORIZONS}  # (day, model-space actual high - mu)
    rows, skipped = [], {"no_forecast": 0, "no_outcome": 0, "warm_up": 0}
    for day in days:
        month = int(day[5:7])
        event = fetch_event(cfg, day) if with_market and day >= FIRST_EVENTS else None
        for h in HORIZONS:
            mu = mus[h].get(day)
            if mu is None:
                skipped["no_forecast"] += 1
                continue
            # Day d settles at the local midnight ending d; the decision is 06:00 on
            # day-(h-1), so only days up to day-h had settled.
            cutoff = (datetime.date.fromisoformat(day) - datetime.timedelta(days=h)).isoformat()
            past = [r for d, r in residuals[h] if d <= cutoff]
            if day not in highs:
                skipped["no_outcome"] += 1
                continue
            residuals[h].append((day, highs[day] + 0.5 - mu))
            if len(past) < MIN_HISTORY:
                skipped["warm_up"] += 1
                continue
            bias = sum(past[-BIAS_WINDOW:]) / len(past[-BIAS_WINDOW:])
            window = past[-SIGMA_WINDOW:]
            mean = sum(window) / len(window)
            sigma = min(max(math.sqrt(sum((r - mean) ** 2 for r in window) / (len(window) - 1)),
                            SIGMA_BOUNDS[0]), SIGMA_BOUNDS[1])
            probs = bracket_probs(cfg, SCORING_BRACKETS, mu, sigma, bias, month)
            row = {"icao": cfg.icao, "date": day, "horizon": h, "month": month,
                   "mu": mu, "sigma": sigma, "bias": bias,
                   "outcome": outcome_vector(highs[day], fixed),
                   "model": normalise(list(probs.values())),
                   "climatology": climatology_probs(clim, month, fixed),
                   "market_eval": None}
            if event:
                row["market_eval"] = market_eval(cfg, event, day, h, mu, sigma, bias, month, clim)
            rows.append(row)
    logger.info(f"{cfg.icao}: proxy replay {len(rows)} rows, skipped {skipped}")
    return rows


def market_eval(cfg, event, day, h, mu, sigma, bias, month, clim) -> Optional[dict]:
    """Model, market and climatology on the event's own brackets at the decision time."""
    decision = decision_ts(cfg, day, h)
    market = []
    for b in event["brackets"]:
        points = [p for p in fetch_price_history(b["token"], day) if p["t"] <= decision]
        market.append(points[-1]["p"] if points and decision - points[-1]["t"] <= MAX_PRICE_AGE_S else None)
    if None in market or sum(market) <= 0:
        return None
    bounds = {b["label"]: b["bounds"] for b in event["brackets"]}
    probs = bracket_probs(cfg, bounds, mu, sigma, bias, month)
    return {"outcome": [int(b["won"]) for b in event["brackets"]],
            "model": normalise([probs[b["label"]] for b in event["brackets"]]),
            "market": normalise(market),
            "climatology": climatology_probs(clim, month, list(bounds.values()))}


# ── B. Live forward score ────────────────────────────────────────────────────

def live_rows(db_path: str, cfg, highs: Dict[str, int], clim) -> List[dict]:
    """Same-day rows for the exact live model: the last scan before 06:00 local on each market date."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        scans = conn.execute("SELECT * FROM scan_snapshots WHERE icao_code = ? ORDER BY scan_at",
                             (cfg.icao,)).fetchall()
        rows = []
        for day in sorted({s["market_date"] for s in scans}):
            if day not in highs:
                continue  # not settled (or no station reading) yet
            cutoff = datetime.datetime.fromtimestamp(decision_ts(cfg, day, 1), datetime.timezone.utc) \
                .replace(tzinfo=None).isoformat()
            before = [s for s in scans if s["market_date"] == day and s["scan_at"] <= cutoff]
            if not before:
                continue
            scan = before[-1]
            probs = json.loads(scan["model_probs"])
            labels = [label for label in cfg.bracket_bounds if label in probs]
            mids = dict(conn.execute("SELECT bracket_label, market_price FROM signal_log WHERE scan_id = ?",
                                     (scan["id"],)).fetchall())
            bounds = [cfg.bracket_bounds[label] for label in labels]
            market = [mids.get(label) for label in labels]
            rows.append({"icao": cfg.icao, "date": day, "horizon": 1, "month": int(day[5:7]),
                         "sigma": scan["sigma"], "bias": scan["trailing_bias"],
                         "outcome": outcome_vector(highs[day], bounds),
                         "model": normalise([probs[label] for label in labels]),
                         "climatology": climatology_probs(clim, int(day[5:7]), bounds),
                         "market": normalise(market) if None not in market and sum(market) > 0 else None})
        return rows
    finally:
        conn.close()


# ── Report ───────────────────────────────────────────────────────────────────

def mean_brier(rows: List[dict], source: str) -> float:
    return sum(brier(r[source], r["outcome"]) for r in rows) / len(rows)


def compare(rows: List[dict], base: str) -> Optional[tuple]:
    """(mean model-minus-base Brier diff, CI low, CI high, n) on rows that have `base`."""
    scored = [r for r in rows if r.get(base)]
    if len(scored) < MIN_VERDICT_DAYS:
        return None
    diffs = [brier(r["model"], r["outcome"]) - brier(r[base], r["outcome"]) for r in scored]
    lo, hi = bootstrap_ci(diffs, [r["date"] for r in scored])
    return sum(diffs) / len(diffs), lo, hi, len(scored)


BLEND_WEIGHTS = [i / 10 for i in range(11)]


def blend_brier(row: dict, w: float) -> float:
    return brier([(1 - w) * m + w * x for m, x in zip(row["market"], row["model"])], row["outcome"])


def blend_lines(rows: List[dict]) -> List[str]:
    """
    Does the model add information the market lacks? Score (1-w)*market +
    w*model over w. Losing to the market head-to-head doesn't rule out edge:
    a best w above zero means disagreements carry signal, and w sizes how
    much to trust the model against the price. The out-of-sample diff picks
    w on every other date (leave-one-date-out), so tuning w can't flatter it.
    """
    rows = [r for r in rows if r.get("market")]
    if len(rows) < MIN_VERDICT_DAYS:
        return [f"  blend: INSUFFICIENT DATA — {len(rows)} days with prices (< {MIN_VERDICT_DAYS})"]
    dates = sorted({r["date"] for r in rows})
    totals = {d: [sum(blend_brier(r, w) for r in rows if r["date"] == d) for w in BLEND_WEIGHTS] for d in dates}
    overall = [sum(t[i] for t in totals.values()) for i in range(len(BLEND_WEIGHTS))]
    best_w = {d: BLEND_WEIGHTS[min(range(len(BLEND_WEIGHTS)), key=lambda i: overall[i] - totals[d][i])]
              for d in dates}
    diffs = [blend_brier(r, best_w[r["date"]]) - brier(r["market"], r["outcome"]) for r in rows]
    lo, hi = bootstrap_ci(diffs, [r["date"] for r in rows])
    w_star = BLEND_WEIGHTS[overall.index(min(overall))]
    adds = hi < 0
    return ["  blend Brier by model weight w: "
            + "  ".join(f"{w:.1f}={t / len(rows):.4f}" for w, t in zip(BLEND_WEIGHTS, overall)),
            f"  best w={w_star:.1f}; out-of-sample blend - market Brier diff {sum(diffs) / len(diffs):+.4f}"
            f"  95% CI [{lo:+.4f}, {hi:+.4f}]  n={len(rows)}",
            "  -> model ADDS information to the market: trade disagreements, trusting the model at weight w"
            if adds else "  -> no evidence the model adds information to the market"]


def reliability(rows: List[dict], source: str) -> List[str]:
    edges = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001]
    out = []
    for lo, hi in zip(edges, edges[1:]):
        pairs = [(p, o) for r in rows for p, o in zip(r[source], r["outcome"]) if lo <= p < hi]
        if pairs:
            mean_p = sum(p for p, _ in pairs) / len(pairs)
            freq = sum(o for _, o in pairs) / len(pairs)
            out.append(f"  [{lo:.2f},{min(hi, 1):.2f})  n={len(pairs):>5}  forecast={mean_p:.3f}  observed={freq:.3f}")
    return out


def verdict_lines(rows: List[dict], label: str) -> List[str]:
    """
    Gate P4: the model must beat climatology (95% CI of the Brier diff
    entirely below zero) and must not be significantly worse than the
    market. Beating the market is evidence of edge (Gate C), not required.
    """
    if len(rows) < MIN_VERDICT_DAYS:
        return [f"{label}: INSUFFICIENT DATA — {len(rows)} days (< {MIN_VERDICT_DAYS})"]
    out = []
    vs = {}
    for base in ("climatology", "market"):
        vs[base] = compare(rows, base)
        if vs[base]:
            d, lo, hi, n = vs[base]
            out.append(f"  model - {base:<11} Brier diff {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  n={n}")
        else:
            out.append(f"  model - {base:<11} pending: {sum(bool(r.get(base)) for r in rows)} days with data")
    beats_clim = bool(vs["climatology"]) and vs["climatology"][2] < 0
    if not vs["market"]:
        return out + [f"{label}: PENDING — {'beats' if beats_clim else 'does NOT beat'} climatology; "
                      "market comparison needs more days with prices"]
    worse = vs["market"][1] > 0
    edge = "BETTER than" if vs["market"][2] < 0 else "WORSE than" if worse else "not distinguishable from"
    out.append(f"  model is {edge} the market")
    passed = beats_clim and not worse
    return out + [f"{label}: {'PASS' if passed else 'FAIL'}"
                  + ("" if beats_clim else " — does not beat climatology")
                  + (" — significantly worse than market" if worse else "")]


def summary_line(rows: List[dict], label: str, sources=("model", "climatology")) -> str:
    if not rows:
        return f"{label:<26} (no data)"
    return f"{label:<26} n={len(rows):>4}  " + "  ".join(f"{s}={mean_brier(rows, s):.4f}" for s in sources)


def report(proxy: List[dict], live: List[dict]) -> str:
    out = ["P4 CALIBRATION REPORT",
           "Multi-category Brier, lower is better; probabilities normalised per day.", "",
           "A. PROXY REPLAY — deterministic day-ahead runs, sigma from past error (not the live sigma)"]
    for h in HORIZONS:
        hr = [r for r in proxy if r["horizon"] == h]
        out.append(summary_line(hr, "same-day (06:00 D)" if h == 1 else "day-ahead (06:00 D-1)"))
        for icao in sorted({r["icao"] for r in hr}):
            city = [r for r in hr if r["icao"] == icao]
            out.append(summary_line(city, f"  {icao}"))
            for year in sorted({r["date"][:4] for r in city}):
                out.append(summary_line([r for r in city if r["date"][:4] == year], f"    {year}"))
    same_day = [r for r in proxy if r["horizon"] == 1]
    priced = [dict(r["market_eval"], date=r["date"]) for r in same_day if r["market_eval"]]
    out += ["", summary_line(priced, "same-day, days with prices", ("model", "market", "climatology"))]
    out += ["", "Reliability, proxy model (same-day, all brackets pooled):"] + reliability(same_day, "model")
    if same_day:
        out.append(f"Mean proxy sigma {sum(r['sigma'] for r in same_day) / len(same_day):.2f}°C, "
                   f"mean |bias| {sum(abs(r['bias']) for r in same_day) / len(same_day):.2f}°C")
    out += ["", "Proxy verdict (same-day):"]
    clim_vs = compare(same_day, "climatology")
    if clim_vs:
        d, lo, hi, n = clim_vs
        out.append(f"  model - climatology Brier diff {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  n={n}"
                   f"  -> {'beats' if hi < 0 else 'does NOT beat'} climatology")
    out += verdict_lines(priced, "  PROXY vs market (priced days)")
    out += ["", "Market blend (same-day, priced days):"] + blend_lines(priced)

    out += ["", "B. LIVE FORWARD — exact live model from scan_snapshots (same-day)"]
    if live:
        out.append(summary_line(live, "all", ("model", "climatology")))
        out.append(summary_line([r for r in live if r["market"]], "days with logged mids",
                                ("model", "market", "climatology")))
        out += ["Reliability, live model:"] + reliability(live, "model")
        out += ["Market blend, live model:"] + blend_lines(live)
    out += verdict_lines(live, "GATE P4 (live model)")
    if _polymarket["missing"]:
        out += ["", f"NOTE: {_polymarket['missing']} Polymarket downloads skipped (blocked) — rerun to complete."]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--icao", choices=sorted(CITIES), help="one city (default: all)")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", help="last market date (default: 2 days ago, so every day has settled)")
    p.add_argument("--db", help="bot database for the live forward score (e.g. a copy from the VPS)")
    p.add_argument("--no-market", action="store_true", help="skip Polymarket downloads")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("hermes.model", "hermes.discovery"):
        logging.getLogger(name).setLevel(logging.ERROR)
    end = args.end or (datetime.date.today() - datetime.timedelta(days=2)).isoformat()
    proxy, live = [], []
    for icao in [args.icao] if args.icao else sorted(CITIES):
        cfg = CITIES[icao]
        proxy += replay(cfg, args.start, end, with_market=not args.no_market)
        if args.db:
            clim = fetch_climatology(cfg)
            live += live_rows(args.db, cfg, fetch_station_highs(cfg, "2026-07-01", end), clim)
    print(report(proxy, live))


if __name__ == "__main__":
    main()
