"""Fresh-run edge research: does the market move toward a new model run only after it is public?

For each run R (ECMWF IFS 0.25 and GFS, 00/06/12/18Z) and market day D:
  signal  = this run's forecast high for D minus the previous run's (same model)
  market  = expected temperature under the bracket prices, E = sum p_i * mid_i
Regress the market's move on the signal before and after the run's public time A.
A slope after A means the market is slow to absorb new runs: an edge window.
"actual - market" slopes test whether the run change still predicts the outcome
beyond the price.

Usage (from the repo root; downloads ~2,050 runs into .cache/calibration/run_*,
about 6h the first time):
    python research/freshrun.py [availability shift hours]

Finding (2026-10-09, 2,999 releases over 250 market days): per 1°C of
run-to-run change, actual minus market at release +0.054°C [+0.027, +0.085];
the market absorbs about half within 6h. Real but worth ~1 point of bracket
probability, below taker fee plus spread: not tradable by taking liquidity.
"""
import collections
import datetime
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pytz

import calibration_report as cr
from config.cities import CITIES

cr._polymarket["blocked"] = True   # prices from cache only
SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
MODELS = {"ecmwf_ifs025": 8, "gfs_seamless": 5}   # hours from run time to public availability (approx.)
AVAIL_SHIFT_H = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0   # sensitivity on availability time


def run_temps(cfg, model, run):
    """{UTC hour datetime: temperature} for one run, cached."""
    name = f"run_{cfg.icao}_{model}_{run:%Y%m%d%H}"
    path = cr.CACHE / f"{name}.json"
    if not path.exists():
        time.sleep(0.3)
    try:
        h = cr.cached(name, lambda: (cr._get(SINGLE_RUNS_URL, {
            "latitude": cfg.lat, "longitude": cfg.lon, "hourly": "temperature_2m", "models": model,
            "run": run.strftime("%Y-%m-%dT%H:%M"), "forecast_hours": 96}) or {}).get("hourly"))
    except Exception as e:
        print("skip", name, e)
        return None
    if not h:
        return None
    return {datetime.datetime.fromisoformat(t): v for t, v in zip(h["time"], h["temperature_2m"]) if v is not None}


def forecast_high(temps, cfg, day):
    tz = pytz.timezone(cfg.timezone)
    start = tz.localize(datetime.datetime.fromisoformat(day)).astimezone(pytz.utc).replace(tzinfo=None)
    vals = [temps[start + datetime.timedelta(hours=i)] for i in range(24) if start + datetime.timedelta(hours=i) in temps]
    return max(vals) if len(vals) == 24 else None


def mid(lo, hi):
    return hi - 0.5 if lo == float("-inf") else lo + 0.5   # upper open bracket: lo + 0.5 too


def market_e(histories, bounds, ts):
    ps = []
    for hist in histories:
        pts = [p for p in hist if ts - 3600 < p["t"] <= ts + 120]
        if not pts:
            return None
        ps.append(pts[-1]["p"])
    if sum(ps) <= 0:
        return None
    return sum(p * mid(lo, hi) for p, (lo, hi) in zip(cr.normalise(ps), bounds))


def slope(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0


def slope_ci(rows, key, n=2000):
    """Slope of market move on signal, 95% CI resampling whole market days."""
    by_day = collections.defaultdict(list)
    for r in rows:
        by_day[(r["icao"], r["day"])].append(r)
    clusters = list(by_day.values())
    rng = random.Random(0)
    est = slope([r["signal"] for r in rows], [r[key] for r in rows])
    boots = []
    for _ in range(n):
        s = [r for _ in clusters for r in rng.choice(clusters)]
        boots.append(slope([r["signal"] for r in s], [r[key] for r in s]))
    boots.sort()
    return est, boots[int(0.025 * n)], boots[int(0.975 * n)]


rows = []
for icao, cfg in CITIES.items():
    tz = pytz.timezone(cfg.timezone)
    for path in sorted(cr.CACHE.glob(f"event_{icao}_*.json")):
        day = path.stem.split("_")[2]
        event = cr.fetch_event(cfg, day)
        if not event:
            continue
        histories = [cr.fetch_price_history(b["token"], day) for b in event["brackets"]]
        bounds = [b["bounds"] for b in event["brackets"]]
        actual = mid(*next(b["bounds"] for b in event["brackets"] if b["won"]))
        d0 = datetime.datetime.fromisoformat(day)
        cutoff = tz.localize(d0 + datetime.timedelta(hours=10)).timestamp()   # before station obs dominate
        for model, lag in MODELS.items():
            runs = [d0 - datetime.timedelta(hours=6 * k) for k in range(2, 10)][::-1]   # D-2 12Z .. D-1 18Z
            prev = None
            for run in runs:
                temps = run_temps(cfg, model, run)
                high = forecast_high(temps, cfg, day) if temps else None
                if high is not None and prev is not None:
                    a = (run + datetime.timedelta(hours=lag + AVAIL_SHIFT_H)).replace(tzinfo=pytz.utc).timestamp()
                    if a <= cutoff:
                        e = {k: market_e(histories, bounds, a + k * 3600) for k in (-3, 0, 1, 3, 6)}
                        if None not in e.values():
                            rows.append({"icao": icao, "day": day, "model": model, "signal": high - prev,
                                         "pre": e[0] - e[-3], "post1": e[1] - e[0], "post3": e[3] - e[0],
                                         "post6": e[6] - e[0],
                                         # still predictive after release? actual minus market at A / A+6h
                                         "resid0": actual - e[0], "resid6": actual - e[6]})
                prev = high if high is not None else prev

print(f"rows={len(rows)}  days={len({(r['icao'], r['day']) for r in rows})}  availability shift {AVAIL_SHIFT_H:+.0f}h")
print("Slope of market expected-temp move (°C) per °C of run-to-run change, 95% CI by market day:")
for model in list(MODELS) + ["both"]:
    sub = [r for r in rows if model in ("both", r["model"])]
    if len(sub) < 30:
        continue
    print(f"  {model}  n={len(sub)}  mean |signal|={sum(abs(r['signal']) for r in sub) / len(sub):.2f}°C")
    for key, label in (("pre", "A-3h -> A  (before public)"), ("post1", "A -> A+1h"),
                       ("post3", "A -> A+3h"), ("post6", "A -> A+6h"),
                       ("resid0", "actual - market at A"), ("resid6", "actual - market at A+6h")):
        est, lo, hi = slope_ci(sub, key)
        print(f"    {label:<28} slope {est:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
