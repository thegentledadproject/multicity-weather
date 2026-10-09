"""Intraday edge research: does the market lag the station's observations?

At hour t local on market day D, the only thing known beyond the forecast is
the running max M_t of METAR readings so far. Model: P(final = M_t + j) from
2016-2023 history at the same local hour (no look-ahead into 2026 events).
Compared with the market's price at t, as in P4.

Usage (from the repo root, after calibration_report.py has filled .cache/calibration/):
    python research/intraday.py [half-spread cost per share] [METAR lag minutes]

Finding (2026-10-09, 250 market days Jun-Oct 2026): no edge at hourly
resolution. The blend test's CIs include zero at every hour, and brackets the
readings rule out are priced at ~0 within the hour. Price points are stamped
hh:00:0x — taking "last point <= hh:00" uses an hour-old price and fakes a
large afternoon edge.
"""
import collections
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pytz

import calibration_report as cr
from config.cities import CITIES

cr._polymarket["blocked"] = True   # cache only, no downloads
OBS_LAG_MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 10  # a METAR is public a few minutes after its valid time
HOURS = (8, 10, 12, 13, 14, 15, 16, 17, 18, 20)


def obs(icao):
    """{local date: [(local datetime, whole °C)]} across all cached ASOS files."""
    out = collections.defaultdict(dict)
    for path in sorted(cr.CACHE.glob(f"asos_{icao}_*.json")):
        text = json.loads(path.read_text())
        for line in text.strip().splitlines()[1:]:
            _, valid, tmpf = line.split(",")
            if tmpf in ("M", ""):
                continue
            t = datetime.datetime.strptime(valid, "%Y-%m-%d %H:%M")
            out[t.date().isoformat()][t] = round((float(tmpf) - 32) * 5 / 9)
    return {d: sorted(v.items()) for d, v in out.items()}


def running_max(day_obs, day, hour):
    cutoff = datetime.datetime.fromisoformat(day) + datetime.timedelta(hours=hour, minutes=-OBS_LAG_MIN)
    seen = [c for t, c in day_obs if t <= cutoff]
    return max(seen) if seen else None


def delta_dist(o, hour):
    """P(final - M_t = j) at this hour, 2016-2023, add-half smoothing over j in 0..8."""
    counts = collections.Counter()
    for day, day_obs in o.items():
        if not ("2016" <= day[:4] <= "2023") or len(day_obs) < 30:
            continue
        m = running_max(day_obs, day, hour)
        if m is not None:
            counts[min(max(max(c for _, c in day_obs) - m, 0), 8)] += 1
    total = sum(counts.values()) + 0.5 * 9
    return {j: (counts[j] + 0.5) / total for j in range(9)}


def bracket_probs(m, dist, bounds):
    return cr.normalise([sum(p for j, p in dist.items() if lo <= m + j < hi) + 1e-6 for lo, hi in bounds])


rows = collections.defaultdict(list)
dead = collections.defaultdict(list)
for icao, cfg in CITIES.items():
    o = obs(icao)
    tz = pytz.timezone(cfg.timezone)
    dists = {h: delta_dist(o, h) for h in HOURS}
    for path in sorted(cr.CACHE.glob(f"event_{icao}_*.json")):
        day = path.stem.split("_")[2]
        event = cr.fetch_event(cfg, day)
        if not event or day not in o:
            continue
        histories = [cr.fetch_price_history(b["token"], day) for b in event["brackets"]]
        bounds = [b["bounds"] for b in event["brackets"]]
        outcome = [int(b["won"]) for b in event["brackets"]]
        for h in HOURS:
            ts = tz.localize(datetime.datetime.fromisoformat(day) + datetime.timedelta(hours=h)).timestamp()
            market = []
            for hist in histories:
                # points are stamped a few seconds after the hour: the hh:00:07 point is the hh:00 price
                pts = [p for p in hist if ts - 3600 < p["t"] <= ts + 120]
                market.append(pts[-1]["p"] if pts else None)
            m = running_max(o[day], day, h)
            if None in market or sum(market) <= 0 or m is None:
                continue
            rows[h].append({"date": day, "icao": icao, "outcome": outcome, "market": cr.normalise(market),
                            "model": bracket_probs(m, dists[h], bounds), "raw": market, "m": m, "bounds": bounds})
            for (lo, hi), p in zip(bounds, market):
                if hi <= m:  # observations already rule this bracket out
                    dead[h].append(p)

print("hour  n    model   market  blend best w / out-of-sample diff [CI]")
for h in HOURS:
    r = rows[h]
    if not r:
        continue
    bl = cr.blend_lines(r)
    print(f"{h:02d}:00 {len(r):>3}  {cr.mean_brier(r, 'model'):.4f}  {cr.mean_brier(r, 'market'):.4f}  "
          + bl[1].strip().replace("out-of-sample blend - market Brier diff ", ""))

print("\nDead brackets (upper bound <= running max) still priced, YES price:")
for h in HOURS:
    d = dead[h]
    if d:
        print(f"{h:02d}:00  n={len(d):>4}  mean={sum(d) / len(d):.4f}  >=0.02: {sum(p >= 0.02 for p in d):>3}  "
              f">=0.05: {sum(p >= 0.05 for p in d):>3}  max={max(d):.3f}")

# Naive trade on the running-max bracket: buy YES at the market price when the
# model says it is at least EDGE more likely; settle at 1/0. No fees or depth.
COST = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0   # half-spread paid over the mid, per share
print(f"\nNaive YES trades where model - price >= edge; pay price + {COST:.2f} + 2% fee, settle 1/0:")
for edge in (0.10, 0.20):
    for h in HOURS:
        trades = [(r["model"][i] - r["raw"][i], r["outcome"][i] - (r["raw"][i] + COST) * 1.02)
                  for r in rows[h] for i in range(len(r["raw"])) if r["model"][i] - r["raw"][i] >= edge]
        if trades:
            pnl = sum(t[1] for t in trades)
            print(f"edge>={edge:.2f} {h:02d}:00  trades={len(trades):>3}  "
                  f"win rate={sum(t[1] > 0 for t in trades) / len(trades):.2f}  pnl/share={pnl / len(trades):+.3f}")
