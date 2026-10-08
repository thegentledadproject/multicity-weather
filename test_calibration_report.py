"""Offline checks for calibration_report.py: scoring and point-in-time rules. Run: python test_calibration_report.py"""
import datetime
import json
import logging
import os
import tempfile
from unittest.mock import patch

import calibration_report as cr
from config.cities import CITIES
from core.model import ForecastResult
from db.ledger import Ledger

CFG = CITIES["WSSS"]


def days(n, start="2026-07-01"):
    return [(datetime.date.fromisoformat(start) + datetime.timedelta(i)).isoformat() for i in range(n)]


def hourly(mu_by_day):
    """Previous-runs block whose GFS and ECMWF 13:00 temps equal mu (other hours colder)."""
    times = [f"{d}T{h:02d}:00" for d in mu_by_day for h in range(24)]
    out = {"time": times}
    for h in cr.HORIZONS:
        for model in ("gfs_seamless", "ecmwf_ifs025"):
            out[f"temperature_2m_previous_day{h}_{model}"] = [
                mu_by_day[t[:10]] if t[11:13] == "13" else 25.0 for t in times]
    return out


def main():
    logging.basicConfig(level=logging.CRITICAL)
    # ── scoring helpers ───────────────────────────────────────────────────────
    assert cr.brier([0, 1, 0], [0, 1, 0]) == 0 and cr.brier([1, 0], [0, 1]) == 2
    assert cr.outcome_vector(28, [(float("-inf"), 29), (29, 30)]) == [1, 0]
    clim = cr.climatology_probs({8: [31, 32, 32, 34]}, 8, [(float("-inf"), 32), (32, 33), (33, float("inf"))])
    assert clim == [2 / 7, 3 / 7, 2 / 7]  # add-one smoothing
    lo, hi = cr.bootstrap_ci([0.0, 1.0] * 20, [f"d{i // 2}" for i in range(40)])
    assert lo == hi == 0.5  # resampling by date: one date's rows move together
    hourly_one = hourly({"2026-07-01": 31.0})
    hourly_one["temperature_2m_previous_day1_gfs_seamless"][13] = 30.0
    assert abs(cr.daily_mu(hourly_one, 1)["2026-07-01"] - (0.6 * 31.0 + 0.4 * 30.0)) < 1e-9

    # ── proxy replay: bias/sigma only from days settled before each decision ──
    span = days(14)
    mus = {d: 31.0 for d in span}
    highs = {d: (33 if i % 2 else 31) for i, d in enumerate(span)}  # residuals alternate +2.5 / +0.5
    with patch.object(cr, "fetch_previous_runs", return_value=hourly(mus)), \
            patch.object(cr, "fetch_station_highs", side_effect=lambda c, s, e: highs if s == span[0] else {"2016-07-01": 32}):
        rows = cr.replay(CFG, span[0], span[-1], with_market=False)
    first = {h: min(r["date"] for r in rows if r["horizon"] == h) for h in cr.HORIZONS}
    assert first == {1: span[10], 2: span[11]}  # day-ahead needs one more settled day
    row = next(r for r in rows if r["date"] == span[10] and r["horizon"] == 1)
    past = [highs[d] + 0.5 - 31.0 for d in span[:10]]  # strictly before the day
    assert abs(row["bias"] - sum(past) / 10) < 1e-9
    assert abs(sum(row["model"]) - 1) < 1e-9 and sum(row["outcome"]) == 1
    # Changing the scored day's own outcome must not change its forecast.
    leaked = dict(highs, **{span[10]: 38})
    with patch.object(cr, "fetch_previous_runs", return_value=hourly(mus)), \
            patch.object(cr, "fetch_station_highs", side_effect=lambda c, s, e: leaked if s == span[0] else {"2016-07-01": 32}):
        again = next(r for r in cr.replay(CFG, span[0], span[-1], with_market=False)
                     if r["date"] == span[10] and r["horizon"] == 1)
    assert again["model"] == row["model"] and again["outcome"] != row["outcome"]

    # ── market prices never come from after the decision time ─────────────────
    event = {"brackets": [{"label": "31°C", "bounds": (float("-inf"), 32.0), "token": "a", "won": False},
                          {"label": "32°C", "bounds": (32.0, float("inf")), "token": "b", "won": True}]}
    decision = cr.decision_ts(CFG, span[10], 1)
    prices = {"a": [{"t": decision - 60, "p": 0.6}, {"t": decision + 1, "p": 0.0}],
              "b": [{"t": decision - 60, "p": 0.4}, {"t": decision + 1, "p": 1.0}]}
    with patch.object(cr, "fetch_price_history", side_effect=lambda token, day: prices[token]):
        ev = cr.market_eval(CFG, event, span[10], 1, 31.0, 1.0, 0.0, 7, {7: [31, 32]})
    assert ev["market"] == [0.6, 0.4] and ev["outcome"] == [0, 1]
    stale = {k: [p for p in v if p["t"] > decision] for k, v in prices.items()}
    with patch.object(cr, "fetch_price_history", side_effect=lambda token, day: stale[token]):
        assert cr.market_eval(CFG, event, span[10], 1, 31.0, 1.0, 0.0, 7, {7: [31]}) is None

    # ── live forward: last scan before 06:00 local, with the mids it logged ────
    with tempfile.TemporaryDirectory() as temp:
        ledger = Ledger(os.path.join(temp, "live.db"))
        fc = ForecastResult(31.0, 0.8, "ensemble_blend")
        early = ledger.log_scan("WSSS", "2026-07-01T21:30:00", "2026-07-02", fc, 0.1, {"31°C": 0.7, "32°C": 0.3})
        late = ledger.log_scan("WSSS", "2026-07-02T03:00:00", "2026-07-02", fc, 0.1, {"31°C": 0.1, "32°C": 0.9})
        for label, mid in (("31°C", 0.55), ("32°C", 0.45)):
            ledger.log_signal("2026-07-02", label, 0, mid, 0, "HOLD_EDGE", icao="WSSS", scan_id=early)
        live = cr.live_rows(os.path.join(temp, "live.db"), CFG, {"2026-07-02": 32}, {7: [32]})
    assert late and len(live) == 1  # 03:00 UTC = 11:00 SGT, after the decision: ignored
    labels = [label for label in CFG.bracket_bounds if label in ("31°C", "32°C")]
    assert live[0]["model"] == [0.7, 0.3] and live[0]["market"] == [0.55, 0.45]
    assert live[0]["outcome"] == [0, 1] and labels == ["31°C", "32°C"]
    assert "INSUFFICIENT DATA" in cr.verdict_lines(live, "x")[0]

    # Blend: a model carrying information the market lacks is detected
    # out of sample; one that is pure noise is not.
    import random
    rng = random.Random(1)
    informed, noise = [], []
    for i in range(60):
        outcome = [0, 0, 0]
        outcome[rng.randrange(3)] = 1
        market = [1 / 3] * 3
        date = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"
        informed.append({"date": date, "outcome": outcome, "market": market,
                         "model": [0.6 if o else 0.2 for o in outcome]})
        noise.append({"date": date, "outcome": outcome, "market": [0.7 if o else 0.15 for o in outcome],
                      "model": cr.normalise([rng.random() for _ in range(3)])})
    assert "ADDS information" in cr.blend_lines(informed)[2]
    assert "best w=0.0" in cr.blend_lines(noise)[1] and "no evidence" in cr.blend_lines(noise)[2]
    assert "INSUFFICIENT" in cr.blend_lines(informed[:5])[0]

    # Breaker: one failure is skipped, MAX_CONSECUTIVE_FAILURES in a row stop downloads.
    def fail():
        raise cr.requests.ConnectionError("flaky")
    with tempfile.TemporaryDirectory() as temp, patch.object(cr, "CACHE", cr.pathlib.Path(temp)), \
            patch.object(cr, "PRICE_REQUEST_GAP_S", 0), \
            patch.dict(cr._polymarket, {"blocked": False, "missing": 0, "failures": 0}):
        assert cr._polymarket_cached("a", fail) is None and not cr._polymarket["blocked"]
        assert cr._polymarket_cached("b", lambda: [1]) == [1] and cr._polymarket["failures"] == 0
        for name in "cde":
            cr._polymarket_cached(name, fail)
        assert cr._polymarket["blocked"] and cr._polymarket_cached("f", lambda: [1]) is None
    print("Calibration report checks passed: scoring, no look-ahead in forecasts, prices or bias.")


if __name__ == "__main__":
    main()
