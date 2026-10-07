"""
core/portfolio.py — P7: correlation-aware exposure limits.

Risk-factor hierarchy:  city (ICAO) -> market date -> daily Tmax -> bracket.
Every bracket of one city/date is an outcome of the same daily high, so they
are one risk: the EVENT. Neighbouring cities on the same date share weather
regimes (WSSS and WMKK are ~300 km apart), so same-date exposure is capped
across cities too. Correlation shapes how much may be held — it never
alters the model's probabilities.

Per-entry quantity limits, each in USD (the caller adds Qkelly and Qdepth;
the final size is the minimum of all of them):

  Qcorrelation  event cap: MAX_EVENT_PCT of the city vault, shared by all
                brackets and sides of that city/date
  Qgeo          same-date cap across all cities: MAX_SAME_DAY_PCT of the
                combined vaults
  Qbankroll     the city vault minus everything already open for the city

Holding YES and NO on the same bracket is a pointless self-hedge, so the
opposite side of a held bracket is blocked outright.
"""

import os
from typing import Dict, Iterable

MAX_EVENT_PCT    = float(os.getenv("MAX_EVENT_PCT", "0.15"))
MAX_SAME_DAY_PCT = float(os.getenv("MAX_SAME_DAY_PCT", "0.20"))

OPPOSITE_SIDE_HELD = "OPPOSITE_SIDE_HELD"


def position_limits(
    positions: Iterable, icao: str, market_date: str, bracket: str, direction: str,
    vault_usd: float, all_vaults_usd: float,
) -> Dict:
    """
    positions: open_positions rows (all cities) with icao_code, market_date,
    bracket_label ("31°C:YES"), size_usd. direction: "BUY" (YES) or "SELL" (NO).
    Returns {"Qcorrelation", "Qgeo", "Qbankroll", "blocked": reason or ""}.
    """
    positions = list(positions)
    side = "YES" if direction == "BUY" else "NO"
    event = [p for p in positions if p["icao_code"] == icao and p["market_date"] == market_date]
    held = {p["bracket_label"] for p in event}
    blocked = OPPOSITE_SIDE_HELD if f"{bracket}:{'NO' if side == 'YES' else 'YES'}" in held else ""

    exposure = lambda rows: sum(float(p["size_usd"]) for p in rows)  # noqa: E731
    return {
        "Qcorrelation": max(0.0, vault_usd * MAX_EVENT_PCT - exposure(event)),
        "Qgeo": max(0.0, all_vaults_usd * MAX_SAME_DAY_PCT
                    - exposure(p for p in positions if p["market_date"] == market_date)),
        "Qbankroll": max(0.0, vault_usd - exposure(p for p in positions if p["icao_code"] == icao)),
        "blocked": blocked,
    }


if __name__ == "__main__":
    row = lambda icao, day, label, usd: {"icao_code": icao, "market_date": day,  # noqa: E731
                                         "bracket_label": label, "size_usd": usd}
    open_ = [row("WSSS", "D", "31°C:YES", 20), row("WSSS", "D", "32°C:NO", 5),
             row("WMKK", "D", "33°C:YES", 30), row("WSSS", "D-1", "30°C:YES", 10)]
    lim = position_limits(open_, "WSSS", "D", "33°C", "BUY", vault_usd=200, all_vaults_usd=300)
    assert lim["Qcorrelation"] == 200 * 0.15 - 25        # other brackets of the event count
    assert lim["Qgeo"] == 300 * 0.20 - 55                # WMKK same date counts, D-1 doesn't
    assert lim["Qbankroll"] == 200 - 35 and lim["blocked"] == ""
    # Gate P7: a second bracket can't take a fresh full allocation.
    assert position_limits([row("WSSS", "D", "31°C:YES", 30)], "WSSS", "D", "32°C", "BUY", 200, 300)["Qcorrelation"] == 0
    assert position_limits(open_, "WSSS", "D", "31°C", "SELL", 200, 300)["blocked"] == OPPOSITE_SIDE_HELD
    print("portfolio checks passed")
