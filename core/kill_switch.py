"""
core/kill_switch.py — P13: hard stops for controlled production.

Evaluated per city every Job 3 / Job 5 cycle from the bot's own records.
Any tripped switch BLOCKs new entries (core/decision.py reason
KILL_SWITCH:<name>); a switch marked flatten also closes open positions
through the RISK_EXIT family (core/position_monitor.py). A switch clears
itself once its condition does: the stop is on the data, not a latch.

  FORECAST_FEED     no successful scan for MAX_SCAN_GAP_S (feed failing,
                    or Job 2 aborting), or the last scan ran on the hard prior
  MARKET_FEED       over half the brackets in the last scan had no price
  ORDER_UNRESOLVED  an order is SUBMITTED/UNKNOWN: it may have filled, so the
                    ledger may not match the wallet (API inconsistency /
                    position mismatch) until it is reconciled
  DRAWDOWN          today's realized P&L below -MAX_DAILY_LOSS_PCT of the
                    vault, or realized P&L down MAX_DRAWDOWN_PCT of the vault
                    from its peak — flattens
  SLIPPAGE          the last SLIPPAGE_WINDOW live fills averaged more than
                    MAX_MEAN_SLIPPAGE above their quotes
  CALIBRATION       |trailing bias| above MAX_ABS_BIAS: forecasts have
                    drifted from the station beyond what bias correction
                    should be absorbing

Settlement ambiguity needs no switch: core/discovery.py's gate already
leaves the city with no token matrix, so nothing can be entered.

Deployment stages map onto existing per-city settings, never automatic:
  SHADOW      config.cities paper_trading=True   (live data, simulated fills)
  TINY        paper_trading=False + VALIDATION_MODE_<ICAO>=true ($1 orders)
  FRACTIONAL  paper_trading=False, validation off: Kelly sizing at
              KELLY_FRACTION with the P7 portfolio caps
Promote only on gate evidence (calibration_report, replay, shadow_report),
never on days online — and never straight from paper to full Kelly.
"""

import datetime
import os
from typing import List, Tuple

MAX_SCAN_GAP_S      = float(os.getenv("KILL_MAX_SCAN_GAP_S", str(45 * 60)))
MAX_NO_PRICE_FRAC   = 0.5
MAX_DAILY_LOSS_PCT  = float(os.getenv("KILL_MAX_DAILY_LOSS_PCT", "0.10"))
MAX_DRAWDOWN_PCT    = float(os.getenv("KILL_MAX_DRAWDOWN_PCT", "0.20"))
SLIPPAGE_WINDOW     = 10
MAX_MEAN_SLIPPAGE   = 0.02
MAX_ABS_BIAS        = float(os.getenv("KILL_MAX_ABS_BIAS", "2.0"))

FLATTEN = {"DRAWDOWN"}


def tripped(ledger, icao: str, vault_usd: float, market_date: str,
            now: datetime.datetime = None) -> List[Tuple[str, bool]]:
    """[(switch name, flatten?)] currently tripped for this city."""
    now = now or datetime.datetime.utcnow()
    icao = icao.upper()
    out = []
    with ledger._conn() as conn:
        scan = conn.execute("SELECT id, scan_at, forecast_source FROM scan_snapshots WHERE icao_code = ? "
                            "ORDER BY scan_at DESC, id DESC LIMIT 1", (icao,)).fetchone()
        if scan is None or (now - datetime.datetime.fromisoformat(scan["scan_at"])).total_seconds() > MAX_SCAN_GAP_S \
                or scan["forecast_source"] in ("fallback", "none"):
            out.append("FORECAST_FEED")
        if scan is not None:
            actions = [r[0] for r in conn.execute("SELECT action FROM signal_log WHERE scan_id = ?", (scan["id"],))]
            if actions and sum(a == "NO_PRICE" for a in actions) / len(actions) > MAX_NO_PRICE_FRAC:
                out.append("MARKET_FEED")
        if conn.execute("SELECT 1 FROM order_log WHERE icao_code = ? AND state IN ('SUBMITTED', 'UNKNOWN') "
                        "LIMIT 1", (icao,)).fetchone():
            out.append("ORDER_UNRESOLVED")
        pnls = [r[0] for r in conn.execute("SELECT realised_pnl FROM exit_log WHERE icao_code = ? ORDER BY id", (icao,))]
        today = sum(r[0] for r in conn.execute(
            "SELECT realised_pnl FROM exit_log WHERE icao_code = ? AND market_date = ?", (icao, market_date)))
        equity, peak, worst = 0.0, 0.0, 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            worst = max(worst, peak - equity)
        if today < -MAX_DAILY_LOSS_PCT * vault_usd or worst > MAX_DRAWDOWN_PCT * vault_usd:
            out.append("DRAWDOWN")
        fills = conn.execute("SELECT filled_usd, filled_shares, expected_vwap FROM order_log WHERE icao_code = ? "
                             "AND state = 'FILLED' AND filled_shares > 0 ORDER BY id DESC LIMIT ?",
                             (icao, SLIPPAGE_WINDOW)).fetchall()
        if fills and sum(f[0] / f[1] - f[2] for f in fills) / len(fills) > MAX_MEAN_SLIPPAGE:
            out.append("SLIPPAGE")
    if abs(ledger.fetch_trailing_bias(icao)) > MAX_ABS_BIAS:
        out.append("CALIBRATION")
    return [(name, name in FLATTEN) for name in out]
