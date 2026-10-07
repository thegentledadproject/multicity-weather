"""
replay.py — P11: event-driven replay of recorded scans through the production decision engine.

Walks a bot database's scan_snapshots in time order (per city) and, for
every bracket whose recorded edge clears the threshold, rebuilds exactly
what Job 3 would have seen and calls the same core.decision.decide():

  - edge lifecycle   core.edge_state over that bracket's signal_log history
                     up to this scan
  - book             the scan's recorded YES book (book_snapshots). A NO
                     entry buys at the NO asks, which on Polymarket's CLOB
                     mirror the YES bids: price 1 - p, same size.
  - sizing           core.sizing.compute_size (or the fixed validation size)
  - portfolio        core.portfolio over the replay's own simulated positions
  - freshness        FRESH (the decision is replayed at the scan's own time)

ENTER decisions fill at the decided VWAP and are held to settlement (the
replay does not re-run intraday exits; that needs price paths between
scans). Outcomes are the settlement station's METAR high, as in
calibration_report.py.

Per trade it measures the edge chain:
  theoretical  win_prob - mid
  executable   win_prob - VWAP*(1+fee) - buffer   (what decide() required)
  realized     outcome - VWAP*(1+fee)             (per share)
and decomposes P&L into spread+slippage (VWAP vs mid), fees, and forecast
error (realized vs expected value).

Gate P11: mean P&L per trade must be positive with its 95% CI (resampling
whole market dates) above zero, over at least 30 trades.

Usage:
    python replay.py --db hermes_vps_copy.db [--icao WSSS] [--validation]
"""

import argparse
import json
import logging
import sqlite3
from typing import Dict, List

from calibration_report import bootstrap_ci, fetch_station_highs
from config.cities import CITIES, resolve_vault_usd
from core.decision import ENTER, DecisionInputs, decide
from core.edge_state import edge_lifecycle
from core.execution_curve import EXEC_BUFFER
from core.portfolio import position_limits
from core.sizing import TAKER_FEE_RATE, compute_size, compute_validation_size

MIN_TRADES = 30
logger = logging.getLogger("hermes.replay")


def _no_asks(yes_bids) -> List[tuple]:
    return [(round(1 - p, 6), s) for p, s in yes_bids]


def replay_city(conn, cfg, edge_threshold: float, validation: bool, highs: Dict[str, int]) -> List[dict]:
    vault = resolve_vault_usd(cfg)
    all_vaults = sum(resolve_vault_usd(c) for c in CITIES.values())
    positions: List[dict] = []   # simulated open_positions rows
    trades: List[dict] = []
    history: Dict[tuple, list] = {}
    scans = conn.execute("SELECT * FROM scan_snapshots WHERE icao_code = ? ORDER BY scan_at, id",
                         (cfg.icao,)).fetchall()
    for scan in scans:
        probs = json.loads(scan["model_probs"])
        rows = conn.execute("SELECT * FROM signal_log WHERE scan_id = ? AND action != 'NO_PRICE'",
                            (scan["id"],)).fetchall()
        books = {b["bracket"]: b for b in conn.execute(
            "SELECT * FROM book_snapshots WHERE scan_id = ? AND purpose = 'scan'", (scan["id"],))}
        for row in rows:
            key = (scan["market_date"], row["bracket_label"])
            history.setdefault(key, []).append((row["timestamp"], row["edge"]))
            edge = row["edge"]
            if abs(edge) < edge_threshold or row["bracket_label"] not in probs:
                continue
            direction = "BUY" if edge > 0 else "SELL"
            p = probs[row["bracket_label"]]
            win_prob = p if direction == "BUY" else 1 - p
            book = books.get(row["bracket_label"])
            if book is None:
                continue  # no recorded book for this bracket: Job 3 couldn't have priced it either
            bids, asks = json.loads(book["bids"]), json.loads(book["asks"])
            side_asks = [tuple(a) for a in asks] if direction == "BUY" else _no_asks(bids)
            if not side_asks:
                continue
            best_ask = min(a for a, _ in side_asks)
            sizing = (compute_validation_size(win_prob, best_ask, direction) if validation
                      else compute_size(win_prob, best_ask, vault, direction, scan["trailing_bias"]))
            side = "YES" if direction == "BUY" else "NO"
            inputs = DecisionInputs(
                icao=cfg.icao, market_date=scan["market_date"], bracket=row["bracket_label"],
                direction=direction, scan_id=scan["id"], win_prob=win_prob, edge_threshold=edge_threshold,
                edge_state=edge_lifecycle(history[key], edge_threshold).state, freshness="FRESH",
                settlement_valid=True,
                position_open=any(x["bracket_label"] == f"{row['bracket_label']}:{side}"
                                  and x["market_date"] == scan["market_date"] for x in positions),
                q_kelly=sizing.size_usd if sizing.verdict == "EXECUTE" else 0.0,
                limits=position_limits(positions, cfg.icao, scan["market_date"], row["bracket_label"],
                                       direction, vault, all_vaults),
                asks=side_asks,
            )
            decision = decide(inputs)
            if decision.action != ENTER:
                continue
            usd, vwap = decision.quantities["final"], decision.expected["vwap"]
            positions.append({"icao_code": cfg.icao, "market_date": scan["market_date"],
                              "bracket_label": f"{row['bracket_label']}:{side}", "size_usd": usd})
            mid = row["market_price"] if direction == "BUY" else 1 - row["market_price"]
            trades.append({"icao": cfg.icao, "date": scan["market_date"], "bracket": row["bracket_label"],
                           "direction": direction, "usd": usd, "shares": usd / vwap, "vwap": vwap,
                           "mid": mid, "win_prob": win_prob, "expected_ev": decision.expected["ev"]})
    # Settle: a YES share pays 1 if the bracket contains the station high; a NO share if it doesn't.
    settled = []
    for t in trades:
        if t["date"] not in highs:
            continue
        lo, hi = cfg.bracket_bounds[t["bracket"]]
        in_bracket = lo <= highs[t["date"]] < hi
        t["won"] = in_bracket if t["direction"] == "BUY" else not in_bracket
        t["fees"] = t["usd"] * TAKER_FEE_RATE
        t["pnl"] = t["shares"] * t["won"] - t["usd"] - t["fees"]
        t["spread_cost"] = (t["vwap"] - t["mid"]) * t["shares"]
        settled.append(t)
    return settled


def report(trades: List[dict]) -> str:
    out = ["P11 REPLAY — recorded scans through core.decision.decide(), held to settlement", ""]
    if not trades:
        return "\n".join(out + ["No settled trades replayed (needs a bot DB with scan_snapshots and book_snapshots)."])
    n = len(trades)
    total = lambda k: sum(t[k] for t in trades)  # noqa: E731
    mean_per_share = lambda f: sum(f(t) * t["shares"] for t in trades) / total("shares")  # noqa: E731
    out += [f"trades {n}   staked ${total('usd'):.2f}   win rate {sum(t['won'] for t in trades) / n:.1%}",
            "",
            "Edge chain (per share, share-weighted):",
            f"  theoretical  {mean_per_share(lambda t: t['win_prob'] - t['mid']):+.4f}",
            f"  executable   {mean_per_share(lambda t: t['win_prob'] - t['vwap'] * (1 + TAKER_FEE_RATE) - EXEC_BUFFER):+.4f}",
            f"  realized     {mean_per_share(lambda t: t['won'] - t['vwap'] * (1 + TAKER_FEE_RATE)):+.4f}",
            "",
            "P&L decomposition (USD):",
            f"  expected EV at decision   {total('expected_ev'):+.2f}",
            f"  spread + slippage paid    {-total('spread_cost'):+.2f}",
            f"  fees                      {-total('fees'):+.2f}",
            f"  forecast/calibration err  {total('pnl') - total('expected_ev'):+.2f}  (realized - expected)",
            f"  realized P&L              {total('pnl'):+.2f}"]
    for icao in sorted({t["icao"] for t in trades}):
        city = [t for t in trades if t["icao"] == icao]
        out.append(f"  {icao}: {len(city)} trades, P&L {sum(t['pnl'] for t in city):+.2f}")
    out.append("")
    if n < MIN_TRADES:
        out.append(f"GATE P11: INSUFFICIENT DATA — {n} trades (< {MIN_TRADES})")
    else:
        lo, hi = bootstrap_ci([t["pnl"] for t in trades], [t["date"] for t in trades])
        out.append(f"mean P&L/trade {total('pnl') / n:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
        out.append(f"GATE P11: {'PASS' if lo > 0 else 'FAIL'}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", required=True, help="bot database (e.g. a copy from the VPS)")
    p.add_argument("--icao", choices=sorted(CITIES))
    p.add_argument("--edge-threshold", type=float, default=0.08)
    p.add_argument("--validation", action="store_true", help="replay VALIDATION_MODE fixed sizing")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    trades = []
    try:
        for icao in [args.icao] if args.icao else sorted(CITIES):
            cfg = CITIES[icao]
            first = conn.execute("SELECT MIN(market_date), MAX(market_date) FROM scan_snapshots WHERE icao_code = ?",
                                 (icao,)).fetchone()
            if first[0] is None:
                continue
            highs = fetch_station_highs(cfg, first[0], first[1])
            trades += replay_city(conn, cfg, args.edge_threshold, args.validation, highs)
    finally:
        conn.close()
    print(report(trades))


if __name__ == "__main__":
    main()
