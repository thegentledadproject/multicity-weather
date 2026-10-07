"""
shadow_report.py — P12: do the engine's assumptions survive live (paper/shadow) conditions?

Reads a bot database and compares what each ENTER decision assumed with
what happened afterwards. Run it against the VPS DB while cities are in
paper trading; the plan says to judge on these discrepancies, not on P&L.

  execution drift   order_log.expected_vwap (book at execution) minus the
                    decision's expected VWAP (book at decision): the cost of
                    the delay between deciding and ordering
  fill vs quote     live fills only: actual price paid vs expected_vwap
                    (paper fills are at the quote by construction)
  depth             share of entries where book depth, not risk or Kelly,
                    set the size
  EV vs realized    per closed trade, realized P&L minus the decision's
                    expected EV, with a date-clustered 95% CI
  edge lifetime     how long ACTIONABLE edges stayed actionable (signal_log)
  orders            counts by state; any UNKNOWN order needs reconciling

Model calibration (probability vs outcome) is calibration_report.py --db.

Gate P12: PASS once >= 30 closed trades show mean |execution drift| <= 1c
and a realized-minus-expected CI that does not sit entirely below zero;
FAIL if it does (live is systematically worse than the decisions assumed).

Usage:
    python shadow_report.py --db hermes_vps_copy.db [--icao WSSS]
"""

import argparse
import datetime
import json
import sqlite3
import statistics
from typing import List

from calibration_report import bootstrap_ci

MIN_TRADES = 30
MAX_MEAN_DRIFT = 0.01


def _rows(conn, sql, *params):
    return conn.execute(sql, params).fetchall()


def entries(conn, icao=None) -> List[dict]:
    """ENTER decisions joined to their order and, if closed, their exit."""
    where, params = ("AND d.icao_code = ?", [icao]) if icao else ("", [])
    out = []
    for d in _rows(conn, f"SELECT * FROM decision_log d WHERE action = 'ENTER' {where} ORDER BY id", *params):
        snap = json.loads(d["snapshot"])["decision"]
        order = conn.execute(
            "SELECT * FROM order_log WHERE icao_code = ? AND scan_id IS ? AND bracket = ? AND direction = ? "
            "AND created_at >= ? ORDER BY id LIMIT 1",
            (d["icao_code"], d["scan_id"], d["bracket"], d["direction"], d["decided_at"])).fetchone()
        exit_row = conn.execute(
            "SELECT * FROM exit_log WHERE icao_code = ? AND scan_id IS ? AND bracket_label = ? ORDER BY id LIMIT 1",
            (d["icao_code"], d["scan_id"], d["bracket"])).fetchone()
        q = snap["quantities"]
        out.append({"date": d["market_date"], "expected_vwap": snap["expected"]["vwap"],
                    "expected_ev": snap["expected"]["ev"],
                    "depth_bound": q["Qdepth"] <= min(q["Qkelly"], q["Qcorrelation"], q["Qgeo"], q["Qbankroll"]),
                    "order": dict(order) if order else None,
                    "pnl": exit_row["realised_pnl"] if exit_row else None})
    return out


def edge_lifetimes(conn, icao=None) -> List[float]:
    """Hours each ACTIONABLE run lasted before the bracket's edge stopped being ACTIONABLE."""
    where, params = ("AND icao_code = ?", [icao]) if icao else ("", [])
    rows = _rows(conn, f"SELECT icao_code, date, bracket_label, timestamp, edge_state FROM signal_log "
                       f"WHERE edge_state != '' {where} ORDER BY icao_code, date, bracket_label, id", *params)
    lifetimes, start, key = [], None, None
    ts = lambda s: datetime.datetime.fromisoformat(s).timestamp()  # noqa: E731
    for r in rows:
        k = (r["icao_code"], r["date"], r["bracket_label"])
        if k != key:
            start, key = None, k
        if r["edge_state"] == "ACTIONABLE" and start is None:
            start = r["timestamp"]
        elif r["edge_state"] != "ACTIONABLE" and start is not None:
            lifetimes.append((ts(r["timestamp"]) - ts(start)) / 3600)
            start = None
    return lifetimes


def report(conn, icao=None) -> str:
    rows = entries(conn, icao)
    out = ["P12 SHADOW REPORT — decision assumptions vs what happened", ""]
    orders = [r["order"] for r in rows if r["order"]]
    drift = [o["expected_vwap"] - r["expected_vwap"] for r in rows for o in [r["order"]] if o]
    out.append(f"ENTER decisions {len(rows)}, with an order {len(orders)}")
    if drift:
        out.append(f"execution drift (exec-book VWAP - decision VWAP): mean {statistics.mean(drift):+.4f}, "
                   f"mean |drift| {statistics.mean(abs(x) for x in drift):.4f}")
    live = [o for o in orders if o["state"] in ("FILLED",) and o["filled_shares"]]
    if live:
        slip = [o["filled_usd"] / o["filled_shares"] - o["expected_vwap"] for o in live]
        out.append(f"live fill vs quote: mean {statistics.mean(slip):+.4f} over {len(live)} fills")
    if rows:
        out.append(f"depth-limited entries: {sum(r['depth_bound'] for r in rows) / len(rows):.0%}")
    where, params = ("WHERE icao_code = ?", [icao]) if icao else ("", [])
    states = _rows(conn, f"SELECT state, COUNT(*) n FROM order_log {where} GROUP BY state", *params)
    out.append("orders by state: " + (", ".join(f"{s['state']}={s['n']}" for s in states) or "none"))
    unknown = sum(s["n"] for s in states if s["state"] in ("SUBMITTED", "UNKNOWN"))
    if unknown:
        out.append(f"!! {unknown} unresolved order(s): reconcile against the wallet (Ledger.update_order)")
    life = edge_lifetimes(conn, icao)
    if life:
        out.append(f"ACTIONABLE edge lifetime: median {statistics.median(life):.2f}h over {len(life)} runs")

    closed = [r for r in rows if r["pnl"] is not None]
    out += ["", f"closed trades {len(closed)}"]
    if closed:
        exp, real = sum(r["expected_ev"] for r in closed), sum(r["pnl"] for r in closed)
        out.append(f"expected EV {exp:+.2f}  realized {real:+.2f}  (realized - expected {real - exp:+.2f})")
    if len(closed) < MIN_TRADES:
        out.append(f"GATE P12: PENDING — {len(closed)} closed trades (< {MIN_TRADES})")
        return "\n".join(out)
    diffs = [r["pnl"] - r["expected_ev"] for r in closed]
    lo, hi = bootstrap_ci(diffs, [r["date"] for r in closed])
    out.append(f"realized - expected per trade {statistics.mean(diffs):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    drift_ok = not drift or statistics.mean(abs(x) for x in drift) <= MAX_MEAN_DRIFT
    if hi < 0:
        out.append("GATE P12: FAIL — live results are systematically worse than the decisions assumed")
    elif not drift_ok:
        out.append(f"GATE P12: FAIL — mean execution drift above {MAX_MEAN_DRIFT}")
    else:
        out.append("GATE P12: PASS")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", required=True)
    p.add_argument("--icao")
    args = p.parse_args()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        print(report(conn, args.icao.upper() if args.icao else None))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
