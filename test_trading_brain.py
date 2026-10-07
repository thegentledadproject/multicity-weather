"""P5-P13 regressions: edge lifecycle, execution economics, portfolio, decisions, orders, exits, kill switches.
Run: python test_trading_brain.py (offline: mocked HTTP/orders, temporary SQLite)."""
import datetime
import logging
import os
import tempfile
import time
from unittest.mock import Mock, patch

from config.cities import CITIES
from core.edge import EdgeSignal, MarketPrice
from core.model import ForecastResult
from db.ledger import Ledger

DAY = "2026-10-07"
WSSS = CITIES["WSSS"]


def signal(edge=0.20, model_prob=0.60, mid=0.40):
    price = MarketPrice("yes", mid, mid - 0.01, mid + 0.01, 0.02, 400)
    return EdgeSignal("31°C", "yes", model_prob, price, edge, 0.08, no_token_id="no")


def recent_scan(ledger, icao="WSSS"):
    """What Job 2 leaves behind; without a recent scan the FORECAST_FEED kill switch blocks entries."""
    import datetime as dt
    return ledger.log_scan(icao, dt.datetime.utcnow().isoformat(), DAY,
                           ForecastResult(31.0, 0.8, "ensemble_blend"), 0.0, {"31°C": 0.6})


def p5_edge_lifecycle(ledger):
    from core.city_runner import CityRunner
    runner = CityRunner(WSSS, ledger, 100, 0.08, 0.2)
    runner._state.update(token_matrix={"31°C": {"yes": "yes", "no": "no"}}, market_date=DAY)
    fc = ForecastResult(31.0, 0.8, "ensemble_blend")
    with patch("core.city_runner.fetch_gfs_forecast", return_value=fc), \
            patch("core.city_runner.scan_all_brackets", side_effect=lambda **kw: {"31°C": signal()}):
        runner.job_signal_scan()
        first = runner._state["signals"]["31°C"]
        assert not first.actionable and first.gate_reason == "EDGE_EMERGING"
        with ledger._conn() as conn:  # pretend that scan happened 20 minutes ago
            conn.execute("UPDATE signal_log SET timestamp = datetime(timestamp, '-20 minutes')")
        runner.job_signal_scan()
    second = runner._state["signals"]["31°C"]
    assert second.actionable and second.edge_state.state == "ACTIONABLE" and second.edge_state.persistence == 2
    with ledger._conn() as conn:
        assert [r[0] for r in conn.execute("SELECT edge_state FROM signal_log ORDER BY id")] == ["EMERGING", "ACTIONABLE"]


def p8_decision_engine(ledger):
    import json
    from core import decision as dm
    base = dict(icao="WSSS", market_date=DAY, bracket="31°C", direction="BUY", scan_id=1, win_prob=0.60,
                edge_threshold=0.08, edge_state="ACTIONABLE", freshness="FRESH", settlement_valid=True,
                position_open=False, q_kelly=15.0,
                limits={"Qcorrelation": 30.0, "Qgeo": 60.0, "Qbankroll": 100.0, "blocked": ""},
                asks=[(0.41, 1000)], kill_switches=[])
    inputs = dm.DecisionInputs(**base)
    d = dm.decide(inputs)
    assert d.action == dm.ENTER and d.quantities["final"] == 15.0
    assert abs(d.expected["vwap"] - 0.41) < 1e-12 and d.expected["exec_edge"] > 0.08
    # Gate P8: same inputs (even after a JSON round trip) → same decision.
    replayed = dm.DecisionInputs(**json.loads(json.dumps(dm.snapshot(inputs, d)))["inputs"])
    assert dm.decide(replayed) == d == dm.decide(inputs)
    # Final quantity is the binding constraint.
    assert dm.decide(dm.DecisionInputs(**dict(base, limits=dict(base["limits"], Qcorrelation=7.0)))).quantities["final"] == 5.0
    cases = [(dict(kill_switches=["DRAWDOWN"]), dm.BLOCK, "KILL_SWITCH:DRAWDOWN"),
             (dict(freshness="STALE"), dm.BLOCK, "INFO_STALE"),
             (dict(settlement_valid=False), dm.BLOCK, "SETTLEMENT_INVALID"),
             (dict(position_open=True), dm.BLOCK, "POSITION_OPEN"),
             (dict(limits=dict(base["limits"], blocked="OPPOSITE_SIDE_HELD")), dm.BLOCK, "OPPOSITE_SIDE_HELD"),
             (dict(edge_state="EMERGING"), dm.HOLD, "EDGE_EMERGING"),
             (dict(q_kelly=0.0), dm.HOLD, "SIZING_HOLD"),
             (dict(limits=dict(base["limits"], Qcorrelation=0.0)), dm.HOLD, "RISK_CAPACITY"),
             (dict(asks=[(0.55, 1000)]), dm.HOLD, "NO_EXECUTABLE_EDGE"),  # mid-based edge, but not at the ask
             (dict(asks=[]), dm.HOLD, "NO_EXECUTABLE_EDGE")]
    for change, action, reason in cases:
        got = dm.decide(dm.DecisionInputs(**dict(base, **change)))
        assert got.action == action and reason in got.reasons, (change, got)

    # Job 3 end to end (paper): first bracket enters at the decided size; a second
    # bracket of the same event finds the event cap used up (Gate P7).
    from core.city_runner import CityRunner
    runner = CityRunner(WSSS, ledger, 100, 0.08, 0.2)
    runner.client = Mock()
    runner.client.get_order_book.return_value = {"bids": [{"price": "0.39", "size": "1000"}],
                                                 "asks": [{"price": "0.41", "size": "1000"}]}
    s31, s32 = signal(), signal()
    s32.bracket_label, s32.token_id, s32.no_token_id = "32°C", "yes32", "no32"
    for sig in (s31, s32):
        sig.market_date, sig.scanned_at, sig.scan_id = DAY, time.time(), 1
        sig.edge_state = Mock(state="ACTIONABLE")
    runner._state.update(token_matrix={"31°C": {}, "32°C": {}}, market_date=DAY, signals={"31°C": s31, "32°C": s32})
    recent_scan(ledger)
    with patch.object(WSSS, "paper_trading", True):
        runner.job_order_execution()
    positions = ledger.get_open_positions("WSSS")
    assert [(p["bracket_label"], p["size_usd"]) for p in positions] == [("31°C:YES", 15.0)]
    with ledger._conn() as conn:
        rows = conn.execute("SELECT bracket, action, reasons FROM decision_log ORDER BY id").fetchall()
    assert [(r[0], r[1]) for r in rows] == [("31°C", "ENTER"), ("32°C", "HOLD")] and "RISK_CAPACITY" in rows[1][2]


def p9_order_lifecycle(ledger):
    from core.execution import ExecutionEngine
    from core.sizing import SizingResult

    def client(ask=0.41, response=None, raises=None):
        c = Mock()
        c.get_order_book.return_value = {"bids": [{"price": str(ask - 0.02), "size": "1000"}],
                                         "asks": [{"price": str(ask), "size": "1000"}]}
        c.get_balance_allowance.return_value = {"balance": "1000"}
        c.post_order.side_effect = raises
        c.post_order.return_value = response
        return c

    def fresh(token="yes"):
        sig = signal()
        sig.token_id = token
        sig.market_date, sig.scanned_at, sig.scan_id = DAY, time.time(), 1
        return sig

    sizing = SizingResult("EXECUTE", "BUY", 10.0, 0.1, 0.1, 0.1, "test")
    def states():
        with ledger._conn() as conn:
            return [r["state"] for r in conn.execute("SELECT state FROM order_log ORDER BY id")]
    # API timeout: outcome unknown, no position, and the token is blocked from a resend.
    c = client(raises=TimeoutError("read timed out"))
    assert not ExecutionEngine(c, ledger, 100, "WSSS").execute(fresh(), sizing, DAY)
    assert not ledger.get_open_positions() and states() == ["UNKNOWN"]
    assert not ExecutionEngine(c, ledger, 100, "WSSS").execute(fresh(), sizing, DAY)
    assert c.post_order.call_count == 1 and len(ledger.unresolved_orders("WSSS")) == 1
    # Rejection.
    assert not ExecutionEngine(client(response={"status": "unmatched", "success": False}),
                               ledger, 100, "WSSS").execute(fresh("t2"), sizing, DAY)
    # Fill with amounts: the position uses what was actually paid and received.
    filled = {"status": "matched", "success": True, "makingAmount": "9.9", "takingAmount": "24.0"}
    assert ExecutionEngine(client(response=filled), ledger, 100, "WSSS").execute(fresh("t3"), sizing, DAY)
    pos = next(p for p in ledger.get_open_positions() if p["token_id"] == "t3")
    assert pos["size_usd"] == 9.9 and abs(pos["entry_price"] - 9.9 / 24) < 1e-12
    # Fill confirmed without amounts: managed at the quote, flagged.
    no_amounts = {"status": "matched", "success": True, "size_matched": "24"}
    assert ExecutionEngine(client(response=no_amounts), ledger, 100, "WSSS").execute(fresh("t4"), sizing, DAY)
    # Price moved: ask 0.55 is inside the staleness tolerance, but the edge is gone at that price.
    moved = client(ask=0.55, response=filled)
    assert not ExecutionEngine(moved, ledger, 100, "WSSS").execute(fresh("t5"), sizing, DAY)
    moved.post_order.assert_not_called()
    # Paper fills are recorded too.
    assert ExecutionEngine(client(), ledger, 100, "WSSS", paper_trading=True).execute(fresh("t6"), sizing, DAY)
    assert states() == ["UNKNOWN", "REJECTED", "FILLED", "FILLED_ESTIMATED", "PAPER_FILLED"]


def p10_exits(ledger):
    from core import position_monitor as pm
    R = pm.ExitReason

    def exit_for(mid, bid=None, win=None, peak=0.40, **flags):
        price = MarketPrice("tok", mid, bid if bid is not None else mid - 0.01, mid + 0.01, 0.02, 400)
        with patch.object(pm, "fetch_market_price", return_value=price):
            return pm.evaluate_exit("tok", "31°C:YES", 0.40, peak, 0.5, 0.08, 0.2, ledger,
                                    win_prob=win, **flags).reason

    # Gate P10: every exit family, deterministically. Entry 0.40, threshold 0.08.
    assert exit_for(0.45, win=0.70) == R.NONE
    assert exit_for(0.55, peak=0.70, win=0.90) == R.TRAILING_STOP       # 0.55 <= 0.70 * 0.8
    assert exit_for(0.30, win=0.90) == R.STOP_LOSS                      # <= 0.40 - 0.08
    assert exit_for(0.50, win=0.35) == R.MODEL_EXIT                     # 0.35 < 0.50 - 0.08
    assert exit_for(0.55, bid=0.54, win=0.52) == R.SETTLEMENT_EXIT      # bid beats hold value
    assert exit_for(0.55, bid=0.50, win=0.56) == R.PRICE_EXIT           # converged, in profit
    assert exit_for(0.45, win=None) == R.NONE                           # no model view: no model exits
    assert exit_for(0.45, win=0.70, force_time_exit=True) == R.TIME_EXIT
    assert exit_for(0.45, win=0.70, risk_flatten=True) == R.RISK_EXIT
    assert exit_for(0.45, win=0.70, info_stale=True) == R.INFORMATION_EXIT

    # Through PositionMonitor.run (paper): a NO position is judged on 1 - P(bracket),
    # and stops still run when Job 2 has produced no probabilities.
    client = Mock()
    client.get_order_book.return_value = {"bids": [{"price": "0.39", "size": "1000"}],
                                          "asks": [{"price": "0.41", "size": "1000"}]}
    ledger.record_position("no31", "31°C:NO", "WSSS", 0.40, 10.0, DAY, is_paper=True)
    monitor = pm.PositionMonitor(client, ledger, 0.08, 0.2, "WSSS", paper_trading=True)
    price = MarketPrice("no31", 0.40, 0.39, 0.41, 0.02, 400)
    with patch.object(pm, "fetch_market_price", return_value=price):
        held = monitor.run({"31°C": 0.30}, market_date=DAY, probs_age_s=60)   # NO wins 70%: hold
        assert held == [] and ledger.get_open_positions("WSSS")
        out = monitor.run({"31°C": 0.80}, market_date=DAY, probs_age_s=60)    # NO wins 20%: exit
    assert [r["reason"] for r in out] == [R.MODEL_EXIT] and not ledger.get_open_positions("WSSS")
    ledger.record_position("yes32", "32°C:YES", "WSSS", 0.40, 10.0, DAY, is_paper=True)
    with patch.object(pm, "fetch_market_price", return_value=MarketPrice("yes32", 0.30, 0.29, 0.31, 0.02, 400)):
        out = monitor.run({}, market_date=DAY, probs_age_s=None)
    assert [r["reason"] for r in out] == [R.STOP_LOSS]


def p11_replay(ledger):
    import sqlite3
    import replay
    fc = ForecastResult(31.0, 0.8, "ensemble_blend")
    for k, at in enumerate(["2026-10-07T00:00:00", "2026-10-07T01:00:00", "2026-10-07T02:00:00"]):
        sid = ledger.log_scan("WSSS", at, DAY, fc, 0.0, {"31°C": 0.60})
        ledger.log_book(sid, "yes31", "scan", [(0.39, 1000)], [(0.41, 1000)], at, bracket="31°C")
        ledger.log_signal(DAY, "31°C", 0.60, 0.40, 0.20, "SIGNAL_BUY", icao="WSSS", scan_id=sid)
        with ledger._conn() as conn:  # the signal row's time is the scan's time
            conn.execute("UPDATE signal_log SET timestamp = ? WHERE scan_id = ?", (at, sid))
    conn = sqlite3.connect(ledger.db_path)
    conn.row_factory = sqlite3.Row
    with patch.dict(os.environ, {"VAULT_USD_WSSS": "200", "VAULT_USD_WMKK": "100"}):
        trades = replay.replay_city(conn, WSSS, 0.08, False, {DAY: 31})
    conn.close()
    # Scan 1 is EMERGING (held), scan 2 is ACTIONABLE (enters), scan 3 finds the position open.
    assert len(trades) == 1
    t = trades[0]
    # Kelly 30 (15% cap of $200) = event cap 30 ≤ Qgeo 60, depth ample → $30 at 0.41.
    assert t["usd"] == 30.0 and abs(t["vwap"] - 0.41) < 1e-12 and t["won"]
    assert abs(t["pnl"] - (30 / 0.41 - 30 - 30 * 0.02)) < 1e-9
    assert "INSUFFICIENT DATA" in replay.report(trades)
    assert replay._no_asks([(0.49, 100)]) == [(0.51, 100)]  # NO asks mirror YES bids


def p12_shadow(ledger):
    import sqlite3
    import shadow_report
    from core.city_runner import CityRunner
    from core.settlement import SettlementEngine
    runner = CityRunner(WSSS, ledger, 100, 0.08, 0.2)
    runner.client = Mock()
    runner.client.get_order_book.return_value = {"bids": [{"price": "0.39", "size": "1000"}],
                                                 "asks": [{"price": "0.41", "size": "1000"}]}
    sig = signal()
    sig.market_date, sig.scanned_at, sig.scan_id = DAY, time.time(), 1
    sig.edge_state = Mock(state="ACTIONABLE")
    runner._state.update(token_matrix={"31°C": {}}, market_date=DAY, signals={"31°C": sig})
    recent_scan(ledger)
    with patch.object(WSSS, "paper_trading", True):
        runner.job_order_execution()
    pos = ledger.get_open_positions("WSSS")[0]
    # The market resolves YES for 31°C: settlement logs the held YES as an exit at 1.0.
    resolved = Mock()
    resolved.json.return_value = [{"closed": True, "outcomePrices": '["1", "0"]'}]
    with patch("core.settlement.requests.get", return_value=resolved):
        assert SettlementEngine(ledger, WSSS)._check_resolution("yes", "31°C:YES", DAY, position=pos)
    exit_row = ledger.get_exit_log(icao="WSSS")[0]
    shares = 15.0 / 0.41
    assert exit_row["reason"] == "SETTLED" and exit_row["exit_price"] == 1.0
    assert abs(exit_row["realised_pnl"] - (1 - 0.41) * shares) < 1e-9 and exit_row["scan_id"] == 1
    # A NO held into a YES resolution settles at 0.
    ledger.record_position("no32", "32°C:NO", "WSSS", 0.60, 6.0, DAY, is_paper=True, scan_id=2)
    with patch("core.settlement.requests.get", return_value=resolved):
        SettlementEngine(ledger, WSSS)._check_resolution("no32", "32°C:NO", DAY,
                                                          position=ledger.get_open_positions("WSSS")[0])
    assert ledger.get_exit_log(icao="WSSS")[0]["realised_pnl"] == -6.0

    conn = sqlite3.connect(ledger.db_path)
    conn.row_factory = sqlite3.Row
    rows = shadow_report.entries(conn, "WSSS")
    text = shadow_report.report(conn, "WSSS")
    conn.close()
    assert len(rows) == 1 and rows[0]["order"]["state"] == "PAPER_FILLED"
    assert rows[0]["order"]["expected_vwap"] == rows[0]["expected_vwap"]  # paper: no drift
    assert abs(rows[0]["pnl"] - (1 - 0.41) * shares) < 1e-9
    assert "PENDING" in text and "PAPER_FILLED=1" in text


def p13_kill_switches(temp):
    import datetime as dt
    from core import kill_switch as ks
    from core import position_monitor as pm
    from core.city_runner import CityRunner
    names = lambda ledger, **kw: [n for n, _ in ks.tripped(ledger, "WSSS", 200, DAY, **kw)]  # noqa: E731
    db = lambda name: Ledger(os.path.join(temp, f"p13_{name}.db"))  # noqa: E731

    healthy = db("ok")
    recent_scan(healthy)
    assert names(healthy) == []
    assert names(db("none")) == ["FORECAST_FEED"]                                   # never scanned
    assert names(healthy, now=dt.datetime.utcnow() + dt.timedelta(hours=1)) == ["FORECAST_FEED"]  # gone quiet
    prior = db("prior")
    prior.log_scan("WSSS", dt.datetime.utcnow().isoformat(), DAY, ForecastResult(31.5, 1, "fallback"), 0, {})
    assert names(prior) == ["FORECAST_FEED"]                                        # hard prior
    feed = db("feed")
    sid = recent_scan(feed)
    for label, action in (("29°C", "NO_PRICE"), ("30°C", "NO_PRICE"), ("31°C", "HOLD_EDGE")):
        feed.log_signal(DAY, label, 0.3, 0, 0, action, icao="WSSS", scan_id=sid)
    assert names(feed) == ["MARKET_FEED"]
    orders = db("orders")
    recent_scan(orders)
    cid = orders.open_order(icao="WSSS", market_date=DAY, token_id="t", bracket="31°C", direction="BUY",
                            scan_id=1, amount_usd=10, limit_price=0.42, expected_vwap=0.41)
    assert names(orders) == ["ORDER_UNRESOLVED"]
    orders.update_order(cid, "FILLED", 10.0, 10.0 / 0.45)                           # reconciled, but 4c worse
    assert names(orders) == ["SLIPPAGE"]
    calib = db("calib")
    recent_scan(calib)
    calib.log_outcome("WSSS", 31.0, 34.0, DAY)                                      # bias +3°C
    assert names(calib) == ["CALIBRATION"]
    loss = db("loss")
    recent_scan(loss)
    loss.log_exit("a", "31°C", "BUY", "STOP_LOSS", 0.4, 0.1, 30, -25.0, "x", DAY, icao="WSSS")
    assert ks.tripped(loss, "WSSS", 200, DAY) == [("DRAWDOWN", True)]               # > 10% of vault today
    peak = db("peak")
    recent_scan(peak)
    for pnl, day in ((10.0, "2026-10-01"), (-45.0, "2026-10-02")):                  # 45 off a +10 peak > 20%
        peak.log_exit("a", "31°C", "BUY", "STOP_LOSS", 0.4, 0.1, 30, pnl, "x", day, icao="WSSS")
    assert names(peak) == ["DRAWDOWN"]

    # Job 3 blocks with the switch named; Job 5 flattens through RISK_EXIT.
    runner = CityRunner(WSSS, loss, 200, 0.08, 0.2)
    runner.client = Mock()
    runner.client.get_order_book.return_value = {"bids": [{"price": "0.39", "size": "1000"}],
                                                 "asks": [{"price": "0.41", "size": "1000"}]}
    sig = signal()
    sig.market_date, sig.scanned_at, sig.scan_id, sig.edge_state = DAY, time.time(), 1, Mock(state="ACTIONABLE")
    runner._state.update(token_matrix={"31°C": {}}, market_date=DAY, signals={"31°C": sig})
    with patch.object(WSSS, "paper_trading", True):
        runner.job_order_execution()
        with loss._conn() as conn:
            row = conn.execute("SELECT action, reasons FROM decision_log").fetchone()
        assert row["action"] == "BLOCK" and "KILL_SWITCH:DRAWDOWN" in row["reasons"]
        loss.record_position("held", "30°C:YES", "WSSS", 0.40, 10.0, DAY, is_paper=True)
        with patch.object(pm, "fetch_market_price", return_value=MarketPrice("held", 0.45, 0.44, 0.46, 0.02, 400)):
            runner.job_position_monitor()
    assert not loss.get_open_positions("WSSS")
    assert loss.get_exit_log(icao="WSSS")[0]["reason"] == "RISK_EXIT"


def main():
    logging.basicConfig(level=logging.CRITICAL)
    with tempfile.TemporaryDirectory() as temp:
        p5_edge_lifecycle(Ledger(os.path.join(temp, "p5.db")))
        p8_decision_engine(Ledger(os.path.join(temp, "p8.db")))
        p9_order_lifecycle(Ledger(os.path.join(temp, "p9.db")))
        p10_exits(Ledger(os.path.join(temp, "p10.db")))
        p11_replay(Ledger(os.path.join(temp, "p11.db")))
        p12_shadow(Ledger(os.path.join(temp, "p12.db")))
        p13_kill_switches(temp)
    print("Trading-brain checks passed: P5, P8, P9, P10, P11, P12, P13")


if __name__ == "__main__":
    main()
