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
    with patch.object(WSSS, "paper_trading", True):
        runner.job_order_execution()
    positions = ledger.get_open_positions("WSSS")
    assert [(p["bracket_label"], p["size_usd"]) for p in positions] == [("31°C:YES", 15.0)]
    with ledger._conn() as conn:
        rows = conn.execute("SELECT bracket, action, reasons FROM decision_log ORDER BY id").fetchall()
    assert [(r[0], r[1]) for r in rows] == [("31°C", "ENTER"), ("32°C", "HOLD")] and "RISK_CAPACITY" in rows[1][2]


def main():
    logging.basicConfig(level=logging.CRITICAL)
    with tempfile.TemporaryDirectory() as temp:
        p5_edge_lifecycle(Ledger(os.path.join(temp, "p5.db")))
        p8_decision_engine(Ledger(os.path.join(temp, "p8.db")))
    print("Trading-brain checks passed: P5, P8")


if __name__ == "__main__":
    main()
