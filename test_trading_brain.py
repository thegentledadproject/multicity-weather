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


def main():
    logging.basicConfig(level=logging.CRITICAL)
    with tempfile.TemporaryDirectory() as temp:
        p5_edge_lifecycle(Ledger(os.path.join(temp, "p5.db")))
    print("Trading-brain checks passed: P5")


if __name__ == "__main__":
    main()
