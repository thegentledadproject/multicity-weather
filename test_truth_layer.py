"""M1 truth-layer gates (settlement, point-in-time data, freshness). Run: python test_truth_layer.py"""
import logging
import os
import sqlite3
import tempfile
import time
from unittest.mock import Mock, patch

from config.cities import CITIES
from core import discovery, execution
from core.edge import EdgeSignal, MarketPrice
from core.model import ForecastResult
from core.sizing import SizingResult
from db.ledger import Ledger

DAY = "2026-10-07"
WSSS = CITIES["WSSS"]


def describe(station="Singapore Changi Airport", site="wsss", date="7 Oct '26"):
    # Wording copied from the live Gamma events for 2026-10-07.
    return (f"This market will resolve to the temperature range that contains the highest temperature "
            f"recorded by NOAA at the {station} Station in degrees Celsius on {date}. ... available here: "
            f"https://www.weather.gov/wrh/timeseries?site={site} ... The resolution source for this market "
            f"measures temperatures to whole degrees Celsius (eg, 9°C).")


def market(title, question=None):
    t = title.split("°")[0]
    return {"groupItemTitle": title, "clobTokenIds": f'["{t}-yes", "{t}-no"]',
            "question": question or f"Will the highest temperature in Singapore be {title} on October 7?"}


def event(titles=None, **overrides):
    titles = titles or ["28°C or below"] + [f"{t}°C" for t in range(29, 38)] + ["38°C or higher"]
    e = {"description": describe(), "resolutionSource": "https://www.weather.gov/wrh/timeseries?site=wsss",
         "markets": [market(t) for t in titles]}
    e.update(overrides)
    return e


def invalid(fn):
    try:
        fn()
    except discovery.SettlementAmbiguous:
        return True
    return False


def book(bid=0.39, ask=0.41, depth=1000):
    return {"bids": [{"price": str(bid), "size": str(depth)}], "asks": [{"price": str(ask), "size": str(depth)}]}


def signal(scanned_ago=0.0, market_date=DAY):
    price = MarketPrice("yes", 0.40, 0.39, 0.41, 0.02, 400)
    sig = EdgeSignal("31°C", "yes", 0.60, price, 0.20, 0.08, no_token_id="no")
    sig.market_date, sig.scanned_at, sig.scan_id = market_date, time.time() - scanned_ago, 7
    return sig


def main():
    logging.basicConfig(level=logging.CRITICAL)
    finder = discovery.MarketDiscovery(Mock(), WSSS)

    # ── P1: settlement ────────────────────────────────────────────────────────
    found = finder._extract_markets_from_event(event(), DAY)
    assert len(found) == 11 and found["28°C"] == {"yes": "28-yes", "no": "28-no"}
    assert discovery.bracket_from_title("28°C or below") == ("28°C", (float("-inf"), 29.0))
    assert discovery.bracket_from_title("38°C or higher") == ("38°C", (38.0, float("inf")))
    assert discovery.bracket_from_title("82°F") is None
    # Each city's own station and NOAA site.
    kl = event(description=describe("Kuala Lumpur Intl Airport", "wmkk"),
               resolutionSource="https://www.weather.gov/wrh/timeseries?site=wmkk")
    assert len(discovery.MarketDiscovery(Mock(), CITIES["WMKK"])._extract_markets_from_event(kl, DAY)) == 11
    assert discovery.settlement_problem(kl, DAY, WSSS) == "station"
    for bad, reason in [(event(description=describe(site="wmkk"), resolutionSource=""), "source"),
                        (event(description=describe().replace("whole degrees", "tenths of degrees")), "precision"),
                        (event(description=describe(date="8 Oct '26")), "date")]:
        assert discovery.settlement_problem(bad, DAY, WSSS) == reason, reason
        assert invalid(lambda: finder._extract_markets_from_event(bad, DAY))
    # July's 26-36 range against the 28-38 config: tails and buckets disagree.
    july = event(["26°C or below"] + [f"{t}°C" for t in range(27, 36)] + ["36°C or higher"])
    assert invalid(lambda: finder._extract_markets_from_event(july, DAY))
    # "28°C or below" must never price as the [28, 29) bucket, nor °F/odd titles pass.
    assert invalid(lambda: finder._extract_markets_from_event(event(["28°C", "29°C"]), DAY))
    assert invalid(lambda: finder._extract_markets_from_event(event(["84°F"]), DAY))
    disagree = event(markets=[market("31°C", "Will ... be 32°C on October 7?")])
    assert invalid(lambda: finder._extract_markets_from_event(disagree, DAY))

    with tempfile.TemporaryDirectory() as temp:
        ledger = Ledger(os.path.join(temp, "t.db"))
        # INVALID never falls back to a matrix cached while the event was valid.
        ledger.upsert_token_matrix("31°C", "cached", "cached-no", DAY, icao="WSSS")
        finder = discovery.MarketDiscovery(ledger, WSSS)
        with patch.object(discovery.requests, "get") as get:
            get.return_value.json.return_value = [event(description="")]
            assert finder.run(DAY) == {} and "station" in finder.invalid_reason
            get.return_value.json.return_value = [event()]
            assert len(finder.run(DAY)) == 11 and not finder.invalid_reason

        # Job 1 drops its cached matrix (and signals) when the gate rejects the live event.
        from core.city_runner import CityRunner
        runner = CityRunner(WSSS, ledger, 100, 0.08, 0.2)
        runner._state.update(token_matrix={"31°C": {"yes": "y", "no": "n"}}, market_date=DAY,
                             signals={"31°C": signal()})
        rejected = Mock(invalid_reason="WSSS: settlement station mismatch")
        rejected.run.return_value = {}
        with patch("core.city_runner.MarketDiscovery", return_value=rejected), \
                patch.object(runner, "_local_now", return_value=__import__("datetime").datetime(2026, 10, 7, 12)):
            runner.job_market_discovery()
        assert runner._state["token_matrix"] == {} and runner._state["signals"] == {}

        # ── P2: point-in-time snapshots ───────────────────────────────────────
        fc = ForecastResult(31.2, 0.6, "ensemble_blend", 31.0, 31.3, 0.5, 0.6)
        s1 = ledger.log_scan("WSSS", "2026-10-07T06:00:00", DAY, fc, 0.1, {"31°C": 0.4})
        ledger.log_book(s1, "y31", "scan", [(0.39, 100)], [(0.41, 50), (0.43, 10)], "2026-10-07T06:00:01")
        s2 = ledger.log_scan("WSSS", "2026-10-07T06:15:00", DAY, fc, 0.1, {"31°C": 0.5})
        ledger.log_scan("WMKK", "2026-10-07T06:10:00", DAY, fc, 0.0, {"31°C": 0.9})  # other city
        assert ledger.scan_as_of("WSSS", DAY, "2026-10-07T05:59:59") is None
        known = ledger.scan_as_of("WSSS", DAY, "2026-10-07T06:14:59")
        assert known["id"] == s1 and known["model_probs"] == {"31°C": 0.4}
        assert known["books"][0]["asks"] == [[0.41, 50], [0.43, 10]]
        assert ledger.scan_as_of("WSSS", DAY, "2026-10-07T07:00:00")["id"] == s2

        # Execution records the book it decided on and links the paper trade to its scan.
        client = Mock()
        client.get_order_book.return_value = book()
        engine = execution.ExecutionEngine(client, ledger, 100, "WSSS", paper_trading=True)
        sizing = SizingResult("EXECUTE", "BUY", 1.0, 0.1, 0.1, 0.1, "test")
        assert engine.execute(signal(), sizing, market_date=DAY)
        assert ledger.get_open_positions("WSSS")[0]["scan_id"] == 7
        with ledger._conn() as conn:
            assert conn.execute("SELECT purpose, scan_id FROM book_snapshots WHERE token_id='yes'").fetchall()[-1][:] == ("exec", 7)
        ledger.close_position("yes")

        # Pre-M1 databases gain scan_id columns on startup.
        old = os.path.join(temp, "old.db")
        conn = sqlite3.connect(old)
        conn.execute("CREATE TABLE signal_log (id INTEGER PRIMARY KEY, timestamp TEXT, date TEXT, bracket_label TEXT, "
                     "model_prob REAL, market_price REAL, edge REAL, action TEXT, settled_outcome TEXT, gate_reason TEXT)")
        conn.commit()
        conn.close()
        Ledger(old)
        conn = sqlite3.connect(old)
        assert "scan_id" in {r[1] for r in conn.execute("PRAGMA table_info(signal_log)")}
        conn.close()
        # A DB from the single-city wsss-weatherbot has scan_snapshots without icao_code.
        legacy = os.path.join(temp, "legacy.db")
        conn = sqlite3.connect(legacy)
        conn.execute("CREATE TABLE scan_snapshots (id INTEGER PRIMARY KEY, scan_at TEXT, market_date TEXT, "
                     "forecast_source TEXT, mu REAL, sigma REAL, mu_gfs REAL, mu_ecmwf REAL, sigma_gfs REAL, "
                     "sigma_ecmwf REAL, trailing_bias REAL, model_probs TEXT)")
        conn.execute("INSERT INTO scan_snapshots VALUES (1, '2026-10-06T00:00:00', '2026-10-06', 'x', 31, 1, "
                     "NULL, NULL, NULL, NULL, 0, '{}')")
        conn.commit()
        conn.close()
        assert Ledger(legacy).scan_as_of("WSSS", "2026-10-06", "9999")["id"] == 1

        # ── P3: freshness / information integrity ─────────────────────────────
        now = time.time()
        f = execution.freshness
        assert f("scan", now - 60, now) == execution.FRESH
        assert f("scan", now - 901, now) == execution.STALE
        assert f("scan", now + 5, now) == execution.INVALID  # clock mismatch
        assert f("scan", None, now) == execution.INVALID
        untagged = signal()
        untagged.scanned_at = None
        for stale in (signal(scanned_ago=901), signal(market_date="2026-10-06"), untagged):
            client = Mock()
            client.get_order_book.return_value = book()
            assert not execution.ExecutionEngine(client, ledger, 100, "WSSS", paper_trading=True).execute(
                stale, sizing, market_date=DAY)
            client.get_order_book.assert_not_called()
        assert not ledger.get_open_positions("WSSS")

        # Job 2 tags signals with scan provenance, and clears them when it can't scan.
        runner._state.update(token_matrix={"31°C": {"yes": "yes", "no": "no"}}, market_date=DAY)
        with patch("core.city_runner.fetch_gfs_forecast", return_value=fc), \
                patch("core.city_runner.scan_all_brackets", return_value={"31°C": signal()}):
            runner.job_signal_scan()
        tagged = runner._state["signals"]["31°C"]
        assert tagged.market_date == DAY and execution.signal_is_current(tagged, DAY)
        with ledger._conn() as conn:
            row = conn.execute("SELECT icao_code, mu FROM scan_snapshots WHERE id = ?", (tagged.scan_id,)).fetchone()
        assert tuple(row) == ("WSSS", 31.2)
        runner._state["token_matrix"] = {}
        runner.job_signal_scan()
        assert runner._state["signals"] == {}

        # Calibration waits for the local day to end, and maps the whole-°C
        # METAR high onto the model's [X, X+1) bracket midpoint.
        from core.settlement import SettlementEngine
        fetcher = Mock(return_value=33.0000001)  # ASOS °F→°C conversion noise
        with patch.object(WSSS, "official_station_fetcher", fetcher):
            engine = SettlementEngine(ledger, WSSS)
            today = runner._local_now().date().isoformat()
            assert engine.run(31.0, today)["actual_temp"] is None and not fetcher.called
            assert engine.run(31.0, "2026-01-02")["actual_temp"] == 33.5
        # No station reading → no calibration, and no fallback to another source.
        with patch.object(WSSS, "official_station_fetcher", Mock(return_value=None)), \
                patch("core.settlement.requests.get", side_effect=AssertionError("fallback fetch")):
            assert SettlementEngine(ledger, WSSS)._fetch_actual_temperature("2026-01-03") is None

    print("Truth-layer gates passed: settlement, point-in-time snapshots, freshness.")


if __name__ == "__main__":
    main()
