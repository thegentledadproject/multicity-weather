"""
core/settlement.py — S1: Settlement detection + calibration write-back
v5.0: City-agnostic — coordinates and the optional official-station
fallback now come from a config.cities.CityConfig instead of hardcoded
WSSS/NEA constants.

Two sub-tasks run in Job 4 (every 15 min, 24/7 as of the round-the-clock
scheduler redesign — checks both today's and yesterday's market_date
every cycle, since a position opened late one day can still resolve in
the early hours of the next):

  Task A — Resolution detection:
    Poll Gamma API outcomePrices for each open position's token_id.
    Terminal: outcomePrices[0] > 0.99 (YES) or < 0.01 (NO).
    On terminal state → close the position and mark the signal settled.

  Task B — Actual temperature fetch (separate from resolution):
    Fetches the true observed daily max at the city's coordinates and
    writes it to calibration_logs regardless of whether we had an open
    position — it feeds the trailing bias.

    THIS IS THE FIX for the settlement inference bug:
    We do NOT infer actual temp from the bracket midpoint.
    We fetch it directly from a meteorological source.

  Source: city_config.official_station_fetcher only — for WSSS/WMKK an
    ASOS/METAR archive fetch keyed by the city's ICAO code, the same
    reports the market settles on. There is deliberately no fallback:
    Open-Meteo's archive (gridded reanalysis) and NEA S24 (another sensor)
    are numbers the market never resolves on, so calibrating against them
    corrupts the trailing bias. No reading → no calibration row this
    cycle; Job 4 retries every 15 minutes.
"""

import json
import logging
import datetime
import pytz
import requests
from typing import Dict, Optional

from db.ledger import Ledger

logger = logging.getLogger("hermes.settlement")

GAMMA_MARKETS_URL   = "https://gamma-api.polymarket.com/markets"


class SettlementEngine:
    def __init__(self, ledger: Ledger, city_config, timeout: int = 15):
        self.ledger      = ledger
        self.city_config = city_config
        self.icao        = city_config.icao
        self.timeout     = timeout

    def run(self, model_mu: float, market_date: Optional[str] = None) -> Dict:
        """
        Main entry point for Job 4.
        model_mu: the GFS mu used in today's signal (passed from scheduler state)
        market_date: date string "YYYY-MM-DD", defaults to today SGT
        """
        if market_date is None:
            sg_now      = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
            market_date = sg_now.strftime("%Y-%m-%d")

        results = {
            "date":            market_date,
            "positions_checked": 0,
            "positions_settled": 0,
            "actual_temp":     None,
            "calibration_logged": False,
        }

        # Task A: Check all open positions for resolution
        open_positions = self.ledger.get_open_positions(self.icao)
        results["positions_checked"] = len(open_positions)

        for pos in open_positions:
            # Use the position's own stored SGT market_date (added when this
            # scheduler redesign threaded market_date through record_position)
            # rather than parsing Gamma's `endDate` field — that field's
            # timezone/format isn't guaranteed to match our SGT calendar-date
            # convention, and could silently fail to match any signal_log row
            # (mark_signal_settled's WHERE clause just matches zero rows,
            # no error raised). Falls back to this run()'s market_date for
            # legacy rows recorded before the market_date column existed.
            pos_market_date = pos["market_date"] if pos["market_date"] else market_date
            settled = self._check_resolution(
                token_id      = pos["token_id"],
                bracket_label = pos["bracket_label"],
                market_date   = pos_market_date,
                position      = pos,
            )
            if settled:
                results["positions_settled"] += 1

        # Task B: Fetch actual observed temperature and write calibration log ONCE per day
        # Two guards added after Jul 2 incident:
        #   1. Idempotency — skip if this ICAO already has a row for today's date.
        #      Without this, Job 4 (every 10 min, 17:00-23:50 SGT) writes a fresh
        #      row on every cycle — 15-20+ duplicate rows per real trading day.
        #   2. Fallback rejection — skip if model_mu == city_config.hard_prior_mu,
        #      which means Job 2 never ran that day (e.g. discovery found no
        #      token matrix). Logging a residual against a hard prior is
        #      meaningless and corrupts fetch_trailing_bias() for every future day.
        local_today = datetime.datetime.now(pytz.timezone(self.city_config.timezone)).date().isoformat()
        if market_date >= local_today:
            # The station feed only has the high SO FAR; logging it would lock
            # a partial-day max into calibration (has_calibration_for_date
            # then blocks the correction). Settle once the local day is over.
            logger.info(f"[SETTLE] {self.icao}: {market_date} not over yet — calibration deferred.")
        elif self.ledger.has_calibration_for_date(self.icao, market_date):
            logger.info(
                f"[SETTLE] {self.icao}: calibration already logged for {market_date} — skipping Task B."
            )
        elif abs(model_mu - self.city_config.hard_prior_mu) < 0.01:
            logger.warning(
                f"[SETTLE] {self.icao}: model_mu={model_mu:.2f} is the fallback prior — "
                f"Job 2 likely never ran today (check discovery/token_matrix). "
                f"Skipping calibration write for {market_date} to avoid corrupting trailing bias."
            )
        else:
            actual_temp = self._fetch_actual_temperature(market_date)
            results["actual_temp"] = actual_temp

            if actual_temp is not None:
                self.ledger.log_outcome(self.icao, model_mu, actual_temp, market_date=market_date)
                results["calibration_logged"] = True
                logger.info(
                    f"[SETTLE] {self.icao}: calibration written: date={market_date} "
                    f"model_mu={model_mu:.2f} actual={actual_temp:.2f} "
                    f"residual={actual_temp - model_mu:+.2f}°C"
                )
            else:
                logger.warning(
                    f"[SETTLE] {self.icao}: could not fetch actual temp for {market_date} — "
                    f"calibration not written. Will retry next cycle."
                )

        return results

    def _check_resolution(self, token_id: str, bracket_label: str, market_date: str = "",
                          position=None) -> bool:
        """
        S1: Check Gamma API outcomePrices for terminal state.
        Returns True if the market has resolved and position was closed.

        market_date: the position's own SGT calendar date (from open_positions),
        used for mark_signal_settled instead of parsing Gamma's endDate field.
        """
        url = f"{GAMMA_MARKETS_URL}?clob_token_ids={token_id}"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data    = resp.json()
            markets = data if isinstance(data, list) else data.get("markets", [])

            if not markets:
                return False

            market = markets[0]

            # Check resolution flags first
            if not (market.get("closed") or market.get("resolved")):
                return False

            raw_prices = market.get("outcomePrices")
            if not raw_prices:
                return False

            prices    = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            yes_price = float(prices[0])

            if yes_price > 0.99:
                outcome = "YES"
            elif yes_price < 0.01:
                outcome = "NO"
            else:
                return False  # Not yet terminal

            logger.info(
                f"[SETTLE] {self.icao} {bracket_label} resolved {outcome} "
                f"(outcomePrices[0]={yes_price:.4f})"
            )

            bracket = bracket_label.split(":")[0]  # positions are labelled "31°C:YES" / "31°C:NO"
            if position is not None:
                # A position held to resolution is an exit at 1 (held side won) or 0;
                # without this row its P&L never reaches exit_log or the dashboard.
                held_no = bracket_label.endswith(":NO")
                exit_price = 1.0 if (outcome == "NO") == held_no else 0.0
                entry, size = float(position["entry_price"]), float(position["size_usd"])
                self.ledger.log_exit(
                    token_id=token_id, bracket_label=bracket, direction="SELL" if held_no else "BUY",
                    reason="SETTLED", entry_price=entry, exit_price=exit_price, size_usd=size,
                    realised_pnl=(exit_price - entry) * size / entry, opened_at=position["opened_at"],
                    market_date=market_date, icao=self.icao, is_paper=bool(position["is_paper"]),
                    scan_id=position["scan_id"],
                )
            self.ledger.close_position(token_id)
            self.ledger.mark_signal_settled(
                date          = market_date or market.get("endDate", "")[:10],
                bracket_label = bracket,
                outcome       = outcome,
                icao          = self.icao,
            )
            return True

        except Exception as e:
            logger.error(f"[SETTLE] {self.icao}: resolution check failed for {bracket_label}: {e}")
            return False

    def _fetch_actual_temperature(self, date: str) -> Optional[float]:
        """
        The city's settlement-station daily max in model space, or None to
        retry next cycle (no station fetcher, no reading yet, or fetch error).
        See the module docstring for why there is no fallback source.
        """
        fetcher = self.city_config.official_station_fetcher
        if fetcher is None:
            logger.warning(f"[SETTLE] {self.icao}: no official_station_fetcher configured — no calibration")
            return None
        try:
            actual = fetcher(date, self.timeout)
        except Exception as e:
            logger.error(f"[SETTLE] {self.icao}: official station fetch failed: {e}")
            return None
        if actual is None:
            return None
        # METARs report whole °C (ASOS's °F is a conversion of that), and the
        # model prices bracket "X°C" as [X, X+1) — so a reported X sits at
        # X+0.5 in model space on average. Logging the bare integer biases
        # every forecast 0.5°C low.
        return round(actual) + 0.5

    def find_stuck(self) -> int:
        """
        PM-5: count (never delete) this city's positions open longer than
        28h — see db/ledger.py's find_stuck_positions() docstring.

        NOTE: currently dead code — core/city_runner.py's Job 5 calls
        ledger.find_stuck_positions() directly, bypassing this wrapper
        entirely. Harmless (just unused), kept here in case a future
        caller wants it via SettlementEngine rather than the raw ledger.
        """
        stuck = self.ledger.find_stuck_positions(self.icao, ttl_hours=28)
        return len(stuck)
