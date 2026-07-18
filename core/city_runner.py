"""
core/city_runner.py — per-city job set for the multi-city scheduler.

v5.0: extracted from scheduler.py's module-level job functions + global
_state dict (v4.6) so a single process can run N cities concurrently.
Each CityRunner owns its own in-process _state and wraps the same 5-job
pipeline (discovery -> signal scan -> execution -> settlement -> position
monitor) that scheduler.py used to run directly against module globals.
Behavior per job is unchanged from v4.6 — this is a mechanical extraction,
parameterized by a config.cities.CityConfig instead of hardcoded WSSS
constants and a single shared vault.

State shared across a city's own jobs (in-process dict, not DB):
  _state = {
      "token_matrix":  {label: token_id},
      "signals":       {label: EdgeSignal},
      "forecast":      ForecastResult,
      "model_probs":   {label: float},
      "model_mu":      float,
      "market_date":   str,   # refreshed every 20 min by Job 1 — rolls over
                               # across local midnight automatically.
  }

The CLOB client and Ledger are shared singletons injected by scheduler.py
(one client, one DB, across every city) — only `client` is set after
construction (scheduler.main() builds it once, then hands it to every
runner), since build_client() must succeed before any runner can use it.
"""

import logging
import datetime

import pytz

from db.ledger       import Ledger
from core.discovery  import MarketDiscovery
from core.model      import BracketModel, fetch_gfs_forecast
from core.edge       import scan_all_brackets
from core.sizing     import compute_size, compute_validation_size
from core.execution  import ExecutionEngine
from core.settlement    import SettlementEngine
from core.position_monitor import PositionMonitor

logger = logging.getLogger("hermes.city_runner")


class CityRunner:
    def __init__(
        self,
        config,
        ledger: Ledger,
        vault_usd: float,
        edge_threshold: float,
        trail_pct: float,
        validation_mode: bool = False,
    ):
        self.config          = config
        self.icao            = config.icao
        self.ledger          = ledger
        self.vault_usd       = vault_usd            # already this city's allocated slice
        self.edge_threshold  = edge_threshold
        self.trail_pct       = trail_pct
        self.validation_mode = validation_mode

        self.client = None   # set by scheduler.main() once build_client() succeeds

        self._state: dict = {
            "token_matrix": {},
            "signals":      {},
            "forecast":     None,
            "model_probs":  {},
            "model_mu":     config.hard_prior_mu,
            "market_date":  "",
        }

    def _local_now(self) -> datetime.datetime:
        return datetime.datetime.now(pytz.timezone(self.config.timezone))

    # ══════════════════════════════════════════════════════════════════════
    # JOB 1 — Market Discovery (every 20 min, 24/7 — self-healing)
    # ══════════════════════════════════════════════════════════════════════
    def job_market_discovery(self):
        local_now  = self._local_now()
        date       = local_now.strftime("%Y-%m-%d")
        prior_date = self._state.get("market_date", "")

        logger.info(f"[JOB1:{self.icao}] ── Market Discovery @ {local_now.strftime('%H:%M %Z')} ──")

        if prior_date and prior_date != date:
            logger.info(f"[JOB1:{self.icao}] Date rollover detected: {prior_date} → {date}")

        discovery = MarketDiscovery(self.ledger, self.config)
        matrix    = discovery.run()

        if not matrix:
            # Not an error on every tick — this runs every 20 min, so a miss
            # just means retry in 20 min rather than a full day gap.
            if self._state.get("token_matrix"):
                logger.warning(
                    f"[JOB1:{self.icao}] No fresh tokens found this cycle — keeping "
                    "previously cached matrix until next retry."
                )
                return
            logger.error(f"[JOB1:{self.icao}] No tokens found and no cached matrix — Jobs 2/3 will skip.")
        else:
            valid = discovery.validate_against_live(matrix)
            if not valid:
                logger.warning(f"[JOB1:{self.icao}] Validation failed — re-running discovery.")
                matrix = discovery.run()

        self._state["token_matrix"] = matrix
        self._state["market_date"]  = date
        logger.info(f"[JOB1:{self.icao}] Token matrix: {list(matrix.keys())}")

    # ══════════════════════════════════════════════════════════════════════
    # JOB 2 — Signal Scan (every 15 min, 24/7)
    # ══════════════════════════════════════════════════════════════════════
    def job_signal_scan(self):
        local_now_str = self._local_now().strftime("%H:%M %Z")
        logger.info(f"[JOB2:{self.icao}] ── Signal Scan @ {local_now_str} ──")

        token_matrix = self._state.get("token_matrix", {})
        if not token_matrix:
            logger.warning(f"[JOB2:{self.icao}] No token matrix — skipping scan. Run Job 1 first.")
            return

        forecast = fetch_gfs_forecast(
            lat=self.config.lat, lon=self.config.lon,
            timezone=self.config.timezone,
            hard_prior_mu=self.config.hard_prior_mu,
            hard_prior_sigma=self.config.hard_prior_sigma,
        )
        self._state["forecast"] = forecast

        if forecast.source == "fallback":
            logger.error(f"[JOB2:{self.icao}] Forecast on hard prior — aborting scan.")
            self._state["signals"] = {}
            return

        trailing_bias = self.ledger.fetch_trailing_bias(self.icao)
        model         = BracketModel(self.config, trailing_bias=trailing_bias)
        model_probs   = model.compute(forecast)

        if not model_probs:
            logger.error(f"[JOB2:{self.icao}] Model returned empty probs — aborting scan.")
            self._state["signals"] = {}
            return

        self._state["model_probs"] = model_probs
        self._state["model_mu"]    = forecast.mu

        signals = scan_all_brackets(
            token_matrix   = token_matrix,
            model_probs    = model_probs,
            edge_threshold = self.edge_threshold,
        )
        self._state["signals"] = signals

        # Log ALL signals to DB — including non-actionable, gated, and held —
        # so the dashboard can show the full scan picture.
        date = self._state.get("market_date", self._local_now().strftime("%Y-%m-%d"))
        for label, sig in signals.items():
            mid = sig.market_price.mid_price if sig.market_price else 0.0
            self.ledger.log_signal(
                date          = date,
                bracket_label = label,
                model_prob    = sig.model_prob,
                market_price  = mid,
                edge          = sig.edge,
                action        = sig.action_label,
                icao          = self.icao,
            )

        buys  = [l for l, s in signals.items() if s.direction == "BUY"  and s.actionable]
        sells = [l for l, s in signals.items() if s.direction == "SELL" and s.actionable]
        held  = [l for l, s in signals.items() if not s.actionable and not s.gate_reason]
        gated = [l for l, s in signals.items() if s.gate_reason]
        logger.info(
            f"[JOB2:{self.icao}] Scan complete — "
            f"BUY:{buys or 'none'} SELL:{sells or 'none'} "
            f"HOLD_EDGE:{held or 'none'} GATED:{gated or 'none'}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # JOB 3 — Order Execution (every 15 min, 24/7, offset +2 min from Job 2)
    # ══════════════════════════════════════════════════════════════════════
    def job_order_execution(self):
        local_now_str = self._local_now().strftime("%H:%M %Z")
        mode = " [PAPER TRADING — no real orders]" if self.config.paper_trading else ""
        logger.info(f"[JOB3:{self.icao}] ── Order Execution @ {local_now_str}{mode} ──")

        if self.client is None:
            logger.error(f"[JOB3:{self.icao}] CLOB client not initialised — skipping.")
            return

        signals = self._state.get("signals", {})
        if not signals:
            logger.info(f"[JOB3:{self.icao}] No signals from Job 2 — nothing to execute.")
            return

        actionable = {l: s for l, s in signals.items() if s.actionable}
        if not actionable:
            logger.info(f"[JOB3:{self.icao}] No actionable signals this cycle.")
            return

        # Expire stale positions before execution — TTL-based, city-agnostic,
        # harmless to call once per city's cycle (idempotent no-op past the
        # first city that ticks in a given window).
        self.ledger.expire_stale_positions(ttl_hours=28)

        trailing_bias = self.ledger.fetch_trailing_bias(self.icao)
        engine        = ExecutionEngine(
            self.client, self.ledger, self.vault_usd, self.icao,
            paper_trading=self.config.paper_trading,
        )

        for label, signal in actionable.items():
            direction = signal.direction  # "BUY" or "SELL"

            # For BUY YES: Kelly uses best_ask (cost to buy)
            # For SELL YES (NO): Kelly uses effective_ask = 1 - best_bid
            #   because buying NO at implied price (1 - bid) is what we're sizing.
            if direction == "BUY":
                effective_ask = signal.market_price.best_ask
            else:
                effective_ask = 1.0 - signal.market_price.best_bid

            # signal.model_prob is always P(bracket occurs) — i.e. P(YES).
            # Kelly's p must be the win probability of the SIDE BEING SIZED:
            #   BUY YES → wins if the bracket occurs         → p = model_prob
            #   SELL/NO → wins if the bracket does NOT occur  → p = 1 - model_prob
            # (see commit 6634da5 — passing model_prob unflipped for SELL fed
            # Kelly/EV the probability of the side we're betting AGAINST.)
            win_prob = signal.model_prob if direction == "BUY" else 1.0 - signal.model_prob

            if self.validation_mode:
                sizing = compute_validation_size(
                    model_prob = win_prob,
                    market_ask = effective_ask,
                    direction  = direction,
                )
                logger.warning(f"[JOB3:{self.icao}] ⚠️  VALIDATION_MODE — {label} [{direction}]: {sizing}")
            else:
                sizing = compute_size(
                    model_prob    = win_prob,
                    market_ask    = effective_ask,
                    vault_usd     = self.vault_usd,
                    direction     = direction,
                    trailing_bias = trailing_bias,
                )
                logger.info(f"[JOB3:{self.icao}] {label} [{direction}]: {sizing}")

            if sizing.verdict == "EXECUTE":
                market_date_for_entry = self._state.get(
                    "market_date", self._local_now().strftime("%Y-%m-%d")
                )
                filled = engine.execute(signal, sizing, market_date=market_date_for_entry)
                if filled:
                    logger.info(
                        f"[JOB3:{self.icao}] ✓ Position opened: {label} "
                        f"{'YES' if direction == 'BUY' else 'NO'} ${sizing.size_usd:.2f}"
                    )
                else:
                    logger.warning(f"[JOB3:{self.icao}] ✗ Execution failed or rejected: {label} [{direction}]")
            else:
                logger.info(f"[JOB3:{self.icao}] Sizing HOLD for {label} [{direction}]: {sizing.reason}")

    # ══════════════════════════════════════════════════════════════════════
    # JOB 4 — Settlement Check (every 15 min, 24/7 — checks today AND yesterday)
    # ══════════════════════════════════════════════════════════════════════
    def job_settlement_check(self):
        local_now_str = self._local_now().strftime("%H:%M %Z")
        logger.info(f"[JOB4:{self.icao}] ── Settlement Check @ {local_now_str} ──")

        engine    = SettlementEngine(self.ledger, self.config)
        model_mu  = self._state.get("model_mu", self.config.hard_prior_mu)
        today     = self._state.get("market_date", self._local_now().strftime("%Y-%m-%d"))
        yesterday = (datetime.datetime.strptime(today, "%Y-%m-%d")
                     - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        for market_date in (today, yesterday):
            results = engine.run(model_mu=model_mu, market_date=market_date)
            logger.info(
                f"[JOB4:{self.icao}] date={results['date']} "
                f"checked={results['positions_checked']} "
                f"settled={results['positions_settled']} "
                f"actual_temp={results['actual_temp']} "
                f"calibration={'✓' if results['calibration_logged'] else '✗'}"
            )

    # ══════════════════════════════════════════════════════════════════════
    # JOB 5 — Position Monitor (every 5 min, 24/7)
    # ══════════════════════════════════════════════════════════════════════
    def job_position_monitor(self):
        logger.info(f"[JOB5:{self.icao}] ── Position Monitor ──")

        if self.client is None:
            logger.error(f"[JOB5:{self.icao}] CLOB client not initialised — skipping.")
            return

        open_positions = self.ledger.get_open_positions(self.icao)
        if not open_positions:
            logger.info(f"[JOB5:{self.icao}] No open positions.")
            return

        model_probs = self._state.get("model_probs", {})
        if not model_probs:
            logger.warning(f"[JOB5:{self.icao}] No model probs in state — Job 2 may not have run yet.")
            return

        monitor = PositionMonitor(
            client         = self.client,
            ledger         = self.ledger,
            edge_threshold = self.edge_threshold,
            trail_pct      = self.trail_pct,
            icao           = self.icao,
            paper_trading  = self.config.paper_trading,
            timezone       = self.config.timezone,
        )
        market_date = self._state.get("market_date", self._local_now().strftime("%Y-%m-%d"))
        results = monitor.run(model_probs, market_date=market_date)

        exits_filled = [r for r in results if r["filled"]]
        exits_failed = [r for r in results if not r["filled"]]
        total_pnl    = sum(r["realised_pnl"] for r in exits_filled)
        daily_pnl    = self.ledger.daily_pnl(market_date, self.icao)

        logger.info(
            f"[JOB5:{self.icao}] Monitor complete: {len(open_positions)} checked | "
            f"{len(exits_filled)} exited | {len(exits_failed)} failed | "
            f"Cycle P&L={total_pnl:+.4f} | Day P&L={daily_pnl:+.4f}"
        )

        for r in exits_filled:
            pnl_pct   = r["realised_pnl"] / r["size_usd"] * 100
            trail_str = f" peak={r['peak_price']:.4f}" if r.get("peak_price") else ""
            logger.info(
                f"[JOB5:{self.icao}]   ✓ {r['label']} [{r['direction']}] {r['reason']}"
                f"{trail_str} exit={r['exit_price']:.4f} "
                f"P&L={r['realised_pnl']:+.4f} ({pnl_pct:+.1f}%)"
            )
        for r in exits_failed:
            logger.warning(f"[JOB5:{self.icao}]   ✗ {r['label']} exit failed: {r['reason']}")
