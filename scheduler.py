"""
scheduler.py — S2: APScheduler round-the-clock job architecture, multi-city

v5.0 — generalized to run N cities in one process. Each city in
config.cities.CITIES gets its own CityRunner (core/city_runner.py) with its
own in-process _state, but all runners share one Ledger (hermes.db), one
CLOB client, and one vault split by each city's config.vault_allocation_pct.
Only WSSS is configured today; adding a city is a config.cities.py change,
not a scheduler change — every job here just loops over CITIES.

v4.6 REDESIGN — round-the-clock coverage to match how Polymarket weather
markets actually trade. Confirmed from live observation across a prior
session: markets for date D launch the evening BEFORE D (~23:00 local),
trade continuously through D, and can resolve/settle late in the evening
or after midnight D+1 depending on when the final reading posts. The old
schedule (daytime-only windows, single daily discovery trigger) had three
consequences, all now fixed:

  1. Job 1 fired once at 07:30 with no retry. If Gamma was down or the
     slug format didn't match that single time (exactly what happened in
     production on Jul 2-3 — two full days of zero trades), the bot had
     no token matrix until the NEXT day's 07:30 trigger — a ~24h dead zone.
     FIX: Job 1 now runs every 20 min, all day, self-healing within
     minutes instead of waiting a full day.

  2. Jobs 2/3/5 only ran daytime hours, missing the evening launch window
     and any edge that appears overnight as forecasts update or the market
     re-prices. FIX: Jobs 2/3/5 now run every 15/15/5 min, 24/7. Market
     quality gates (liquidity floor, spread cap) already suppress bad
     signals on thin overnight books — a clock-based window isn't needed.

  3. Job 5's old window never overlapped position_monitor.py's own
     HARD_EXIT_HOUR_SGT=16 check, meaning the hard time-exit could never
     actually fire. FIX: Job 5 now runs 24/7, so force_time_exit's own
     internal logic (unchanged) can actually execute. Consequence: once
     Jobs 2/3 can open NEW positions at any hour (including after 16:00
     local), a purely wall-clock-hour force-exit would immediately close
     a position opened at, say, 20:00 on its very next Job 5 tick — a
     self-defeating trade. position_monitor.py's force_time_exit is
     therefore scoped to positions whose opened_at date is STRICTLY
     BEFORE today's local calendar date (i.e. genuinely stale), not simply
     "it's currently past 16:00."

  4. Job 4 only settled _state["market_date"] (today). A position opened
     late yesterday could still be resolving in the early hours of today,
     after market_date has already rolled over — that outcome was never
     checked. FIX: Job 4 now settles BOTH today's and yesterday's date
     every cycle. Safe because settlement.py's has_calibration_for_date
     guard (db/ledger.py) makes this idempotent regardless of how many
     times or how many dates are checked per cycle.

Jobs (per city, all times in that city's configured timezone, 24/7 unless noted):
  Job 1 — market_discovery   : every 20 min — self-healing token matrix
  Job 2 — signal_scan        : every 15 min — forecast + edge scan
  Job 3 — order_execution    : every 15 min (offset +2 min from Job 2)
  Job 4 — settlement_check   : every 15 min — checks today AND yesterday
  Job 5 — position_monitor   : every 5 min  — trailing stop/stop loss/time exit

APScheduler runs jobs in a thread pool (max_workers scales with city count).
The CLOB client and Ledger are instantiated once at startup and shared
across every city's runner.
"""

import os
import logging
import sys

from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("hermes.log"),
    ],
)
logger = logging.getLogger("hermes.scheduler")

# ── Lazy imports (so missing deps fail loudly at runtime, not import time) ────
from db.ledger        import Ledger
from core.execution    import build_client
from core.sizing       import check_sizing_config
from core.city_runner  import CityRunner
from config.cities     import CITIES

# ── Config from environment ────────────────────────────────────────────────────
DB_PATH        = os.getenv("DB_PATH",              "hermes.db")
VAULT_USD      = float(os.getenv("MAX_VAULT_ALLOCATION") or 200.0)
EDGE_THRESHOLD = float(os.getenv("EDGE_THRESHOLD", 0.05))
TRAIL_PCT      = float(os.getenv("TRAIL_PCT", 0.20))

# VALIDATION_MODE: forces $1 trades on any actionable signal, bypassing
# Kelly sizing and EV hurdles entirely. Full lifecycle (entry, trailing
# stop, stop loss, settlement) still runs for real — only sizing/gating
# is bypassed. Use to prove mechanics work before deploying real capital.
# Set VALIDATION_MODE=false in .env once validation run is complete.
VALIDATION_MODE = os.getenv("VALIDATION_MODE", "false").lower() == "true"

# ── Shared singletons ──────────────────────────────────────────────────────────
_ledger  = Ledger(DB_PATH)
_client  = None   # initialised in main() to catch auth errors early
_runners: dict = {}   # icao -> CityRunner, populated in main()


def _build_runners() -> dict:
    """One CityRunner per configured city, each allocated its slice of VAULT_USD."""
    runners = {}
    for icao, config in CITIES.items():
        allocated_vault = VAULT_USD * config.vault_allocation_pct
        runners[icao] = CityRunner(
            config          = config,
            ledger          = _ledger,
            vault_usd       = allocated_vault,
            edge_threshold  = EDGE_THRESHOLD,
            trail_pct       = TRAIL_PCT,
            validation_mode = VALIDATION_MODE,
        )
        logger.info(
            f"[INIT] City configured: {icao} ({config.display_name}) — "
            f"vault=${allocated_vault:.2f} ({config.vault_allocation_pct:.0%} of ${VAULT_USD:.0f})"
        )
    return runners


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global _client, _runners

    _runners = _build_runners()

    logger.info("═" * 60)
    logger.info(f"  HERMES v5.0 — Multi-City Weather Bracket Trader (round-the-clock)")
    logger.info(f"  Cities: {', '.join(_runners.keys())} | Vault: ${VAULT_USD:.0f} | Edge threshold: {EDGE_THRESHOLD*100:.0f}%")
    if VALIDATION_MODE:
        logger.warning("  ⚠️  VALIDATION_MODE ACTIVE — all trades forced to $1, EV gating bypassed")
        logger.warning("  ⚠️  Set VALIDATION_MODE=false in .env to resume normal Kelly sizing")
    logger.info("═" * 60)

    # Warn if vault/cap/floor collapse the sizing range (see sizing.py) —
    # checked per-city allocation, since that's the vault each runner actually sizes against.
    for icao, runner in _runners.items():
        check_sizing_config(runner.vault_usd)

    # Initialise CLOB client once — shared across every city's jobs
    try:
        _client = build_client()
        logger.info("[INIT] CLOB client authenticated ✓")
    except Exception as e:
        logger.error(f"[INIT] CLOB client failed: {e}")
        logger.error("[INIT] Check POLYMARKET_PRIVATE_KEY / CLOB_* env vars.")
        sys.exit(1)

    for runner in _runners.values():
        runner.client = _client

    # Run discovery immediately on startup so jobs 2/3 have tokens from the start
    for icao, runner in _runners.items():
        logger.info(f"[INIT] Running initial market discovery for {icao}...")
        runner.job_market_discovery()

    # ── Scheduler setup ────────────────────────────────────────────────────────
    max_workers = max(3, 3 * len(_runners))
    executors   = {"default": ThreadPoolExecutor(max_workers=max_workers)}
    scheduler   = BlockingScheduler(executors=executors, timezone="UTC")

    for icao, runner in _runners.items():
        tz = runner.config.timezone

        # Job 1 — Market discovery: every 20 min, 24/7 — self-healing.
        scheduler.add_job(
            runner.job_market_discovery,
            trigger   = "cron",
            minute    = "0,20,40",
            timezone  = tz,
            id        = f"market_discovery_{icao}",
            name      = f"Market Discovery [{icao}]",
            max_instances = 1,
        )

        # Job 2 — Signal scan: every 15 min, 24/7.
        scheduler.add_job(
            runner.job_signal_scan,
            trigger   = "cron",
            minute    = "0,15,30,45",
            timezone  = tz,
            id        = f"signal_scan_{icao}",
            name      = f"Signal Scan [{icao}]",
            max_instances = 1,
        )

        # Job 3 — Execution: every 15 min, 24/7, offset +2 min from Job 2
        # so _state["signals"] is always freshly computed before Job 3 reads it.
        scheduler.add_job(
            runner.job_order_execution,
            trigger   = "cron",
            minute    = "2,17,32,47",
            timezone  = tz,
            id        = f"order_execution_{icao}",
            name      = f"Order Execution [{icao}]",
            max_instances = 1,
        )

        # Job 4 — Settlement: every 15 min, 24/7. Checks both today's and
        # yesterday's market_date every cycle (see CityRunner.job_settlement_check).
        scheduler.add_job(
            runner.job_settlement_check,
            trigger   = "cron",
            minute    = "5,20,35,50",
            timezone  = tz,
            id        = f"settlement_check_{icao}",
            name      = f"Settlement Check [{icao}]",
            max_instances = 1,
        )

        # Job 5 — Position monitor: every 5 min, 24/7.
        scheduler.add_job(
            runner.job_position_monitor,
            trigger   = "cron",
            minute    = "*/5",
            timezone  = tz,
            id        = f"position_monitor_{icao}",
            name      = f"Position Monitor [{icao}]",
            max_instances = 1,
        )

    # Graceful shutdown on SIGTERM / SIGINT
    import signal
    def _shutdown(signum, frame):
        logger.info("[SHUTDOWN] Signal received — stopping scheduler.")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    logger.info("[INIT] Scheduler armed. Jobs registered:")
    for job in scheduler.get_jobs():
        logger.info(f"  {job.name}: {job.trigger}")

    logger.info("[INIT] Starting. Ctrl+C to stop.")
    scheduler.start()


if __name__ == "__main__":
    main()
