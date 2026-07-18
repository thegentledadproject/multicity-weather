"""
dashboard_api.py — Hermes Dashboard REST API
FastAPI backend. Reads hermes.db and serves JSON to the web frontend.

v5.0: multi-city — every trade-data endpoint now takes an optional
?icao= query param (default "WSSS", preserving pre-v5.0 behavior for any
caller that omits it) and filters by icao_code. Added /api/cities (city
registry for the frontend's switcher) and /api/overview (aggregated KPIs
across every configured city, for the "center dashboard").

Run:  uvicorn dashboard_api:app --host 0.0.0.0 --port 8000
Access: http://YOUR_VPS_IP:8000

Install: pip install fastapi uvicorn[standard]
(already in requirements.txt)

SECURITY NOTE: this API has no authentication and binds 0.0.0.0 (all
interfaces) per deploy/hermes-dashboard.service. Anyone who can reach
port 8000 on the VPS can see full trade history, P&L, and open positions
(read-only — no route can modify state). If the VPS has a public IP,
firewall port 8000 to trusted IPs only, or put it behind a reverse proxy
with auth (nginx + basic auth, or a Cloudflare Tunnel/Access policy).
"""

import os
import time
import sqlite3
import datetime
import logging
import threading
import contextlib
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core.model import ECMWF_ENSEMBLE_URL, GFS_ENSEMBLE_URL
from config.cities import CITIES, get_city, resolve_vault_usd

logger = logging.getLogger("hermes.dashboard_api")
logging.basicConfig(level=logging.INFO)

DB_PATH        = os.getenv("DB_PATH", "hermes.db")
DEFAULT_ICAO   = "WSSS"
_DASHBOARD_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")

# ── Upstream (Open-Meteo GFS/ECMWF) health check cache ───────────────────────
# Keyed by "{icao}:{source}" since each city polls its own coordinates/timezone
# and GFS/ECMWF are independent upstreams with independent uptime.
UPSTREAM_CACHE_TTL = 30.0
_upstream_cache: Dict[str, Dict[str, Any]] = {}

# ── Wallet balance (deposit wallet, on-chain via CLOB balance-allowance) ─────
# One shared Polymarket wallet funds every city's real (non-paper) trading —
# see core/execution.py:build_client(). Surfaced here so an operator can see
# the bot's own view of its funded balance without grepping journalctl for
# "balance: 0" after the fact. The CLOB client is built lazily (only once
# this endpoint is first hit) and reused, since construction does an auth
# round-trip; a lock keeps concurrent dashboard polls from racing to build
# it twice.
WALLET_BALANCE_CACHE_TTL = 20.0
_wallet_client = None
_wallet_client_lock = threading.Lock()
_wallet_balance_cache: Dict[str, Any] = {
    "result":     None,  # last successful {"balance_usd": ..., "synced_at": iso}
    "checked_at": 0.0,   # monotonic time of last attempt (success or fail)
}

app = FastAPI(title="Hermes Dashboard API", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Serve the single-file frontend ───────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_dashboard():
    # Absolute path anchored to this file's directory — a relative
    # "dashboard.html" would 404 if uvicorn is ever launched from a
    # different working directory than deploy/hermes-dashboard.service's
    # WorkingDirectory=/opt/hermes (e.g. manual local testing).
    return FileResponse(_DASHBOARD_HTML)

# ── DB helper ─────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _conn():
    """
    sqlite3.Connection used as `with conn:` only wraps a transaction
    (commit/rollback) — it never calls close(). This wrapper guarantees
    close() in a finally block for every call site in this file.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def _rows(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    if not os.path.exists(DB_PATH):
        return []
    try:
        with _conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        # Log first, then degrade — a bare `except: return []` would silently
        # swallow genuine bugs (a typo'd column name, a missing table) as
        # indistinguishable from "DB not ready yet", invisible in journalctl.
        logger.error(f"[DASHBOARD] Query failed: {query[:80]}... — {e}")
        return []

def _scalar(query: str, params: tuple = (), default: Any = 0.0) -> Any:
    if not os.path.exists(DB_PATH):
        return default
    try:
        with _conn() as conn:
            row = conn.execute(query, params).fetchone()
            return row[0] if row and row[0] is not None else default
    except Exception as e:
        logger.error(f"[DASHBOARD] Query failed: {query[:80]}... — {e}")
        return default

def _vault_start(icao: str) -> float:
    """This city's own vault size (see config.cities.resolve_vault_usd)."""
    config = CITIES.get(icao.upper())
    return resolve_vault_usd(config) if config else 0.0

def _is_paper(icao: str) -> int:
    """
    1 if this city trades in paper mode (config.cities.CityConfig.paper_trading),
    else 0. Every exit_log/open_positions query for money figures (P&L, vault,
    ROI) filters on this so a paper city's simulated results never mix with a
    real city's real ones — see core/execution.py's is_paper writes.
    """
    config = CITIES.get(icao.upper())
    return 1 if (config and config.paper_trading) else 0

# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    """Health check + live/mock indicator."""
    live = os.path.exists(DB_PATH)
    sg   = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    total_vault = sum(resolve_vault_usd(c) for c in CITIES.values())
    return {
        "live":       live,
        "db_path":    DB_PATH,
        "vault_start": total_vault,   # combined across every configured city's own vault
        "sgt_now":    sg.strftime("%Y-%m-%d %H:%M SGT"),
        "utc_now":    datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }

@app.get("/api/cities")
def cities():
    """City registry for the frontend's overview/switcher — no DB dependency."""
    return {
        "cities": [
            {
                "icao":          c.icao,
                "display_name":  c.display_name,
                "vault_start":   _vault_start(c.icao),
                "paper_trading": c.paper_trading,
            }
            for c in CITIES.values()
        ]
    }

def _check_ensemble_source(url_template: str, source: str, icao: str) -> Dict[str, Any]:
    """
    Live health check for one Open-Meteo ensemble source (GFS or ECMWF) at
    one city's coordinates — the upstream forecast sources core/model.py
    blends into mu/sigma. If either is down, the bot falls back to the other
    alone (or the hard prior if both fail), which is exactly the kind of
    degradation an operator wants surfaced on the dashboard rather than
    discovered later in the logs.

    ok=True requires BOTH a 200 response AND >=3 ensemble members in the
    body (the same threshold core/model.py._fetch_ensemble_members() gates
    on). A bare status-code check isn't enough: Open-Meteo can return 200
    with zero temperature_2m_member* keys (observed in production for
    ECMWF — the bot logged "only 0 members returned" and fell back while
    a status-code-only check would have reported "reachable"). n_members
    is returned so a degraded-but-200 response is visibly distinct from a
    fully healthy one, not just lumped into "ok".
    """
    icao = icao.upper()
    cache_key = f"{icao}:{source}"
    now = time.monotonic()
    entry = _upstream_cache.setdefault(cache_key, {"result": None, "checked_at": 0.0})
    cached = entry["result"]
    if cached is not None and (now - entry["checked_at"]) < UPSTREAM_CACHE_TTL:
        return {**cached, "cached": True}

    try:
        config = get_city(icao)
    except KeyError:
        return {"ok": False, "error": f"unknown city '{icao}'", "cached": False}

    url = url_template.format(lat=config.lat, lon=config.lon, timezone=quote(config.timezone, safe=""))
    t0 = time.monotonic()
    try:
        r = requests.get(url, timeout=5)
        n_members = None
        if r.status_code == 200:
            try:
                hourly = r.json().get("hourly", {})
                n_members = sum(1 for k in hourly if k.startswith("temperature_2m_member"))
            except ValueError as e:
                logger.error(f"[DASHBOARD] {source} response not valid JSON for {icao}: {e}")

        result = {
            "ok":           r.status_code == 200 and (n_members or 0) >= 3,
            "status_code":  r.status_code,
            "n_members":    n_members,
            "latency_ms":   round((time.monotonic() - t0) * 1000),
            "checked_at":   datetime.datetime.utcnow().isoformat(),
            "source":       f"{source}_ensemble",
        }
    except requests.RequestException as e:
        logger.error(f"[DASHBOARD] Upstream {source} check failed for {icao}: {e}")
        result = {
            "ok":           False,
            "status_code":  None,
            "n_members":    None,
            "latency_ms":   round((time.monotonic() - t0) * 1000),
            "checked_at":   datetime.datetime.utcnow().isoformat(),
            "source":       f"{source}_ensemble",
            "error":        str(e),
        }

    entry["result"]     = result
    entry["checked_at"] = now
    return {**result, "cached": False}

@app.get("/api/upstream_status")
def upstream_status(icao: str = DEFAULT_ICAO):
    """Live health check for the Open-Meteo ECMWF ensemble API (60% blend weight) at this city."""
    return _check_ensemble_source(ECMWF_ENSEMBLE_URL, "ecmwf", icao)

@app.get("/api/gfs_status")
def gfs_status(icao: str = DEFAULT_ICAO):
    """Live health check for the Open-Meteo GFS ensemble API (40% blend weight) at this city."""
    return _check_ensemble_source(GFS_ENSEMBLE_URL, "gfs", icao)

def _get_wallet_client():
    """Build the CLOB client once and reuse it. Read-only usage here — this
    endpoint never signs or posts orders, only queries balance/allowance."""
    global _wallet_client
    if _wallet_client is not None:
        return _wallet_client
    with _wallet_client_lock:
        if _wallet_client is None:
            from core.execution import build_client
            _wallet_client = build_client()
        return _wallet_client

@app.get("/api/wallet_balance")
def wallet_balance():
    """
    Live deposit-wallet COLLATERAL balance, straight from the same CLOB
    balance-allowance endpoint core/execution.py syncs immediately before
    every order. One wallet is shared across every configured city (see
    core/execution.py:build_client()), so this is deliberately global, not
    icao-scoped. Surfacing it here lets an operator see "does the bot
    currently believe it has funds?" without waiting for a trade attempt to
    fail with "balance: 0" in the logs.

    Cached for WALLET_BALANCE_CACHE_TTL so dashboard polling doesn't spam
    the CLOB API. On failure, returns the last known-good balance (if any)
    alongside ok=False so the frontend can show a stale-but-informative
    value instead of blanking out.
    """
    now = time.monotonic()
    cached = _wallet_balance_cache["result"]
    if cached is not None and (now - _wallet_balance_cache["checked_at"]) < WALLET_BALANCE_CACHE_TTL:
        return {**cached, "ok": True, "cached": True}

    try:
        from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType

        client = _get_wallet_client()
        client.update_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        raw = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))

        raw_balance = raw.get("balance") if isinstance(raw, dict) else getattr(raw, "balance", None)
        balance_usd = round(float(raw_balance) / 1_000_000, 2) if raw_balance is not None else None

        synced_at = datetime.datetime.utcnow().isoformat()
        result = {
            "balance_usd": balance_usd,
            "synced_at":   synced_at,
        }
        _wallet_balance_cache["result"]     = result
        _wallet_balance_cache["checked_at"] = now
        return {**result, "ok": True, "cached": False}

    except Exception as e:
        logger.error(f"[DASHBOARD] Wallet balance sync failed: {e}")
        _wallet_balance_cache["checked_at"] = now
        if cached is not None:
            return {**cached, "ok": False, "cached": True, "error": str(e)}
        return {"balance_usd": None, "synced_at": None, "ok": False, "cached": False, "error": str(e)}

@app.get("/api/portfolio")
def portfolio():
    """
    Live total portfolio: on-chain wallet balance (idle USDC) + current
    market value of all open REAL positions (across every non-paper city —
    see config.cities.CityConfig.paper_trading) + current market value of
    all open positions. portfolio_usd = balance_usd + positions_value_usd
    — the bot's real net worth right now, not just the idle cash sitting
    in the deposit wallet between trades.

    Paper-trading cities (e.g. WMKK) are excluded from positions_value_usd:
    their simulated positions never drew on the shared wallet, so including
    them would inflate this figure with capital that was never actually at
    risk. Use /api/open_positions?icao=<paper city> to see paper P&L.
    """
    from core.edge import fetch_market_price

    wallet      = wallet_balance()
    balance_usd = wallet.get("balance_usd")

    rows = _rows(
        "SELECT icao_code, token_id, bracket_label, entry_price, size_usd "
        "FROM open_positions WHERE is_paper = 0"
    )
    positions_value = 0.0
    for r in rows:
        entry    = float(r["entry_price"])
        size_usd = float(r["size_usd"])
        if entry <= 0:
            continue
        shares_held = size_usd / entry
        try:
            live_price = fetch_market_price(r["token_id"])
            price = live_price.mid_price if live_price else entry
        except Exception as e:
            logger.warning(f"[DASHBOARD] portfolio price fetch failed for {r['token_id']}: {e}")
            price = entry
        positions_value += shares_held * price

    portfolio_usd = (balance_usd or 0.0) + positions_value
    return {
        "balance_usd":         balance_usd,
        "positions_value_usd": round(positions_value, 2),
        "portfolio_usd":       round(portfolio_usd, 2),
        "n_positions":         len(rows),
        "ok":                  wallet.get("ok", False),
    }

@app.get("/api/kpis")
def kpis(icao: str = DEFAULT_ICAO):
    """Headline KPI numbers for one city."""
    icao        = icao.upper()
    is_paper    = _is_paper(icao)
    vault_start = _vault_start(icao)
    net_pnl      = _scalar(
        "SELECT COALESCE(SUM(realised_pnl),0) FROM exit_log WHERE icao_code = ? AND is_paper = ?",
        (icao, is_paper),
    )
    total_trades = _scalar(
        "SELECT COUNT(*) FROM exit_log WHERE icao_code = ? AND is_paper = ?", (icao, is_paper), default=0,
    )
    wins         = _scalar(
        "SELECT COUNT(*) FROM exit_log WHERE icao_code = ? AND is_paper = ? AND realised_pnl > 0",
        (icao, is_paper), default=0,
    )
    trail_bias   = _scalar("""
        SELECT AVG(residual) FROM (
            SELECT residual FROM calibration_logs WHERE icao_code = ? ORDER BY id DESC LIMIT 10
        )
    """, (icao,), default=0.0)
    mae          = _scalar("SELECT AVG(ABS(residual)) FROM calibration_logs WHERE icao_code = ?", (icao,), default=0.0)
    n_calib      = _scalar("SELECT COUNT(*) FROM calibration_logs WHERE icao_code = ?", (icao,), default=0)
    n_open       = _scalar(
        "SELECT COUNT(*) FROM open_positions WHERE icao_code = ? AND is_paper = ?", (icao, is_paper), default=0,
    )
    actionable   = _scalar(
        "SELECT COUNT(*) FROM signal_log WHERE icao_code = ? AND action IN ('SIGNAL_BUY','SIGNAL_SELL_NO')",
        (icao,), default=0,
    )
    avg_edge     = _scalar(
        "SELECT AVG(ABS(edge)) FROM signal_log WHERE icao_code = ? AND action IN ('SIGNAL_BUY','SIGNAL_SELL_NO')",
        (icao,), default=0.0,
    )
    losses       = total_trades - wins
    win_rate     = (wins / total_trades * 100) if total_trades else 0.0
    roi          = (net_pnl / vault_start * 100) if vault_start else 0.0
    return {
        "icao":           icao,
        "paper_trading":  bool(is_paper),
        "vault_current":  round(vault_start + net_pnl, 2),
        "vault_start":    vault_start,
        "net_pnl":        round(net_pnl, 4),
        "roi_pct":        round(roi, 2),
        "total_trades":   int(total_trades),
        "wins":           int(wins),
        "losses":         int(losses),
        "win_rate_pct":   round(win_rate, 1),
        "trailing_bias":  round(trail_bias, 4),
        "mae_celsius":    round(mae, 4),
        "n_calibrations": int(n_calib),
        "open_positions": int(n_open),
        "actionable_signals": int(actionable),
        "avg_edge_pct":   round(avg_edge * 100, 2),
    }

@app.get("/api/overview")
def overview():
    """
    Aggregated view across every configured city — the "center dashboard".
    One row per city (allocation, current vault, ROI, win rate, open
    positions) plus combined totals.

    Combined totals only sum REAL (non-paper) cities — a paper city's
    simulated P&L must never inflate/deflate the real "Combined Vault" figure.
    Paper cities still get their own row (with paper_trading=true) so their
    simulated results are visible, just not blended into the real total.
    """
    rows = []
    combined_start = 0.0
    combined_pnl   = 0.0
    for config in CITIES.values():
        icao     = config.icao
        is_paper = _is_paper(icao)
        vstart = _vault_start(icao)
        pnl    = _scalar(
            "SELECT COALESCE(SUM(realised_pnl),0) FROM exit_log WHERE icao_code = ? AND is_paper = ?",
            (icao, is_paper),
        )
        total_trades = _scalar(
            "SELECT COUNT(*) FROM exit_log WHERE icao_code = ? AND is_paper = ?", (icao, is_paper), default=0,
        )
        wins   = _scalar(
            "SELECT COUNT(*) FROM exit_log WHERE icao_code = ? AND is_paper = ? AND realised_pnl > 0",
            (icao, is_paper), default=0,
        )
        n_open = _scalar(
            "SELECT COUNT(*) FROM open_positions WHERE icao_code = ? AND is_paper = ?", (icao, is_paper), default=0,
        )
        avg_edge = _scalar(
            "SELECT AVG(ABS(edge)) FROM signal_log WHERE icao_code = ? AND action IN ('SIGNAL_BUY','SIGNAL_SELL_NO')",
            (icao,), default=0.0,
        )
        win_rate = (wins / total_trades * 100) if total_trades else 0.0
        roi      = (pnl / vstart * 100) if vstart else 0.0

        rows.append({
            "icao":           icao,
            "display_name":   config.display_name,
            "paper_trading":  bool(is_paper),
            "vault_start":    round(vstart, 2),
            "vault_current":  round(vstart + pnl, 2),
            "net_pnl":        round(pnl, 4),
            "roi_pct":        round(roi, 2),
            "total_trades":   int(total_trades),
            "win_rate_pct":   round(win_rate, 1),
            "open_positions": int(n_open),
            "avg_edge_pct":   round(avg_edge * 100, 2),
        })
        if not is_paper:
            combined_start += vstart
            combined_pnl   += pnl

    combined_roi = (combined_pnl / combined_start * 100) if combined_start else 0.0
    return {
        "cities": rows,
        "combined": {
            "vault_start":   round(combined_start, 2),
            "vault_current": round(combined_start + combined_pnl, 2),
            "net_pnl":       round(combined_pnl, 4),
            "roi_pct":       round(combined_roi, 2),
        },
    }

@app.get("/api/equity")
def equity(icao: str = DEFAULT_ICAO):
    """Cumulative vault equity per closed trade, for one city."""
    icao = icao.upper()
    vault_start = _vault_start(icao)
    rows = _rows(
        "SELECT id, bracket_label, direction, reason, "
        "entry_price, exit_price, size_usd, realised_pnl, closed_at "
        "FROM exit_log WHERE icao_code = ? AND is_paper = ? ORDER BY id ASC",
        (icao, _is_paper(icao)),
    )
    running = vault_start
    for r in rows:
        running += r["realised_pnl"]
        r["vault"] = round(running, 4)
    return {"vault_start": vault_start, "trades": rows}

@app.get("/api/signals")
def signals(limit: int = 80, icao: str = DEFAULT_ICAO):
    """Recent signal scan results for one city — ALL brackets including non-actionable."""
    rows = _rows(
        "SELECT id, date, bracket_label, model_prob, market_price, "
        "edge, action, settled_outcome, COALESCE(gate_reason,'') as gate_reason "
        "FROM signal_log WHERE icao_code = ? ORDER BY id DESC LIMIT ?",
        (icao.upper(), limit),
    )
    rows.reverse()
    return {"signals": rows}


@app.get("/api/signal_summary")
def signal_summary(icao: str = DEFAULT_ICAO):
    """
    Breakdown of signal action labels for one city, across all time.
    Lets the dashboard show a full scan funnel:
      Total priced → Edge threshold met → EV/sizing passed → Executed
    """
    rows = _rows(
        "SELECT action, COUNT(*) as count, "
        "AVG(ABS(edge)) as avg_abs_edge "
        "FROM signal_log WHERE icao_code = ? GROUP BY action ORDER BY count DESC",
        (icao.upper(),),
    )
    LABEL_MAP = {
        "SIGNAL_BUY":     "BUY signal",
        "SIGNAL_SELL_NO": "SELL signal",
        "HOLD_EDGE":      "Held — edge below 5%",
        "SKIP_ILLIQUID":  "Skipped — illiquid",
        "SKIP_SPREAD":    "Skipped — wide spread",
        "NO_PRICE":       "No price fetched",
    }
    for r in rows:
        r["display_label"] = LABEL_MAP.get(r["action"], r["action"])
        r["avg_abs_edge"]  = round(r["avg_abs_edge"] or 0.0, 4)
    return {"breakdown": rows}

@app.get("/api/latest_scan")
def latest_scan(icao: str = DEFAULT_ICAO):
    """
    Every bracket from the MOST RECENT scan cycle for one city — passing
    AND non-passing. The live snapshot: for the current market, which
    brackets cleared the edge gate, which were held below threshold, and
    which were gated on liquidity/spread — each with model prob, market
    price, and edge.
    """
    latest = _rows(
        "SELECT id, date, bracket_label, model_prob, market_price, edge, "
        "action, COALESCE(gate_reason,'') as gate_reason "
        "FROM signal_log WHERE icao_code = ? ORDER BY id DESC LIMIT 40",
        (icao.upper(),),
    )
    if not latest:
        return {"scan": [], "scan_date": None, "n_brackets": 0,
                "n_passed": 0, "n_blocked": 0, "edge_threshold": 0.05}

    # Keep the most recent row per bracket (highest id)
    seen = {}
    for r in latest:
        if r["bracket_label"] not in seen:
            seen[r["bracket_label"]] = r
    scan = sorted(seen.values(), key=lambda r: r["bracket_label"])

    THRESH = float(os.getenv("EDGE_THRESHOLD", 0.05))
    STATUS = {
        "SIGNAL_BUY":     ("PASS",  "BUY YES",            True),
        "SIGNAL_SELL_NO": ("PASS",  "SELL NO",            True),
        "HOLD_EDGE":      ("HOLD",  f"edge < {THRESH*100:.0f}%", False),
        "SKIP_ILLIQUID":  ("GATED", "illiquid book",      False),
        "SKIP_SPREAD":    ("GATED", "spread > 8c",        False),
        "NO_PRICE":       ("GATED", "no price",           False),
    }
    for r in scan:
        status, label, passed = STATUS.get(r["action"], ("HOLD", "hold", False))
        r["status"]       = status
        r["status_label"] = label
        r["passed"]       = passed
        r["edge_pct"]     = round(r["edge"] * 100, 2)
        r["model_pct"]    = round(r["model_prob"] * 100, 1)
        r["market_pct"]   = round(r["market_price"] * 100, 1)

    n_pass = sum(1 for r in scan if r["passed"])
    return {
        "scan":           scan,
        "scan_date":      latest[0]["date"],
        "edge_threshold": THRESH,
        "n_brackets":     len(scan),
        "n_passed":       n_pass,
        "n_blocked":      len(scan) - n_pass,
    }

@app.get("/api/calibration")
def calibration(icao: str = DEFAULT_ICAO):
    """
    All calibration residuals for one city, in the same id-ordered sequence
    the live bot itself processes them in (db/ledger.py:fetch_trailing_bias
    orders by id, not market_date — id is "the order calibration rows were
    written in", which is usually but not always the same as calendar-date
    order: Job 4 checks both today's and yesterday's date every cycle, so a
    backfilled prior-day entry can land right after a same-day entry.
    Rather than silently reorder by market_date (which would also disagree
    with what the bot actually used at trade time), this returns market_date
    alongside each row so the frontend can label bars with real dates.

    trailing_bias here is a ROLLING last-10 window, matching
    fetch_trailing_bias(icao, n=10) exactly (same id order, same window
    size) — NOT an expanding all-time average. An expanding mean smooths
    over the city's entire history forever and increasingly diverges from
    what the bot is actually using to calibrate live trades as more data
    accumulates; this makes the dashboard line track the real number.
    """
    rows = _rows(
        "SELECT id, timestamp, market_date, icao_code, model_mu, actual_settled, residual "
        "FROM calibration_logs WHERE icao_code = ? ORDER BY id ASC",
        (icao.upper(),),
    )
    window = []
    for r in rows:
        window.append(r["residual"])
        if len(window) > 10:
            window.pop(0)
        r["trailing_bias"] = round(sum(window) / len(window), 4)
    return {"calibrations": rows}

@app.get("/api/pnl_by_bracket")
def pnl_by_bracket(icao: str = DEFAULT_ICAO):
    """P&L grouped by bracket + direction, for one city."""
    icao = icao.upper()
    rows = _rows(
        "SELECT bracket_label, direction, "
        "SUM(realised_pnl) AS total_pnl, COUNT(*) AS n_trades, "
        "SUM(CASE WHEN realised_pnl > 0 THEN 1 ELSE 0 END) AS wins "
        "FROM exit_log WHERE icao_code = ? AND is_paper = ? GROUP BY bracket_label, direction ORDER BY bracket_label",
        (icao, _is_paper(icao)),
    )
    return {"groups": rows}

@app.get("/api/exit_reasons")
def exit_reasons(icao: str = DEFAULT_ICAO):
    """Exit reason counts for one city."""
    icao = icao.upper()
    rows = _rows(
        "SELECT reason, COUNT(*) AS count, "
        "SUM(realised_pnl) AS total_pnl "
        "FROM exit_log WHERE icao_code = ? AND is_paper = ? GROUP BY reason ORDER BY count DESC",
        (icao, _is_paper(icao)),
    )
    return {"reasons": rows}

@app.get("/api/open_positions")
def open_positions(icao: str = DEFAULT_ICAO):
    """All currently open positions for one city, with live trail/stop
    levels, current market value, and unrealised P&L."""
    from core.edge import fetch_market_price

    icao = icao.upper()
    rows = _rows(
        "SELECT * FROM open_positions WHERE icao_code = ? AND is_paper = ? ORDER BY opened_at ASC",
        (icao, _is_paper(icao)),
    )
    trail_pct   = float(os.getenv("TRAIL_PCT", 0.20))
    edge_thresh = float(os.getenv("EDGE_THRESHOLD", 0.05))
    for r in rows:
        label     = r["bracket_label"]
        direction = "NO" if ":NO" in label else "YES"
        entry     = float(r["entry_price"])
        size_usd  = float(r["size_usd"])
        peak_raw  = r.get("peak_price")
        peak      = float(peak_raw) if peak_raw is not None else entry
        shares_held = size_usd / entry if entry > 0 else 0.0
        r["direction"]   = direction
        r["trail_pct"]   = trail_pct
        # core/position_monitor.py: YES/BUY positions profit as price rises
        # (trail follows the peak down); NO/SELL positions are a synthetic
        # short on the YES token (see core/edge.py's SELL YES = BUY NO
        # comment) and profit as price FALLS, so "peak" there means the
        # lowest mid seen and the trail level sits ABOVE it.
        if direction == "YES":
            r["trail_level"]  = round(peak * (1 - trail_pct), 5) if peak > entry else None
            r["trail_armed"]  = peak > entry
            r["stop_level"]   = round(entry - edge_thresh, 5)
        else:
            r["trail_level"]  = round(peak * (1 + trail_pct), 5) if peak < entry else None
            r["trail_armed"]  = peak < entry
            r["stop_level"]   = round(entry + edge_thresh, 5)
        # Config thresholds as percentages — the trailing-stop drawdown is
        # already scale-free (20% off peak regardless of price level), but
        # the stop-loss is a fixed price delta (EDGE_THRESHOLD), so express
        # it relative to entry so it reads the same way ("triggers at -X%").
        r["profit_taking_pct"] = round(trail_pct * 100, 2)
        r["stop_loss_pct"]     = round((edge_thresh / entry) * 100, 2) if entry > 0 else None

        # Current market value + unrealised P&L — a fresh price fetch per
        # position (same call position_monitor.py makes each cycle); a
        # failure here just falls back to entry (flat, 0% unrealised)
        # rather than breaking the whole endpoint for one bad lookup.
        try:
            live_price = fetch_market_price(r["token_id"])
            current_price = live_price.mid_price if live_price else entry
        except Exception as e:
            logger.warning(f"[DASHBOARD] price fetch failed for {icao}:{label}: {e}")
            current_price = entry

        current_value = shares_held * current_price
        if direction == "YES":
            unrealized_pnl_usd = (current_price - entry) * shares_held
            unrealized_pnl_pct = ((current_price - entry) / entry) * 100 if entry > 0 else None
        else:
            # SELL/NO: profits when the YES mid falls below entry.
            unrealized_pnl_usd = (entry - current_price) * shares_held
            unrealized_pnl_pct = ((entry - current_price) / entry) * 100 if entry > 0 else None

        r["current_price"]      = round(current_price, 5)
        r["current_value"]      = round(current_value, 2)
        r["unrealized_pnl_usd"] = round(unrealized_pnl_usd, 4)
        r["unrealized_pnl_pct"] = round(unrealized_pnl_pct, 2) if unrealized_pnl_pct is not None else None

        # Hold duration
        try:
            opened = datetime.datetime.fromisoformat(r["opened_at"])
            delta  = datetime.datetime.utcnow() - opened
            hours  = delta.total_seconds() / 3600
            r["hold_hours"] = round(hours, 1)
        except Exception:
            r["hold_hours"] = 0.0
    return {"positions": rows}

@app.get("/api/trades")
def trades(limit: int = 100, icao: str = DEFAULT_ICAO):
    """Recent trade history for one city."""
    icao = icao.upper()
    rows = _rows(
        "SELECT id, closed_at, bracket_label, direction, reason, "
        "entry_price, exit_price, size_usd, realised_pnl, opened_at "
        "FROM exit_log WHERE icao_code = ? AND is_paper = ? ORDER BY id DESC LIMIT ?",
        (icao, _is_paper(icao), limit),
    )
    return {"trades": rows}
