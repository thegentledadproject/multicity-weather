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
import contextlib
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core.model import ECMWF_ENSEMBLE_URL
from config.cities import CITIES, get_city, resolve_vault_usd

logger = logging.getLogger("hermes.dashboard_api")
logging.basicConfig(level=logging.INFO)

DB_PATH        = os.getenv("DB_PATH", "hermes.db")
DEFAULT_ICAO   = "WSSS"
_DASHBOARD_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")

# ── Upstream (Open-Meteo ECMWF) health check cache ───────────────────────────
# Keyed by icao since each city polls its own coordinates.
UPSTREAM_CACHE_TTL = 30.0
_upstream_cache: Dict[str, Dict[str, Any]] = {}

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
                "icao":         c.icao,
                "display_name": c.display_name,
                "vault_start":  _vault_start(c.icao),
            }
            for c in CITIES.values()
        ]
    }

@app.get("/api/upstream_status")
def upstream_status(icao: str = DEFAULT_ICAO):
    """
    Live reachability check for the Open-Meteo ECMWF ensemble API at this
    city's coordinates — the upstream forecast source core/model.py depends
    on for mu_ecmwf/sigma_ecmwf. If this is down, the bot falls back to
    GFS-only (or the hard prior), which is exactly the kind of degradation
    an operator wants surfaced on the dashboard rather than discovered later.
    """
    icao = icao.upper()
    now = time.monotonic()
    cached = _upstream_cache.get(icao, {}).get("result")
    checked_at = _upstream_cache.get(icao, {}).get("checked_at", 0.0)
    if cached is not None and (now - checked_at) < UPSTREAM_CACHE_TTL:
        return {**cached, "cached": True}

    try:
        config = get_city(icao)
    except KeyError:
        return {"ok": False, "error": f"unknown city '{icao}'", "cached": False}

    url = ECMWF_ENSEMBLE_URL.format(lat=config.lat, lon=config.lon)
    t0 = time.monotonic()
    try:
        r = requests.get(url, timeout=5)
        result = {
            "ok":           r.status_code == 200,
            "status_code":  r.status_code,
            "latency_ms":   round((time.monotonic() - t0) * 1000),
            "checked_at":   datetime.datetime.utcnow().isoformat(),
            "source":       "ecmwf_ensemble",
        }
    except requests.RequestException as e:
        logger.error(f"[DASHBOARD] Upstream ECMWF check failed for {icao}: {e}")
        result = {
            "ok":           False,
            "status_code":  None,
            "latency_ms":   round((time.monotonic() - t0) * 1000),
            "checked_at":   datetime.datetime.utcnow().isoformat(),
            "source":       "ecmwf_ensemble",
            "error":        str(e),
        }

    _upstream_cache[icao] = {"result": result, "checked_at": now}
    return {**result, "cached": False}

@app.get("/api/kpis")
def kpis(icao: str = DEFAULT_ICAO):
    """Headline KPI numbers for one city."""
    icao        = icao.upper()
    vault_start = _vault_start(icao)
    net_pnl      = _scalar("SELECT COALESCE(SUM(realised_pnl),0) FROM exit_log WHERE icao_code = ?", (icao,))
    total_trades = _scalar("SELECT COUNT(*) FROM exit_log WHERE icao_code = ?", (icao,), default=0)
    wins         = _scalar(
        "SELECT COUNT(*) FROM exit_log WHERE icao_code = ? AND realised_pnl > 0", (icao,), default=0,
    )
    trail_bias   = _scalar("""
        SELECT AVG(residual) FROM (
            SELECT residual FROM calibration_logs WHERE icao_code = ? ORDER BY id DESC LIMIT 10
        )
    """, (icao,), default=0.0)
    mae          = _scalar("SELECT AVG(ABS(residual)) FROM calibration_logs WHERE icao_code = ?", (icao,), default=0.0)
    n_calib      = _scalar("SELECT COUNT(*) FROM calibration_logs WHERE icao_code = ?", (icao,), default=0)
    n_open       = _scalar("SELECT COUNT(*) FROM open_positions WHERE icao_code = ?", (icao,), default=0)
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
    """
    rows = []
    combined_start = 0.0
    combined_pnl   = 0.0
    for config in CITIES.values():
        icao   = config.icao
        vstart = _vault_start(icao)
        pnl    = _scalar("SELECT COALESCE(SUM(realised_pnl),0) FROM exit_log WHERE icao_code = ?", (icao,))
        total_trades = _scalar("SELECT COUNT(*) FROM exit_log WHERE icao_code = ?", (icao,), default=0)
        wins   = _scalar("SELECT COUNT(*) FROM exit_log WHERE icao_code = ? AND realised_pnl > 0", (icao,), default=0)
        n_open = _scalar("SELECT COUNT(*) FROM open_positions WHERE icao_code = ?", (icao,), default=0)
        avg_edge = _scalar(
            "SELECT AVG(ABS(edge)) FROM signal_log WHERE icao_code = ? AND action IN ('SIGNAL_BUY','SIGNAL_SELL_NO')",
            (icao,), default=0.0,
        )
        win_rate = (wins / total_trades * 100) if total_trades else 0.0
        roi      = (pnl / vstart * 100) if vstart else 0.0

        rows.append({
            "icao":           icao,
            "display_name":   config.display_name,
            "vault_start":    round(vstart, 2),
            "vault_current":  round(vstart + pnl, 2),
            "net_pnl":        round(pnl, 4),
            "roi_pct":        round(roi, 2),
            "total_trades":   int(total_trades),
            "win_rate_pct":   round(win_rate, 1),
            "open_positions": int(n_open),
            "avg_edge_pct":   round(avg_edge * 100, 2),
        })
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
        "FROM exit_log WHERE icao_code = ? ORDER BY id ASC",
        (icao,),
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
    """All calibration residuals for one city."""
    rows = _rows(
        "SELECT id, timestamp, icao_code, model_mu, actual_settled, residual "
        "FROM calibration_logs WHERE icao_code = ? ORDER BY id ASC",
        (icao.upper(),),
    )
    # Compute expanding trailing bias
    running_sum = 0.0
    for i, r in enumerate(rows):
        running_sum += r["residual"]
        r["trailing_bias"] = round(running_sum / (i + 1), 4)
    return {"calibrations": rows}

@app.get("/api/pnl_by_bracket")
def pnl_by_bracket(icao: str = DEFAULT_ICAO):
    """P&L grouped by bracket + direction, for one city."""
    rows = _rows(
        "SELECT bracket_label, direction, "
        "SUM(realised_pnl) AS total_pnl, COUNT(*) AS n_trades, "
        "SUM(CASE WHEN realised_pnl > 0 THEN 1 ELSE 0 END) AS wins "
        "FROM exit_log WHERE icao_code = ? GROUP BY bracket_label, direction ORDER BY bracket_label",
        (icao.upper(),),
    )
    return {"groups": rows}

@app.get("/api/exit_reasons")
def exit_reasons(icao: str = DEFAULT_ICAO):
    """Exit reason counts for one city."""
    rows = _rows(
        "SELECT reason, COUNT(*) AS count, "
        "SUM(realised_pnl) AS total_pnl "
        "FROM exit_log WHERE icao_code = ? GROUP BY reason ORDER BY count DESC",
        (icao.upper(),),
    )
    return {"reasons": rows}

@app.get("/api/open_positions")
def open_positions(icao: str = DEFAULT_ICAO):
    """All currently open positions for one city, with live trail/stop levels."""
    rows = _rows(
        "SELECT * FROM open_positions WHERE icao_code = ? ORDER BY opened_at ASC",
        (icao.upper(),),
    )
    trail_pct   = float(os.getenv("TRAIL_PCT", 0.20))
    edge_thresh = float(os.getenv("EDGE_THRESHOLD", 0.05))
    for r in rows:
        label     = r["bracket_label"]
        direction = "NO" if ":NO" in label else "YES"
        entry     = float(r["entry_price"])
        peak_raw  = r.get("peak_price")
        peak      = float(peak_raw) if peak_raw is not None else entry
        r["direction"]   = direction
        r["trail_pct"]   = trail_pct
        if direction == "YES":
            r["trail_level"]  = round(peak * (1 - trail_pct), 5) if peak > entry else None
            r["trail_armed"]  = peak > entry
            r["stop_level"]   = round(entry - edge_thresh, 5)
        else:
            r["trail_level"]  = round(peak * (1 + trail_pct), 5) if peak < entry else None
            r["trail_armed"]  = peak < entry
            r["stop_level"]   = round(entry + edge_thresh, 5)
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
    rows = _rows(
        "SELECT id, closed_at, bracket_label, direction, reason, "
        "entry_price, exit_price, size_usd, realised_pnl, opened_at "
        "FROM exit_log WHERE icao_code = ? ORDER BY id DESC LIMIT ?",
        (icao.upper(), limit),
    )
    return {"trades": rows}
