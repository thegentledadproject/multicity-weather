"""
core/decision.py — P8: one deterministic entry decision with reason codes.

decide() is a pure function of a DecisionInputs snapshot: no clock, network
or DB access, so the same inputs always produce the same decision (Gate P8),
and historical replay (P11) runs this exact function on recorded snapshots.

Checks, in order (hard blocks first, then soft holds):
  BLOCK  KILL_SWITCH:<name>, SETTLEMENT_INVALID, INFO_<STATE> (not fresh),
         POSITION_OPEN, OPPOSITE_SIDE_HELD
  HOLD   EDGE_<STATE> (not ACTIONABLE), SIZING_HOLD (no Kelly size),
         NO_EXECUTABLE_EDGE (Qdepth below the minimum order), RISK_CAPACITY
  ENTER  with EDGE_ACTIONABLE, INFO_FRESH, EXECUTABLE_EV_POSITIVE,
         LIQUIDITY_OK, CORRELATION_OK, RISK_OK

Final quantity = min(Qkelly, Qdepth, Qcorrelation, Qgeo, Qbankroll).
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

from core.edge_state import ACTIONABLE
from core.execution_curve import CURVE_STEP_USD, entry_capacity, execution_curve

ENTER, HOLD, BLOCK = "ENTER", "HOLD", "BLOCK"
MIN_ORDER_USD = 1.0  # Polymarket's minimum marketable order


@dataclass
class DecisionInputs:
    icao: str
    market_date: str
    bracket: str
    direction: str                 # "BUY" (YES) or "SELL" (NO)
    scan_id: int
    win_prob: float                # model probability that the side bought wins
    edge_threshold: float
    edge_state: str
    freshness: str                 # core.execution FRESH/DEGRADED/STALE/INVALID
    settlement_valid: bool
    position_open: bool
    q_kelly: float                 # 0 when sizing says HOLD
    limits: Dict                   # core.portfolio.position_limits() output
    asks: List[Tuple[float, float]]  # execution token's ask side at decision time
    kill_switches: List[str] = field(default_factory=list)


@dataclass
class Decision:
    action: str
    reasons: List[str]
    quantities: Dict[str, float]
    expected: Dict[str, float]     # at the final quantity: vwap, slippage, exec_edge, ev


def decide(i: DecisionInputs) -> Decision:
    q = {"Qkelly": i.q_kelly, "Qdepth": 0.0, "Qcorrelation": i.limits["Qcorrelation"],
         "Qgeo": i.limits["Qgeo"], "Qbankroll": i.limits["Qbankroll"], "final": 0.0}

    blocks = [f"KILL_SWITCH:{k}" for k in i.kill_switches]
    if not i.settlement_valid:
        blocks.append("SETTLEMENT_INVALID")
    if i.freshness not in ("FRESH", "DEGRADED"):
        blocks.append(f"INFO_{i.freshness}")
    if i.position_open:
        blocks.append("POSITION_OPEN")
    if i.limits.get("blocked"):
        blocks.append(i.limits["blocked"])
    if blocks:
        return Decision(BLOCK, blocks, q, {})

    holds = []
    if i.edge_state != ACTIONABLE:
        holds.append(f"EDGE_{i.edge_state}")
    if i.q_kelly < MIN_ORDER_USD:
        holds.append("SIZING_HOLD")
    risk_cap = min(q["Qcorrelation"], q["Qgeo"], q["Qbankroll"])
    if risk_cap < MIN_ORDER_USD:
        holds.append("RISK_CAPACITY")

    ceiling = min(i.q_kelly, risk_cap)
    curve = execution_curve(i.asks, i.win_prob, max_usd=ceiling,
                            step_usd=min(CURVE_STEP_USD, max(ceiling, MIN_ORDER_USD)))
    q["Qdepth"] = entry_capacity(curve, i.edge_threshold)
    if q["Qdepth"] < MIN_ORDER_USD:
        holds.append("NO_EXECUTABLE_EDGE")
    if holds:
        return Decision(HOLD, holds, q, {})

    q["final"] = min(ceiling, q["Qdepth"])
    point = next(p for p in reversed(curve) if p.usd <= q["final"] + 1e-9)
    expected = {"vwap": point.vwap, "slippage": point.slippage,
                "exec_edge": point.exec_edge, "ev": point.ev, "marginal_ev": point.marginal_ev}
    reasons = ["EDGE_ACTIONABLE", f"INFO_{i.freshness}", "EXECUTABLE_EV_POSITIVE",
               "LIQUIDITY_OK", "CORRELATION_OK", "RISK_OK"]
    return Decision(ENTER, reasons, q, expected)


def snapshot(inputs: DecisionInputs, decision: Decision) -> Dict:
    """JSON-ready record of everything the decision depended on (decision_log)."""
    return {"inputs": asdict(inputs), "decision": asdict(decision)}
