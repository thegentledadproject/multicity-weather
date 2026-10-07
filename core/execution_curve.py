"""
core/execution_curve.py — P6: theoretical edge -> executable edge.

The scan's edge is measured against the mid. What a trade actually earns
depends on walking the ask side for the size bought, paying the fee, and a
safety buffer. For each spend u (USD) this computes:

  shares(u)          shares bought walking the asks cheapest-first
  vwap(u)            u / shares(u)
  slippage(u)        vwap(u) - best ask
  exec_edge(u)       win_prob - vwap(u) * (1 + fee) - buffer   (per share)
  ev(u)              shares(u) * (win_prob - buffer) - u * (1 + fee)   (USD)
  marginal_ev(u)     d ev / d u between this step and the previous one

Entry capacity (Qdepth) is the largest u for which every step so far still
has exec_edge >= the entry threshold and marginal_ev >= MIN_MARGINAL_EV:
buying more past that point adds risk faster than it adds expected value.

Exit capacity sells into the bids: normal, and stressed (depth halved and
every bid STRESS_PRICE_SHIFT worse) for "can we get out if the book thins
or reprices against us".

Prices/sizes are [(price, size_in_shares)], the shape core.edge.book_levels returns.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.sizing import TAKER_FEE_RATE

EXEC_BUFFER = 0.01          # per-share safety margin beyond fees
MIN_MARGINAL_EV = 0.02      # each extra $ must add >= 2c of expected value
CURVE_STEP_USD = 5.0
STRESS_DEPTH_FACTOR = 0.5
STRESS_PRICE_SHIFT = 0.05

Levels = List[Tuple[float, float]]


@dataclass
class CurvePoint:
    usd: float
    shares: float
    vwap: float
    slippage: float
    exec_edge: float
    ev: float
    marginal_ev: Optional[float]


def _buy(asks: Levels, usd: float) -> Optional[float]:
    """Shares bought for `usd`, walking asks cheapest-first; None if the book is too thin."""
    shares, left = 0.0, usd
    for price, size in sorted(asks):
        take = min(left, price * size)
        shares += take / price
        left -= take
        if left <= 1e-9:
            return shares
    return None


def execution_curve(asks: Levels, win_prob: float, max_usd: float,
                    step_usd: float = CURVE_STEP_USD, fee: float = TAKER_FEE_RATE,
                    buffer: float = EXEC_BUFFER) -> List[CurvePoint]:
    if not asks:
        return []
    best = min(p for p, _ in asks)
    points: List[CurvePoint] = []
    usd = step_usd
    while usd <= max_usd + 1e-9:
        shares = _buy(asks, usd)
        if shares is None:
            break
        vwap = usd / shares
        ev = shares * (win_prob - buffer) - usd * (1 + fee)
        marginal = (ev - points[-1].ev) / step_usd if points else ev / usd
        points.append(CurvePoint(usd, shares, vwap, vwap - best,
                                 win_prob - vwap * (1 + fee) - buffer, ev, marginal))
        usd += step_usd
    return points


def entry_capacity(curve: List[CurvePoint], edge_threshold: float,
                   min_marginal_ev: float = MIN_MARGINAL_EV) -> float:
    """Qdepth: largest spend whose every step keeps executable edge and marginal EV above the bars."""
    capacity = 0.0
    for point in curve:
        if point.exec_edge < edge_threshold or point.marginal_ev < min_marginal_ev:
            break
        capacity = point.usd
    return capacity


def liquidation(bids: Levels, shares: float, depth_factor: float = 1.0,
                price_shift: float = 0.0) -> Tuple[float, float]:
    """(proceeds USD, shares sold) selling `shares` into the bids, best first."""
    proceeds, left = 0.0, shares
    for price, size in sorted(bids, reverse=True):
        price = max(price - price_shift, 0.0)
        take = min(left, size * depth_factor)
        proceeds += take * price
        left -= take
        if left <= 1e-9:
            break
    return proceeds, shares - left


def exit_capacity(bids: Levels, shares: float) -> dict:
    """Normal and stressed liquidation of a position of `shares`."""
    normal = liquidation(bids, shares)
    stress = liquidation(bids, shares, STRESS_DEPTH_FACTOR, STRESS_PRICE_SHIFT)
    return {"normal_proceeds": normal[0], "normal_filled": normal[1],
            "stress_proceeds": stress[0], "stress_filled": stress[1]}


if __name__ == "__main__":
    # 100 shares at 0.40, then 100 at 0.50, then 1000 at 0.70.
    asks = [(0.50, 100), (0.40, 100), (0.70, 1000)]
    curve = execution_curve(asks, win_prob=0.65, max_usd=200)
    assert abs(curve[0].vwap - 0.40) < 1e-12 and curve[0].slippage == 0
    assert abs(curve[8].vwap - 45 / (100 + 5 / 0.5)) < 1e-12  # $45: 100 @0.40 + 10 @0.50
    assert all(a.marginal_ev >= b.marginal_ev - 1e-12 for a, b in zip(curve, curve[1:]))  # deeper = worse
    # Marginal EV per $ is 0.64/0.40-1.02 = +0.58, then 0.64/0.50-1.02 = +0.26, then
    # 0.64/0.70-1.02 < 0: capacity stops where the 0.70 asks begin ($40 + $50 = $90).
    assert entry_capacity(curve, edge_threshold=0.08) == 90.0
    # $65 buys 100 + 25/0.5 = 150 shares at 0.4333: 0.65 - 0.4333*1.02 - 0.01 = 0.198 < 0.20.
    assert entry_capacity(curve, edge_threshold=0.20) == 60.0
    assert entry_capacity(execution_curve(asks, 0.30, 200), 0.08) == 0  # no edge at any size
    assert execution_curve([(0.4, 10)], 0.6, 100) == []  # $4 of depth can't fill a $5 step
    bids = [(0.38, 50), (0.35, 100)]
    cap = exit_capacity(bids, 100)
    assert abs(cap["normal_proceeds"] - (50 * 0.38 + 50 * 0.35)) < 1e-12
    assert cap["stress_filled"] == 75 and abs(cap["stress_proceeds"] - (25 * 0.33 + 50 * 0.30)) < 1e-12
    print("execution_curve checks passed")
