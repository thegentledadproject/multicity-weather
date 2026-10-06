"""
core/edge_state.py — P5: an edge is a time series, not a single reading.

A bracket's edge (model prob − market mid) is tracked across Job 2 scans of
the same market date. Entries are only allowed once an edge has persisted
and is not collapsing — a one-scan edge is as likely to be a stale quote or
a forecast blip as an opportunity.

Lifecycle (sign-aware; a run is consecutive scans with |edge| >= threshold
and the same sign, ending at the current scan):

  NO_EDGE     |edge| < threshold, and no same-sign edge in the lookback
  EMERGING    first scan of a run
  CONFIRMED   run of >= 2 scans, but spanning < MIN_SPAN_MINUTES
  ACTIONABLE  confirmed over >= MIN_SPAN_MINUTES and not decaying
  DECAYING    in a run, but down > DECAY_FRAC from the run's peak, or on
              course to halve within MIN_HALF_LIFE_HOURS
  EXHAUSTED   |edge| < threshold after a same-sign run in the lookback

Velocity is the change in |edge| per hour over the last two scans (negative
= shrinking); half-life is the hours until |edge| halves at that velocity.
"""

import datetime
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

NO_EDGE, EMERGING, CONFIRMED = "NO_EDGE", "EMERGING", "CONFIRMED"
ACTIONABLE, DECAYING, EXHAUSTED = "ACTIONABLE", "DECAYING", "EXHAUSTED"

LOOKBACK_SCANS = 8          # ~2h at Job 2's 15-min cadence
MIN_SPAN_MINUTES = 14       # two consecutive 15-min scans (minus jitter)
DECAY_FRAC = 0.30           # down 30% from the run's peak = decaying
MIN_HALF_LIFE_HOURS = 0.5   # halving within 30 min = decaying


@dataclass
class EdgeState:
    state: str
    persistence: int                  # scans in the current run
    velocity_per_hour: Optional[float]
    half_life_hours: Optional[float]  # None unless shrinking
    peak: float                       # max |edge| in the current run


def _ts(stamp: str) -> float:
    return datetime.datetime.fromisoformat(stamp).replace(tzinfo=datetime.timezone.utc).timestamp()


def edge_lifecycle(history: List[Tuple[str, float]], threshold: float) -> EdgeState:
    """history: [(utc iso timestamp, edge)] oldest first, ending with the current scan."""
    history = history[-LOOKBACK_SCANS:]
    stamp, edge = history[-1]
    sign = 1 if edge >= 0 else -1

    velocity = half_life = None
    if len(history) >= 2:
        dt_h = (_ts(stamp) - _ts(history[-2][0])) / 3600
        if dt_h > 0:
            # Strength in the current direction, so a sign flip reads as growth.
            velocity = (edge * sign - history[-2][1] * sign) / dt_h
            if velocity < 0:
                half_life = abs(edge) / 2 / -velocity

    if abs(edge) < threshold:
        prior_run = any(abs(e) >= threshold and e * sign > 0 for _, e in history[:-1])
        return EdgeState(EXHAUSTED if prior_run else NO_EDGE, 0, velocity, half_life, abs(edge))

    run = []
    for s, e in reversed(history):
        if abs(e) < threshold or e * sign <= 0:
            break
        run.append((s, e))
    run.reverse()
    peak = max(abs(e) for _, e in run)
    if len(run) == 1:
        return EdgeState(EMERGING, 1, velocity, half_life, peak)
    if abs(edge) < peak * (1 - DECAY_FRAC) or (half_life is not None and half_life < MIN_HALF_LIFE_HOURS):
        return EdgeState(DECAYING, len(run), velocity, half_life, peak)
    span_minutes = (_ts(run[-1][0]) - _ts(run[0][0])) / 60
    state = ACTIONABLE if span_minutes >= MIN_SPAN_MINUTES else CONFIRMED
    return EdgeState(state, len(run), velocity, half_life, peak)


if __name__ == "__main__":
    t = lambda m: (datetime.datetime(2026, 10, 7, 6) + datetime.timedelta(minutes=m)).isoformat()  # noqa: E731
    thr = 0.08
    assert edge_lifecycle([(t(0), 0.02)], thr).state == NO_EDGE
    assert edge_lifecycle([(t(0), 0.10)], thr).state == EMERGING
    assert edge_lifecycle([(t(0), 0.10), (t(5), 0.11)], thr).state == CONFIRMED
    s = edge_lifecycle([(t(0), 0.10), (t(15), 0.11)], thr)
    assert s.state == ACTIONABLE and s.persistence == 2 and math.isclose(s.velocity_per_hour, 0.04)
    assert edge_lifecycle([(t(0), 0.20), (t(15), 0.19), (t(30), 0.12)], thr).state == DECAYING
    assert edge_lifecycle([(t(0), 0.10), (t(15), 0.12), (t(30), 0.03)], thr).state == EXHAUSTED
    # A sign flip starts a new run; the opposite-signed history doesn't confirm it.
    assert edge_lifecycle([(t(0), -0.12), (t(15), 0.10)], thr).state == EMERGING
    assert edge_lifecycle([(t(0), -0.12), (t(15), -0.11), (t(30), -0.10)], thr).state == ACTIONABLE
    print("edge_state checks passed")
