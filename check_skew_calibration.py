"""
check_skew_calibration.py — Sanity-check whether a city's configured
skew-alpha-by-month table is supported by actual settled-day forecast
residuals, or whether it's over-tuning the model into an unrealistically
thin tail.

BACKGROUND: config/cities.py's CityConfig.skew_alpha_by_month (and
default_skew_alpha) shape how confident the model is that the actual high
won't land warmer/colder than its predicted mean — a strongly negative
alpha, for instance, makes the model near-certain the high won't land far
above the mean. That's either a genuine forecast edge, or the skew
parameter making the model falsely certain about tail outcomes it hasn't
actually seen enough real data to justify. This script checks which, using
db/ledger.py's calibration_logs table (real settled outcomes vs what the
model predicted that day) for one city.

This is read-only — no API calls, no orders, just a query against the
local hermes.db and some scipy stats.

Usage:
    python check_skew_calibration.py --icao WSSS
    python check_skew_calibration.py --icao WSSS --month 7   # July-only rows
    python check_skew_calibration.py --icao WMKK
"""

import argparse
import sqlite3

import numpy as np
from scipy.stats import skewnorm
from scipy.stats import skew as scipy_skew


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--icao", default="WSSS")
    p.add_argument("--month", type=int, default=None,
                    help="Filter to calibration_logs rows whose market_date falls in this "
                         "month (1-12). Default: all months (more data, less season-specific).")
    p.add_argument("--db", default="hermes.db")
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT market_date, model_mu, actual_settled, residual "
            "FROM calibration_logs WHERE icao_code = ? ORDER BY id",
            (args.icao.upper(),),
        ).fetchall()
    finally:
        conn.close()

    if args.month:
        rows = [r for r in rows if r["market_date"] and len(r["market_date"]) >= 7
                and int(r["market_date"][5:7]) == args.month]

    n = len(rows)
    scope = f"month={args.month}" if args.month else "all months"
    print(f"{n} calibration rows for {args.icao} ({scope})")
    if n == 0:
        print("No data — nothing to check yet.")
        return
    if n < 8:
        print("Fewer than 8 rows — treat any skew estimate below as very tentative; "
              "this is a rough sanity check, not a robust refit.")

    for r in rows:
        print(f"  {r['market_date']}: model_mu={r['model_mu']:.2f} "
              f"actual={r['actual_settled']:.2f} residual={r['residual']:+.2f}")

    residuals = np.array([r["residual"] for r in rows], dtype=float)
    mean = residuals.mean()
    std  = residuals.std(ddof=1) if n > 1 else float("nan")
    emp_skew = scipy_skew(residuals) if n > 2 else float("nan")

    print(f"\nresidual (actual - model_mu): mean={mean:+.3f}  std={std:.3f}  "
          f"empirical_skewness={emp_skew:+.3f}")
    print("(negative empirical_skewness = real busts skew toward COLDER than predicted, "
          "supporting a negative alpha; near-zero or positive = the negative skew assumption "
          "may be overstated for this data)")

    warm_busts = int((residuals > 2.0).sum())
    cold_busts = int((residuals < -2.0).sum())
    print(f"\nwarm busts (actual > model_mu + 2°C): {warm_busts}/{n}   "
          f"cold busts (actual < model_mu - 2°C): {cold_busts}/{n}")

    if n >= 8:
        try:
            fit_alpha, fit_loc, fit_scale = skewnorm.fit(residuals)
            print(f"\nEmpirical skewnorm fit to residuals: alpha={fit_alpha:.2f} "
                  f"loc={fit_loc:.2f} scale={fit_scale:.2f}")
            print("Compare against config/cities.py's CITIES[icao].skew_alpha_by_month — "
                  "if the fit alpha is notably less negative (or positive) than the "
                  "configured value for this month, the table may be over-tuned for "
                  "this station/period.")
        except Exception as e:
            print(f"skewnorm.fit failed (often just needs more data points): {e}")

    if args.month and std == std and std > 0:
        from config.cities import get_city
        try:
            city_config = get_city(args.icao)
            configured_alpha = city_config.skew_alpha_by_month.get(args.month, city_config.default_skew_alpha)
        except KeyError:
            print(f"\nUnknown city '{args.icao}' in config.cities.CITIES — skipping alpha comparison.")
            return
        p_warm_bust_model = 1 - skewnorm.cdf(2.0, configured_alpha, loc=0, scale=std)
        print(f"\nConfigured alpha={configured_alpha} for {args.icao} month={args.month} "
              f"(at the observed residual std={std:.3f}) "
              f"implies P(residual > +2°C) = {p_warm_bust_model*100:.3f}%")
        print(f"Actual observed rate this period: {warm_busts/n*100:.1f}% ({warm_busts}/{n})")
        if n >= 8 and warm_busts / n > p_warm_bust_model * 3:
            print(
                "WARNING: actual warm-bust rate is several times higher than what alpha implies — "
                "this is evidence the configured skew is over-tuned (too confident the "
                "high won't land warmer than expected)."
            )


if __name__ == "__main__":
    main()
