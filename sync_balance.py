"""
sync_balance.py — Sync the CLOB's internal balance/allowance cache for the
deposit wallet against its actual on-chain state.

BACKGROUND: order placement can fail with "not enough balance / allowance
... balance: 0" even though the wallet genuinely holds funds on-chain. The
CLOB maintains its own internal balance cache that does not automatically
pick up on-chain changes — it must be explicitly told to re-check via
GET /balance-allowance/update?asset_type=COLLATERAL.

The account-wide wallet (see core/execution.py:build_client — one shared
wallet across every configured city) is what this syncs; it is not
per-city/per-icao.

This is read-only from a funds perspective: no gas, no signing beyond the
existing L2 API auth this bot already uses every cycle. Safe to (re)run any
time the CLOB's balance appears stale relative to what's actually on-chain
(e.g. right after funding, after adding more funds later, or after a
"balance: 0" order rejection in the logs). scheduler.py's main() also runs
this same sync automatically on every startup; this script is a manual
fallback for use between restarts.

Usage:
    python sync_balance.py
"""

import sys

from dotenv import load_dotenv


def main():
    load_dotenv()

    from core.execution import build_client
    from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType

    print("Authenticating CLOB client...")
    try:
        client = build_client()
    except Exception as e:
        print(f"ERROR: could not build/authenticate CLOB client: {e}")
        sys.exit(1)

    print("Requesting balance/allowance sync (asset_type=COLLATERAL)...")
    resp = client.update_balance_allowance(
        BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
    )
    print(f"Response: {resp}")

    print()
    print("Re-checking balance to confirm...")
    check = client.get_balance_allowance(
        BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
    )
    print(f"Balance/allowance now: {check}")


if __name__ == "__main__":
    main()
