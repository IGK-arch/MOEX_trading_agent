"""
scripts/refit_pairs.py — Daily cointegration refit for all pairs.

Runs PairTrader.refit_all_pairs() with real ISS daily data.

Usage:
    python3 scripts/refit_pairs.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.pair_trader import get_pair_trader
from app.utils.logging import setup_logging

async def main() -> None:
    """Main."""
    setup_logging()
    print(" Fitting pair cointegration (60d daily data from ISS)...")

    pt = get_pair_trader()
    await pt.startup()
    n_qual = await pt.refit_all_pairs()

    print("\n Pairs status after refit:")
    print(f"{'pair':18s}  {'beta':>9s}  {'ADF p-value':>11s}  {'qualified':>9s}")
    print("-" * 60)
    for key, ps in pt.state.items():
        flag = "" if ps.qualified else ""
        print(f"{key:18s}  {ps.beta:9.4f}  {ps.adf_pvalue:11.4f}  {flag}")

    print(f"\n Qualified: {n_qual}/{len(pt.pairs)}")
    print(f" State saved to: {pt.state}")
    await pt.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
