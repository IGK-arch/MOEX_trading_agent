"""
scripts/train_hmm.py — Fit the HMM regime detector on IMOEX daily candles.

Saves to data/models/hmm.pkl.

Usage:
    python3 scripts/train_hmm.py [--days 90]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.hmm_regime import get_hmm_detector
from app.data.iss_client import get_iss_client
from app.utils.logging import setup_logging

async def main() -> None:
    """Main."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--days", type=int, default=90, help="Days of IMOEX daily candles to train on"
    )
    parser.add_argument("--ticker", default="IMOEX", help="Index ticker (IMOEX or RTSI)")
    args = parser.parse_args()

    setup_logging()

    iss = get_iss_client()
    await iss.startup()

    till = datetime.now(tz=UTC)
    from_dt = till - timedelta(days=args.days + 30)

    print(f" Fetching {args.ticker} daily candles ({args.days} days)...")
    df = await iss.get_candles(args.ticker, interval=24, from_dt=from_dt, till_dt=till)

    import pandas as pd

    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f" No data for {args.ticker}, trying SBER as proxy...")
        df = await iss.get_candles("SBER", interval=24, from_dt=from_dt, till_dt=till)

    print(f"  Got {len(df)} daily candles")
    print(f"  Price range: {df['low'].min():.2f} → {df['high'].max():.2f}")

    if len(df) < 30:
        print(" Too few candles to train HMM")
        await iss.shutdown()
        sys.exit(1)

    print("\n Training HMM (3 states)...")
    hmm = get_hmm_detector()
    ok = await hmm.fit(df)

    if not ok:
        print(" HMM fit failed")
        await iss.shutdown()
        sys.exit(1)

    print(" HMM trained")
    print("\n Current regime prediction:")
    regime = hmm.predict_state(df)
    proba = hmm.predict_proba_last(df)
    print(f"  Regime: {regime}")
    for label, p in proba.items():
        print(f"    {label}: {p:.1%}")

    print(f"\n Saved to: {hmm.model_path}")
    await iss.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
