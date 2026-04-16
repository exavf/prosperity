"""
plot_overview.py

Creates high-level plots for each symbol across all days:
- mid price over time
- spread over time
- imbalance over time
- trade prints overlaid on mid price
- histogram of trade sizes

Usage:
    python plot_overview.py --input-dir ./output --fig-dir ./figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_symbol_day(prices: pd.DataFrame, trades: pd.DataFrame, symbol: str, day: int, fig_dir: Path) -> None:
    p = prices[(prices["symbol"] == symbol) & (prices["day"] == day)].copy()
    t = trades[(trades["symbol"] == symbol) & (trades["day"] == day)].copy()

    if p.empty:
        return

    # 1. mid price + trade prints
    plt.figure(figsize=(14, 6))
    plt.plot(p["timestamp"], p["mid_price"], label="mid_price")
    if not t.empty:
        plt.scatter(t["timestamp"], t["price"], s=t["quantity"] * 4, alpha=0.6, label="trades")
    plt.title(f"{symbol} | day {day} | mid price with trades")
    plt.xlabel("timestamp")
    plt.ylabel("price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_day_{day}_mid_with_trades.png", dpi=180)
    plt.close()

    # 2. spread
    plt.figure(figsize=(14, 4))
    plt.plot(p["timestamp"], p["spread"])
    plt.title(f"{symbol} | day {day} | spread")
    plt.xlabel("timestamp")
    plt.ylabel("spread")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_day_{day}_spread.png", dpi=180)
    plt.close()

    # 3. imbalance
    plt.figure(figsize=(14, 4))
    plt.plot(p["timestamp"], p["imbalance_l1"])
    plt.title(f"{symbol} | day {day} | L1 imbalance")
    plt.xlabel("timestamp")
    plt.ylabel("imbalance_l1")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_day_{day}_imbalance.png", dpi=180)
    plt.close()

    # 4. depth
    plt.figure(figsize=(14, 4))
    plt.plot(p["timestamp"], p["bid_depth_3"], label="bid_depth_3")
    plt.plot(p["timestamp"], p["ask_depth_3"], label="ask_depth_3")
    plt.title(f"{symbol} | day {day} | top 3 depth")
    plt.xlabel("timestamp")
    plt.ylabel("volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_day_{day}_depth.png", dpi=180)
    plt.close()

    # 5. trade size histogram
    if not t.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(t["quantity"], bins=30)
        plt.title(f"{symbol} | day {day} | trade quantity distribution")
        plt.xlabel("quantity")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{symbol}_day_{day}_trade_size_hist.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder made by load_and_pair.py")
    parser.add_argument("--fig-dir", type=Path, required=True, help="Folder to save figures")
    args = parser.parse_args()

    ensure_dirs(args.fig_dir)

    prices = pd.read_csv(args.input_dir / "prices_timeseries_enriched.csv")
    trades = pd.read_csv(args.input_dir / "trades_all.csv")

    symbols = sorted(prices["symbol"].dropna().unique())
    days = sorted(prices["day"].dropna().unique())

    for symbol in symbols:
        for day in days:
            plot_symbol_day(prices, trades, symbol, day, args.fig_dir)

    print(f"Saved figures to: {args.fig_dir}")


if __name__ == "__main__":
    main()
