"""
plot_clean_overview.py

Creates clean overview plots from the cleaned datasets.

Run:
    python scripts/plot_clean_overview.py --input-dir ./output_clean --fig-dir ./figures_clean
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_mid_and_trades(prices: pd.DataFrame, trades: pd.DataFrame, symbol: str, day: int, fig_dir: Path) -> None:
    """
    Plot clean mid-price line plus trade scatter.
    No trade line, because that visually creates fake vertical spikes.
    """
    p = prices[(prices["symbol"] == symbol) & (prices["day"] == day)].copy()
    t = trades[(trades["symbol"] == symbol) & (trades["day"] == day)].copy()

    if p.empty:
        return

    plt.figure(figsize=(14, 6))
    plt.plot(p["timestamp"], p["mid_price_clean"], label="clean mid price")

    if not t.empty:
        # Keep marker sizes under control.
        marker_size = np.clip(t["quantity"].fillna(1).to_numpy() * 4, 8, 120)
        plt.scatter(
            t["timestamp"],
            t["price"],
            s=marker_size,
            alpha=0.65,
            label="trades",
        )

    plt.title(f"{symbol} | day {day} | clean mid price with trades")
    plt.xlabel("timestamp")
    plt.ylabel("price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_day_{day}_mid_with_trades_clean.png", dpi=180)
    plt.close()


def plot_spread(prices: pd.DataFrame, symbol: str, day: int, fig_dir: Path) -> None:
    p = prices[(prices["symbol"] == symbol) & (prices["day"] == day)].copy()
    if p.empty:
        return

    plt.figure(figsize=(14, 4))
    plt.plot(p["timestamp"], p["spread"])
    plt.title(f"{symbol} | day {day} | spread")
    plt.xlabel("timestamp")
    plt.ylabel("spread")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_day_{day}_spread.png", dpi=180)
    plt.close()


def plot_imbalance(prices: pd.DataFrame, symbol: str, day: int, fig_dir: Path) -> None:
    p = prices[(prices["symbol"] == symbol) & (prices["day"] == day)].copy()
    if p.empty:
        return

    plt.figure(figsize=(14, 4))
    plt.plot(p["timestamp"], p["imbalance_l1"])
    plt.axhline(0.0, linewidth=1)
    plt.title(f"{symbol} | day {day} | L1 imbalance")
    plt.xlabel("timestamp")
    plt.ylabel("imbalance_l1")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_day_{day}_imbalance.png", dpi=180)
    plt.close()


def plot_depth(prices: pd.DataFrame, symbol: str, day: int, fig_dir: Path) -> None:
    p = prices[(prices["symbol"] == symbol) & (prices["day"] == day)].copy()
    if p.empty:
        return

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


def plot_return_hist(prices: pd.DataFrame, symbol: str, day: int, fig_dir: Path) -> None:
    p = prices[(prices["symbol"] == symbol) & (prices["day"] == day)].copy()
    if p.empty:
        return

    x = p["mid_return"].dropna()
    if x.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=50)
    plt.title(f"{symbol} | day {day} | mid return distribution")
    plt.xlabel("mid_return")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_day_{day}_mid_return_hist.png", dpi=180)
    plt.close()


def plot_zscore(prices: pd.DataFrame, symbol: str, day: int, fig_dir: Path) -> None:
    p = prices[(prices["symbol"] == symbol) & (prices["day"] == day)].copy()
    if p.empty:
        return

    plt.figure(figsize=(14, 4))
    plt.plot(p["timestamp"], p["zscore_mid_50"])
    plt.axhline(0.0, linewidth=1)
    plt.axhline(2.0, linewidth=1)
    plt.axhline(-2.0, linewidth=1)
    plt.title(f"{symbol} | day {day} | 50-step z-score of mid price")
    plt.xlabel("timestamp")
    plt.ylabel("zscore_mid_50")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_day_{day}_zscore_mid_50.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Cleaned output folder")
    parser.add_argument("--fig-dir", type=Path, required=True, help="Where to save plots")
    args = parser.parse_args()

    ensure_dir(args.fig_dir)

    prices = pd.read_csv(args.input_dir / "prices_clean_enriched.csv")
    trades = pd.read_csv(args.input_dir / "trades_clean.csv")

    symbols = sorted(prices["symbol"].dropna().unique())
    days = sorted(prices["day"].dropna().unique())

    for symbol in symbols:
        for day in days:
            plot_mid_and_trades(prices, trades, symbol, day, args.fig_dir)
            plot_spread(prices, symbol, day, args.fig_dir)
            plot_imbalance(prices, symbol, day, args.fig_dir)
            plot_depth(prices, symbol, day, args.fig_dir)
            plot_return_hist(prices, symbol, day, args.fig_dir)
            plot_zscore(prices, symbol, day, args.fig_dir)

    print("Saved clean figures to:", args.fig_dir)


if __name__ == "__main__":
    main()