"""
plot_pairings.py

Focuses on the paired trade-book dataset.
Helps you inspect how executions sit relative to the visible book.

Plots:
- trade price minus mid
- trade price versus best bid / ask
- aggressor guess counts
- imbalance versus next trade price deviation

Usage:
    python plot_pairings.py --input-dir ./output --fig-dir ./figures_pairings
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_symbol_day(paired: pd.DataFrame, symbol: str, day: int, fig_dir: Path) -> None:
    df = paired[(paired["symbol"] == symbol) & (paired["day"] == day)].copy()
    if df.empty:
        return

    # 1. trade minus mid through time
    plt.figure(figsize=(14, 5))
    plt.scatter(df["timestamp"], df["trade_vs_mid"], s=df["quantity"] * 5, alpha=0.7)
    plt.axhline(0.0, linewidth=1)
    plt.title(f"{symbol} | day {day} | trade price minus mid")
    plt.xlabel("timestamp")
    plt.ylabel("trade_vs_mid")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_day_{day}_trade_vs_mid.png", dpi=180)
    plt.close()

    # 2. trade vs top of book
    plt.figure(figsize=(14, 6))
    plt.plot(df["timestamp"], df["bid_price_1"], label="bid_price_1")
    plt.plot(df["timestamp"], df["ask_price_1"], label="ask_price_1")
    plt.scatter(df["timestamp"], df["price"], s=df["quantity"] * 5, alpha=0.7, label="trade_price")
    plt.title(f"{symbol} | day {day} | trade price against best bid / ask")
    plt.xlabel("timestamp")
    plt.ylabel("price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_day_{day}_trade_vs_book.png", dpi=180)
    plt.close()

    # 3. aggressor guess counts
    counts = df["trade_side_guess"].value_counts()
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(f"{symbol} | day {day} | rough aggressor classification")
    plt.xlabel("trade_side_guess")
    plt.ylabel("count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_day_{day}_aggressor_counts.png", dpi=180)
    plt.close()

    # 4. imbalance vs trade minus mid
    plt.figure(figsize=(8, 6))
    plt.scatter(df["imbalance_l1"], df["trade_vs_mid"], s=df["quantity"] * 5, alpha=0.7)
    plt.axhline(0.0, linewidth=1)
    plt.axvline(0.0, linewidth=1)
    plt.title(f"{symbol} | day {day} | imbalance vs trade minus mid")
    plt.xlabel("imbalance_l1")
    plt.ylabel("trade_vs_mid")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_day_{day}_imbalance_vs_trade_deviation.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder made by load_and_pair.py")
    parser.add_argument("--fig-dir", type=Path, required=True, help="Folder to save figures")
    args = parser.parse_args()

    ensure_dirs(args.fig_dir)

    paired = pd.read_csv(args.input_dir / "trades_paired_to_book.csv")
    symbols = sorted(paired["symbol"].dropna().unique())
    days = sorted(paired["day"].dropna().unique())

    for symbol in symbols:
        for day in days:
            plot_symbol_day(paired, symbol, day, args.fig_dir)

    print(f"Saved figures to: {args.fig_dir}")


if __name__ == "__main__":
    main()
