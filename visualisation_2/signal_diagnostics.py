"""
signal_diagnostics.py

Runs simple statistical diagnostics to help identify whether there is edge in:
- imbalance
- z-score mean reversion
- spread regimes
- trade-side heuristics

Run:
    python scripts/signal_diagnostics.py --input-dir ./output_clean --out-dir ./diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    """
    Small helper to avoid noisy failures.
    """
    x = pd.concat([a, b], axis=1).dropna()
    if len(x) < 5:
        return np.nan
    return x.iloc[:, 0].corr(x.iloc[:, 1])


def bucket_means(df: pd.DataFrame, feature_col: str, target_col: str, n_bins: int = 10) -> pd.DataFrame:
    """
    Put a signal into quantile buckets and compute average future return.
    Great for quickly seeing monotonicity.
    """
    tmp = df[[feature_col, target_col]].dropna().copy()
    if len(tmp) < n_bins:
        return pd.DataFrame()

    tmp["bucket"] = pd.qcut(tmp[feature_col], q=n_bins, duplicates="drop")
    out = (
        tmp.groupby("bucket", observed=False)[target_col]
        .agg(["mean", "count"])
        .reset_index()
    )
    return out


def run_numeric_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric signal summary by day and symbol.
    """
    rows: list[dict] = []

    keys = (
        prices[["day", "symbol"]]
        .drop_duplicates()
        .sort_values(["day", "symbol"])
        .itertuples(index=False)
    )

    for day, symbol in keys:
        p = prices[(prices["day"] == day) & (prices["symbol"] == symbol)].copy()

        row = {
            "day": day,
            "symbol": symbol,
            "corr_imbalance_fwd1": safe_corr(p["imbalance_l1"], p["fwd_mid_return_1"]),
            "corr_imbalance_fwd5": safe_corr(p["imbalance_l1"], p["fwd_mid_return_5"]),
            "corr_imbalance_fwd10": safe_corr(p["imbalance_l1"], p["fwd_mid_return_10"]),
            "corr_zscore_fwd5": safe_corr(p["zscore_mid_50"], p["fwd_mid_return_5"]),
            "corr_zscore_fwd10": safe_corr(p["zscore_mid_50"], p["fwd_mid_return_10"]),
            "spread_mean": p["spread"].mean(),
            "spread_std": p["spread"].std(),
            "mid_return_std": p["mid_return"].std(),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def plot_imbalance_buckets(prices: pd.DataFrame, out_dir: Path) -> None:
    """
    Bucket imbalance and see average future return.
    If monotonic, that's interesting.
    """
    keys = (
        prices[["day", "symbol"]]
        .drop_duplicates()
        .sort_values(["day", "symbol"])
        .itertuples(index=False)
    )

    for day, symbol in keys:
        p = prices[(prices["day"] == day) & (prices["symbol"] == symbol)].copy()
        bucket_df = bucket_means(p, "imbalance_l1", "fwd_mid_return_5", n_bins=10)
        if bucket_df.empty:
            continue

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(bucket_df)), bucket_df["mean"])
        plt.axhline(0.0, linewidth=1)
        plt.title(f"{symbol} | day {day} | avg fwd 5-step return by imbalance bucket")
        plt.xlabel("imbalance bucket (low -> high)")
        plt.ylabel("mean fwd_mid_return_5")
        plt.tight_layout()
        plt.savefig(out_dir / f"{symbol}_day_{day}_imbalance_bucket_vs_fwd5.png", dpi=180)
        plt.close()


def plot_zscore_buckets(prices: pd.DataFrame, out_dir: Path) -> None:
    """
    Bucket z-score and see if large positive deviations mean-revert down,
    and large negative deviations mean-revert up.
    """
    keys = (
        prices[["day", "symbol"]]
        .drop_duplicates()
        .sort_values(["day", "symbol"])
        .itertuples(index=False)
    )

    for day, symbol in keys:
        p = prices[(prices["day"] == day) & (prices["symbol"] == symbol)].copy()
        bucket_df = bucket_means(p, "zscore_mid_50", "fwd_mid_return_10", n_bins=10)
        if bucket_df.empty:
            continue

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(bucket_df)), bucket_df["mean"])
        plt.axhline(0.0, linewidth=1)
        plt.title(f"{symbol} | day {day} | avg fwd 10-step return by z-score bucket")
        plt.xlabel("z-score bucket (low -> high)")
        plt.ylabel("mean fwd_mid_return_10")
        plt.tight_layout()
        plt.savefig(out_dir / f"{symbol}_day_{day}_zscore_bucket_vs_fwd10.png", dpi=180)
        plt.close()


def plot_spread_regime(prices: pd.DataFrame, out_dir: Path) -> None:
    """
    Compare future return variance in low-spread vs high-spread regimes.
    Useful for deciding when not to trade.
    """
    keys = (
        prices[["day", "symbol"]]
        .drop_duplicates()
        .sort_values(["day", "symbol"])
        .itertuples(index=False)
    )

    rows: list[dict] = []

    for day, symbol in keys:
        p = prices[(prices["day"] == day) & (prices["symbol"] == symbol)].copy()
        p = p.dropna(subset=["spread", "fwd_mid_return_5"])

        if len(p) < 20:
            continue

        spread_cut = p["spread"].median()
        low = p[p["spread"] <= spread_cut]
        high = p[p["spread"] > spread_cut]

        rows.append({
            "day": day,
            "symbol": symbol,
            "median_spread": spread_cut,
            "low_spread_mean_abs_fwd5": low["fwd_mid_return_5"].abs().mean(),
            "high_spread_mean_abs_fwd5": high["fwd_mid_return_5"].abs().mean(),
            "low_spread_count": len(low),
            "high_spread_count": len(high),
        })

    regime_df = pd.DataFrame(rows)
    regime_df.to_csv(out_dir / "spread_regime_summary.csv", index=False)


def trade_side_followthrough(paired: pd.DataFrame, out_dir: Path) -> None:
    """
    Check whether rough aggressor classification lines up with next short-term move.
    This is noisy, but still worth inspecting.
    """
    rows: list[dict] = []

    keys = (
        paired[["day", "symbol"]]
        .drop_duplicates()
        .sort_values(["day", "symbol"])
        .itertuples(index=False)
    )

    for day, symbol in keys:
        df = paired[(paired["day"] == day) & (paired["symbol"] == symbol)].copy()

        if df.empty or "fwd_mid_return_1" not in df.columns:
            continue

        grouped = (
            df.groupby("trade_side_guess", observed=False)["fwd_mid_return_1"]
            .agg(["mean", "count"])
            .reset_index()
        )
        grouped["day"] = day
        grouped["symbol"] = symbol
        rows.append(grouped)

    if rows:
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(out_dir / "trade_side_followthrough.csv", index=False)


def merge_forward_returns_into_paired(prices: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """
    Pull forward return features from prices into the paired trade-book dataset.
    """
    keep_cols = [
        "day", "symbol", "timestamp",
        "fwd_mid_return_1", "fwd_mid_return_5", "fwd_mid_return_10"
    ]
    ref = prices[keep_cols].copy()
    merged = paired.merge(ref, on=["day", "symbol", "timestamp"], how="left")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Cleaned output folder")
    parser.add_argument("--out-dir", type=Path, required=True, help="Diagnostics output folder")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    prices = pd.read_csv(args.input_dir / "prices_clean_enriched.csv")
    paired = pd.read_csv(args.input_dir / "trades_paired_to_clean_book.csv")

    paired = merge_forward_returns_into_paired(prices, paired)

    summary = run_numeric_summary(prices)
    summary.to_csv(args.out_dir / "signal_numeric_summary.csv", index=False)

    plot_imbalance_buckets(prices, args.out_dir)
    plot_zscore_buckets(prices, args.out_dir)
    plot_spread_regime(prices, args.out_dir)
    trade_side_followthrough(paired, args.out_dir)

    print("Saved diagnostics to:", args.out_dir)
    print("Files:")
    print(" - signal_numeric_summary.csv")
    print(" - spread_regime_summary.csv")
    print(" - trade_side_followthrough.csv")
    print(" - imbalance bucket plots")
    print(" - z-score bucket plots")


if __name__ == "__main__":
    main()