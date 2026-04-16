"""
load_and_pair.py

Loads all Prosperity round 1 prices + trades CSVs, normalises schemas,
engineers a few basic microstructure fields, and pairs trades to the
nearest available book snapshot on the same day / symbol.

Usage:
    python load_and_pair.py --data-dir /path/to/csvs --out-dir ./output
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


PRICE_FILES = [
    "prices_round_1_day_-2.csv",
    "prices_round_1_day_-1.csv",
    "prices_round_1_day_0.csv",
]

TRADE_FILES = [
    "trades_round_1_day_-2.csv",
    "trades_round_1_day_-1.csv",
    "trades_round_1_day_0.csv",
]


def infer_day_from_name(path: Path) -> int:
    # file name format ends with day_-2.csv / day_-1.csv / day_0.csv
    stem = path.stem
    return int(stem.split("day_")[-1])


def load_prices(data_dir: Path) -> pd.DataFrame:
    frames = []

    for fname in PRICE_FILES:
        path = data_dir / fname
        df = pd.read_csv(path, sep=";")

        # keep naming consistent with trade tape
        df = df.rename(columns={"product": "symbol"})

        # useful basic book fields
        df["spread"] = df["ask_price_1"] - df["bid_price_1"]
        df["microprice"] = (
            (df["ask_price_1"] * df["bid_volume_1"]) +
            (df["bid_price_1"] * df["ask_volume_1"])
        ) / (df["bid_volume_1"] + df["ask_volume_1"])

        df["best_bid_notional"] = df["bid_price_1"] * df["bid_volume_1"]
        df["best_ask_notional"] = df["ask_price_1"] * df["ask_volume_1"]

        # simple L1 imbalance
        denom = df["bid_volume_1"] + df["ask_volume_1"]
        df["imbalance_l1"] = np.where(
            denom.notna() & (denom != 0),
            (df["bid_volume_1"] - df["ask_volume_1"]) / denom,
            np.nan,
        )

        # top 3 depth summaries
        bid_vol_cols = ["bid_volume_1", "bid_volume_2", "bid_volume_3"]
        ask_vol_cols = ["ask_volume_1", "ask_volume_2", "ask_volume_3"]
        df["bid_depth_3"] = df[bid_vol_cols].fillna(0).sum(axis=1)
        df["ask_depth_3"] = df[ask_vol_cols].fillna(0).sum(axis=1)

        frames.append(df)

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["day", "symbol", "timestamp"]).reset_index(drop=True)
    return prices


def load_trades(data_dir: Path) -> pd.DataFrame:
    frames = []

    for fname in TRADE_FILES:
        path = data_dir / fname
        df = pd.read_csv(path, sep=";")

        # trades do not explicitly store "day", so infer from file name
        df["day"] = infer_day_from_name(path)

        # small convenience fields
        df["trade_notional"] = df["price"] * df["quantity"]

        frames.append(df)

    trades = pd.concat(frames, ignore_index=True)
    trades = trades.sort_values(["day", "symbol", "timestamp"]).reset_index(drop=True)
    return trades


def pair_trades_to_book(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    paired_parts = []

    # do the merge per day / symbol because merge_asof needs sorted slices
    keys = (
        trades[["day", "symbol"]]
        .drop_duplicates()
        .sort_values(["day", "symbol"])
        .itertuples(index=False)
    )

    for day, symbol in keys:
        t = trades[(trades["day"] == day) & (trades["symbol"] == symbol)].copy()
        p = prices[(prices["day"] == day) & (prices["symbol"] == symbol)].copy()

        if t.empty or p.empty:
            continue

        t = t.sort_values("timestamp")
        p = p.sort_values("timestamp")

        # nearest prior-or-equal book state is usually the cleanest default
        paired = pd.merge_asof(
            t,
            p,
            on="timestamp",
            by=None,
            direction="backward",
            suffixes=("_trade", "_book"),
        )

        paired["trade_vs_mid"] = paired["price"] - paired["mid_price"]
        paired["trade_vs_bid"] = paired["price"] - paired["bid_price_1"]
        paired["trade_vs_ask"] = paired["price"] - paired["ask_price_1"]

        # rough aggressor guess
        paired["trade_side_guess"] = np.select(
            [
                paired["price"] >= paired["ask_price_1"],
                paired["price"] <= paired["bid_price_1"],
                paired["trade_vs_mid"] > 0,
                paired["trade_vs_mid"] < 0,
            ],
            [
                "buy_aggressor",
                "sell_aggressor",
                "likely_buy",
                "likely_sell",
            ],
            default="unclear",
        )

        paired["day"] = day
        paired["symbol"] = symbol
        paired_parts.append(paired)

    if not paired_parts:
        return pd.DataFrame()

    paired_all = pd.concat(paired_parts, ignore_index=True)
    paired_all = paired_all.sort_values(["day", "symbol", "timestamp"]).reset_index(drop=True)
    return paired_all


def make_symbol_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    # slightly cleaner table for plotting / feature work
    ts = prices.copy()
    ts["log_mid_return"] = ts.groupby(["day", "symbol"])["mid_price"].transform(
        lambda s: np.log(s).diff()
    )
    ts["mid_change"] = ts.groupby(["day", "symbol"])["mid_price"].diff()
    ts["rolling_mid_mean_50"] = ts.groupby(["day", "symbol"])["mid_price"].transform(
        lambda s: s.rolling(50, min_periods=1).mean()
    )
    ts["rolling_mid_std_50"] = ts.groupby(["day", "symbol"])["mid_price"].transform(
        lambda s: s.rolling(50, min_periods=1).std()
    )
    return ts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing the six CSVs")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where to save cleaned outputs")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    prices = load_prices(args.data_dir)
    trades = load_trades(args.data_dir)
    paired = pair_trades_to_book(prices, trades)
    ts = make_symbol_timeseries(prices)

    prices.to_csv(args.out_dir / "prices_all.csv", index=False)
    trades.to_csv(args.out_dir / "trades_all.csv", index=False)
    paired.to_csv(args.out_dir / "trades_paired_to_book.csv", index=False)
    ts.to_csv(args.out_dir / "prices_timeseries_enriched.csv", index=False)

    print("Saved:")
    print(f"  {args.out_dir / 'prices_all.csv'}")
    print(f"  {args.out_dir / 'trades_all.csv'}")
    print(f"  {args.out_dir / 'trades_paired_to_book.csv'}")
    print(f"  {args.out_dir / 'prices_timeseries_enriched.csv'}")


if __name__ == "__main__":
    main()
