from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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


def infer_day_from_filename(path: Path) -> int:
    return int(path.stem.split("day_")[-1])


def read_csv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def load_prices(data_dir: Path) -> pd.DataFrame:
    frames = []

    for fname in PRICE_FILES:
        path = data_dir / fname
        df = read_csv_auto(path).copy()
        df = df.rename(columns={"product": "symbol"})

        if "day" not in df.columns:
            df["day"] = infer_day_from_filename(path)

        frames.append(df)

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["day", "symbol", "timestamp"]).reset_index(drop=True)
    return prices


def load_trades(data_dir: Path) -> pd.DataFrame:
    frames = []

    for fname in TRADE_FILES:
        path = data_dir / fname
        df = read_csv_auto(path).copy()
        df["day"] = infer_day_from_filename(path)
        frames.append(df)

    trades = pd.concat(frames, ignore_index=True)
    trades = trades.sort_values(["day", "symbol", "timestamp"]).reset_index(drop=True)
    return trades


def clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prices.copy()
    audit_rows = []

    numeric_cols = [
        "timestamp",
        "day",
        "bid_price_1", "bid_volume_1",
        "bid_price_2", "bid_volume_2",
        "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1",
        "ask_price_2", "ask_volume_2",
        "ask_price_3", "ask_volume_3",
        "mid_price",
        "profit_and_loss",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_mask = df["symbol"].isna() | df["timestamp"].isna() | df["day"].isna()
    if bad_mask.any():
        audit_rows.append({
            "stage": "prices",
            "reason": "missing_symbol_or_timestamp_or_day",
            "rows_removed": int(bad_mask.sum()),
        })
        df = df.loc[~bad_mask].copy()

    bad_mask = (
        df["bid_price_1"].isna()
        | df["ask_price_1"].isna()
        | (df["bid_price_1"] <= 0)
        | (df["ask_price_1"] <= 0)
    )
    if bad_mask.any():
        audit_rows.append({
            "stage": "prices",
            "reason": "non_positive_or_missing_best_quotes",
            "rows_removed": int(bad_mask.sum()),
        })
        df = df.loc[~bad_mask].copy()

    bad_mask = df["bid_price_1"] > df["ask_price_1"]
    if bad_mask.any():
        audit_rows.append({
            "stage": "prices",
            "reason": "crossed_book_bid_gt_ask",
            "rows_removed": int(bad_mask.sum()),
        })
        df = df.loc[~bad_mask].copy()

    vol_cols = [
        "bid_volume_1", "bid_volume_2", "bid_volume_3",
        "ask_volume_1", "ask_volume_2", "ask_volume_3",
    ]
    for col in vol_cols:
        if col in df.columns:
            bad_mask = df[col].notna() & (df[col] < 0)
            if bad_mask.any():
                audit_rows.append({
                    "stage": "prices",
                    "reason": f"negative_volume_{col}",
                    "rows_removed": int(bad_mask.sum()),
                })
                df = df.loc[~bad_mask].copy()

    df["mid_price_clean"] = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]

    if "mid_price" in df.columns:
        df["mid_diff_vs_raw"] = df["mid_price_clean"] - df["mid_price"]

    df["bid_depth_3"] = df[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].fillna(0).sum(axis=1)
    df["ask_depth_3"] = df[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].fillna(0).sum(axis=1)

    denom = df["bid_volume_1"] + df["ask_volume_1"]
    df["imbalance_l1"] = np.where(
        denom > 0,
        (df["bid_volume_1"] - df["ask_volume_1"]) / denom,
        np.nan,
    )

    micro_denom = df["bid_volume_1"] + df["ask_volume_1"]
    df["microprice"] = np.where(
        micro_denom > 0,
        (
            df["ask_price_1"] * df["bid_volume_1"]
            + df["bid_price_1"] * df["ask_volume_1"]
        ) / micro_denom,
        np.nan,
    )

    df = df.sort_values(["day", "symbol", "timestamp"]).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    return df, audit


def pair_trades_to_book(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    out = []

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

        t = t.sort_values("timestamp").copy()
        p = p.sort_values("timestamp").copy()

        book_cols = [
            "timestamp",
            "bid_price_1", "bid_volume_1",
            "bid_price_2", "bid_volume_2",
            "bid_price_3", "bid_volume_3",
            "ask_price_1", "ask_volume_1",
            "ask_price_2", "ask_volume_2",
            "ask_price_3", "ask_volume_3",
            "mid_price_clean",
            "spread",
            "mid_diff_vs_raw",
            "bid_depth_3",
            "ask_depth_3",
            "imbalance_l1",
            "microprice",
        ]
        book_cols = [col for col in book_cols if col in p.columns]
        p_small = p[book_cols].copy()

        paired = pd.merge_asof(
            t,
            p_small,
            on="timestamp",
            direction="backward",
        )

        paired["day"] = day
        paired["symbol"] = symbol

        paired["trade_vs_mid"] = paired["price"] - paired["mid_price_clean"]
        paired["trade_vs_bid"] = paired["price"] - paired["bid_price_1"]
        paired["trade_vs_ask"] = paired["price"] - paired["ask_price_1"]

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

        out.append(paired)

    if not out:
        return pd.DataFrame()

    paired_all = pd.concat(out, ignore_index=True)
    paired_all = paired_all.sort_values(["day", "symbol", "timestamp"]).reset_index(drop=True)
    return paired_all


def clean_trades(trades: pd.DataFrame, cleaned_prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = trades.copy()
    audit_rows = []

    numeric_cols = ["timestamp", "price", "quantity", "day"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_mask = (
        df["symbol"].isna()
        | df["timestamp"].isna()
        | df["day"].isna()
        | df["price"].isna()
        | df["quantity"].isna()
    )
    if bad_mask.any():
        audit_rows.append({
            "stage": "trades",
            "reason": "missing_symbol_or_timestamp_or_day_or_trade_fields",
            "rows_removed": int(bad_mask.sum()),
        })
        df = df.loc[~bad_mask].copy()

    bad_mask = (df["price"] <= 0) | (df["quantity"] <= 0)
    if bad_mask.any():
        audit_rows.append({
            "stage": "trades",
            "reason": "non_positive_trade_price_or_quantity",
            "rows_removed": int(bad_mask.sum()),
        })
        df = df.loc[~bad_mask].copy()

    df["trade_notional"] = df["price"] * df["quantity"]
    df = df.sort_values(["day", "symbol", "timestamp"]).reset_index(drop=True)

    paired = pair_trades_to_book(cleaned_prices, df)

    bad_mask = (
        paired["mid_price_clean"].isna()
        | (paired["price"] < 0.5 * paired["mid_price_clean"])
        | (paired["price"] > 1.5 * paired["mid_price_clean"])
    )
    if bad_mask.any():
        audit_rows.append({
            "stage": "trades",
            "reason": "trade_price_far_from_visible_mid_context",
            "rows_removed": int(bad_mask.sum()),
        })
        bad_ids = paired.loc[bad_mask, ["day", "symbol", "timestamp", "price", "quantity"]].copy()
        df = df.merge(
            bad_ids.assign(_drop_me=1),
            on=["day", "symbol", "timestamp", "price", "quantity"],
            how="left",
        )
        df = df[df["_drop_me"].isna()].drop(columns="_drop_me")

    df = df.sort_values(["day", "symbol", "timestamp"]).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    return df, audit


def add_time_series_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    group_cols = ["day", "symbol"]

    df["mid_return"] = df.groupby(group_cols)["mid_price_clean"].diff()
    df["log_mid_return"] = df.groupby(group_cols)["mid_price_clean"].transform(
        lambda s: np.log(s).diff()
    )

    df["rolling_mid_mean_50"] = df.groupby(group_cols)["mid_price_clean"].transform(
        lambda s: s.rolling(50, min_periods=5).mean()
    )
    df["rolling_mid_std_50"] = df.groupby(group_cols)["mid_price_clean"].transform(
        lambda s: s.rolling(50, min_periods=5).std()
    )

    df["zscore_mid_50"] = (
        (df["mid_price_clean"] - df["rolling_mid_mean_50"]) / df["rolling_mid_std_50"]
    )

    for horizon in [1, 5, 10, 20]:
        df[f"fwd_mid_return_{horizon}"] = (
            df.groupby(group_cols)["mid_price_clean"].shift(-horizon) - df["mid_price_clean"]
        )

    return df


def build_summary(prices: pd.DataFrame, trades: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []

    keys = (
        prices[["day", "symbol"]]
        .drop_duplicates()
        .sort_values(["day", "symbol"])
        .itertuples(index=False)
    )

    for day, symbol in keys:
        p = prices[(prices["day"] == day) & (prices["symbol"] == symbol)]
        t = trades[(trades["day"] == day) & (trades["symbol"] == symbol)]
        pt = paired[(paired["day"] == day) & (paired["symbol"] == symbol)]

        rows.append({
            "day": day,
            "symbol": symbol,
            "n_price_rows": len(p),
            "n_trade_rows": len(t),
            "mid_min": p["mid_price_clean"].min(),
            "mid_max": p["mid_price_clean"].max(),
            "mid_std": p["mid_price_clean"].std(),
            "spread_mean": p["spread"].mean(),
            "spread_median": p["spread"].median(),
            "imbalance_mean": p["imbalance_l1"].mean(),
            "trade_price_min": t["price"].min() if len(t) else np.nan,
            "trade_price_max": t["price"].max() if len(t) else np.nan,
            "trade_qty_mean": t["quantity"].mean() if len(t) else np.nan,
            "avg_abs_trade_vs_mid": pt["trade_vs_mid"].abs().mean() if len(pt) else np.nan,
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder holding raw CSVs")
    parser.add_argument("--out-dir", type=Path, required=True, help="Folder to save cleaned outputs")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    prices_raw = load_prices(args.data_dir)
    trades_raw = load_trades(args.data_dir)

    prices_clean, prices_audit = clean_prices(prices_raw)
    trades_clean, trades_audit = clean_trades(trades_raw, prices_clean)

    prices_enriched = add_time_series_features(prices_clean)
    paired = pair_trades_to_book(prices_enriched, trades_clean)
    summary = build_summary(prices_enriched, trades_clean, paired)

    prices_raw.to_csv(args.out_dir / "prices_raw_all.csv", index=False)
    trades_raw.to_csv(args.out_dir / "trades_raw_all.csv", index=False)

    prices_enriched.to_csv(args.out_dir / "prices_clean_enriched.csv", index=False)
    trades_clean.to_csv(args.out_dir / "trades_clean.csv", index=False)
    paired.to_csv(args.out_dir / "trades_paired_to_clean_book.csv", index=False)

    prices_audit.to_csv(args.out_dir / "prices_cleaning_audit.csv", index=False)
    trades_audit.to_csv(args.out_dir / "trades_cleaning_audit.csv", index=False)
    summary.to_csv(args.out_dir / "summary_by_day_symbol.csv", index=False)

    print("Saved cleaned datasets to:", args.out_dir)
    print("Files:")
    print(" - prices_raw_all.csv")
    print(" - trades_raw_all.csv")
    print(" - prices_clean_enriched.csv")
    print(" - trades_clean.csv")
    print(" - trades_paired_to_clean_book.csv")
    print(" - prices_cleaning_audit.csv")
    print(" - trades_cleaning_audit.csv")
    print(" - summary_by_day_symbol.csv")


if __name__ == "__main__":
    main()