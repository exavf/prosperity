# IMC Prosperity 4 — Data Pipeline & Visualisation Toolkit

## Overview

This repository contains a structured data pipeline and visualisation toolkit for analysing the IMC Prosperity 4 trading environment.

The goal of this project is not just to visualise data, but to:

* reconstruct market microstructure from partial data
* align trade executions with order book states
* extract statistically meaningful signals
* build a foundation for systematic trading strategies

This is designed as a **research-first environment**, mirroring workflows used in quantitative trading.

---

## Repository Structure

```
prosperity/
├── data/                  # Raw IMC datasets (prices + trades)
├── output/                # Processed datasets (paired + enriched)
├── figures_overview/      # Generated visualisations
├── scripts/               # Core data + visualisation pipeline
│   ├── load_and_pair.py
│   ├── plot_overview.py
│   └── plot_pairings.py
├── .gitignore
└── README.md
```

---

## Data Description

### 1. Prices Data (Order Book Snapshots)

Each `prices_round_1_day_*` file represents **limit order book snapshots**.

Key fields:

* `bid_price_1 / ask_price_1`: best bid/ask
* `bid_volume_* / ask_volume_*`: depth (top 3 levels)
* `mid_price`: derived midpoint
* `timestamp`, `day`, `product`

This data represents:

> **market state at discrete time intervals**

---

### 2. Trades Data (Executed Trades)

Each `trades_round_1_day_*` file represents **executed trades**.

Key fields:

* `price`, `quantity`
* `timestamp`
* `symbol`

This data represents:

> **actual market activity (tape data)**

---

### Core Insight

The dataset is intentionally incomplete:

* no direct trade direction
* limited order book depth
* discrete timestamps

Therefore:

> edge must be extracted via reconstruction and inference

---

## Data Pipeline

### Step 1 — Load & Normalise

Script: `load_and_pair.py`

* Loads all price and trade CSVs
* Normalises schema (`product → symbol`)
* Combines all days into unified datasets

---

### Step 2 — Feature Engineering

Key engineered features:

* **Spread**

  ```
  spread = ask_price_1 - bid_price_1
  ```

* **Microprice**

  ```
  microprice = (ask_price_1 * bid_volume_1 + bid_price_1 * ask_volume_1)
               / (bid_volume_1 + ask_volume_1)
  ```

* **Order Book Imbalance**

  ```
  imbalance = (bid_volume_1 - ask_volume_1) / (bid_volume_1 + ask_volume_1)
  ```

* **Depth (Top 3 Levels)**

  ```
  bid_depth_3 = sum(bid_volume_1..3)
  ask_depth_3 = sum(ask_volume_1..3)
  ```

* **Returns**

  * log returns
  * mid price changes
  * rolling statistics

---

### Step 3 — Trade ↔ Order Book Pairing

Core logic:

```python
pd.merge_asof(...)
```

Each trade is matched to the **most recent prior order book snapshot**.

Why?

* avoids lookahead bias
* reflects real trading conditions
* aligns execution with available liquidity

---

### Step 4 — Trade Classification (Heuristic)

Since direction is not given, we infer it:

* `price >= ask` → aggressive buy
* `price <= bid` → aggressive sell
* otherwise → proximity to mid

This produces:

```
trade_side_guess ∈ {buy_aggressor, sell_aggressor, likely_buy, likely_sell}
```

---

### Output Files

```
output/
├── prices_all.csv
├── trades_all.csv
├── trades_paired_to_book.csv
├── prices_timeseries_enriched.csv
```

---

## Visualisation

### 1. Overview Plots (`plot_overview.py`)

For each symbol and day:

* mid price over time
* trade prints (size-scaled)
* spread evolution
* order book imbalance
* depth (bid vs ask)
* trade size distribution

Purpose:

> understand price behaviour and liquidity structure

---

### 2. Microstructure Analysis (`plot_pairings.py`)

Using paired data:

* trade price vs mid
* trade vs bid/ask
* aggressor classification distribution
* imbalance vs trade deviation

Purpose:

> identify execution patterns and short-term signals

---

## Key Insights Enabled

This pipeline enables:

### 1. Mean Reversion Detection

* deviations from rolling mid
* spread dynamics

### 2. Order Book Signal Extraction

* imbalance → short-term direction
* depth asymmetry

### 3. Trade Impact Analysis

* how trades move price
* aggressor behaviour

### 4. Feature Prototyping

* foundation for alpha signals
* inputs for backtesting models

---

## Design Principles

This project follows core quant principles:

* **Edge-first thinking**
* **No lookahead bias**
* **Data-driven feature engineering**
* **Simple → complex model progression**
* **Reproducibility and modularity**

---

## How to Run

### 1. Process Data

```bash
python scripts/load_and_pair.py \
  --data-dir ./data \
  --out-dir ./output
```

---

### 2. Generate Visualisations

```bash
python scripts/plot_overview.py \
  --input-dir ./output \
  --fig-dir ./figures_overview

python scripts/plot_pairings.py \
  --input-dir ./output \
  --fig-dir ./figures_pairings
```

---

## Next Steps

This repo is designed as a foundation for:

* signal testing
* backtesting framework integration
* strategy development (mean reversion, market making, etc.)
* dashboarding (e.g. Streamlit)

---

## Author

Built as part of IMC Prosperity 4 preparation.

Focus:

> extracting edge from incomplete, adversarial market data
