from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Any
import json
import math


class Trader:
    """
    Round 1 trader for:
    - ASH_COATED_OSMIUM
    - INTARIAN_PEPPER_ROOT

    Core idea:
    - imbalance predicts short-horizon continuation
    - z-score predicts medium-horizon mean reversion
    - combine both into an adjusted reservation price
    - take mispriced quotes aggressively
    - otherwise make inventory-aware passive markets

    This is intentionally simple, robust, and easy to tune.
    """

    PRODUCTS = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]

    # Replace with exact official limits if you have them.
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 50,
        "INTARIAN_PEPPER_ROOT": 50,
    }

    # Product-specific tuning.
    # Osmium gets stronger alpha weights because your analysis says its signal is stronger.
    PARAMS = {
        "ASH_COATED_OSMIUM": {
            "history_len": 80,
            "z_window": 40,
            "fair_alpha": 0.18,
            "imbalance_weight": 4.2,
            "zscore_weight": 1.8,
            "take_threshold": 1.0,
            "make_edge": 0.40,
            "base_order_size": 8,
            "max_take_size": 18,
            "inventory_aversion": 0.12,
            "min_std": 0.8,
        },
        "INTARIAN_PEPPER_ROOT": {
            "history_len": 100,
            "z_window": 50,
            "fair_alpha": 0.10,
            "imbalance_weight": 2.8,
            "zscore_weight": 1.2,
            "take_threshold": 0.8,
            "make_edge": 0.30,
            "base_order_size": 6,
            "max_take_size": 14,
            "inventory_aversion": 0.08,
            "min_std": 0.8,
        },
    }

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)

        result: Dict[str, List[Order]] = {}

        for product in self.PRODUCTS:
            if product not in state.order_depths:
                result[product] = []
                continue

            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)

            if product not in memory:
                memory[product] = {
                    "mid_history": [],
                    "fair_value": None,
                    "last_mid": None,
                }

            orders = self._trade_product(
                product=product,
                order_depth=order_depth,
                position=position,
                mem=memory[product],
            )
            result[product] = orders

        trader_data = self._dump_memory(memory)
        conversions = 0
        return result, conversions, trader_data

    def _trade_product(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        mem: Dict[str, Any],
    ) -> List[Order]:
        orders: List[Order] = []
        params = self.PARAMS[product]
        limit = self.POSITION_LIMITS[product]

        best_bid = self._best_bid(order_depth)
        best_ask = self._best_ask(order_depth)

        if best_bid is None or best_ask is None:
            return orders

        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])

        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        # -----------------------------
        # Update state
        # -----------------------------
        mid_hist = mem["mid_history"]
        mid_hist.append(mid)
        if len(mid_hist) > params["history_len"]:
            mid_hist.pop(0)

        old_fair = mem["fair_value"]
        if old_fair is None:
            fair_value = mid
        else:
            fair_value = (1.0 - params["fair_alpha"]) * old_fair + params["fair_alpha"] * mid
        mem["fair_value"] = fair_value

        # -----------------------------
        # Signal construction
        # -----------------------------
        imbalance = self._l1_imbalance(best_bid_vol, best_ask_vol)
        zscore = self._rolling_zscore(mid_hist, params["z_window"], params["min_std"])

        # Momentum from imbalance, mean reversion from z-score.
        # Positive imbalance is bullish.
        # Positive z-score means price is rich relative to recent history, so bearish.
        alpha = (
            params["imbalance_weight"] * imbalance
            - params["zscore_weight"] * zscore
        )

        # Inventory-aware reservation price.
        inv_skew = params["inventory_aversion"] * position
        reservation_price = fair_value + alpha - inv_skew

        # Capacities after accounting for current position.
        max_buy = max(0, limit - position)
        max_sell = max(0, limit + position)

        # -----------------------------
        # 1) Aggressive taking
        # -----------------------------
        # Cross only when quoted edge is real.
        buy_edge = reservation_price - best_ask
        sell_edge = best_bid - reservation_price

        if buy_edge >= params["take_threshold"] and max_buy > 0:
            take_qty = min(max_buy, best_ask_vol, params["max_take_size"])
            if take_qty > 0:
                orders.append(Order(product, best_ask, take_qty))
                max_buy -= take_qty

        if sell_edge >= params["take_threshold"] and max_sell > 0:
            take_qty = min(max_sell, best_bid_vol, params["max_take_size"])
            if take_qty > 0:
                orders.append(Order(product, best_bid, -take_qty))
                max_sell -= take_qty

        # -----------------------------
        # 2) Passive quoting
        # -----------------------------
        # Use reservation price and keep quotes non-crossing.
        # Wider spread lets us quote more safely.
        if spread >= 2:
            # Basic target quotes around reservation price.
            bid_px = math.floor(reservation_price - params["make_edge"])
            ask_px = math.ceil(reservation_price + params["make_edge"])

            # Never cross the current book when passively quoting.
            bid_px = min(bid_px, best_ask - 1)
            ask_px = max(ask_px, best_bid + 1)

            # Avoid obviously bad quote inversion.
            if bid_px < ask_px:
                # Stronger alpha -> slightly larger quoting size.
                alpha_strength = min(2.0, abs(alpha) / 2.0)
                base = params["base_order_size"]
                bid_size = int(round(base * (1.0 + 0.4 * alpha_strength)))
                ask_size = int(round(base * (1.0 + 0.4 * alpha_strength)))

                # Inventory shaping.
                # If long, reduce bids and increase asks.
                # If short, reduce asks and increase bids.
                inv_ratio = position / limit if limit > 0 else 0.0

                if inv_ratio > 0:
                    bid_size = int(round(bid_size * max(0.2, 1.0 - 1.5 * inv_ratio)))
                    ask_size = int(round(ask_size * (1.0 + 0.8 * inv_ratio)))
                elif inv_ratio < 0:
                    bid_size = int(round(bid_size * (1.0 + 0.8 * (-inv_ratio))))
                    ask_size = int(round(ask_size * max(0.2, 1.0 - 1.5 * (-inv_ratio))))

                bid_size = min(bid_size, max_buy)
                ask_size = min(ask_size, max_sell)

                # Optional directional bias:
                # if alpha is strongly positive, quote bid a bit more aggressively;
                # if strongly negative, quote ask a bit more aggressively.
                if alpha > 1.5:
                    bid_px = min(bid_px + 1, best_ask - 1)
                elif alpha < -1.5:
                    ask_px = max(ask_px - 1, best_bid + 1)

                if bid_size > 0:
                    orders.append(Order(product, bid_px, bid_size))
                if ask_size > 0:
                    orders.append(Order(product, ask_px, -ask_size))

        mem["last_mid"] = mid
        return orders

    # =========================================================
    # Helpers
    # =========================================================

    def _best_bid(self, order_depth: OrderDepth):
        if not order_depth.buy_orders:
            return None
        return max(order_depth.buy_orders.keys())

    def _best_ask(self, order_depth: OrderDepth):
        if not order_depth.sell_orders:
            return None
        return min(order_depth.sell_orders.keys())

    def _l1_imbalance(self, bid_vol: int, ask_vol: int) -> float:
        denom = bid_vol + ask_vol
        if denom <= 0:
            return 0.0
        return (bid_vol - ask_vol) / denom

    def _rolling_zscore(self, values: List[float], window: int, min_std: float) -> float:
        if len(values) < max(5, window // 3):
            return 0.0

        sample = values[-window:] if len(values) >= window else values
        mean = sum(sample) / len(sample)

        var = sum((x - mean) ** 2 for x in sample) / len(sample)
        std = math.sqrt(max(var, min_std ** 2))

        if std <= 0:
            return 0.0

        return (sample[-1] - mean) / std

    def _load_memory(self, trader_data: str) -> Dict[str, Any]:
        if trader_data is None or trader_data == "":
            return {}
        try:
            return json.loads(trader_data)
        except Exception:
            return {}

    def _dump_memory(self, memory: Dict[str, Any]) -> str:
        try:
            return json.dumps(memory, separators=(",", ":"))
        except Exception:
            return ""