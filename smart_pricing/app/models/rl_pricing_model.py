import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random


class RLDynamicPricingModel:
    def __init__(
        self,
        learning_rate: float = 0.05,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.1,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.q_tables: Dict[str, Dict] = {}
        self.state_counts: Dict[str, int] = defaultdict(int)
        self.is_trained: Dict[str, bool] = {}

        self.price_bins = 10
        self.stock_bins = 5
        self.expiry_bins = 4
        self.time_bins = 4

    def train(self, sales_data: pd.DataFrame, episodes: int = 100) -> Dict:
        results = {}
        sku_groups = sales_data.groupby("sku_id")

        for sku_id, sku_data in sku_groups:
            if len(sku_data) < 30:
                continue

            category = sku_data["category"].iloc[0]
            base_price = sku_data["base_price"].iloc[0] if "base_price" in sku_data.columns else sku_data["price"].mean()
            cost = sku_data["cost"].iloc[0] if "cost" in sku_data.columns else base_price * 0.6

            q_table = self._train_sku(sku_data, base_price, cost, episodes)

            self.q_tables[sku_id] = {
                "q_table": q_table,
                "base_price": base_price,
                "cost": cost,
                "category": category,
                "avg_sales": float(sku_data["sales_volume"].mean()),
            }
            self.is_trained[sku_id] = True

        results = {
            "skus_trained": len(self.q_tables),
            "episodes_per_sku": episodes,
        }
        return results

    def _train_sku(
        self,
        sku_data: pd.DataFrame,
        base_price: float,
        cost: float,
        episodes: int,
    ) -> Dict:
        q_table = defaultdict(lambda: 0.0)

        sorted_data = sku_data.sort_values("date").reset_index(drop=True)
        avg_sales = sorted_data["sales_volume"].mean()
        avg_stock = 500

        for episode in range(episodes):
            state = self._get_initial_state(avg_sales, avg_stock, base_price)

            for step in range(30):
                action = self._select_action(state, q_table, base_price, episode, episodes)
                next_state, reward = self._simulate_step(
                    state, action, base_price, cost, avg_sales, sorted_data
                )

                current_q = q_table[(state, action)]
                max_next_q = self._max_q(next_state, q_table, base_price)

                new_q = current_q + self.learning_rate * (
                    reward + self.discount_factor * max_next_q - current_q
                )
                q_table[(state, action)] = new_q

                state = next_state

                if state[1] <= 0:
                    break

        return dict(q_table)

    def _get_initial_state(self, avg_sales: float, avg_stock: float, base_price: float) -> Tuple:
        stock_level = min(avg_stock * 0.8, 500)
        expiry_days = 30
        time_period = 0

        stock_bin = self._bin_value(stock_level, 0, avg_stock * 2, self.stock_bins)
        expiry_bin = self._bin_value(expiry_days, 0, 60, self.expiry_bins)
        time_bin = time_period % self.time_bins

        return (stock_bin, stock_level, expiry_bin, time_bin)

    def _select_action(
        self,
        state: Tuple,
        q_table: Dict,
        base_price: float,
        episode: int,
        total_episodes: int,
    ) -> int:
        exploration = self.exploration_rate * (1 - episode / total_episodes)

        if random.random() < exploration:
            return random.randint(0, self.price_bins - 1)
        else:
            return self._best_action(state, q_table, base_price)

    def _best_action(self, state: Tuple, q_table: Dict, base_price: float) -> int:
        best_action = self.price_bins // 2
        best_q = float("-inf")

        for action in range(self.price_bins):
            q_value = q_table.get((state, action), 0.0)
            if q_value > best_q:
                best_q = q_value
                best_action = action

        return best_action

    def _max_q(self, state: Tuple, q_table: Dict, base_price: float) -> float:
        max_q = float("-inf")
        for action in range(self.price_bins):
            q_value = q_table.get((state, action), 0.0)
            max_q = max(max_q, q_value)
        return max_q if max_q != float("-inf") else 0.0

    def _simulate_step(
        self,
        state: Tuple,
        action: int,
        base_price: float,
        cost: float,
        avg_sales: float,
        history_data: pd.DataFrame,
    ) -> Tuple[Tuple, float]:
        stock_bin, stock_level, expiry_bin, time_bin = state

        price_ratio = 0.6 + (action / (self.price_bins - 1)) * 0.8
        current_price = base_price * price_ratio

        discount = 1 - current_price / base_price if base_price > 0 else 0
        elasticity = -1.2
        demand_multiplier = 1 + discount * abs(elasticity)

        base_demand = avg_sales
        if expiry_bin <= 1:
            base_demand *= 0.7
        if stock_bin <= 1:
            base_demand *= 0.8

        expected_sales = base_demand * demand_multiplier
        actual_sales = min(stock_level, expected_sales * random.uniform(0.8, 1.2))
        actual_sales = max(0, actual_sales)

        revenue = actual_sales * current_price
        cost_total = actual_sales * cost
        profit = revenue - cost_total

        if stock_level > 0 and expiry_bin == 0:
            spoilage = stock_level * 0.3
            spoilage_cost = spoilage * cost * 0.5
            profit -= spoilage_cost

        next_stock = max(0, stock_level - actual_sales)
        next_expiry_bin = max(0, expiry_bin - 1)
        next_time_bin = (time_bin + 1) % self.time_bins

        next_stock_bin = self._bin_value(next_stock, 0, avg_sales * 10, self.stock_bins)

        next_state = (next_stock_bin, next_stock, next_expiry_bin, next_time_bin)

        reward = profit

        return next_state, reward

    def _bin_value(self, value: float, min_val: float, max_val: float, bins: int) -> int:
        if max_val == min_val:
            return 0
        ratio = (value - min_val) / (max_val - min_val)
        return max(0, min(bins - 1, int(ratio * bins)))

    def get_optimal_price(
        self,
        sku_id: str,
        stock_quantity: int,
        days_to_expiry: Optional[int] = None,
        competitor_price: Optional[float] = None,
        time_period: int = 0,
        constraints: Optional[Dict] = None,
    ) -> Dict:
        model_info = self.q_tables.get(sku_id)

        if model_info is None:
            return self._naive_pricing(sku_id, stock_quantity, days_to_expiry, constraints)

        base_price = model_info["base_price"]
        cost = model_info["cost"]
        q_table = model_info["q_table"]
        avg_sales = model_info["avg_sales"]

        if constraints is None:
            constraints = {}

        min_price = constraints.get("min_price", cost * 1.15)
        max_price = constraints.get("max_price", base_price * 1.2)
        min_margin = constraints.get("min_margin_rate", 0.15)

        min_margin_price = cost / (1 - min_margin) if (1 - min_margin) > 0 else min_price
        min_price = max(min_price, min_margin_price)

        stock_bin = self._bin_value(stock_quantity, 0, avg_sales * 10, self.stock_bins)

        if days_to_expiry is None:
            days_to_expiry = 30
        expiry_bin = self._bin_value(days_to_expiry, 0, 60, self.expiry_bins)
        time_bin = time_period % self.time_bins

        state = (stock_bin, stock_quantity, expiry_bin, time_bin)

        best_action = self._best_action(state, q_table, base_price)

        price_ratio = 0.6 + (best_action / (self.price_bins - 1)) * 0.8
        optimal_price = base_price * price_ratio

        optimal_price = max(min_price, min(max_price, optimal_price))

        expiry_factor = 1.0
        if days_to_expiry is not None and days_to_expiry < 7:
            expiry_factor = 0.7 + (days_to_expiry / 7) * 0.3
        elif days_to_expiry is not None and days_to_expiry < 14:
            expiry_factor = 0.85 + (days_to_expiry - 7) / 7 * 0.15

        if days_to_expiry is not None and days_to_expiry < 3:
            stock_pressure = 1.0
        else:
            stock_pressure = min(1.0, stock_quantity / (avg_sales * 5))

        adjustment_factor = 0.5 * (1 - expiry_factor) + 0.5 * (1 - stock_pressure)
        adjustment_factor = max(0, min(0.5, adjustment_factor))

        final_price = optimal_price * (1 - adjustment_factor * 0.3)
        final_price = max(min_price, min(max_price, final_price))

        if competitor_price is not None:
            if final_price > competitor_price * 1.05:
                final_price = competitor_price * 0.98
                final_price = max(min_price, final_price)

        margin_rate = (final_price - cost) / final_price if final_price > 0 else 0

        expected_demand = self._estimate_demand(final_price, base_price, model_info)

        tiered_pricing = self._calculate_tiered_pricing(
            base_price, cost, days_to_expiry, stock_quantity, avg_sales, min_price, max_price
        )

        time_based_pricing = self._calculate_time_based_pricing(
            base_price, final_price, time_bin
        )

        return {
            "sku_id": sku_id,
            "base_price": round(base_price, 2),
            "cost": round(cost, 2),
            "optimal_price": round(final_price, 2),
            "discount_rate": round(1 - final_price / base_price if base_price > 0 else 0, 4),
            "margin_rate": round(margin_rate, 4),
            "expected_daily_sales": round(expected_demand, 1),
            "expected_daily_profit": round((final_price - cost) * expected_demand, 2),
            "stock_quantity": stock_quantity,
            "days_to_expiry": days_to_expiry,
            "competitor_price": competitor_price,
            "pricing_strategy": self._get_strategy_label(
                stock_quantity, days_to_expiry, base_price, final_price
            ),
            "tiered_pricing": tiered_pricing,
            "time_based_pricing": time_based_pricing,
            "confidence": 0.78,
            "constraints": {
                "min_price": min_price,
                "max_price": max_price,
                "min_margin_rate": min_margin,
            },
        }

    def _estimate_demand(self, price: float, base_price: float, model_info: Dict) -> float:
        elasticity = -1.2
        base_demand = model_info.get("avg_sales", 50)
        discount = 1 - price / base_price if base_price > 0 else 0
        demand = base_demand * (1 + discount * abs(elasticity))
        return max(1, demand)

    def _calculate_tiered_pricing(
        self,
        base_price: float,
        cost: float,
        days_to_expiry: Optional[int],
        stock_quantity: int,
        avg_sales: float,
        min_price: float,
        max_price: float,
    ) -> List[Dict]:
        tiers = []

        if days_to_expiry is None or days_to_expiry > 30:
            steps = [0, 7, 3, 1]
        else:
            steps = [days_to_expiry, max(1, days_to_expiry // 2), max(1, days_to_expiry // 4), 1]

        for i, days_left in enumerate(sorted(set(steps), reverse=True)):
            discount_level = min(0.6, i * 0.15 + 0.05)
            price = base_price * (1 - discount_level)
            price = max(min_price, min(max_price, price))
            margin = (price - cost) / price if price > 0 else 0

            tiers.append({
                "tier": f"第{i+1}档",
                "days_to_expiry_threshold": days_left,
                "price": round(price, 2),
                "discount_rate": round(discount_level, 4),
                "margin_rate": round(margin, 4),
            })

        return tiers

    def _calculate_time_based_pricing(
        self,
        base_price: float,
        optimal_price: float,
        time_bin: int,
    ) -> List[Dict]:
        time_periods = [
            {"name": "早高峰", "factor": 1.05, "hours": "7:00-9:00"},
            {"name": "午间时段", "factor": 1.0, "hours": "11:00-14:00"},
            {"name": "下午时段", "factor": 0.98, "hours": "14:00-18:00"},
            {"name": "晚高峰", "factor": 1.02, "hours": "18:00-21:00"},
        ]

        result = []
        for period in time_periods:
            price = optimal_price * period["factor"]
            discount = 1 - price / base_price if base_price > 0 else 0
            result.append({
                "time_period": period["name"],
                "hours": period["hours"],
                "price": round(price, 2),
                "discount_rate": round(discount, 4),
                "adjustment_factor": period["factor"],
            })

        return result

    def _get_strategy_label(
        self,
        stock_qty: int,
        days_to_expiry: Optional[int],
        base_price: float,
        optimal_price: float,
    ) -> str:
        discount = 1 - optimal_price / base_price if base_price > 0 else 0

        if days_to_expiry is not None and days_to_expiry < 3:
            return "临期清仓"
        elif stock_qty > 300 and discount > 0.2:
            return "去库存促销"
        elif discount < 0.05:
            return "常规价"
        elif discount < 0.15:
            return "轻度促销"
        else:
            return "中度促销"

    def _naive_pricing(
        self,
        sku_id: str,
        stock_quantity: int,
        days_to_expiry: Optional[int],
        constraints: Optional[Dict],
    ) -> Dict:
        base_price = 100.0
        cost = 60.0

        if constraints:
            base_price = constraints.get("base_price", base_price)
            cost = constraints.get("cost", cost)

        discount = 0.1
        if stock_quantity > 200:
            discount = 0.2
        if days_to_expiry is not None and days_to_expiry < 7:
            discount = max(discount, 0.3)
        if days_to_expiry is not None and days_to_expiry < 3:
            discount = max(discount, 0.5)

        final_price = base_price * (1 - discount)
        margin_rate = (final_price - cost) / final_price if final_price > 0 else 0

        return {
            "sku_id": sku_id,
            "base_price": round(base_price, 2),
            "cost": round(cost, 2),
            "optimal_price": round(final_price, 2),
            "discount_rate": round(discount, 4),
            "margin_rate": round(margin_rate, 4),
            "expected_daily_sales": 50.0,
            "expected_daily_profit": round((final_price - cost) * 50, 2),
            "stock_quantity": stock_quantity,
            "days_to_expiry": days_to_expiry,
            "competitor_price": None,
            "pricing_strategy": "经验定价",
            "tiered_pricing": [],
            "time_based_pricing": [],
            "confidence": 0.4,
            "constraints": constraints or {},
        }

    def batch_pricing(
        self,
        sku_list: List[Dict],
    ) -> List[Dict]:
        results = []
        for sku_info in sku_list:
            result = self.get_optimal_price(
                sku_id=sku_info["sku_id"],
                stock_quantity=sku_info.get("stock_quantity", 100),
                days_to_expiry=sku_info.get("days_to_expiry"),
                competitor_price=sku_info.get("competitor_price"),
                time_period=sku_info.get("time_period", 0),
                constraints=sku_info.get("constraints"),
            )
            results.append(result)
        return results

    def update(self, sku_id: str, state: Tuple, action: int, reward: float, next_state: Tuple):
        model_info = self.q_tables.get(sku_id)
        if model_info is None:
            return

        q_table = model_info["q_table"]
        base_price = model_info["base_price"]

        current_q = q_table.get((state, action), 0.0)
        max_next_q = self._max_q(next_state, q_table, base_price)

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        q_table[(state, action)] = new_q


_default_rl_model = None


def get_rl_pricing_model() -> RLDynamicPricingModel:
    global _default_rl_model
    if _default_rl_model is None:
        _default_rl_model = RLDynamicPricingModel()
    return _default_rl_model
