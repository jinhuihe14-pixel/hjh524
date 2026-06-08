import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class DataGenerator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.categories = {
            "生鲜": {"sku_count": 50, "base_demand": 200, "price_elasticity": -1.8, "shelf_life_days": 3},
            "食品": {"sku_count": 100, "base_demand": 150, "price_elasticity": -1.2, "shelf_life_days": 180},
            "日化": {"sku_count": 80, "base_demand": 80, "price_elasticity": -0.9, "shelf_life_days": 365},
            "家电": {"sku_count": 30, "base_demand": 10, "price_elasticity": -0.6, "shelf_life_days": 730},
            "服饰": {"sku_count": 60, "base_demand": 40, "price_elasticity": -1.5, "shelf_life_days": 540},
        }
        self.promotion_types = ["常规定价", "限时折扣", "满减", "捆绑销售", "会员专享", "阶梯促销"]

    def generate_sku_catalog(self) -> pd.DataFrame:
        skus = []
        sku_id = 1
        for category, info in self.categories.items():
            for i in range(info["sku_count"]):
                base_price = np.random.uniform(5, 500)
                cost = base_price * np.random.uniform(0.4, 0.7)
                skus.append({
                    "sku_id": f"SKU{sku_id:06d}",
                    "category": category,
                    "product_name": f"{category}商品{i+1}",
                    "base_price": round(base_price, 2),
                    "cost": round(cost, 2),
                    "shelf_life_days": info["shelf_life_days"],
                    "price_elasticity_true": info["price_elasticity"] + np.random.normal(0, 0.2),
                    "base_demand": info["base_demand"] * np.random.uniform(0.5, 1.5),
                    "is_traffic_driver": np.random.choice([True, False], p=[0.2, 0.8]),
                    "is_must_have": np.random.choice([True, False], p=[0.3, 0.7]),
                })
                sku_id += 1
        return pd.DataFrame(skus)

    def generate_sales_history(self, sku_catalog: pd.DataFrame, days: int = 1095) -> pd.DataFrame:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        sales_records = []

        for _, sku in sku_catalog.iterrows():
            daily_demand_base = sku["base_demand"]
            elasticity = sku["price_elasticity_true"]

            for date in date_range:
                day_of_week = date.dayofweek
                is_weekend = 1 if day_of_week >= 5 else 0
                month = date.month
                is_holiday = self._is_holiday(date)

                seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * month / 12)
                weekend_factor = 1.3 if is_weekend else 1.0
                holiday_factor = 1.5 if is_holiday else 1.0

                temperature = self._get_temperature(date)
                temp_factor = 1.0 + 0.05 * (temperature - 20) / 10 if sku["category"] == "生鲜" else 1.0

                traffic_factor = np.random.normal(1.0, 0.1)

                price = sku["base_price"] * np.random.uniform(0.85, 1.05)
                discount = 1 - price / sku["base_price"]
                price_effect = (1 + discount * abs(elasticity)) if discount > 0 else 1.0

                promotion_type = np.random.choice(
                    self.promotion_types,
                    p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]
                )
                promo_boost = 1.0
                if promotion_type == "限时折扣":
                    promo_boost = 1.3
                elif promotion_type == "满减":
                    promo_boost = 1.15
                elif promotion_type == "捆绑销售":
                    promo_boost = 1.2
                elif promotion_type == "会员专享":
                    promo_boost = 1.1
                elif promotion_type == "阶梯促销":
                    promo_boost = 1.25

                expected_demand = (
                    daily_demand_base
                    * seasonal_factor
                    * weekend_factor
                    * holiday_factor
                    * temp_factor
                    * traffic_factor
                    * price_effect
                    * promo_boost
                )

                actual_sales = np.random.poisson(max(1, expected_demand))
                stock_level = np.random.randint(100, 1000)

                sales_records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "sku_id": sku["sku_id"],
                    "category": sku["category"],
                    "sales_volume": actual_sales,
                    "price": round(price, 2),
                    "base_price": sku["base_price"],
                    "cost": sku["cost"],
                    "discount_rate": round(discount, 4),
                    "promotion_type": promotion_type,
                    "stock_level": stock_level,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "is_holiday": is_holiday,
                    "month": month,
                    "temperature": round(temperature, 1),
                    "traffic_index": round(traffic_factor, 4),
                })

        return pd.DataFrame(sales_records)

    def generate_competitor_prices(self, sku_catalog: pd.DataFrame, days: int = 90) -> pd.DataFrame:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        records = []
        for _, sku in sku_catalog.iterrows():
            for date in date_range:
                for competitor in ["竞品A", "竞品B", "竞品C"]:
                    comp_price = sku["base_price"] * np.random.uniform(0.9, 1.1)
                    records.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "sku_id": sku["sku_id"],
                        "competitor": competitor,
                        "competitor_price": round(comp_price, 2),
                    })

        return pd.DataFrame(records)

    def generate_inventory_data(self, sku_catalog: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, sku in sku_catalog.iterrows():
            stock_qty = np.random.randint(50, 500)
            production_date = datetime.now() - timedelta(days=np.random.randint(0, int(sku["shelf_life_days"] * 0.5)))
            days_to_expiry = sku["shelf_life_days"] - (datetime.now() - production_date).days
            records.append({
                "sku_id": sku["sku_id"],
                "stock_quantity": stock_qty,
                "production_date": production_date.strftime("%Y-%m-%d"),
                "days_to_expiry": max(1, days_to_expiry),
                "expiry_ratio": round(max(0, 1 - days_to_expiry / sku["shelf_life_days"]), 4),
                "warehouse": np.random.choice(["中心仓", "前置仓1", "前置仓2"]),
            })
        return pd.DataFrame(records)

    def _is_holiday(self, date: datetime) -> int:
        month_day = (date.month, date.day)
        holidays = [
            (1, 1), (1, 2), (1, 3),
            (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17),
            (4, 4), (4, 5), (4, 6),
            (5, 1), (5, 2), (5, 3),
            (6, 10), (6, 11), (6, 12),
            (9, 15), (9, 16), (9, 17),
            (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7),
            (12, 25),
        ]
        return 1 if month_day in holidays else 0

    def _get_temperature(self, date: datetime) -> float:
        month = date.month
        base_temp = 10 + 15 * np.sin(2 * np.pi * (month - 3) / 12)
        return base_temp + np.random.normal(0, 3)


_default_generator = None


def get_data_generator() -> DataGenerator:
    global _default_generator
    if _default_generator is None:
        _default_generator = DataGenerator()
    return _default_generator
