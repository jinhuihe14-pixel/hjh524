import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class PriceElasticityModel:
    def __init__(self):
        self.elasticity_coefs: Dict[str, Dict] = {}
        self.price_response_curves: Dict[str, object] = {}
        self.sku_classification: Dict[str, Dict] = {}
        self.is_trained: bool = False

    def train(self, sales_data: pd.DataFrame) -> Dict:
        results = {}
        sku_groups = sales_data.groupby("sku_id")

        for sku_id, sku_data in sku_groups:
            if len(sku_data) < 30:
                continue

            category = sku_data["category"].iloc[0]
            elasticities = self._calculate_elasticity(sku_data)
            response_curve = self._build_response_curve(sku_data, elasticities)
            classification = self._classify_sku(sku_data, elasticities)

            self.elasticity_coefs[sku_id] = {
                **elasticities,
                "category": category,
            }
            self.price_response_curves[sku_id] = response_curve
            self.sku_classification[sku_id] = classification

        self.is_trained = len(self.elasticity_coefs) > 0

        category_summary = self._summarize_by_category()
        results = {
            "skus_analyzed": len(self.elasticity_coefs),
            "category_summary": category_summary,
        }
        return results

    def _calculate_elasticity(self, sku_data: pd.DataFrame) -> Dict:
        df = sku_data.copy()
        df = df.sort_values("date")

        df["price_change"] = df["price"].pct_change().fillna(0)
        df["sales_change"] = df["sales_volume"].pct_change().fillna(0)

        df["price_change"] = df["price_change"].replace([np.inf, -np.inf], 0)
        df["sales_change"] = df["sales_change"].replace([np.inf, -np.inf], 0)

        valid = df[(df["price_change"] != 0) & (df["sales_change"] != 0)]
        if len(valid) < 5:
            valid = df

        if len(valid) == 0:
            return {
                "price_elasticity": -1.0,
                "cross_elasticity": 0.0,
                "promo_elasticity": 0.5,
                "r_squared": 0.0,
                "confidence": 0.3,
            }

        X = valid[["price_change"]].values
        y = valid["sales_change"].values

        model = LinearRegression()
        model.fit(X, y)

        elasticity = model.coef_[0]

        if elasticity > 0:
            elasticity = -abs(elasticity) * 0.5

        r_squared = model.score(X, y) if len(valid) > 2 else 0

        promo_data = df[df["promotion_type"] != "常规定价"]
        regular_data = df[df["promotion_type"] == "常规定价"]
        if len(promo_data) > 0 and len(regular_data) > 0:
            promo_lift = (promo_data["sales_volume"].mean() / regular_data["sales_volume"].mean()) - 1
        else:
            promo_lift = 0.3

        base_price = df["base_price"].iloc[0] if "base_price" in df.columns else df["price"].mean()
        mid_price = df["price"].mean()
        price_ratio = mid_price / base_price if base_price > 0 else 1

        if abs(elasticity) < 0.3:
            confidence = 0.4
        elif abs(elasticity) < 1.0:
            confidence = 0.7
        else:
            confidence = 0.85

        return {
            "price_elasticity": round(float(elasticity), 4),
            "promo_elasticity": round(float(promo_lift), 4),
            "r_squared": round(float(max(0, r_squared)), 4),
            "confidence": round(confidence, 2),
            "avg_discount_rate": round(float(df["discount_rate"].mean()), 4),
            "base_price": float(base_price),
        }

    def _build_response_curve(self, sku_data: pd.DataFrame, elasticities: Dict) -> Dict:
        base_price = elasticities["base_price"]
        elasticity = elasticities["price_elasticity"]
        base_demand = sku_data["sales_volume"].mean()

        price_points = np.linspace(base_price * 0.5, base_price * 1.3, 50)

        demand_points = base_demand * (price_points / base_price) ** elasticity
        demand_points = np.maximum(1, demand_points)

        cost = sku_data["cost"].iloc[0] if "cost" in sku_data.columns else base_price * 0.6
        revenue_points = price_points * demand_points
        profit_points = (price_points - cost) * demand_points

        max_profit_idx = np.argmax(profit_points)
        max_profit_price = price_points[max_profit_idx]
        max_profit = profit_points[max_profit_idx]

        max_revenue_idx = np.argmax(revenue_points)
        max_revenue_price = price_points[max_revenue_idx]

        price_min_margin = cost * 1.2
        if price_min_margin < base_price * 0.5:
            price_min_margin = base_price * 0.5

        return {
            "base_price": base_price,
            "base_demand": base_demand,
            "elasticity": elasticity,
            "cost": float(cost),
            "price_range": [float(price_points[0]), float(price_points[-1])],
            "optimal_profit_price": float(max_profit_price),
            "optimal_profit": float(max_profit),
            "optimal_revenue_price": float(max_revenue_price),
            "price_min_margin": float(price_min_margin),
            "curve_points": [
                {"price": float(p), "demand": float(d), "profit": float(pr), "revenue": float(r)}
                for p, d, pr, r in zip(
                    price_points[::5],
                    demand_points[::5],
                    profit_points[::5],
                    revenue_points[::5],
                )
            ],
        }

    def _classify_sku(self, sku_data: pd.DataFrame, elasticities: Dict) -> Dict:
        elasticity = abs(elasticities["price_elasticity"])
        avg_sales = sku_data["sales_volume"].mean()
        margin_rate = sku_data.get("margin_rate", pd.Series([0.3])).mean()
        if isinstance(margin_rate, pd.Series):
            margin_rate = margin_rate.mean() if len(margin_rate) > 0 else 0.3

        if elasticity >= 1.5 or avg_sales > 200:
            sku_type = "引流款"
        elif elasticity < 0.5:
            sku_type = "刚需款"
        else:
            sku_type = "利润款"

        price_sensitivity = "高" if elasticity >= 1.2 else ("中" if elasticity >= 0.6 else "低")

        return {
            "sku_type": sku_type,
            "price_sensitivity": price_sensitivity,
            "elasticity_magnitude": round(elasticity, 4),
            "avg_daily_sales": round(float(avg_sales), 1),
            "margin_rate": round(float(margin_rate), 4),
        }

    def _summarize_by_category(self) -> Dict:
        categories = {}
        for sku_id, coefs in self.elasticity_coefs.items():
            cat = coefs.get("category", "未知")
            if cat not in categories:
                categories[cat] = {"elasticities": [], "count": 0}
            categories[cat]["elasticities"].append(coefs["price_elasticity"])
            categories[cat]["count"] += 1

        summary = {}
        for cat, data in categories.items():
            summary[cat] = {
                "sku_count": data["count"],
                "avg_elasticity": round(float(np.mean(data["elasticities"])), 4),
                "min_elasticity": round(float(np.min(data["elasticities"])), 4),
                "max_elasticity": round(float(np.max(data["elasticities"])), 4),
            }
        return summary

    def analyze_price_change(
        self,
        sku_id: str,
        new_price: float,
        current_price: Optional[float] = None,
        current_sales: Optional[float] = None,
    ) -> Dict:
        coefs = self.elasticity_coefs.get(sku_id)
        curve = self.price_response_curves.get(sku_id)
        classification = self.sku_classification.get(sku_id)

        if coefs is None or curve is None:
            return self._naive_analysis(new_price, current_price, current_sales)

        elasticity = coefs["price_elasticity"]
        base_price = curve["base_price"]
        base_demand = curve["base_demand"]
        cost = curve["cost"]

        if current_price is None:
            current_price = base_price
        if current_sales is None:
            current_sales = base_demand

        price_change_pct = (new_price - current_price) / current_price if current_price > 0 else 0
        predicted_sales_change = elasticity * price_change_pct
        new_sales = current_sales * (1 + predicted_sales_change)
        new_sales = max(1, new_sales)

        current_revenue = current_price * current_sales
        current_profit = (current_price - cost) * current_sales
        current_margin = (current_price - cost) / current_price if current_price > 0 else 0

        new_revenue = new_price * new_sales
        new_profit = (new_price - cost) * new_sales
        new_margin = (new_price - cost) / new_price if new_price > 0 else 0

        revenue_change_pct = (new_revenue - current_revenue) / current_revenue if current_revenue > 0 else 0
        profit_change_pct = (new_profit - current_profit) / current_profit if current_profit > 0 else 0
        sales_change_pct = (new_sales - current_sales) / current_sales if current_sales > 0 else 0

        recommendation = "维持"
        if new_price > current_price:
            if profit_change_pct > 0.02:
                recommendation = "建议提价"
            elif profit_change_pct < -0.05:
                recommendation = "不建议提价"
        else:
            if profit_change_pct > 0.02:
                recommendation = "建议降价"
            elif profit_change_pct < -0.05:
                recommendation = "不建议降价"

        risk_level = "低"
        if abs(price_change_pct) > 0.3:
            risk_level = "高"
        elif abs(price_change_pct) > 0.15:
            risk_level = "中"

        return {
            "sku_id": sku_id,
            "sku_type": classification["sku_type"] if classification else "未知",
            "price_sensitivity": classification["price_sensitivity"] if classification else "中",
            "current_price": round(float(current_price), 2),
            "new_price": round(float(new_price), 2),
            "price_change_pct": round(float(price_change_pct * 100), 2),
            "current_sales": round(float(current_sales), 1),
            "predicted_sales": round(float(new_sales), 1),
            "sales_change_pct": round(float(sales_change_pct * 100), 2),
            "current_revenue": round(float(current_revenue), 2),
            "predicted_revenue": round(float(new_revenue), 2),
            "revenue_change_pct": round(float(revenue_change_pct * 100), 2),
            "current_profit": round(float(current_profit), 2),
            "predicted_profit": round(float(new_profit), 2),
            "profit_change_pct": round(float(profit_change_pct * 100), 2),
            "current_margin_rate": round(float(current_margin * 100), 2),
            "new_margin_rate": round(float(new_margin * 100), 2),
            "price_elasticity": round(float(elasticity), 4),
            "recommendation": recommendation,
            "risk_level": risk_level,
            "confidence": coefs.get("confidence", 0.5),
        }

    def _naive_analysis(
        self,
        new_price: float,
        current_price: Optional[float] = None,
        current_sales: Optional[float] = None,
    ) -> Dict:
        if current_price is None:
            current_price = new_price
        if current_sales is None:
            current_sales = 100.0

        default_elasticity = -1.0
        price_change_pct = (new_price - current_price) / current_price if current_price > 0 else 0
        sales_change_pct = default_elasticity * price_change_pct
        new_sales = max(1, current_sales * (1 + sales_change_pct))

        return {
            "sku_id": "",
            "sku_type": "未知",
            "price_sensitivity": "中",
            "current_price": round(float(current_price), 2),
            "new_price": round(float(new_price), 2),
            "price_change_pct": round(float(price_change_pct * 100), 2),
            "current_sales": round(float(current_sales), 1),
            "predicted_sales": round(float(new_sales), 1),
            "sales_change_pct": round(float(sales_change_pct * 100), 2),
            "current_revenue": 0,
            "predicted_revenue": 0,
            "revenue_change_pct": 0,
            "current_profit": 0,
            "predicted_profit": 0,
            "profit_change_pct": 0,
            "current_margin_rate": 0,
            "new_margin_rate": 0,
            "price_elasticity": default_elasticity,
            "recommendation": "数据不足，建议补充历史数据",
            "risk_level": "高",
            "confidence": 0.3,
        }

    def get_optimal_price(
        self,
        sku_id: str,
        objective: str = "profit",
        constraints: Optional[Dict] = None,
    ) -> Dict:
        curve = self.price_response_curves.get(sku_id)
        classification = self.sku_classification.get(sku_id)

        if curve is None:
            return {
                "sku_id": sku_id,
                "optimal_price": 0,
                "expected_sales": 0,
                "expected_profit": 0,
                "objective": objective,
                "confidence": 0.3,
            }

        if constraints is None:
            constraints = {}

        min_price = constraints.get("min_price", curve["price_min_margin"])
        max_price = constraints.get("max_price", curve["price_range"][1])
        min_margin = constraints.get("min_margin_rate", 0.15)

        min_margin_price = curve["cost"] / (1 - min_margin) if (1 - min_margin) > 0 else min_price
        min_price = max(min_price, min_margin_price)

        points = curve["curve_points"]
        valid_points = [p for p in points if min_price <= p["price"] <= max_price]

        if not valid_points:
            valid_points = points

        if objective == "profit":
            optimal = max(valid_points, key=lambda p: p["profit"])
        elif objective == "revenue":
            optimal = max(valid_points, key=lambda p: p["revenue"])
        else:
            optimal = max(valid_points, key=lambda p: p["profit"])

        return {
            "sku_id": sku_id,
            "sku_type": classification["sku_type"] if classification else "未知",
            "optimal_price": round(optimal["price"], 2),
            "expected_sales": round(optimal["demand"], 1),
            "expected_profit": round(optimal["profit"], 2),
            "expected_revenue": round(optimal["revenue"], 2),
            "objective": objective,
            "cost": curve["cost"],
            "constraints": {
                "min_price": min_price,
                "max_price": max_price,
                "min_margin_rate": min_margin,
            },
            "confidence": 0.75,
        }

    def get_sku_classification(self, sku_id: str) -> Dict:
        return self.sku_classification.get(sku_id, {
            "sku_type": "未知",
            "price_sensitivity": "中",
        })

    def batch_analyze(self, sku_prices: List[Dict]) -> List[Dict]:
        results = []
        for item in sku_prices:
            result = self.analyze_price_change(
                sku_id=item["sku_id"],
                new_price=item["new_price"],
                current_price=item.get("current_price"),
                current_sales=item.get("current_sales"),
            )
            results.append(result)
        return results


_default_elasticity_model = None


def get_price_elasticity_model() -> PriceElasticityModel:
    global _default_elasticity_model
    if _default_elasticity_model is None:
        _default_elasticity_model = PriceElasticityModel()
    return _default_elasticity_model
