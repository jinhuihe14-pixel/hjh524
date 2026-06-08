import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from itertools import combinations


class PromotionOptimizerModel:
    def __init__(self):
        self.promotion_types = {
            "discount": {"name": "限时折扣", "description": "直接价格折扣", "impact_factor": 1.3},
            "full_reduction": {"name": "满减", "description": "满X减Y", "impact_factor": 1.15},
            "bundle": {"name": "捆绑销售", "description": "多件组合优惠", "impact_factor": 1.2},
            "member_price": {"name": "会员专享价", "description": "会员专属折扣", "impact_factor": 1.1},
            "tiered": {"name": "阶梯促销", "description": "买越多越便宜", "impact_factor": 1.25},
        }
        self.historical_performance: Dict[str, List] = {}
        self.is_trained: bool = False

    def train(self, sales_data: pd.DataFrame, promotion_data: Optional[pd.DataFrame] = None) -> Dict:
        promo_groups = sales_data.groupby("promotion_type")

        for promo_type, group in promo_groups:
            if promo_type == "常规定价":
                continue
            self.historical_performance[promo_type] = {
                "avg_lift_rate": float(group["sales_volume"].mean() / sales_data[sales_data["promotion_type"] == "常规定价"]["sales_volume"].mean() - 1) if len(sales_data[sales_data["promotion_type"] == "常规定价"]) > 0 else 0.2,
                "sample_size": len(group),
                "category_performance": self._category_performance(group),
            }

        self.is_trained = len(self.historical_performance) > 0
        return {
            "promotion_types_analyzed": len(self.historical_performance),
            "types": list(self.historical_performance.keys()),
        }

    def _category_performance(self, promo_group: pd.DataFrame) -> Dict:
        if "category" not in promo_group.columns:
            return {}
        cat_perf = promo_group.groupby("category")["sales_volume"].agg(["mean", "count"]).to_dict("index")
        return {cat: {"avg_sales": float(vals["mean"]), "count": int(vals["count"])} for cat, vals in cat_perf.items()}

    def recommend_promotion(
        self,
        sku_info: Dict,
        objective: str = "profit",
        budget: Optional[float] = None,
        duration_days: int = 7,
        inventory_constraint: Optional[int] = None,
    ) -> Dict:
        sku_id = sku_info.get("sku_id", "")
        base_price = sku_info.get("base_price", 100.0)
        cost = sku_info.get("cost", 60.0)
        base_demand = sku_info.get("base_demand", 50)
        category = sku_info.get("category", "食品")
        current_stock = sku_info.get("stock_quantity", 200)

        candidates = []

        single_promos = self._generate_single_promotions(
            base_price, cost, base_demand, duration_days, objective
        )
        candidates.extend(single_promos)

        combo_promos = self._generate_combo_promotions(
            base_price, cost, base_demand, duration_days, objective
        )
        candidates.extend(combo_promos)

        valid_candidates = []
        for promo in candidates:
            if budget is not None and promo.get("estimated_cost", 0) > budget:
                continue
            if inventory_constraint is not None and promo.get("estimated_sales", 0) > inventory_constraint:
                continue
            if promo.get("estimated_profit", 0) <= 0:
                continue
            valid_candidates.append(promo)

        if not valid_candidates:
            valid_candidates = candidates

        if objective == "profit":
            valid_candidates.sort(key=lambda x: x.get("estimated_profit", 0), reverse=True)
        elif objective == "revenue":
            valid_candidates.sort(key=lambda x: x.get("estimated_revenue", 0), reverse=True)
        elif objective == "roi":
            valid_candidates.sort(key=lambda x: x.get("roi", 0), reverse=True)
        else:
            valid_candidates.sort(key=lambda x: x.get("estimated_profit", 0), reverse=True)

        top3 = valid_candidates[:3]

        return {
            "sku_id": sku_id,
            "category": category,
            "objective": objective,
            "duration_days": duration_days,
            "base_price": base_price,
            "cost": cost,
            "base_demand": base_demand,
            "recommendations": [
                {
                    "rank": i + 1,
                    **promo,
                }
                for i, promo in enumerate(top3)
            ],
            "baseline": {
                "no_promo_sales": base_demand * duration_days,
                "no_promo_revenue": base_price * base_demand * duration_days,
                "no_promo_profit": (base_price - cost) * base_demand * duration_days,
            },
        }

    def _generate_single_promotions(
        self,
        base_price: float,
        cost: float,
        base_demand: float,
        duration: int,
        objective: str,
    ) -> List[Dict]:
        promos = []

        for discount in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            promo_price = base_price * (1 - discount)
            margin = promo_price - cost
            if margin <= 0:
                continue

            lift_rate = discount * 1.5 + discount * discount * 2
            expected_daily_sales = base_demand * (1 + lift_rate)
            total_sales = expected_daily_sales * duration
            revenue = promo_price * total_sales
            profit = margin * total_sales
            promo_cost = (base_price - promo_price) * total_sales
            roi = (profit - (base_price - cost) * base_demand * duration) / max(1, promo_cost) if promo_cost > 0 else 0

            promos.append({
                "type": "限时折扣",
                "type_code": "discount",
                "detail": f"{int(discount*100)}%折扣",
                "params": {"discount_rate": discount},
                "promo_price": round(promo_price, 2),
                "estimated_daily_sales": round(expected_daily_sales, 1),
                "estimated_total_sales": round(total_sales, 1),
                "estimated_revenue": round(revenue, 2),
                "estimated_profit": round(profit, 2),
                "estimated_cost": round(promo_cost, 2),
                "profit_increment": round(profit - (base_price - cost) * base_demand * duration, 2),
                "sales_increment_rate": round(lift_rate, 4),
                "roi": round(roi, 4),
                "margin_rate": round(margin / promo_price, 4),
                "risk_level": "低" if discount < 0.15 else ("中" if discount < 0.25 else "高"),
            })

        for threshold in [base_price * 2, base_price * 3, base_price * 5]:
            reduction = threshold * 0.1
            effective_discount = reduction / threshold
            lift_rate = effective_discount * 1.2

            avg_pieces = threshold / base_price
            expected_daily_sales = base_demand * (1 + lift_rate)
            total_sales = expected_daily_sales * duration
            avg_order_value = threshold
            orders = total_sales / avg_pieces
            total_reduction = orders * reduction
            revenue = base_price * total_sales - total_reduction
            profit = revenue - cost * total_sales
            promo_cost = total_reduction
            baseline_profit = (base_price - cost) * base_demand * duration
            roi = (profit - baseline_profit) / max(1, promo_cost) if promo_cost > 0 else 0

            promos.append({
                "type": "满减",
                "type_code": "full_reduction",
                "detail": f"满{int(threshold)}减{int(reduction)}",
                "params": {"full_amount": round(threshold, 2), "reduction_amount": round(reduction, 2)},
                "effective_discount_rate": round(effective_discount, 4),
                "estimated_daily_sales": round(expected_daily_sales, 1),
                "estimated_total_sales": round(total_sales, 1),
                "estimated_revenue": round(revenue, 2),
                "estimated_profit": round(profit, 2),
                "estimated_cost": round(promo_cost, 2),
                "profit_increment": round(profit - baseline_profit, 2),
                "sales_increment_rate": round(lift_rate, 4),
                "roi": round(roi, 4),
                "risk_level": "低",
            })

        for bundle_size in [2, 3, 4]:
            bundle_price = base_price * bundle_size * 0.9
            unit_price = bundle_price / bundle_size
            discount = 1 - unit_price / base_price
            lift_rate = discount * 1.3

            expected_daily_sales = base_demand * (1 + lift_rate) * bundle_size
            total_sales = expected_daily_sales * duration
            bundles_sold = total_sales / bundle_size
            revenue = bundle_price * bundles_sold
            profit = (unit_price - cost) * total_sales
            promo_cost = (base_price - unit_price) * total_sales
            baseline_profit = (base_price - cost) * base_demand * duration
            roi = (profit - baseline_profit) / max(1, promo_cost) if promo_cost > 0 else 0

            promos.append({
                "type": "捆绑销售",
                "type_code": "bundle",
                "detail": f"{bundle_size}件{round(bundle_price,2)}元",
                "params": {"bundle_size": bundle_size, "bundle_price": round(bundle_price, 2)},
                "unit_price": round(unit_price, 2),
                "effective_discount_rate": round(discount, 4),
                "estimated_daily_sales": round(expected_daily_sales, 1),
                "estimated_total_sales": round(total_sales, 1),
                "estimated_revenue": round(revenue, 2),
                "estimated_profit": round(profit, 2),
                "estimated_cost": round(promo_cost, 2),
                "profit_increment": round(profit - baseline_profit, 2),
                "sales_increment_rate": round(lift_rate, 4),
                "roi": round(roi, 4),
                "risk_level": "低" if bundle_size <= 2 else "中",
            })

        member_discount = 0.1
        member_price = base_price * (1 - member_discount)
        member_ratio = 0.3
        overall_discount = member_discount * member_ratio
        lift_rate = overall_discount * 1.1

        expected_daily_sales = base_demand * (1 + lift_rate)
        total_sales = expected_daily_sales * duration
        member_sales = total_sales * member_ratio
        regular_sales = total_sales * (1 - member_ratio)
        revenue = member_price * member_sales + base_price * regular_sales
        profit = revenue - cost * total_sales
        promo_cost = (base_price - member_price) * member_sales
        baseline_profit = (base_price - cost) * base_demand * duration
        roi = (profit - baseline_profit) / max(1, promo_cost) if promo_cost > 0 else 0

        promos.append({
            "type": "会员专享价",
            "type_code": "member_price",
            "detail": f"会员价{round(member_price,2)}元",
            "params": {"member_price": round(member_price, 2), "member_discount_rate": member_discount},
            "estimated_daily_sales": round(expected_daily_sales, 1),
            "estimated_total_sales": round(total_sales, 1),
            "estimated_revenue": round(revenue, 2),
            "estimated_profit": round(profit, 2),
            "estimated_cost": round(promo_cost, 2),
            "profit_increment": round(profit - baseline_profit, 2),
            "sales_increment_rate": round(lift_rate, 4),
            "roi": round(roi, 4),
            "risk_level": "低",
        })

        return promos

    def _generate_combo_promotions(
        self,
        base_price: float,
        cost: float,
        base_demand: float,
        duration: int,
        objective: str,
    ) -> List[Dict]:
        promos = []

        combo_configs = [
            {"discount": 0.1, "full_threshold": 3, "name": "折扣+满减"},
            {"discount": 0.05, "bundle_size": 2, "name": "折扣+捆绑"},
            {"member_discount": 0.15, "bundle_size": 2, "name": "会员+捆绑"},
        ]

        for config in combo_configs:
            name = config["name"]

            base_discount = config.get("discount", config.get("member_discount", 0.1))
            bundle_size = config.get("bundle_size", 1)
            full_threshold = config.get("full_threshold", 0)

            effective_discount = base_discount
            if bundle_size > 1:
                effective_discount = max(effective_discount, 0.15)
            if full_threshold > 0:
                effective_discount = max(effective_discount, 0.1)

            combo_boost = 1.15
            lift_rate = effective_discount * 1.5 * combo_boost

            promo_price = base_price * (1 - effective_discount)
            margin = promo_price - cost
            if margin <= 0:
                continue

            expected_daily_sales = base_demand * (1 + lift_rate)
            total_sales = expected_daily_sales * duration
            revenue = promo_price * total_sales
            profit = margin * total_sales
            promo_cost = (base_price - promo_price) * total_sales
            baseline_profit = (base_price - cost) * base_demand * duration
            roi = (profit - baseline_profit) / max(1, promo_cost) if promo_cost > 0 else 0

            promos.append({
                "type": "组合促销",
                "type_code": "combo",
                "detail": name,
                "params": config,
                "effective_discount_rate": round(effective_discount, 4),
                "estimated_daily_sales": round(expected_daily_sales, 1),
                "estimated_total_sales": round(total_sales, 1),
                "estimated_revenue": round(revenue, 2),
                "estimated_profit": round(profit, 2),
                "estimated_cost": round(promo_cost, 2),
                "profit_increment": round(profit - baseline_profit, 2),
                "sales_increment_rate": round(lift_rate, 4),
                "roi": round(roi, 4),
                "risk_level": "中",
            })

        return promos

    def evaluate_promotion(
        self,
        promotion_config: Dict,
        actual_data: pd.DataFrame,
        baseline_data: pd.DataFrame,
    ) -> Dict:
        actual_sales = actual_data["sales_volume"].sum()
        actual_revenue = (actual_data["price"] * actual_data["sales_volume"]).sum()
        actual_cost_total = (actual_data["cost"] * actual_data["sales_volume"]).sum() if "cost" in actual_data.columns else actual_revenue * 0.6
        actual_profit = actual_revenue - actual_cost_total

        baseline_sales = baseline_data["sales_volume"].sum()
        baseline_revenue = (baseline_data.get("price", pd.Series([100])) * baseline_data["sales_volume"]).sum()
        baseline_cost = baseline_revenue * 0.6
        baseline_profit = baseline_revenue - baseline_cost

        sales_lift = (actual_sales - baseline_sales) / baseline_sales if baseline_sales > 0 else 0
        revenue_lift = (actual_revenue - baseline_revenue) / baseline_revenue if baseline_revenue > 0 else 0
        profit_lift = (actual_profit - baseline_profit) / baseline_profit if baseline_profit > 0 else 0

        daily_data = actual_data.groupby("date").agg(
            daily_sales=("sales_volume", "sum"),
            daily_revenue=("price", lambda x: (x * actual_data.loc[x.index, "sales_volume"]).sum()),
        ).reset_index()

        daily_data["cumulative_sales"] = daily_data["daily_sales"].cumsum()
        daily_data["cumulative_revenue"] = daily_data["daily_revenue"].cumsum()

        estimated_sales = promotion_config.get("estimated_total_sales", actual_sales)
        estimated_profit = promotion_config.get("estimated_profit", actual_profit)

        sales_achievement = actual_sales / estimated_sales if estimated_sales > 0 else 0
        profit_achievement = actual_profit / estimated_profit if estimated_profit > 0 else 0

        if sales_achievement >= 1.0 and profit_lift > 0:
            performance = "优秀"
        elif sales_achievement >= 0.8 and profit_lift > -0.05:
            performance = "良好"
        elif sales_achievement >= 0.6:
            performance = "一般"
        else:
            performance = "不佳"

        insights = self._generate_insights(
            actual_sales, baseline_sales, actual_profit, baseline_profit,
            sales_achievement, profit_achievement, daily_data
        )

        suggestions = self._generate_suggestions(
            performance, sales_lift, profit_lift, promotion_config
        )

        return {
            "promotion_name": promotion_config.get("detail", "促销活动"),
            "promotion_type": promotion_config.get("type", ""),
            "duration_days": len(daily_data),
            "actual": {
                "total_sales": float(actual_sales),
                "total_revenue": float(actual_revenue),
                "total_profit": float(actual_profit),
                "avg_daily_sales": float(actual_sales / max(1, len(daily_data))),
            },
            "baseline": {
                "total_sales": float(baseline_sales),
                "total_revenue": float(baseline_revenue),
                "total_profit": float(baseline_profit),
            },
            "lift": {
                "sales_lift_rate": round(sales_lift, 4),
                "revenue_lift_rate": round(revenue_lift, 4),
                "profit_lift_rate": round(profit_lift, 4),
            },
            "target_achievement": {
                "sales_achievement_rate": round(sales_achievement, 4),
                "profit_achievement_rate": round(profit_achievement, 4),
            },
            "performance": performance,
            "daily_trend": daily_data.to_dict("records"),
            "insights": insights,
            "suggestions": suggestions,
        }

    def _generate_insights(
        self,
        actual_sales: float,
        baseline_sales: float,
        actual_profit: float,
        baseline_profit: float,
        sales_achievement: float,
        profit_achievement: float,
        daily_data: pd.DataFrame,
    ) -> List[str]:
        insights = []

        sales_lift = (actual_sales - baseline_sales) / baseline_sales if baseline_sales > 0 else 0
        if sales_lift > 0.3:
            insights.append(f"销量提升显著，达{round(sales_lift*100,1)}%，促销拉动效果明显")
        elif sales_lift > 0.1:
            insights.append(f"销量有一定提升，为{round(sales_lift*100,1)}%，促销有效果")
        else:
            insights.append(f"销量提升有限，仅{round(sales_lift*100,1)}%，促销效果不及预期")

        profit_lift = (actual_profit - baseline_profit) / baseline_profit if baseline_profit > 0 else 0
        if profit_lift > 0:
            insights.append(f"利润实现正增长({round(profit_lift*100,1)}%)，促销投入产出比良好")
        else:
            insights.append(f"利润出现下滑({round(profit_lift*100,1)}%)，需要权衡促销力度")

        if sales_achievement > 1.0:
            insights.append("销量超额完成目标，库存消耗速度可能快于预期")
        elif sales_achievement < 0.7:
            insights.append("销量未达目标，建议评估促销力度或加大宣传")

        if len(daily_data) > 3:
            first_half = daily_data.iloc[:len(daily_data)//2]["daily_sales"].mean()
            second_half = daily_data.iloc[len(daily_data)//2:]["daily_sales"].mean()
            trend = (second_half - first_half) / first_half if first_half > 0 else 0
            if trend < -0.2:
                insights.append("促销后半程热度衰减明显，建议考虑中期加码")
            elif trend > 0.2:
                insights.append("促销热度持续上升，口碑传播效应显现")

        return insights

    def _generate_suggestions(
        self,
        performance: str,
        sales_lift: float,
        profit_lift: float,
        promo_config: Dict,
    ) -> List[str]:
        suggestions = []

        if performance == "优秀":
            suggestions.append("本次促销效果优秀，可作为同类商品参考模板")
            suggestions.append("建议复制成功经验，推广至同品类其他商品")
        elif performance == "良好":
            suggestions.append("促销效果良好，可微调参数进一步优化")
            if profit_lift < 0.05:
                suggestions.append("利润提升有限，建议优化折扣力度")
        elif performance == "一般":
            suggestions.append("促销效果一般，建议重新评估目标客群")
            suggestions.append("可考虑更换促销类型或调整折扣幅度")
        else:
            suggestions.append("促销效果不佳，建议终止或大幅调整方案")
            suggestions.append("深入分析原因：是价格不敏感还是宣传不到位")

        if sales_lift > 0.2 and profit_lift < 0:
            suggestions.append("销量增长但利润下降，注意避免过度降价")
            suggestions.append("建议搭配毛利较高的关联商品做捆绑")

        return suggestions

    def real_time_adjustment(
        self,
        promotion_config: Dict,
        current_progress: Dict,
    ) -> Dict:
        target_sales = promotion_config.get("estimated_total_sales", 1000)
        duration = promotion_config.get("duration_days", 7)

        current_sales = current_progress.get("current_sales", 0)
        days_passed = current_progress.get("days_passed", 1)
        current_stock = current_progress.get("current_stock", 500)

        expected_sales_by_now = target_sales * (days_passed / duration)
        achievement_rate = current_sales / expected_sales_by_now if expected_sales_by_now > 0 else 0

        stock_run_rate = current_sales / max(1, days_passed)
        days_of_stock = current_stock / max(1, stock_run_rate)

        adjustment = "维持力度"
        new_discount = promotion_config.get("params", {}).get("discount_rate", 0.1)

        if achievement_rate < 0.7:
            if days_of_stock > duration - days_passed + 3:
                adjustment = "加大促销力度"
                new_discount = min(0.5, new_discount * 1.3)
            else:
                adjustment = "维持力度，观察趋势"
        elif achievement_rate > 1.3:
            if days_of_stock < duration - days_passed:
                adjustment = "降低促销力度，防止断货"
                new_discount = max(0.02, new_discount * 0.7)
            else:
                adjustment = "维持力度，效果良好"

        remaining_days = duration - days_passed
        projected_total = current_sales + stock_run_rate * remaining_days

        return {
            "current_achievement_rate": round(achievement_rate, 4),
            "days_of_stock_remaining": round(days_of_stock, 1),
            "current_run_rate": round(stock_run_rate, 1),
            "projected_total_sales": round(projected_total, 1),
            "adjustment_recommendation": adjustment,
            "suggested_discount_rate": round(new_discount, 4),
            "advisement": self._get_adjustment_advice(adjustment, achievement_rate, days_of_stock),
        }

    def _get_adjustment_advice(self, adjustment: str, achievement: float, stock_days: float) -> str:
        if adjustment == "加大促销力度":
            return f"当前进度仅完成{round(achievement*100,1)}%，库存还可支撑{round(stock_days,1)}天，建议适当加大折扣"
        elif adjustment == "降低促销力度，防止断货":
            return f"进度超预期({round(achievement*100,1)}%)，库存仅够{round(stock_days,1)}天，建议收窄折扣防止断货"
        elif adjustment == "维持力度，效果良好":
            return f"促销进度良好({round(achievement*100,1)}%)，库存充足，维持当前策略"
        else:
            return f"当前进度{round(achievement*100,1)}%，建议持续观察再做决策"

    def batch_recommend(self, skus: List[Dict], objective: str = "profit") -> List[Dict]:
        results = []
        for sku in skus:
            result = self.recommend_promotion(sku, objective=objective)
            results.append(result)
        return results


_default_promo_model = None


def get_promotion_optimizer_model() -> PromotionOptimizerModel:
    global _default_promo_model
    if _default_promo_model is None:
        _default_promo_model = PromotionOptimizerModel()
    return _default_promo_model
