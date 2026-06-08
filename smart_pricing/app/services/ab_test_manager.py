import uuid
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class ABTestStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ABTestManager:
    def __init__(self):
        self.tests: Dict[str, Dict] = {}
        self.test_results: Dict[str, Dict] = {}

    def create_test(
        self,
        name: str,
        sku_ids: List[str],
        control_group_price: Dict[str, float],
        experiment_group_price: Dict[str, float],
        duration_days: int = 7,
        description: str = "",
    ) -> Dict:
        test_id = str(uuid.uuid4())[:8]

        sku_assignment = {}
        half = len(sku_ids) // 2
        for i, sku_id in enumerate(sku_ids):
            group = "control" if i < half else "experiment"
            sku_assignment[sku_id] = {
                "group": group,
                "price": control_group_price.get(sku_id) if group == "control" else experiment_group_price.get(sku_id),
            }

        test = {
            "test_id": test_id,
            "name": name,
            "description": description,
            "sku_count": len(sku_ids),
            "duration_days": duration_days,
            "status": ABTestStatus.DRAFT,
            "sku_assignment": sku_assignment,
            "control_group_price": control_group_price,
            "experiment_group_price": experiment_group_price,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "started_at": None,
            "ended_at": None,
        }

        self.tests[test_id] = test
        return test

    def start_test(self, test_id: str) -> Dict:
        if test_id not in self.tests:
            raise ValueError(f"测试不存在: {test_id}")

        test = self.tests[test_id]
        if test["status"] == ABTestStatus.RUNNING:
            return test

        test["status"] = ABTestStatus.RUNNING
        test["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return test

    def stop_test(self, test_id: str) -> Dict:
        if test_id not in self.tests:
            raise ValueError(f"测试不存在: {test_id}")

        test = self.tests[test_id]
        test["status"] = ABTestStatus.CANCELLED
        test["ended_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return test

    def record_data(self, test_id: str, sku_id: str, sales_data: Dict) -> bool:
        if test_id not in self.tests:
            return False

        if test_id not in self.test_results:
            self.test_results[test_id] = {}

        if sku_id not in self.test_results[test_id]:
            self.test_results[test_id][sku_id] = []

        self.test_results[test_id][sku_id].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **sales_data,
        })
        return True

    def analyze_results(self, test_id: str) -> Dict:
        if test_id not in self.tests:
            raise ValueError(f"测试不存在: {test_id}")

        test = self.tests[test_id]
        results = self.test_results.get(test_id, {})

        control_sales = []
        control_revenue = []
        experiment_sales = []
        experiment_revenue = []

        for sku_id, sku_data in results.items():
            assignment = test["sku_assignment"].get(sku_id, {})
            group = assignment.get("group", "control")

            for record in sku_data:
                sales = record.get("sales_volume", 0)
                revenue = record.get("revenue", 0)

                if group == "control":
                    control_sales.append(sales)
                    control_revenue.append(revenue)
                else:
                    experiment_sales.append(sales)
                    experiment_revenue.append(revenue)

        control_avg_sales = sum(control_sales) / len(control_sales) if control_sales else 0
        experiment_avg_sales = sum(experiment_sales) / len(experiment_sales) if experiment_sales else 0
        control_avg_revenue = sum(control_revenue) / len(control_revenue) if control_revenue else 0
        experiment_avg_revenue = sum(experiment_revenue) / len(experiment_revenue) if experiment_revenue else 0

        sales_lift = (
            (experiment_avg_sales - control_avg_sales) / control_avg_sales
            if control_avg_sales > 0 else 0
        )
        revenue_lift = (
            (experiment_avg_revenue - control_avg_revenue) / control_avg_revenue
            if control_avg_revenue > 0 else 0
        )

        if len(control_sales) > 0 and len(experiment_sales) > 0:
            import math
            n1, n2 = len(control_sales), len(experiment_sales)
            mean1, mean2 = control_avg_sales, experiment_avg_sales
            var1 = sum((x - mean1) ** 2 for x in control_sales) / max(1, n1 - 1)
            var2 = sum((x - mean2) ** 2 for x in experiment_sales) / max(1, n2 - 1)

            pooled_se = math.sqrt(var1 / n1 + var2 / n2)
            t_stat = (mean2 - mean1) / pooled_se if pooled_se > 0 else 0
            p_value = min(1.0, max(0.0, 1 - abs(t_stat) / 3))

            statistical_significance = 1 - p_value
        else:
            statistical_significance = 0.5

        if statistical_significance > 0.95:
            conclusion = "实验组效果显著优于对照组" if sales_lift > 0 else "实验组效果显著差于对照组"
            confidence = "高"
        elif statistical_significance > 0.8:
            conclusion = "实验组有一定优势" if sales_lift > 0 else "实验组有一定劣势"
            confidence = "中"
        else:
            conclusion = "两组差异不显著"
            confidence = "低"

        recommendation = ""
        if sales_lift > 0.05 and statistical_significance > 0.9:
            recommendation = "建议采纳实验方案，全量推广"
        elif sales_lift < -0.05 and statistical_significance > 0.9:
            recommendation = "建议维持现有方案，不采纳实验方案"
        else:
            recommendation = "建议延长测试周期或扩大样本量"

        return {
            "test_id": test_id,
            "test_name": test["name"],
            "status": test["status"],
            "sample_size": {
                "control_group": len(control_sales),
                "experiment_group": len(experiment_sales),
            },
            "metrics": {
                "control_avg_sales": round(control_avg_sales, 2),
                "experiment_avg_sales": round(experiment_avg_sales, 2),
                "sales_lift_pct": round(sales_lift * 100, 2),
                "control_avg_revenue": round(control_avg_revenue, 2),
                "experiment_avg_revenue": round(experiment_avg_revenue, 2),
                "revenue_lift_pct": round(revenue_lift * 100, 2),
            },
            "statistical_significance": round(statistical_significance, 4),
            "confidence_level": confidence,
            "conclusion": conclusion,
            "recommendation": recommendation,
        }

    def list_tests(self, status: Optional[str] = None, limit: int = 20) -> List[Dict]:
        tests = []
        for test_id, test in self.tests.items():
            if status and test["status"] != status:
                continue
            tests.append({
                "test_id": test_id,
                "name": test["name"],
                "status": test["status"],
                "sku_count": test["sku_count"],
                "duration_days": test["duration_days"],
                "created_at": test["created_at"],
            })
        return sorted(tests, key=lambda x: x["created_at"], reverse=True)[:limit]

    def get_test(self, test_id: str) -> Optional[Dict]:
        return self.tests.get(test_id)


_default_ab_manager = None


def get_ab_test_manager() -> ABTestManager:
    global _default_ab_manager
    if _default_ab_manager is None:
        _default_ab_manager = ABTestManager()
    return _default_ab_manager
