import uuid
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque


class AuditLogger:
    def __init__(self, max_logs: int = 10000):
        self.logs: deque = deque(maxlen=max_logs)

    def log_pricing_decision(
        self,
        sku_id: str,
        old_price: float,
        new_price: float,
        reason: str,
        model_type: str,
        metadata: Optional[Dict] = None,
        operator: str = "system",
    ) -> Dict:
        log = {
            "log_id": str(uuid.uuid4())[:12],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action_type": "pricing_change",
            "sku_id": sku_id,
            "old_price": old_price,
            "new_price": new_price,
            "price_change_pct": round((new_price - old_price) / old_price, 4) if old_price > 0 else 0,
            "reason": reason,
            "model_type": model_type,
            "operator": operator,
            "metadata": metadata or {},
        }
        self.logs.append(log)
        return log

    def log_promotion_decision(
        self,
        sku_id: str,
        promotion_type: str,
        action: str,
        detail: str,
        operator: str = "system",
        metadata: Optional[Dict] = None,
    ) -> Dict:
        log = {
            "log_id": str(uuid.uuid4())[:12],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action_type": "promotion_change",
            "sku_id": sku_id,
            "promotion_type": promotion_type,
            "action": action,
            "detail": detail,
            "operator": operator,
            "metadata": metadata or {},
        }
        self.logs.append(log)
        return log

    def log_lock_action(
        self,
        sku_id: str,
        action: str,
        price: float,
        reason: str,
        operator: str = "system",
    ) -> Dict:
        log = {
            "log_id": str(uuid.uuid4())[:12],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action_type": "price_lock",
            "sku_id": sku_id,
            "lock_action": action,
            "price": price,
            "reason": reason,
            "operator": operator,
        }
        self.logs.append(log)
        return log

    def log_version_action(
        self,
        version_id: str,
        action: str,
        version_name: str,
        operator: str = "system",
        metadata: Optional[Dict] = None,
    ) -> Dict:
        log = {
            "log_id": str(uuid.uuid4())[:12],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action_type": "version_change",
            "version_id": version_id,
            "version_action": action,
            "version_name": version_name,
            "operator": operator,
            "metadata": metadata or {},
        }
        self.logs.append(log)
        return log

    def log_ab_test_action(
        self,
        test_id: str,
        action: str,
        test_name: str,
        operator: str = "system",
    ) -> Dict:
        log = {
            "log_id": str(uuid.uuid4())[:12],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action_type": "ab_test",
            "test_id": test_id,
            "ab_action": action,
            "test_name": test_name,
            "operator": operator,
        }
        self.logs.append(log)
        return log

    def query_logs(
        self,
        sku_id: Optional[str] = None,
        action_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        results = []

        for log in reversed(self.logs):
            if sku_id and log.get("sku_id") != sku_id:
                continue
            if action_type and log.get("action_type") != action_type:
                continue
            if start_date and log["timestamp"] < start_date:
                continue
            if end_date and log["timestamp"] > end_date + " 23:59:59":
                continue

            results.append(log)
            if len(results) >= limit:
                break

        return results

    def get_statistics(self) -> Dict:
        action_counts = {}
        sku_counts = {}
        model_counts = {}

        for log in self.logs:
            action = log.get("action_type", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1

            sku = log.get("sku_id")
            if sku:
                sku_counts[sku] = sku_counts.get(sku, 0) + 1

            model = log.get("model_type")
            if model:
                model_counts[model] = model_counts.get(model, 0) + 1

        top_skus = sorted(sku_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_logs": len(self.logs),
            "action_distribution": action_counts,
            "model_distribution": model_counts,
            "top_skus_by_changes": [
                {"sku_id": sku, "change_count": count}
                for sku, count in top_skus
            ],
        }

    def export_logs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict]:
        return self.query_logs(start_date=start_date, end_date=end_date, limit=10000)


_default_audit_logger = None


def get_audit_logger() -> AuditLogger:
    global _default_audit_logger
    if _default_audit_logger is None:
        _default_audit_logger = AuditLogger()
    return _default_audit_logger
