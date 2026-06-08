from datetime import datetime
from typing import Dict, List, Optional


class PriceLockManager:
    def __init__(self):
        self.locked_prices: Dict[str, Dict] = {}

    def lock_price(
        self,
        sku_id: str,
        locked_price: float,
        reason: str = "",
        lock_until: Optional[str] = None,
        locked_by: str = "system",
    ) -> Dict:
        lock = {
            "sku_id": sku_id,
            "locked_price": locked_price,
            "is_locked": True,
            "reason": reason,
            "locked_by": locked_by,
            "locked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lock_until": lock_until,
        }
        self.locked_prices[sku_id] = lock
        return lock

    def unlock_price(self, sku_id: str, unlock_reason: str = "") -> bool:
        if sku_id in self.locked_prices:
            self.locked_prices[sku_id]["is_locked"] = False
            self.locked_prices[sku_id]["unlocked_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.locked_prices[sku_id]["unlock_reason"] = unlock_reason
            return True
        return False

    def get_locked_price(self, sku_id: str) -> Optional[Dict]:
        lock = self.locked_prices.get(sku_id)
        if lock and lock["is_locked"]:
            if lock.get("lock_until"):
                try:
                    unlock_date = datetime.strptime(lock["lock_until"], "%Y-%m-%d")
                    if datetime.now() > unlock_date:
                        lock["is_locked"] = False
                        return None
                except ValueError:
                    pass
            return lock
        return None

    def is_locked(self, sku_id: str) -> bool:
        return self.get_locked_price(sku_id) is not None

    def apply_price_with_lock(
        self,
        sku_id: str,
        calculated_price: float,
    ) -> Dict:
        lock = self.get_locked_price(sku_id)
        if lock:
            return {
                "sku_id": sku_id,
                "final_price": lock["locked_price"],
                "is_locked": True,
                "locked_price": lock["locked_price"],
                "calculated_price": calculated_price,
                "lock_reason": lock["reason"],
                "locked_by": lock["locked_by"],
                "locked_at": lock["locked_at"],
            }
        return {
            "sku_id": sku_id,
            "final_price": calculated_price,
            "is_locked": False,
            "calculated_price": calculated_price,
        }

    def list_locked_prices(self) -> List[Dict]:
        result = []
        for sku_id, lock in self.locked_prices.items():
            if lock["is_locked"]:
                result.append(lock)
        return result

    def batch_lock(self, sku_prices: Dict[str, float], reason: str = "") -> Dict:
        results = {}
        for sku_id, price in sku_prices.items():
            results[sku_id] = self.lock_price(sku_id, price, reason)
        return {
            "total_locked": len(results),
            "results": results,
        }

    def batch_unlock(self, sku_ids: List[str]) -> Dict:
        success_count = 0
        for sku_id in sku_ids:
            if self.unlock_price(sku_id):
                success_count += 1
        return {
            "requested": len(sku_ids),
            "successfully_unlocked": success_count,
        }


_default_lock_manager = None


def get_price_lock_manager() -> PriceLockManager:
    global _default_lock_manager
    if _default_lock_manager is None:
        _default_lock_manager = PriceLockManager()
    return _default_lock_manager
