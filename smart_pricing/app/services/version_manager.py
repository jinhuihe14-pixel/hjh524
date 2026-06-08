import uuid
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque


class PricingVersionManager:
    def __init__(self):
        self.versions: Dict[str, Dict] = {}
        self.version_history: List[Dict] = []
        self.active_version: Optional[str] = None

    def create_version(
        self,
        version_name: str,
        sku_prices: Dict[str, float],
        description: str = "",
        is_active: bool = False,
    ) -> Dict:
        version_id = str(uuid.uuid4())[:8]
        version = {
            "version_id": version_id,
            "version_name": version_name,
            "description": description,
            "sku_count": len(sku_prices),
            "sku_prices": sku_prices.copy(),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_active": is_active,
            "status": "draft",
        }

        self.versions[version_id] = version
        self.version_history.append({
            "version_id": version_id,
            "version_name": version_name,
            "created_at": version["created_at"],
            "is_active": is_active,
            "sku_count": len(sku_prices),
        })

        if is_active:
            self._deactivate_others(version_id)
            self.active_version = version_id

        return version

    def activate_version(self, version_id: str) -> Dict:
        if version_id not in self.versions:
            raise ValueError(f"版本不存在: {version_id}")

        self._deactivate_others(version_id)
        self.versions[version_id]["is_active"] = True
        self.versions[version_id]["status"] = "active"
        self.active_version = version_id

        for v in self.version_history:
            if v["version_id"] == version_id:
                v["is_active"] = True
            else:
                v["is_active"] = False

        return self.versions[version_id]

    def _deactivate_others(self, keep_id: str):
        for vid, version in self.versions.items():
            if vid != keep_id:
                version["is_active"] = False
                if version["status"] == "active":
                    version["status"] = "historical"

    def get_version(self, version_id: str) -> Optional[Dict]:
        return self.versions.get(version_id)

    def get_active_version(self) -> Optional[Dict]:
        if self.active_version:
            return self.versions.get(self.active_version)
        return None

    def list_versions(self, limit: int = 20) -> List[Dict]:
        return self.version_history[-limit:][::-1]

    def get_sku_price(self, sku_id: str, version_id: Optional[str] = None) -> Optional[float]:
        if version_id is None:
            version_id = self.active_version

        if version_id is None:
            return None

        version = self.versions.get(version_id)
        if version is None:
            return None

        return version["sku_prices"].get(sku_id)

    def compare_versions(self, version_id_a: str, version_id_b: str) -> Dict:
        ver_a = self.versions.get(version_id_a)
        ver_b = self.versions.get(version_id_b)

        if ver_a is None or ver_b is None:
            raise ValueError("版本不存在")

        prices_a = ver_a["sku_prices"]
        prices_b = ver_b["sku_prices"]

        all_skus = set(prices_a.keys()) | set(prices_b.keys())

        changes = []
        added = []
        removed = []

        for sku in all_skus:
            pa = prices_a.get(sku)
            pb = prices_b.get(sku)

            if pa is None and pb is not None:
                added.append({"sku_id": sku, "new_price": pb})
            elif pa is not None and pb is None:
                removed.append({"sku_id": sku, "old_price": pa})
            elif pa != pb:
                change_pct = (pb - pa) / pa if pa > 0 else 0
                changes.append({
                    "sku_id": sku,
                    "old_price": pa,
                    "new_price": pb,
                    "change_pct": round(change_pct, 4),
                })

        avg_change = sum(c["change_pct"] for c in changes) / len(changes) if changes else 0
        increased = [c for c in changes if c["change_pct"] > 0]
        decreased = [c for c in changes if c["change_pct"] < 0]

        return {
            "version_a": version_id_a,
            "version_b": version_id_b,
            "total_changes": len(changes),
            "price_increases": len(increased),
            "price_decreases": len(decreased),
            "avg_change_pct": round(avg_change, 4),
            "added_skus": len(added),
            "removed_skus": len(removed),
            "changes": changes[:50],
        }

    def rollback(self, version_id: str) -> Dict:
        return self.activate_version(version_id)


_default_version_manager = None


def get_version_manager() -> PricingVersionManager:
    global _default_version_manager
    if _default_version_manager is None:
        _default_version_manager = PricingVersionManager()
    return _default_version_manager
