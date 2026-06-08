from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional

from app.services.version_manager import get_version_manager
from app.services.ab_test_manager import get_ab_test_manager
from app.services.price_lock_manager import get_price_lock_manager
from app.services.audit_logger import get_audit_logger
from app.schemas.models import (
    PricingVersionCreateRequest,
    ABTestCreateRequest,
    PriceLockRequest,
    AuditLogQuery,
)

router = APIRouter(prefix="/api/v1/admin", tags=["工程管理接口"])


@router.post("/versions", summary="创建定价版本")
async def create_version(request: PricingVersionCreateRequest):
    manager = get_version_manager()
    audit = get_audit_logger()

    version = manager.create_version(
        version_name=request.version_name,
        sku_prices=request.sku_prices,
        description=request.description or "",
        is_active=request.is_active,
    )

    audit.log_version_action(
        version_id=version["version_id"],
        action="create",
        version_name=version["version_name"],
    )

    return version


@router.get("/versions", summary="获取版本列表")
async def list_versions(limit: int = 20):
    manager = get_version_manager()
    versions = manager.list_versions(limit=limit)
    return {"total": len(versions), "versions": versions}


@router.get("/versions/{version_id}", summary="获取版本详情")
async def get_version(version_id: str):
    manager = get_version_manager()
    version = manager.get_version(version_id)
    if not version:
        raise HTTPException(status_code=404, detail="版本不存在")
    return version


@router.post("/versions/{version_id}/activate", summary="激活版本")
async def activate_version(version_id: str):
    manager = get_version_manager()
    audit = get_audit_logger()

    try:
        version = manager.activate_version(version_id)
        audit.log_version_action(
            version_id=version_id,
            action="activate",
            version_name=version["version_name"],
        )
        return version
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/versions/compare/{version_a}/{version_b}", summary="版本对比")
async def compare_versions(version_a: str, version_b: str):
    manager = get_version_manager()
    try:
        result = manager.compare_versions(version_a, version_b)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/ab-tests", summary="创建A/B测试")
async def create_ab_test(request: ABTestCreateRequest):
    manager = get_ab_test_manager()
    audit = get_audit_logger()

    test = manager.create_test(
        name=request.name,
        sku_ids=request.sku_ids,
        control_group_price=request.control_group_price,
        experiment_group_price=request.experiment_group_price,
        duration_days=request.duration_days,
        description=request.description or "",
    )

    audit.log_ab_test_action(
        test_id=test["test_id"],
        action="create",
        test_name=test["name"],
    )

    return test


@router.get("/ab-tests", summary="获取A/B测试列表")
async def list_ab_tests(status: Optional[str] = None, limit: int = 20):
    manager = get_ab_test_manager()
    tests = manager.list_tests(status=status, limit=limit)
    return {"total": len(tests), "tests": tests}


@router.post("/ab-tests/{test_id}/start", summary="启动A/B测试")
async def start_ab_test(test_id: str):
    manager = get_ab_test_manager()
    audit = get_audit_logger()

    try:
        test = manager.start_test(test_id)
        audit.log_ab_test_action(
            test_id=test_id,
            action="start",
            test_name=test["name"],
        )
        return test
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/ab-tests/{test_id}/stop", summary="停止A/B测试")
async def stop_ab_test(test_id: str):
    manager = get_ab_test_manager()
    audit = get_audit_logger()

    try:
        test = manager.stop_test(test_id)
        audit.log_ab_test_action(
            test_id=test_id,
            action="stop",
            test_name=test["name"],
        )
        return test
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/ab-tests/{test_id}/results", summary="获取A/B测试结果")
async def get_ab_test_results(test_id: str):
    manager = get_ab_test_manager()
    try:
        results = manager.analyze_results(test_id)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/price-locks", summary="锁定商品价格")
async def lock_price(request: PriceLockRequest):
    manager = get_price_lock_manager()
    audit = get_audit_logger()

    lock = manager.lock_price(
        sku_id=request.sku_id,
        locked_price=request.locked_price,
        reason=request.reason or "",
        lock_until=request.lock_until,
    )

    audit.log_lock_action(
        sku_id=request.sku_id,
        action="lock",
        price=request.locked_price,
        reason=request.reason or "",
    )

    return lock


@router.delete("/price-locks/{sku_id}", summary="解锁商品价格")
async def unlock_price(sku_id: str, reason: str = ""):
    manager = get_price_lock_manager()
    audit = get_audit_logger()

    success = manager.unlock_price(sku_id, reason)
    if not success:
        raise HTTPException(status_code=404, detail="未找到该商品的价格锁定记录")

    audit.log_lock_action(
        sku_id=sku_id,
        action="unlock",
        price=0,
        reason=reason,
    )

    return {"status": "success", "sku_id": sku_id, "unlocked": True}


@router.get("/price-locks", summary="获取所有锁定价格")
async def list_locked_prices():
    manager = get_price_lock_manager()
    locks = manager.list_locked_prices()
    return {"total": len(locks), "locked_prices": locks}


@router.get("/audit-logs", summary="查询审计日志")
async def query_audit_logs(
    sku_id: Optional[str] = None,
    action_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
):
    logger = get_audit_logger()
    logs = logger.query_logs(
        sku_id=sku_id,
        action_type=action_type,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    return {"total": len(logs), "logs": logs}


@router.get("/audit-logs/statistics", summary="获取审计统计")
async def get_audit_statistics():
    logger = get_audit_logger()
    stats = logger.get_statistics()
    return stats
