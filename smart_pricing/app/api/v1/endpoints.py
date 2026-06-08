from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from app.models.sales_forecast_model import get_sales_forecast_model
from app.models.price_elasticity_model import get_price_elasticity_model
from app.models.rl_pricing_model import get_rl_pricing_model
from app.models.promotion_optimizer_model import get_promotion_optimizer_model
from app.data.data_generator import get_data_generator
from app.data.feature_engineering import get_feature_engineer
from app.data.data_preprocessing import get_data_preprocessor
from app.services.price_lock_manager import get_price_lock_manager
from app.services.audit_logger import get_audit_logger
from app.schemas.models import (
    SalesForecastRequest,
    SalesForecastBatchRequest,
    SalesForecastResponse,
    PriceAnalysisRequest,
    PriceAnalysisBatchRequest,
    PriceAnalysisResponse,
    OptimalPriceRequest,
    DynamicPricingRequest,
    DynamicPricingBatchRequest,
    DynamicPricingResponse,
    PromotionRecommendRequest,
    PromotionRecommendBatchRequest,
    PromotionRecommendResponse,
    PromotionEvaluationRequest,
    PromotionAdjustmentRequest,
    PromotionAdjustmentResponse,
)

router = APIRouter(prefix="/api/v1", tags=["智能定价核心接口"])


@router.post("/forecast/sales", response_model=SalesForecastResponse, summary="单品销量预测")
async def forecast_sales(request: SalesForecastRequest):
    model = get_sales_forecast_model()
    future_dates = model._generate_future_dates(request.days, request.sku_id)

    result = model.predict(
        sku_id=request.sku_id,
        category=request.category,
        future_dates=future_dates,
        price=request.price,
        stock_level=request.stock_level,
        days=request.days,
    )
    result["sku_id"] = request.sku_id
    return result


@router.post("/forecast/sales/batch", summary="批量销量预测")
async def forecast_sales_batch(request: SalesForecastBatchRequest):
    model = get_sales_forecast_model()
    results = model.predict_batch(request.skus, days=request.days)
    return {"total": len(results), "predictions": results}


@router.post("/elasticity/analyze", response_model=PriceAnalysisResponse, summary="价格弹性分析")
async def analyze_price_change(request: PriceAnalysisRequest):
    model = get_price_elasticity_model()
    result = model.analyze_price_change(
        sku_id=request.sku_id,
        new_price=request.new_price,
        current_price=request.current_price,
        current_sales=request.current_sales,
    )
    return result


@router.post("/elasticity/analyze/batch", summary="批量价格弹性分析")
async def analyze_price_change_batch(request: PriceAnalysisBatchRequest):
    model = get_price_elasticity_model()
    results = model.batch_analyze(request.items)
    return {"total": len(results), "analyses": results}


@router.post("/elasticity/optimal-price", summary="计算最优价格")
async def get_optimal_price(request: OptimalPriceRequest):
    model = get_price_elasticity_model()
    result = model.get_optimal_price(
        sku_id=request.sku_id,
        objective=request.objective,
        constraints=request.constraints,
    )
    return result


@router.post("/pricing/dynamic", response_model=DynamicPricingResponse, summary="强化学习动态定价")
async def dynamic_pricing(request: DynamicPricingRequest):
    model = get_rl_pricing_model()
    lock_manager = get_price_lock_manager()
    audit_logger = get_audit_logger()

    result = model.get_optimal_price(
        sku_id=request.sku_id,
        stock_quantity=request.stock_quantity,
        days_to_expiry=request.days_to_expiry,
        competitor_price=request.competitor_price,
        time_period=request.time_period,
        constraints=request.constraints,
    )

    lock_result = lock_manager.apply_price_with_lock(
        request.sku_id,
        result["optimal_price"],
    )

    if lock_result["is_locked"]:
        result["optimal_price"] = lock_result["locked_price"]
        result["price_locked"] = True
        result["lock_info"] = {
            "locked_price": lock_result["locked_price"],
            "lock_reason": lock_result["lock_reason"],
            "locked_by": lock_result["locked_by"],
        }
    else:
        result["price_locked"] = False

    return result


@router.post("/pricing/dynamic/batch", summary="批量动态定价")
async def dynamic_pricing_batch(request: DynamicPricingBatchRequest):
    model = get_rl_pricing_model()
    lock_manager = get_price_lock_manager()

    results = model.batch_pricing(request.skus)

    for result in results:
        lock_result = lock_manager.apply_price_with_lock(
            result["sku_id"],
            result["optimal_price"],
        )
        if lock_result["is_locked"]:
            result["optimal_price"] = lock_result["locked_price"]
            result["price_locked"] = True
            result["original_calculated_price"] = lock_result["calculated_price"]
        else:
            result["price_locked"] = False

    return {"total": len(results), "pricing_results": results}


@router.post("/promotion/recommend", response_model=PromotionRecommendResponse, summary="促销组合推荐")
async def recommend_promotion(request: PromotionRecommendRequest):
    model = get_promotion_optimizer_model()

    sku_info = {
        "sku_id": request.sku_id,
        "base_price": request.base_price,
        "cost": request.cost,
        "base_demand": request.base_demand,
        "category": request.category,
        "stock_quantity": request.stock_quantity,
    }

    result = model.recommend_promotion(
        sku_info=sku_info,
        objective=request.objective,
        budget=request.budget,
        duration_days=request.duration_days,
    )
    return result


@router.post("/promotion/recommend/batch", summary="批量促销推荐")
async def recommend_promotion_batch(request: PromotionRecommendBatchRequest):
    model = get_promotion_optimizer_model()
    results = model.batch_recommend(request.skus, objective=request.objective)
    return {"total": len(results), "recommendations": results}


@router.post("/promotion/evaluate", summary="促销效果评估")
async def evaluate_promotion(request: PromotionEvaluationRequest):
    model = get_promotion_optimizer_model()
    import pandas as pd

    actual_df = pd.DataFrame(request.actual_data)
    baseline_df = pd.DataFrame(request.baseline_data)

    result = model.evaluate_promotion(
        promotion_config=request.promotion_config,
        actual_data=actual_df,
        baseline_data=baseline_df,
    )
    return result


@router.post("/promotion/adjust", response_model=PromotionAdjustmentResponse, summary="促销实时调价建议")
async def adjust_promotion(request: PromotionAdjustmentRequest):
    model = get_promotion_optimizer_model()
    result = model.real_time_adjustment(
        promotion_config=request.promotion_config,
        current_progress=request.current_progress,
    )
    return result


@router.post("/model/train-all", summary="训练所有模型")
async def train_all_models(days: int = 365):
    data_gen = get_data_generator()
    feature_engineer = get_feature_engineer()
    preprocessor = get_data_preprocessor()

    sku_catalog = data_gen.generate_sku_catalog()
    sales_data = data_gen.generate_sales_history(sku_catalog, days=days)
    sales_data_clean = preprocessor.clean_sales_data(sales_data)
    features = feature_engineer.build_features(sales_data_clean)

    forecast_model = get_sales_forecast_model()
    forecast_result = forecast_model.train(features)

    elasticity_model = get_price_elasticity_model()
    elasticity_result = elasticity_model.train(features)

    rl_model = get_rl_pricing_model()
    rl_result = rl_model.train(features, episodes=50)

    promo_model = get_promotion_optimizer_model()
    promo_result = promo_model.train(features)

    return {
        "status": "success",
        "training_days": days,
        "sku_count": len(sku_catalog),
        "sales_forecast": forecast_result,
        "price_elasticity": elasticity_result,
        "rl_pricing": rl_result,
        "promotion_optimizer": promo_result,
    }
