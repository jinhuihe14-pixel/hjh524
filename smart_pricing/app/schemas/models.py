from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class BaseSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class SKUBase(BaseSchema):
    sku_id: str
    category: Optional[str] = None
    product_name: Optional[str] = None
    base_price: Optional[float] = None
    cost: Optional[float] = None


class SalesForecastRequest(BaseSchema):
    sku_id: str
    category: str = "食品"
    days: int = Field(default=14, ge=1, le=90)
    price: Optional[float] = None
    stock_level: Optional[int] = None


class SalesForecastBatchRequest(BaseSchema):
    skus: List[Dict[str, Any]]
    days: int = Field(default=14, ge=1, le=90)


class DailyPrediction(BaseSchema):
    date: str
    predicted_sales: float
    lower_bound: float
    upper_bound: float
    is_weekend: int
    is_holiday: int


class SalesForecastResponse(BaseSchema):
    sku_id: str
    forecast_days: int
    total_predicted_sales: float
    daily_predictions: List[DailyPrediction]
    model_type: str
    confidence: float


class PriceAnalysisRequest(BaseSchema):
    sku_id: str
    new_price: float
    current_price: Optional[float] = None
    current_sales: Optional[float] = None


class PriceAnalysisBatchRequest(BaseSchema):
    items: List[Dict[str, Any]]


class OptimalPriceRequest(BaseSchema):
    sku_id: str
    objective: str = "profit"
    constraints: Optional[Dict[str, Any]] = None


class PriceAnalysisResponse(BaseSchema):
    sku_id: str
    sku_type: str
    price_sensitivity: str
    current_price: float
    new_price: float
    price_change_pct: float
    predicted_sales: float
    sales_change_pct: float
    predicted_profit: float
    profit_change_pct: float
    recommendation: str
    risk_level: str
    confidence: float


class DynamicPricingRequest(BaseSchema):
    sku_id: str
    stock_quantity: int
    days_to_expiry: Optional[int] = None
    competitor_price: Optional[float] = None
    time_period: int = 0
    constraints: Optional[Dict[str, Any]] = None


class DynamicPricingBatchRequest(BaseSchema):
    skus: List[Dict[str, Any]]


class TieredPricing(BaseSchema):
    tier: str
    days_to_expiry_threshold: int
    price: float
    discount_rate: float
    margin_rate: float


class TimeBasedPricing(BaseSchema):
    time_period: str
    hours: str
    price: float
    discount_rate: float
    adjustment_factor: float


class DynamicPricingResponse(BaseSchema):
    sku_id: str
    base_price: float
    cost: float
    optimal_price: float
    discount_rate: float
    margin_rate: float
    expected_daily_sales: float
    expected_daily_profit: float
    stock_quantity: int
    days_to_expiry: Optional[int] = None
    competitor_price: Optional[float] = None
    pricing_strategy: str
    tiered_pricing: List[TieredPricing]
    time_based_pricing: List[TimeBasedPricing]
    confidence: float


class PromotionRecommendRequest(BaseSchema):
    sku_id: str
    base_price: float
    cost: float
    base_demand: float = 50
    category: str = "食品"
    stock_quantity: int = 200
    objective: str = "profit"
    duration_days: int = 7
    budget: Optional[float] = None


class PromotionRecommendBatchRequest(BaseSchema):
    skus: List[Dict[str, Any]]
    objective: str = "profit"


class PromotionRecommendation(BaseSchema):
    rank: int
    type: str
    type_code: str
    detail: str
    estimated_total_sales: float
    estimated_revenue: float
    estimated_profit: float
    estimated_cost: float
    profit_increment: float
    sales_increment_rate: float
    roi: float
    risk_level: str


class PromotionRecommendResponse(BaseSchema):
    sku_id: str
    category: str
    objective: str
    duration_days: int
    base_price: float
    cost: float
    recommendations: List[PromotionRecommendation]
    baseline: Dict[str, Any]


class PromotionEvaluationRequest(BaseSchema):
    promotion_config: Dict[str, Any]
    actual_data: List[Dict[str, Any]]
    baseline_data: List[Dict[str, Any]]


class PromotionAdjustmentRequest(BaseSchema):
    promotion_config: Dict[str, Any]
    current_progress: Dict[str, Any]


class PromotionAdjustmentResponse(BaseSchema):
    current_achievement_rate: float
    days_of_stock_remaining: float
    current_run_rate: float
    projected_total_sales: float
    adjustment_recommendation: str
    suggested_discount_rate: float
    advisement: str


class ABTestCreateRequest(BaseSchema):
    name: str
    sku_ids: List[str]
    control_group_price: Dict[str, float]
    experiment_group_price: Dict[str, float]
    duration_days: int = 7
    description: Optional[str] = None


class PriceLockRequest(BaseSchema):
    sku_id: str
    locked_price: float
    reason: Optional[str] = None
    lock_until: Optional[str] = None


class AuditLogQuery(BaseSchema):
    sku_id: Optional[str] = None
    action_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: int = 100


class PricingVersionCreateRequest(BaseSchema):
    version_name: str
    sku_prices: Dict[str, float]
    description: Optional[str] = None
    is_active: bool = False
