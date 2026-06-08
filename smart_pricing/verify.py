import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  模块导入验证")
print("=" * 60)

try:
    from app.core.config import settings
    print(f"✅ 配置模块 - {settings.app_name} v{settings.app_version}")
except Exception as e:
    print(f"❌ 配置模块 - {e}")

try:
    from app.data.data_generator import get_data_generator
    gen = get_data_generator()
    catalog = gen.generate_sku_catalog()
    print(f"✅ 数据生成器 - 生成 {len(catalog)} 个SKU")
except Exception as e:
    print(f"❌ 数据生成器 - {e}")
    import traceback
    traceback.print_exc()

try:
    from app.data.feature_engineering import get_feature_engineer
    fe = get_feature_engineer()
    print(f"✅ 特征工程模块")
except Exception as e:
    print(f"❌ 特征工程模块 - {e}")

try:
    from app.data.data_preprocessing import get_data_preprocessor
    dp = get_data_preprocessor()
    print(f"✅ 数据预处理模块")
except Exception as e:
    print(f"❌ 数据预处理模块 - {e}")

try:
    from app.models.sales_forecast_model import get_sales_forecast_model
    sf = get_sales_forecast_model()
    print(f"✅ 时序销量预测模型")
except Exception as e:
    print(f"❌ 时序销量预测模型 - {e}")
    import traceback
    traceback.print_exc()

try:
    from app.models.price_elasticity_model import get_price_elasticity_model
    pe = get_price_elasticity_model()
    print(f"✅ 价格弹性分析模型")
except Exception as e:
    print(f"❌ 价格弹性分析模型 - {e}")
    import traceback
    traceback.print_exc()

try:
    from app.models.rl_pricing_model import get_rl_pricing_model
    rl = get_rl_pricing_model()
    print(f"✅ 强化学习动态定价模型")
except Exception as e:
    print(f"❌ 强化学习动态定价模型 - {e}")
    import traceback
    traceback.print_exc()

try:
    from app.models.promotion_optimizer_model import get_promotion_optimizer_model
    promo = get_promotion_optimizer_model()
    print(f"✅ 促销智能组合与效果评估模型")
except Exception as e:
    print(f"❌ 促销智能组合与效果评估模型 - {e}")
    import traceback
    traceback.print_exc()

try:
    from app.services.version_manager import get_version_manager
    vm = get_version_manager()
    print(f"✅ 版本管理服务")
except Exception as e:
    print(f"❌ 版本管理服务 - {e}")

try:
    from app.services.ab_test_manager import get_ab_test_manager
    ab = get_ab_test_manager()
    print(f"✅ A/B测试服务")
except Exception as e:
    print(f"❌ A/B测试服务 - {e}")

try:
    from app.services.price_lock_manager import get_price_lock_manager
    pl = get_price_lock_manager()
    print(f"✅ 价格锁定服务")
except Exception as e:
    print(f"❌ 价格锁定服务 - {e}")

try:
    from app.services.audit_logger import get_audit_logger
    al = get_audit_logger()
    print(f"✅ 审计日志服务")
except Exception as e:
    print(f"❌ 审计日志服务 - {e}")

print()
print("=" * 60)
print("  核心功能快速测试")
print("=" * 60)

try:
    from app.models.sales_forecast_model import get_sales_forecast_model
    sf = get_sales_forecast_model()
    future_dates = sf._generate_future_dates(7, "SKU000001")
    result = sf.predict("SKU000001", "食品", future_dates, days=7)
    print(f"✅ 销量预测测试 - 预测{result['forecast_days']}天，总销量{result['total_predicted_sales']}")
except Exception as e:
    print(f"❌ 销量预测测试 - {e}")

try:
    from app.models.price_elasticity_model import get_price_elasticity_model
    pe = get_price_elasticity_model()
    result = pe.analyze_price_change("test_sku", 90.0, current_price=100.0, current_sales=50)
    print(f"✅ 价格弹性测试 - 降价10%，销量变化{result['sales_change_pct']}%")
except Exception as e:
    print(f"❌ 价格弹性测试 - {e}")

try:
    from app.models.rl_pricing_model import get_rl_pricing_model
    rl = get_rl_pricing_model()
    result = rl.get_optimal_price("test_sku", stock_quantity=200, days_to_expiry=15)
    print(f"✅ 动态定价测试 - 最优价格{result['optimal_price']}，策略：{result['pricing_strategy']}")
except Exception as e:
    print(f"❌ 动态定价测试 - {e}")

try:
    from app.models.promotion_optimizer_model import get_promotion_optimizer_model
    promo = get_promotion_optimizer_model()
    result = promo.recommend_promotion({
        "sku_id": "test_sku",
        "base_price": 100,
        "cost": 60,
        "base_demand": 50,
        "category": "食品",
        "stock_quantity": 200,
    }, objective="profit")
    print(f"✅ 促销推荐测试 - 推荐 {len(result['recommendations'])} 个方案")
    for rec in result['recommendations'][:2]:
        print(f"   - {rec['type']}: {rec['detail']}，ROI: {rec['roi']}")
except Exception as e:
    print(f"❌ 促销推荐测试 - {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("  验证完成")
print("=" * 60)
