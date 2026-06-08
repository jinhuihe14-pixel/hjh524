from fastapi import APIRouter, Query
from typing import Optional
import pandas as pd

from app.data.data_generator import get_data_generator, DataGenerator
from app.data.feature_engineering import get_feature_engineer
from app.data.data_preprocessing import get_data_preprocessor

router = APIRouter(prefix="/api/v1/data", tags=["数据管理"])


@router.post("/generate/sku-catalog", summary="生成SKU商品目录")
async def generate_sku_catalog(seed: int = 42):
    generator = get_data_generator()
    catalog = generator.generate_sku_catalog()
    return {
        "total_skus": len(catalog),
        "categories": list(catalog["category"].unique()),
        "data": catalog.to_dict("records"),
    }


@router.post("/generate/sales-history", summary="生成历史销量数据")
async def generate_sales_history(days: int = 365, seed: int = 42):
    generator = DataGenerator(seed=seed)
    catalog = generator.generate_sku_catalog()
    sales = generator.generate_sales_history(catalog, days=days)
    return {
        "days": days,
        "total_records": len(sales),
        "sku_count": sales["sku_id"].nunique(),
        "date_range": {
            "start": sales["date"].min(),
            "end": sales["date"].max(),
        },
        "data_preview": sales.head(20).to_dict("records"),
    }


@router.post("/generate/competitor-prices", summary="生成竞品价格数据")
async def generate_competitor_prices(days: int = 90):
    generator = get_data_generator()
    catalog = generator.generate_sku_catalog()
    comp = generator.generate_competitor_prices(catalog, days=days)
    return {
        "days": days,
        "total_records": len(comp),
        "competitors": list(comp["competitor"].unique()),
        "data_preview": comp.head(20).to_dict("records"),
    }


@router.post("/generate/inventory", summary="生成库存数据")
async def generate_inventory():
    generator = get_data_generator()
    catalog = generator.generate_sku_catalog()
    inventory = generator.generate_inventory_data(catalog)
    return {
        "total_skus": len(inventory),
        "warehouses": list(inventory["warehouse"].unique()),
        "data_preview": inventory.head(20).to_dict("records"),
    }


@router.post("/generate/full-dataset", summary="生成完整数据集")
async def generate_full_dataset(sales_days: int = 365, competitor_days: int = 90, seed: int = 42):
    generator = DataGenerator(seed=seed)
    catalog = generator.generate_sku_catalog()
    sales = generator.generate_sales_history(catalog, days=sales_days)
    comp = generator.generate_competitor_prices(catalog, days=competitor_days)
    inventory = generator.generate_inventory_data(catalog)

    preprocessor = get_data_preprocessor()
    sales_clean = preprocessor.clean_sales_data(sales)

    feature_engineer = get_feature_engineer()
    features = feature_engineer.build_features(sales_clean, inventory, comp)

    return {
        "status": "success",
        "dataset_info": {
            "sku_count": len(catalog),
            "sales_records": len(sales_clean),
            "sales_days": sales_days,
            "competitor_records": len(comp),
            "feature_count": len(features.columns),
            "feature_columns": list(features.columns),
        },
    }


@router.get("/preview/sales-stats", summary="获取销量统计摘要")
async def get_sales_stats(days: int = 30):
    generator = get_data_generator()
    catalog = generator.generate_sku_catalog()
    sales = generator.generate_sales_history(catalog, days=days)

    stats = {
        "total_records": len(sales),
        "total_sales_volume": int(sales["sales_volume"].sum()),
        "avg_daily_sales": float(sales.groupby("date")["sales_volume"].sum().mean()),
        "avg_price": float(sales["price"].mean()),
        "avg_discount": float(sales["discount_rate"].mean()),
    }

    cat_stats = sales.groupby("category").agg(
        total_sales=("sales_volume", "sum"),
        avg_price=("price", "mean"),
        avg_discount=("discount_rate", "mean"),
    ).to_dict("index")

    return {
        "overall": stats,
        "by_category": cat_stats,
    }
