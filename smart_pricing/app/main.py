from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.v1.endpoints import router as v1_router
from app.api.v1.admin import router as admin_router
from app.api.v1.data_api import router as data_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 {settings.app_name} v{settings.app_version} 启动中...")
    print("📦 初始化数据层和模型...")
    yield
    print("🛑 服务关闭中...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="大型商超智能动态定价与促销决策平台 - 基于机器学习、时序预测和强化学习",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router)
app.include_router(admin_router)
app.include_router(data_router)


@app.get("/", tags=["系统"])
async def root():
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "description": "智能动态定价与促销决策平台",
        "models": [
            "时序销量预测模型",
            "价格弹性分析模型",
            "强化学习动态定价模型",
            "促销智能组合与效果评估模型",
        ],
        "capabilities": [
            "上万SKU批量实时定价",
            "定价策略版本管理",
            "A/B测试对比",
            "人工价格锁定",
            "决策溯源审计",
        ],
        "docs": "/docs",
    }


@app.get("/health", tags=["系统"])
async def health_check():
    return {"status": "healthy", "service": "smart-pricing-platform"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
