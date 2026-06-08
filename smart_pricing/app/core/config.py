from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "智能动态定价与促销决策平台"
    app_version: str = "1.0.0"
    debug: bool = True

    default_forecast_days: int = 14
    default_price_history_days: int = 90
    max_pricing_batch_size: int = 1000

    rl_learning_rate: float = 0.01
    rl_discount_factor: float = 0.95
    rl_exploration_rate: float = 0.1

    min_margin_ratio: float = 0.15
    max_discount_ratio: float = 0.7

    class Config:
        env_file = ".env"


settings = Settings()
