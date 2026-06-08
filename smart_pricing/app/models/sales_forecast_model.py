import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose


class SalesForecastModel:
    def __init__(self):
        self.models: Dict[str, object] = {}
        self.feature_cols: List[str] = []
        self.is_trained: Dict[str, bool] = {}
        self.category_configs = {
            "生鲜": {"model_type": "hybrid", "seasonal_period": 7, "short_shelf": True},
            "食品": {"model_type": "ml", "seasonal_period": 7, "short_shelf": False},
            "日化": {"model_type": "ml", "seasonal_period": 7, "short_shelf": False},
            "家电": {"model_type": "ml", "seasonal_period": 30, "short_shelf": False},
            "服饰": {"model_type": "hybrid", "seasonal_period": 30, "short_shelf": False},
        }

    def train(
        self,
        sales_data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict:
        if feature_cols is None:
            feature_cols = self._default_feature_cols(sales_data)

        self.feature_cols = feature_cols
        results = {}

        for category in sales_data["category"].unique():
            cat_data = sales_data[sales_data["category"] == category].copy()
            cat_skus = cat_data["sku_id"].unique()

            category_models = {}
            for sku_id in cat_skus[:10]:
                sku_data = cat_data[cat_data["sku_id"] == sku_id].copy()
                sku_data = sku_data.sort_values("date")

                if len(sku_data) < 30:
                    continue

                model = self._train_sku_model(sku_data, category)
                category_models[sku_id] = model

            self.models[category] = category_models
            self.is_trained[category] = len(category_models) > 0
            results[category] = {"skus_trained": len(category_models)}

        return results

    def _train_sku_model(self, sku_data: pd.DataFrame, category: str) -> Dict:
        config = self.category_configs.get(category, self.category_configs["食品"])
        model_type = config["model_type"]

        if model_type == "hybrid":
            return self._train_hybrid_model(sku_data, config)
        else:
            return self._train_ml_model(sku_data, config)

    def _train_ml_model(self, sku_data: pd.DataFrame, config: Dict) -> Dict:
        valid_cols = [c for c in self.feature_cols if c in sku_data.columns]

        X = sku_data[valid_cols].fillna(0).values
        y = sku_data["sales_volume"].values

        gbr = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        )
        gbr.fit(X, y)

        lr = LinearRegression()
        lr.fit(X, y)

        feature_importance = dict(zip(valid_cols, gbr.feature_importances_))

        return {
            "type": "ml",
            "gbr_model": gbr,
            "lr_model": lr,
            "feature_cols": valid_cols,
            "feature_importance": feature_importance,
            "baseline_demand": np.mean(y[-30:]),
        }

    def _train_hybrid_model(self, sku_data: pd.DataFrame, config: Dict) -> Dict:
        ts = sku_data.set_index("date")["sales_volume"].astype(float)
        ts = ts.asfreq("D").fillna(method="ffill").fillna(0)

        seasonal_period = min(config["seasonal_period"], len(ts) // 2)

        try:
            hw_model = ExponentialSmoothing(
                ts,
                seasonal_periods=seasonal_period,
                trend="add",
                seasonal="add",
                initialization_method="estimated",
            ).fit(optimized=True)
        except Exception:
            hw_model = None

        valid_cols = [c for c in self.feature_cols if c in sku_data.columns]
        X = sku_data[valid_cols].fillna(0).values
        y = sku_data["sales_volume"].values

        rf = RandomForestRegressor(
            n_estimators=80,
            max_depth=6,
            random_state=42,
        )
        rf.fit(X, y)

        return {
            "type": "hybrid",
            "hw_model": hw_model,
            "ml_model": rf,
            "feature_cols": valid_cols,
            "seasonal_period": seasonal_period,
            "baseline_demand": np.mean(y[-30:]),
            "trend_decomp": self._extract_trend(ts, seasonal_period),
        }

    def _extract_trend(self, ts: pd.Series, period: int) -> Dict:
        try:
            result = seasonal_decompose(ts, model="add", period=min(period, len(ts) // 2))
            return {
                "trend_last": float(result.trend.iloc[-1]) if result.trend is not None else float(ts.iloc[-1]),
                "seasonality_strength": float(np.std(result.seasonal) / np.std(ts)) if np.std(ts) > 0 else 0,
            }
        except Exception:
            return {"trend_last": float(ts.iloc[-1]), "seasonality_strength": 0.1}

    def predict(
        self,
        sku_id: str,
        category: str,
        future_dates: pd.DataFrame,
        price: Optional[float] = None,
        stock_level: Optional[int] = None,
        days: int = 14,
    ) -> Dict:
        model_info = self._get_sku_model(sku_id, category)
        if model_info is None:
            return self._naive_predict(future_dates, days)

        if model_info["type"] == "hybrid":
            return self._predict_hybrid(model_info, future_dates, price, stock_level, days)
        else:
            return self._predict_ml(model_info, future_dates, price, stock_level, days)

    def _get_sku_model(self, sku_id: str, category: str) -> Optional[Dict]:
        cat_models = self.models.get(category, {})
        if sku_id in cat_models:
            return cat_models[sku_id]

        if cat_models:
            first_key = list(cat_models.keys())[0]
            return cat_models[first_key]

        return None

    def _predict_ml(
        self,
        model_info: Dict,
        future_dates: pd.DataFrame,
        price: Optional[float],
        stock_level: Optional[int],
        days: int,
    ) -> Dict:
        model = model_info["gbr_model"]
        lr_model = model_info["lr_model"]
        feature_cols = model_info["feature_cols"]

        X_future = self._prepare_future_features(future_dates, feature_cols, price, stock_level)

        gbr_pred = model.predict(X_future)
        lr_pred = lr_model.predict(X_future)
        final_pred = gbr_pred * 0.7 + lr_pred * 0.3

        predictions = []
        for i, (_, row) in enumerate(future_dates.iterrows()):
            pred = max(1, final_pred[i])
            lower = pred * 0.7
            upper = pred * 1.4
            if stock_level is not None and i > 0:
                cum_pred = sum(p["predicted_sales"] for p in predictions)
                if cum_pred >= stock_level:
                    pred = max(0, stock_level - cum_pred)
                    lower = pred * 0.8
                    upper = pred * 1.2

            predictions.append({
                "date": row["date"].strftime("%Y-%m-%d") if isinstance(row["date"], datetime) else str(row["date"]),
                "predicted_sales": round(float(pred), 1),
                "lower_bound": round(float(lower), 1),
                "upper_bound": round(float(upper), 1),
                "is_weekend": int(row.get("is_weekend", 0)),
                "is_holiday": int(row.get("is_holiday", 0)),
            })

        total = sum(p["predicted_sales"] for p in predictions)

        return {
            "sku_id": future_dates.get("sku_id", [""])[0] if len(future_dates) > 0 else "",
            "forecast_days": days,
            "total_predicted_sales": round(float(total), 1),
            "daily_predictions": predictions,
            "model_type": "ml",
            "confidence": 0.75,
        }

    def _predict_hybrid(
        self,
        model_info: Dict,
        future_dates: pd.DataFrame,
        price: Optional[float],
        stock_level: Optional[int],
        days: int,
    ) -> Dict:
        hw_model = model_info.get("hw_model")
        ml_model = model_info.get("ml_model")
        feature_cols = model_info.get("feature_cols", [])

        hw_forecast = np.zeros(days)
        if hw_model is not None:
            try:
                hw_result = hw_model.forecast(days)
                hw_forecast = hw_result.values
            except Exception:
                hw_forecast = np.full(days, model_info["baseline_demand"])

        ml_forecast = np.zeros(days)
        if ml_model is not None and len(feature_cols) > 0:
            X_future = self._prepare_future_features(future_dates, feature_cols, price, stock_level)
            ml_forecast = ml_model.predict(X_future)

        combined = hw_forecast * 0.4 + ml_forecast * 0.6
        combined = np.maximum(1, combined)

        predictions = []
        for i, (_, row) in enumerate(future_dates.iterrows()):
            pred = float(combined[i]) if i < len(combined) else float(combined[-1])
            lower = pred * 0.65
            upper = pred * 1.45

            if stock_level is not None and i > 0:
                cum_pred = sum(p["predicted_sales"] for p in predictions)
                if cum_pred >= stock_level:
                    pred = max(0, stock_level - cum_pred)
                    lower = pred * 0.8
                    upper = pred * 1.2

            predictions.append({
                "date": row["date"].strftime("%Y-%m-%d") if isinstance(row["date"], datetime) else str(row["date"]),
                "predicted_sales": round(pred, 1),
                "lower_bound": round(lower, 1),
                "upper_bound": round(upper, 1),
                "is_weekend": int(row.get("is_weekend", 0)),
                "is_holiday": int(row.get("is_holiday", 0)),
                "ts_component": round(float(hw_forecast[i]) if i < len(hw_forecast) else 0, 1),
                "ml_component": round(float(ml_forecast[i]) if i < len(ml_forecast) else 0, 1),
            })

        total = sum(p["predicted_sales"] for p in predictions)

        return {
            "sku_id": "",
            "forecast_days": days,
            "total_predicted_sales": round(float(total), 1),
            "daily_predictions": predictions,
            "model_type": "hybrid",
            "confidence": 0.82,
        }

    def _prepare_future_features(
        self,
        future_dates: pd.DataFrame,
        feature_cols: List[str],
        price: Optional[float],
        stock_level: Optional[int],
    ) -> np.ndarray:
        df = future_dates.copy()

        for col in feature_cols:
            if col not in df.columns:
                if col.startswith("sales_lag_") or col.startswith("sales_rolling_"):
                    df[col] = 0
                elif col == "price" and price is not None:
                    df[col] = price
                elif col == "stock_level" and stock_level is not None:
                    df[col] = stock_level
                elif col == "discount_rate":
                    df[col] = 0
                elif col == "is_discounted":
                    df[col] = 0
                elif col == "temperature":
                    df[col] = 20
                elif col == "traffic_index":
                    df[col] = 1.0
                else:
                    df[col] = 0

        X = df[feature_cols].fillna(0).values
        return X

    def _naive_predict(self, future_dates: pd.DataFrame, days: int) -> Dict:
        predictions = []
        for i, (_, row) in enumerate(future_dates.iterrows()):
            base = 50
            is_weekend = int(row.get("is_weekend", 0))
            is_holiday = int(row.get("is_holiday", 0))
            pred = base * (1.3 if is_weekend else 1.0) * (1.5 if is_holiday else 1.0)
            predictions.append({
                "date": row["date"].strftime("%Y-%m-%d") if isinstance(row["date"], datetime) else str(row["date"]),
                "predicted_sales": round(pred, 1),
                "lower_bound": round(pred * 0.6, 1),
                "upper_bound": round(pred * 1.5, 1),
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
            })

        total = sum(p["predicted_sales"] for p in predictions)
        return {
            "sku_id": "",
            "forecast_days": days,
            "total_predicted_sales": round(float(total), 1),
            "daily_predictions": predictions,
            "model_type": "naive",
            "confidence": 0.5,
        }

    def predict_batch(
        self,
        sku_list: List[Dict],
        days: int = 14,
    ) -> List[Dict]:
        results = []
        for sku_info in sku_list:
            future_dates = self._generate_future_dates(days, sku_info.get("sku_id", ""))
            pred = self.predict(
                sku_id=sku_info.get("sku_id", ""),
                category=sku_info.get("category", "食品"),
                future_dates=future_dates,
                price=sku_info.get("price"),
                stock_level=sku_info.get("stock_level"),
                days=days,
            )
            pred["sku_id"] = sku_info.get("sku_id", "")
            results.append(pred)
        return results

    def _generate_future_dates(self, days: int, sku_id: str = "") -> pd.DataFrame:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dates = []
        for i in range(1, days + 1):
            d = today + timedelta(days=i)
            dayofweek = d.weekday()
            dates.append({
                "date": d,
                "sku_id": sku_id,
                "dayofweek": dayofweek,
                "is_weekend": 1 if dayofweek >= 5 else 0,
                "is_holiday": self._is_holiday(d),
                "month": d.month,
                "sin_month": np.sin(2 * np.pi * d.month / 12),
                "cos_month": np.cos(2 * np.pi * d.month / 12),
            })
        return pd.DataFrame(dates)

    def _is_holiday(self, date: datetime) -> int:
        month_day = (date.month, date.day)
        holidays = [
            (1, 1), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17),
            (4, 4), (4, 5), (4, 6), (5, 1), (5, 2), (5, 3), (6, 10), (6, 11), (6, 12),
            (9, 15), (9, 16), (9, 17), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7),
            (12, 25),
        ]
        return 1 if month_day in holidays else 0

    def _default_feature_cols(self, df: pd.DataFrame) -> List[str]:
        candidates = [
            "dayofweek", "is_weekend", "is_holiday", "month",
            "sin_month", "cos_month", "sin_dayofweek", "cos_dayofweek",
            "price", "discount_rate", "is_discounted",
            "sales_lag_1", "sales_lag_3", "sales_lag_7", "sales_lag_14",
            "sales_rolling_mean_7", "sales_rolling_mean_14", "sales_rolling_mean_30",
            "sales_rolling_std_7",
            "temperature", "traffic_index",
            "stock_level",
        ]
        return [c for c in candidates if c in df.columns]


_default_forecast_model = None


def get_sales_forecast_model() -> SalesForecastModel:
    global _default_forecast_model
    if _default_forecast_model is None:
        _default_forecast_model = SalesForecastModel()
    return _default_forecast_model
