import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class FeatureEngineer:
    def build_features(
        self,
        sales_data: pd.DataFrame,
        inventory_data: Optional[pd.DataFrame] = None,
        competitor_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        df = sales_data.copy()
        df = self._add_time_features(df)
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_price_features(df)
        df = self._add_promotion_features(df)

        if inventory_data is not None:
            df = self._merge_inventory_features(df, inventory_data)

        if competitor_data is not None:
            df = self._merge_competitor_features(df, competitor_data)

        df = self._handle_missing_values(df)
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day"] = df["date"].dt.day
            df["dayofweek"] = df["date"].dt.dayofweek
            df["dayofyear"] = df["date"].dt.dayofyear
            df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
            df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
            df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
            df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
            df["quarter"] = df["date"].dt.quarter
            df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
            df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
            df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
            df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["sku_id", "date"])
        lag_days = [1, 3, 7, 14, 28]

        for lag in lag_days:
            df[f"sales_lag_{lag}"] = df.groupby("sku_id")["sales_volume"].shift(lag)
            df[f"price_lag_{lag}"] = df.groupby("sku_id")["price"].shift(lag)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["sku_id", "date"])
        windows = [7, 14, 30, 60]

        for window in windows:
            df[f"sales_rolling_mean_{window}"] = (
                df.groupby("sku_id")["sales_volume"]
                .transform(lambda x: x.rolling(window=window, min_periods=3).mean())
            )
            df[f"sales_rolling_std_{window}"] = (
                df.groupby("sku_id")["sales_volume"]
                .transform(lambda x: x.rolling(window=window, min_periods=3).std())
            )
            df[f"price_rolling_mean_{window}"] = (
                df.groupby("sku_id")["price"]
                .transform(lambda x: x.rolling(window=window, min_periods=3).mean())
            )

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "base_price" in df.columns and "price" in df.columns:
            df["discount_rate"] = 1 - df["price"] / df["base_price"]
            df["price_ratio_vs_base"] = df["price"] / df["base_price"]
            df["is_discounted"] = (df["discount_rate"] > 0).astype(int)

        if "cost" in df.columns and "price" in df.columns:
            df["margin_amount"] = df["price"] - df["cost"]
            df["margin_rate"] = (df["price"] - df["cost"]) / df["price"]

        return df

    def _add_promotion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "promotion_type" in df.columns:
            df["is_promotion"] = (df["promotion_type"] != "常规定价").astype(int)
            promo_dummies = pd.get_dummies(df["promotion_type"], prefix="promo", dtype=int)
            df = pd.concat([df, promo_dummies], axis=1)

        return df

    def _merge_inventory_features(self, df: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
        inv = inventory.copy()
        if "date" in df.columns and "date" not in inv.columns:
            pass
        merged = df.merge(inv, on="sku_id", how="left", suffixes=("", "_inv"))
        return merged

    def _merge_competitor_features(self, df: pd.DataFrame, competitor: pd.DataFrame) -> pd.DataFrame:
        comp_agg = competitor.groupby(["date", "sku_id"])["competitor_price"].agg(
            ["mean", "min", "max"]
        ).reset_index()
        comp_agg.columns = ["date", "sku_id", "comp_price_mean", "comp_price_min", "comp_price_max"]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            comp_agg["date"] = pd.to_datetime(comp_agg["date"])

        merged = df.merge(comp_agg, on=["date", "sku_id"], how="left")

        if "price" in merged.columns:
            merged["price_vs_comp_mean"] = merged["price"] / merged["comp_price_mean"].replace(0, np.nan)
            merged["price_vs_comp_min"] = merged["price"] / merged["comp_price_min"].replace(0, np.nan)

        return merged

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df.groupby("sku_id")[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                    df[col] = df[col].fillna(df[col].median())
        return df

    def extract_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "category" in df.columns:
            cat_dummies = pd.get_dummies(df["category"], prefix="cat", dtype=int)
            df = pd.concat([df, cat_dummies], axis=1)
        return df


_default_engineer = None


def get_feature_engineer() -> FeatureEngineer:
    global _default_engineer
    if _default_engineer is None:
        _default_engineer = FeatureEngineer()
    return _default_engineer
