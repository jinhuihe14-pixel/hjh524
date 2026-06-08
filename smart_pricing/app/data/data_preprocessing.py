import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    def __init__(self):
        self.scalers: Dict[str, object] = {}

    def clean_sales_data(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        df = sales_data.copy()
        df = self._remove_duplicates(df)
        df = self._handle_outliers(df)
        df = self._validate_data(df)
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if {"sku_id", "date"}.issubset(df.columns):
            df = df.drop_duplicates(subset=["sku_id", "date"], keep="last")
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if "sales_volume" in df.columns and "sku_id" in df.columns:
            df = df.groupby("sku_id", group_keys=False).apply(
                self._cap_outliers_iqr, include_groups=False
            )
        return df

    def _cap_outliers_iqr(self, group: pd.DataFrame) -> pd.DataFrame:
        if "sales_volume" not in group.columns:
            return group
        q1 = group["sales_volume"].quantile(0.25)
        q3 = group["sales_volume"].quantile(0.75)
        iqr = q3 - q1
        lower = max(0, q1 - 1.5 * iqr)
        upper = q3 + 3 * iqr
        group["sales_volume"] = group["sales_volume"].clip(lower=lower, upper=upper)
        return group

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "price" in df.columns:
            df["price"] = df["price"].clip(lower=0.01)
        if "sales_volume" in df.columns:
            df["sales_volume"] = df["sales_volume"].clip(lower=0).astype(int)
        return df

    def scale_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        method: str = "standard",
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        result = df.copy()

        if fit:
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")

            scaler.fit(result[feature_cols])
            self.scalers[method] = scaler
        else:
            scaler = self.scalers.get(method)
            if scaler is None:
                raise ValueError(f"No fitted scaler found for method: {method}")

        result[feature_cols] = scaler.transform(result[feature_cols])
        return result, {"method": method, "feature_cols": feature_cols}

    def inverse_scale_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        method: str = "standard",
    ) -> pd.DataFrame:
        result = df.copy()
        scaler = self.scalers.get(method)
        if scaler is None:
            raise ValueError(f"No fitted scaler found for method: {method}")

        result[feature_cols] = scaler.inverse_transform(result[feature_cols])
        return result

    def prepare_train_test(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        test_size_days: int = 30,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if "date" not in df.columns:
            raise ValueError("DataFrame must have 'date' column for time-based split")

        df_sorted = df.sort_values("date")
        unique_dates = df_sorted["date"].unique()

        if len(unique_dates) <= test_size_days:
            test_size_days = max(1, len(unique_dates) // 10)

        split_date = unique_dates[-test_size_days]

        train_df = df_sorted[df_sorted["date"] < split_date]
        test_df = df_sorted[df_sorted["date"] >= split_date]

        X_train = train_df[feature_cols].copy()
        y_train = train_df[target_col].copy()
        X_test = test_df[feature_cols].copy()
        y_test = test_df[target_col].copy()

        return X_train, X_test, y_train, y_test

    def normalize_price_data(self, prices: np.ndarray) -> np.ndarray:
        min_p = np.min(prices)
        max_p = np.max(prices)
        if max_p == min_p:
            return np.ones_like(prices) * 0.5
        return (prices - min_p) / (max_p - min_p)


_default_preprocessor = None


def get_data_preprocessor() -> DataPreprocessor:
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = DataPreprocessor()
    return _default_preprocessor
