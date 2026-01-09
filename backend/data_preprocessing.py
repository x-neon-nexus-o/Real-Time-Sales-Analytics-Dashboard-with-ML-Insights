"""
Data Preprocessing Module.
Handles data cleaning, validation, feature engineering, and aggregation for ML models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

from config import FEATURE_CONFIG, CATEGORIES, REGIONS
from utils import (
    setup_logger, parse_datetime, clean_numeric_column,
    validate_dataframe, detect_outliers_zscore, detect_outliers_iqr,
    calculate_growth_rate, round_currency
)

warnings.filterwarnings('ignore')
logger = setup_logger("preprocessing")


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for sales analytics.
    
    Handles cleaning, validation, feature engineering, and aggregation
    of sales transaction data for ML model consumption.
    """
    
    # Required columns for raw data
    REQUIRED_COLUMNS = [
        "timestamp", "transaction_id", "product_id", "product_name",
        "category", "quantity", "unit_price", "total_amount",
        "region", "customer_id", "payment_method"
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Optional configuration overrides
        """
        self.config = config or FEATURE_CONFIG
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._fitted = False
        
        logger.info("DataPreprocessor initialized")
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate raw sales data.
        
        Args:
            df: Raw sales DataFrame
        
        Returns:
            Validation result dictionary
        """
        result = validate_dataframe(df, self.REQUIRED_COLUMNS)
        
        # Additional validations
        if result["valid"]:
            # Check data types
            if "timestamp" in df.columns:
                try:
                    pd.to_datetime(df["timestamp"])
                except Exception as e:
                    result["valid"] = False
                    result["errors"].append(f"Invalid timestamp format: {e}")
            
            # Check numeric columns
            numeric_cols = ["quantity", "unit_price", "total_amount"]
            for col in numeric_cols:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        result["valid"] = False
                        result["errors"].append(f"Column '{col}' should be numeric")
            
            # Check for negative values
            if "total_amount" in df.columns:
                if (df["total_amount"] < 0).any():
                    result["errors"].append("Warning: Negative values in total_amount")
        
        logger.info(f"Validation result: {result['valid']}, Errors: {len(result['errors'])}")
        return result
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw sales data.
        
        Operations:
        - Parse and validate timestamps
        - Handle missing values
        - Remove duplicates
        - Remove outliers
        - Standardize formats
        
        Args:
            df: Raw sales DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning data: {len(df)} rows")
        df = df.copy()
        
        # 1. Parse timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
        # Remove rows with invalid timestamps
        initial_len = len(df)
        df = df.dropna(subset=["timestamp"])
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} rows with invalid timestamps")
        
        # 2. Handle missing values
        # Numeric columns: fill with median
        for col in ["quantity", "unit_price", "total_amount"]:
            if col in df.columns:
                df[col] = clean_numeric_column(df[col], fill_method="median")
        
        # Categorical columns: fill with 'Unknown'
        for col in ["category", "region", "payment_method"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
        
        # 3. Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates(subset=["transaction_id"], keep="first")
        if len(df) < initial_len:
            logger.info(f"Removed {initial_len - len(df)} duplicate transactions")
        
        # 4. Remove extreme outliers (only for training, keep for detection)
        # Using IQR method with 3x multiplier (very conservative)
        if "total_amount" in df.columns and len(df) > 100:
            q1 = df["total_amount"].quantile(0.01)
            q99 = df["total_amount"].quantile(0.99)
            
            # Flag but don't remove extreme values
            df["is_extreme"] = (df["total_amount"] < q1) | (df["total_amount"] > q99)
        
        # 5. Ensure positive values
        df.loc[df["quantity"] <= 0, "quantity"] = 1
        df.loc[df["unit_price"] <= 0, "unit_price"] = df["unit_price"].median()
        
        # 6. Recalculate total_amount if needed
        df["total_amount_calc"] = df["quantity"] * df["unit_price"]
        
        # Use calculated value if original is suspicious
        discrepancy = abs(df["total_amount"] - df["total_amount_calc"]) / df["total_amount_calc"]
        df.loc[discrepancy > 0.1, "total_amount"] = df.loc[discrepancy > 0.1, "total_amount_calc"]
        df = df.drop(columns=["total_amount_calc"])
        
        # 7. Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"Cleaning complete: {len(df)} rows remaining")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for ML models.
        
        Features created:
        - Time-based: hour, day_of_week, day_of_month, month, quarter, is_weekend
        - Lag features: sales_lag_1, sales_lag_7, etc.
        - Rolling statistics: rolling_mean_7d, rolling_std_7d, etc.
        - Growth features: daily_growth_rate, week_over_week_change
        
        Args:
            df: Cleaned sales DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        df = df.copy()
        
        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # 1. TIME-BASED FEATURES
        if self.config.get("include_time_features", True):
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["day_of_month"] = df["timestamp"].dt.day
            df["month"] = df["timestamp"].dt.month
            df["quarter"] = df["timestamp"].dt.quarter
            df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
            df["year"] = df["timestamp"].dt.year
            
            # Binary features
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df["is_month_start"] = df["timestamp"].dt.is_month_start.astype(int)
            df["is_month_end"] = df["timestamp"].dt.is_month_end.astype(int)
            
            # Time of day categories
            df["time_of_day"] = pd.cut(
                df["hour"],
                bins=[0, 6, 12, 18, 24],
                labels=["night", "morning", "afternoon", "evening"],
                right=False
            )
        
        # 2. CATEGORICAL ENCODING
        # Create numeric codes for categories
        if "category" in df.columns:
            if "category" not in self.label_encoders:
                self.label_encoders["category"] = LabelEncoder()
                df["category_encoded"] = self.label_encoders["category"].fit_transform(df["category"])
            else:
                # Handle unseen categories
                known_cats = set(self.label_encoders["category"].classes_)
                df["category_safe"] = df["category"].apply(lambda x: x if x in known_cats else "Unknown")
                df["category_encoded"] = self.label_encoders["category"].transform(df["category_safe"])
                df = df.drop(columns=["category_safe"])
        
        if "region" in df.columns:
            if "region" not in self.label_encoders:
                self.label_encoders["region"] = LabelEncoder()
                df["region_encoded"] = self.label_encoders["region"].fit_transform(df["region"])
            else:
                known_regions = set(self.label_encoders["region"].classes_)
                df["region_safe"] = df["region"].apply(lambda x: x if x in known_regions else "Unknown")
                df["region_encoded"] = self.label_encoders["region"].transform(df["region_safe"])
                df = df.drop(columns=["region_safe"])
        
        logger.info(f"Engineered {len([c for c in df.columns if c not in self.REQUIRED_COLUMNS])} features")
        return df
    
    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate transaction data to daily level.
        
        Args:
            df: Transaction-level DataFrame
        
        Returns:
            Daily aggregated DataFrame
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        
        daily = df.groupby("date").agg({
            "transaction_id": "count",
            "total_amount": "sum",
            "quantity": "sum",
            "unit_price": "mean",
            "customer_id": "nunique",
        }).reset_index()
        
        daily.columns = [
            "date", "transaction_count", "total_revenue",
            "total_quantity", "avg_unit_price", "unique_customers"
        ]
        
        # Calculate AOV
        daily["avg_order_value"] = daily["total_revenue"] / daily["transaction_count"]
        
        # Add time features
        daily["date"] = pd.to_datetime(daily["date"])
        daily["day_of_week"] = daily["date"].dt.dayofweek
        daily["month"] = daily["date"].dt.month
        daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
        
        # Add lag features
        if self.config.get("include_lag_features", True):
            for lag in self.config.get("lag_periods", [1, 7, 14, 30]):
                daily[f"revenue_lag_{lag}"] = daily["total_revenue"].shift(lag)
                daily[f"transactions_lag_{lag}"] = daily["transaction_count"].shift(lag)
        
        # Add rolling features
        if self.config.get("include_rolling_features", True):
            for window in self.config.get("rolling_windows", [7, 14, 30]):
                daily[f"revenue_rolling_mean_{window}d"] = daily["total_revenue"].rolling(window).mean()
                daily[f"revenue_rolling_std_{window}d"] = daily["total_revenue"].rolling(window).std()
                daily[f"revenue_rolling_min_{window}d"] = daily["total_revenue"].rolling(window).min()
                daily[f"revenue_rolling_max_{window}d"] = daily["total_revenue"].rolling(window).max()
        
        # Add growth features
        if self.config.get("include_growth_features", True):
            daily["daily_growth_rate"] = daily["total_revenue"].pct_change()
            daily["week_over_week_change"] = daily["total_revenue"].pct_change(7)
            
        # Fill NaN values from lag/rolling with 0 or forward fill
        daily = daily.fillna(method="bfill").fillna(0)
        
        return daily
    
    def aggregate_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate transaction data to hourly level.
        
        Args:
            df: Transaction-level DataFrame
        
        Returns:
            Hourly aggregated DataFrame
        """
        df = df.copy()
        df["datetime_hour"] = pd.to_datetime(df["timestamp"]).dt.floor("H")
        
        hourly = df.groupby("datetime_hour").agg({
            "transaction_id": "count",
            "total_amount": "sum",
            "quantity": "sum",
        }).reset_index()
        
        hourly.columns = ["datetime", "transaction_count", "total_revenue", "total_quantity"]
        
        # Add time features
        hourly["hour"] = hourly["datetime"].dt.hour
        hourly["day_of_week"] = hourly["datetime"].dt.dayofweek
        hourly["is_weekend"] = hourly["day_of_week"].isin([5, 6]).astype(int)
        
        return hourly
    
    def aggregate_by_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data by product category.
        
        Args:
            df: Transaction-level DataFrame
        
        Returns:
            Category-level aggregated DataFrame
        """
        category_agg = df.groupby("category").agg({
            "transaction_id": "count",
            "total_amount": "sum",
            "quantity": "sum",
            "unit_price": "mean",
            "customer_id": "nunique",
        }).reset_index()
        
        category_agg.columns = [
            "category", "transaction_count", "total_revenue",
            "total_quantity", "avg_price", "unique_customers"
        ]
        
        # Calculate percentage of total
        total_revenue = category_agg["total_revenue"].sum()
        category_agg["revenue_percentage"] = (category_agg["total_revenue"] / total_revenue * 100).round(2)
        
        # Sort by revenue
        category_agg = category_agg.sort_values("total_revenue", ascending=False).reset_index(drop=True)
        
        return category_agg
    
    def aggregate_by_region(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data by region.
        
        Args:
            df: Transaction-level DataFrame
        
        Returns:
            Region-level aggregated DataFrame
        """
        region_agg = df.groupby("region").agg({
            "transaction_id": "count",
            "total_amount": "sum",
            "quantity": "sum",
            "customer_id": "nunique",
        }).reset_index()
        
        region_agg.columns = [
            "region", "transaction_count", "total_revenue",
            "total_quantity", "unique_customers"
        ]
        
        # Calculate percentage and AOV
        total_revenue = region_agg["total_revenue"].sum()
        region_agg["revenue_percentage"] = (region_agg["total_revenue"] / total_revenue * 100).round(2)
        region_agg["avg_order_value"] = (region_agg["total_revenue"] / region_agg["transaction_count"]).round(2)
        
        # Sort by revenue
        region_agg = region_agg.sort_values("total_revenue", ascending=False).reset_index(drop=True)
        
        return region_agg
    
    def get_ml_dataset(
        self,
        df: pd.DataFrame,
        target_column: str = "total_revenue",
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare dataset for ML model training.
        
        Args:
            df: Preprocessed DataFrame (daily aggregated)
            target_column: Name of target variable
            exclude_columns: Columns to exclude from features
        
        Returns:
            Tuple of (X, y) for model training
        """
        df = df.copy()
        
        # Default columns to exclude
        default_exclude = [
            "date", "datetime", "timestamp", "transaction_id",
            "product_id", "product_name", "customer_id",
            "time_of_day", "is_anomaly", "is_extreme"
        ]
        
        exclude = set(default_exclude + (exclude_columns or []))
        exclude.add(target_column)
        
        # Select numeric columns only
        feature_cols = [
            col for col in df.columns
            if col not in exclude
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        # Scale features
        if not self._fitted:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            self._fitted = True
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        logger.info(f"ML dataset prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        
        return X_scaled, y
    
    def prepare_for_inference(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Prepare new data for model inference.
        
        Args:
            df: New data DataFrame
            feature_columns: List of feature columns used in training
        
        Returns:
            Scaled feature DataFrame
        """
        df = df.copy()
        
        # Ensure all required columns exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        X = df[feature_columns].fillna(0)
        
        # Scale using fitted scaler
        if self._fitted:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            logger.warning("Scaler not fitted, returning unscaled data")
            X_scaled = X
        
        return X_scaled
    
    def get_trend_data(
        self,
        df: pd.DataFrame,
        period: str = "daily",
        last_n_periods: int = 30
    ) -> pd.DataFrame:
        """
        Get trend data for visualization.
        
        Args:
            df: Transaction DataFrame
            period: Aggregation period (hourly, daily, weekly)
            last_n_periods: Number of periods to include
        
        Returns:
            Trend DataFrame
        """
        if period == "hourly":
            trend = self.aggregate_hourly(df)
            date_col = "datetime"
        elif period == "weekly":
            df = df.copy()
            df["week"] = pd.to_datetime(df["timestamp"]).dt.to_period("W").dt.start_time
            trend = df.groupby("week").agg({
                "total_amount": "sum",
                "transaction_id": "count",
            }).reset_index()
            trend.columns = ["date", "total_revenue", "transaction_count"]
            date_col = "date"
        else:  # daily
            trend = self.aggregate_daily(df)
            date_col = "date"
        
        # Take last n periods
        trend = trend.tail(last_n_periods).reset_index(drop=True)
        
        return trend


def preprocess_pipeline(
    df: pd.DataFrame,
    for_training: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, DataPreprocessor]:
    """
    Run full preprocessing pipeline.
    
    Args:
        df: Raw sales DataFrame
        for_training: Whether this is for model training
    
    Returns:
        Tuple of (cleaned_df, daily_df, preprocessor)
    """
    preprocessor = DataPreprocessor()
    
    # Validate
    validation = preprocessor.validate_data(df)
    if not validation["valid"]:
        logger.error(f"Validation failed: {validation['errors']}")
        raise ValueError(f"Data validation failed: {validation['errors']}")
    
    # Clean
    cleaned = preprocessor.clean_data(df)
    
    # Engineer features
    featured = preprocessor.engineer_features(cleaned)
    
    # Aggregate to daily
    daily = preprocessor.aggregate_daily(featured)
    
    return featured, daily, preprocessor
