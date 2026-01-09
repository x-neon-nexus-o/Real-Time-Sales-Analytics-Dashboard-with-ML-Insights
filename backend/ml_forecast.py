"""
Sales Forecasting Module.
Implements ML models for predicting future sales with confidence intervals.
"""

import os
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from config import FORECAST_CONFIG, MODELS_DIR
from utils import setup_logger, round_currency

warnings.filterwarnings('ignore')
logger = setup_logger("ml_forecast")


class SalesForecastModel:
    """
    Multi-algorithm sales forecasting model.
    
    Supports:
    - Random Forest Regressor (primary)
    - Gradient Boosting Regressor
    - ARIMA/SARIMA (time series)
    - Exponential Smoothing
    """
    
    def __init__(
        self,
        algorithm: str = "random_forest",
        config: Optional[Dict] = None
    ):
        """
        Initialize the forecasting model.
        
        Args:
            algorithm: Primary algorithm to use
            config: Optional configuration overrides
        """
        self.algorithm = algorithm
        self.config = config or FORECAST_CONFIG
        
        # Model instances
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        
        # Model state
        self._fitted = False
        self._last_train_date: Optional[datetime] = None
        
        # Performance metrics
        self.metrics: Dict[str, float] = {}
        
        # Initialize the model
        self._initialize_model()
        
        logger.info(f"SalesForecastModel initialized with algorithm: {algorithm}")
    
    def _initialize_model(self) -> None:
        """Initialize the ML model based on algorithm choice."""
        if self.algorithm == "random_forest":
            params = self.config.get("rf_params", {})
            self.model = RandomForestRegressor(**params)
            
        elif self.algorithm == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
            
        elif self.algorithm == "linear":
            self.model = Ridge(alpha=1.0)
            
        else:
            # Default to Random Forest
            self.model = RandomForestRegressor(**self.config.get("rf_params", {}))
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_column: str = "total_revenue"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training/prediction.
        
        Args:
            df: Daily aggregated DataFrame
            target_column: Target variable column
        
        Returns:
            Tuple of (X, y)
        """
        df = df.copy()
        
        # Ensure we have required columns
        required = ["date", target_column]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Select feature columns (exclude non-numeric and target)
        exclude_cols = {
            "date", "datetime", "timestamp", target_column,
            "is_anomaly", "is_extreme"
        }
        
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        # Handle NaN values
        X = X.fillna(0)
        y = y.fillna(y.median())
        
        return X, y
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the forecasting model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion for testing
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Training model with {len(X)} samples...")
        
        # Time-based split (don't shuffle time series data)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        self.metrics = self._calculate_metrics(y_test, y_pred)
        self._fitted = True
        self._last_train_date = datetime.now()
        
        logger.info(f"Training complete. MAPE: {self.metrics['mape']:.2f}%, R²: {self.metrics['r2']:.4f}")
        
        return self.metrics
    
    def train_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Train with time series cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of CV splits
        
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Training with {n_splits}-fold time series CV...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Clone and train model
            model = self._clone_model()
            model.fit(X_train_scaled, y_train)
            
            # Predict and score
            y_pred = model.predict(X_test_scaled)
            metrics = self._calculate_metrics(y_test, y_pred)
            cv_scores.append(metrics)
        
        # Train final model on all data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True
        self._last_train_date = datetime.now()
        
        # Average CV scores
        avg_metrics = {
            key: np.mean([s[key] for s in cv_scores])
            for key in cv_scores[0].keys()
        }
        avg_metrics["cv_std"] = {
            key: np.std([s[key] for s in cv_scores])
            for key in cv_scores[0].keys()
        }
        
        self.metrics = avg_metrics
        
        logger.info(f"CV Training complete. Avg MAPE: {avg_metrics['mape']:.2f}%")
        
        return avg_metrics
    
    def _clone_model(self):
        """Clone the model for CV."""
        if self.algorithm == "random_forest":
            return RandomForestRegressor(**self.config.get("rf_params", {}))
        elif self.algorithm == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42
            )
        else:
            return Ridge(alpha=1.0)
    
    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
        
        Returns:
            Dictionary of metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (handle zeros)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Direction accuracy
        if len(y_true) > 1:
            actual_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            direction_accuracy = 0
        
        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "r2": float(r2),
            "direction_accuracy": float(direction_accuracy),
        }
    
    def predict(
        self,
        X: pd.DataFrame,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Generate predictions.
        
        Args:
            X: Feature DataFrame
            return_confidence: Whether to include confidence intervals
        
        Returns:
            Dictionary with predictions and optional confidence intervals
        """
        if not self._fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Ensure columns match
        X = X[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Point predictions
        predictions = self.model.predict(X_scaled)
        
        result = {
            "predictions": predictions.tolist(),
            "mean": float(np.mean(predictions)),
            "total": float(np.sum(predictions)),
        }
        
        # Confidence intervals (using prediction std if available)
        if return_confidence and hasattr(self.model, "estimators_"):
            # For ensemble models, get predictions from all trees
            tree_preds = np.array([
                tree.predict(X_scaled) for tree in self.model.estimators_
            ])
            
            std = np.std(tree_preds, axis=0)
            confidence_level = self.config.get("confidence_level", 0.95)
            z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
            
            result["confidence_lower"] = (predictions - z_score * std).tolist()
            result["confidence_upper"] = (predictions + z_score * std).tolist()
            result["std"] = std.tolist()
        
        return result
    
    def forecast(
        self,
        historical_data: pd.DataFrame,
        horizon: int = 7,
        target_column: str = "total_revenue"
    ) -> Dict[str, Any]:
        """
        Generate future forecasts.
        
        Args:
            historical_data: Daily aggregated historical data
            horizon: Number of days to forecast
            target_column: Target column name
        
        Returns:
            Forecast results dictionary
        """
        logger.info(f"Generating {horizon}-day forecast...")
        
        df = historical_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        last_date = df["date"].max()
        
        forecasts = []
        forecast_dates = []
        confidence_lower = []
        confidence_upper = []
        
        # Generate forecasts iteratively
        for i in range(horizon):
            future_date = last_date + timedelta(days=i + 1)
            
            # Create feature row for future date
            future_row = self._create_future_features(df, future_date)
            
            # Prepare features
            X_future = pd.DataFrame([future_row])[self.feature_columns].fillna(0)
            X_scaled = self.scaler.transform(X_future)
            
            # Predict
            pred = self.model.predict(X_scaled)[0]
            
            # Confidence interval
            if hasattr(self.model, "estimators_"):
                tree_preds = [tree.predict(X_scaled)[0] for tree in self.model.estimators_]
                std = np.std(tree_preds)
                lower = pred - 1.96 * std
                upper = pred + 1.96 * std
            else:
                # Use historical std as proxy
                hist_std = df[target_column].std() * 0.1
                lower = pred - 1.96 * hist_std
                upper = pred + 1.96 * hist_std
            
            forecasts.append(max(0, pred))  # Ensure non-negative
            forecast_dates.append(future_date.strftime("%Y-%m-%d"))
            confidence_lower.append(max(0, lower))
            confidence_upper.append(upper)
            
            # Add to dataframe for next iteration's lag features
            new_row = future_row.copy()
            new_row["date"] = future_date
            new_row[target_column] = pred
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        return {
            "dates": forecast_dates,
            "values": [round_currency(f) for f in forecasts],
            "confidence_lower": [round_currency(l) for l in confidence_lower],
            "confidence_upper": [round_currency(u) for u in confidence_upper],
            "total_forecast": round_currency(sum(forecasts)),
            "avg_daily_forecast": round_currency(np.mean(forecasts)),
            "horizon_days": horizon,
            "model": self.algorithm,
            "generated_at": datetime.now().isoformat(),
        }
    
    def _create_future_features(
        self,
        df: pd.DataFrame,
        future_date: datetime
    ) -> Dict[str, Any]:
        """
        Create feature dictionary for a future date.
        
        Args:
            df: Historical data
            future_date: Date to create features for
        
        Returns:
            Feature dictionary
        """
        # Time features
        features = {
            "day_of_week": future_date.weekday(),
            "month": future_date.month,
            "is_weekend": 1 if future_date.weekday() >= 5 else 0,
            "day_of_month": future_date.day,
        }
        
        # Lag features from historical data
        if "total_revenue" in df.columns:
            revenue = df["total_revenue"].values
            
            if len(revenue) >= 1:
                features["revenue_lag_1"] = revenue[-1]
            if len(revenue) >= 7:
                features["revenue_lag_7"] = revenue[-7]
            if len(revenue) >= 14:
                features["revenue_lag_14"] = revenue[-14]
            if len(revenue) >= 30:
                features["revenue_lag_30"] = revenue[-30]
            
            # Rolling features
            if len(revenue) >= 7:
                features["revenue_rolling_mean_7d"] = np.mean(revenue[-7:])
                features["revenue_rolling_std_7d"] = np.std(revenue[-7:])
            if len(revenue) >= 14:
                features["revenue_rolling_mean_14d"] = np.mean(revenue[-14:])
            if len(revenue) >= 30:
                features["revenue_rolling_mean_30d"] = np.mean(revenue[-30:])
        
        # Add any missing feature columns with 0
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0
        
        return features
    
    def get_feature_importance(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get feature importance rankings.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            List of feature importance dictionaries
        """
        if not self._fitted:
            return []
        
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_)
        else:
            return []
        
        # Pair with feature names
        importance_pairs = list(zip(self.feature_columns, importances))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {"feature": name, "importance": float(imp)}
            for name, imp in importance_pairs[:top_n]
        ]
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            path: Optional save path
        
        Returns:
            Path where model was saved
        """
        if not self._fitted:
            raise RuntimeError("Cannot save untrained model")
        
        if path is None:
            path = os.path.join(MODELS_DIR, "forecast_model.pkl")
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "algorithm": self.algorithm,
            "config": self.config,
            "metrics": self.metrics,
            "trained_at": self._last_train_date.isoformat() if self._last_train_date else None,
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
        
        return path
    
    def load_model(self, path: Optional[str] = None) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to model file
        """
        if path is None:
            path = os.path.join(MODELS_DIR, "forecast_model.pkl")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        
        model_data = joblib.load(path)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.algorithm = model_data["algorithm"]
        self.config = model_data.get("config", self.config)
        self.metrics = model_data.get("metrics", {})
        self._fitted = True
        
        trained_at = model_data.get("trained_at")
        if trained_at:
            self._last_train_date = datetime.fromisoformat(trained_at)
        
        logger.info(f"Model loaded from {path}")


class ARIMAForecastModel:
    """
    ARIMA-based time series forecasting model.
    
    For comparison with ML models and as a baseline.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0)):
        """
        Initialize ARIMA model.
        
        Args:
            order: ARIMA (p, d, q) parameters
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ARIMA. Install with: pip install statsmodels")
        
        self.order = order
        self.model = None
        self.fitted_model = None
        self._fitted = False
    
    def train(self, y: pd.Series) -> Dict[str, float]:
        """
        Train ARIMA model on time series.
        
        Args:
            y: Time series data
        
        Returns:
            Model information
        """
        logger.info(f"Training ARIMA{self.order} model...")
        
        self.model = ARIMA(y, order=self.order)
        self.fitted_model = self.model.fit()
        self._fitted = True
        
        return {
            "aic": self.fitted_model.aic,
            "bic": self.fitted_model.bic,
        }
    
    def forecast(self, horizon: int = 7) -> Dict[str, Any]:
        """
        Generate forecast.
        
        Args:
            horizon: Days to forecast
        
        Returns:
            Forecast dictionary
        """
        if not self._fitted:
            raise RuntimeError("Model not trained")
        
        forecast = self.fitted_model.forecast(steps=horizon)
        conf_int = self.fitted_model.get_forecast(steps=horizon).conf_int()
        
        return {
            "values": forecast.tolist(),
            "confidence_lower": conf_int.iloc[:, 0].tolist(),
            "confidence_upper": conf_int.iloc[:, 1].tolist(),
        }


class EnsembleForecastModel:
    """
    Ensemble of multiple forecasting models.
    
    Combines predictions from RF, Gradient Boosting, and optionally ARIMA.
    """
    
    def __init__(self, include_arima: bool = False):
        """
        Initialize ensemble model.
        
        Args:
            include_arima: Whether to include ARIMA in ensemble
        """
        self.models = {
            "random_forest": SalesForecastModel("random_forest"),
            "gradient_boosting": SalesForecastModel("gradient_boosting"),
        }
        
        if include_arima and STATSMODELS_AVAILABLE:
            self.models["arima"] = ARIMAForecastModel()
        
        self.weights = {name: 1.0 / len(self.models) for name in self.models}
        self._fitted = False
    
    def train(
        self,
        df: pd.DataFrame,
        target_column: str = "total_revenue"
    ) -> Dict[str, Any]:
        """
        Train all models in ensemble.
        
        Args:
            df: Training data
            target_column: Target column
        
        Returns:
            Training results for all models
        """
        results = {}
        
        for name, model in self.models.items():
            if isinstance(model, SalesForecastModel):
                X, y = model.prepare_features(df, target_column)
                metrics = model.train(X, y)
                results[name] = metrics
            elif isinstance(model, ARIMAForecastModel):
                y = df[target_column]
                info = model.train(y)
                results[name] = info
        
        self._fitted = True
        
        # Adjust weights based on performance
        self._update_weights(results)
        
        return results
    
    def _update_weights(self, results: Dict[str, Dict]) -> None:
        """Update model weights based on performance."""
        # Use inverse MAPE for weighting
        mape_scores = {}
        for name, metrics in results.items():
            if "mape" in metrics:
                mape_scores[name] = metrics["mape"]
        
        if mape_scores:
            total_inv_mape = sum(1 / m for m in mape_scores.values() if m > 0)
            for name, mape in mape_scores.items():
                if mape > 0:
                    self.weights[name] = (1 / mape) / total_inv_mape
    
    def forecast(
        self,
        historical_data: pd.DataFrame,
        horizon: int = 7
    ) -> Dict[str, Any]:
        """
        Generate ensemble forecast.
        
        Args:
            historical_data: Historical data
            horizon: Forecast horizon
        
        Returns:
            Weighted ensemble forecast
        """
        all_forecasts = {}
        
        for name, model in self.models.items():
            if isinstance(model, SalesForecastModel):
                forecast = model.forecast(historical_data, horizon)
                all_forecasts[name] = forecast["values"]
            elif isinstance(model, ARIMAForecastModel):
                forecast = model.forecast(horizon)
                all_forecasts[name] = forecast["values"]
        
        # Weighted average
        ensemble_values = np.zeros(horizon)
        for name, values in all_forecasts.items():
            weight = self.weights.get(name, 1 / len(all_forecasts))
            ensemble_values += weight * np.array(values)
        
        # Calculate uncertainty as std of model predictions
        all_values = np.array(list(all_forecasts.values()))
        uncertainty = np.std(all_values, axis=0)
        
        return {
            "values": ensemble_values.tolist(),
            "confidence_lower": (ensemble_values - 1.96 * uncertainty).tolist(),
            "confidence_upper": (ensemble_values + 1.96 * uncertainty).tolist(),
            "individual_forecasts": all_forecasts,
            "model_weights": self.weights,
        }
