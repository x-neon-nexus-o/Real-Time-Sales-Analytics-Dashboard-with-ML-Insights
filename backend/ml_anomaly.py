"""
Anomaly Detection Module.
Implements ML-based and statistical anomaly detection for sales data.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib

from config import ANOMALY_CONFIG, MODELS_DIR
from utils import (
    setup_logger, calculate_zscore, calculate_iqr,
    round_currency, format_currency, format_percentage
)

logger = setup_logger("ml_anomaly")


class AnomalyDetector:
    """
    Multi-algorithm anomaly detection for sales data.
    
    Supports:
    - Isolation Forest (unsupervised ML)
    - Z-Score (statistical)
    - IQR (statistical)
    - Local Outlier Factor (density-based)
    """
    
    # Severity levels based on deviation
    SEVERITY_LEVELS = {
        "low": (2.0, 3.0),
        "medium": (3.0, 4.0),
        "high": (4.0, 5.0),
        "critical": (5.0, float("inf")),
    }
    
    def __init__(
        self,
        algorithm: str = "isolation_forest",
        contamination: float = 0.05,
        config: Optional[Dict] = None
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            algorithm: Detection algorithm to use
            contamination: Expected proportion of outliers
            config: Optional configuration overrides
        """
        self.algorithm = algorithm
        self.contamination = contamination
        self.config = config or ANOMALY_CONFIG
        
        # Model instances
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        
        # Training statistics
        self._stats: Dict[str, float] = {}
        self._fitted = False
        
        # Detected anomalies history
        self._anomaly_history: List[Dict] = []
        self._max_history = 1000
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"AnomalyDetector initialized with algorithm: {algorithm}")
    
    def _initialize_model(self) -> None:
        """Initialize the anomaly detection model."""
        if self.algorithm == "isolation_forest":
            params = self.config.get("if_params", {})
            self.model = IsolationForest(
                n_estimators=params.get("n_estimators", 100),
                contamination=params.get("contamination", self.contamination),
                random_state=params.get("random_state", 42),
                n_jobs=params.get("n_jobs", -1)
            )
        
        elif self.algorithm == "lof":
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True  # Enable predict on new data
            )
        
        elif self.algorithm == "dbscan":
            self.model = DBSCAN(eps=0.5, min_samples=5)
        
        else:
            # Default to Isolation Forest
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
    
    def train(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train the anomaly detection model.
        
        Args:
            df: Training DataFrame
            feature_columns: Columns to use for detection
        
        Returns:
            Training results
        """
        logger.info(f"Training anomaly detector on {len(df)} samples...")
        
        # Determine feature columns
        if feature_columns:
            self.feature_columns = feature_columns
        else:
            # Default to numeric columns
            self.feature_columns = [
                col for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col])
                and col not in ["is_anomaly", "is_extreme", "anomaly_score"]
            ]
        
        X = df[self.feature_columns].copy()
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate statistics for Z-score method
        self._stats = {
            col: {"mean": df[col].mean(), "std": df[col].std()}
            for col in self.feature_columns
            if pd.api.types.is_numeric_dtype(df[col])
        }
        
        # Overall stats for primary metric
        if "total_revenue" in df.columns:
            self._stats["revenue_mean"] = df["total_revenue"].mean()
            self._stats["revenue_std"] = df["total_revenue"].std()
        if "total_amount" in df.columns:
            self._stats["amount_mean"] = df["total_amount"].mean()
            self._stats["amount_std"] = df["total_amount"].std()
        
        # Train model
        if self.algorithm in ["isolation_forest", "lof"]:
            self.model.fit(X_scaled)
        elif self.algorithm == "dbscan":
            # DBSCAN doesn't need fit for novelty detection
            self.model.fit(X_scaled)
        
        self._fitted = True
        
        logger.info("Anomaly detector training complete")
        
        return {
            "samples_trained": len(df),
            "features_used": len(self.feature_columns),
            "algorithm": self.algorithm,
            "contamination": self.contamination,
        }
    
    def detect(
        self,
        df: pd.DataFrame,
        return_scores: bool = True
    ) -> pd.DataFrame:
        """
        Detect anomalies in data.
        
        Args:
            df: DataFrame to analyze
            return_scores: Whether to include anomaly scores
        
        Returns:
            DataFrame with anomaly labels and scores
        """
        if not self._fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        
        df = df.copy()
        
        # Prepare features
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions based on algorithm
        if self.algorithm == "isolation_forest":
            # predict returns -1 for anomalies, 1 for normal
            predictions = self.model.predict(X_scaled)
            df["is_anomaly_ml"] = (predictions == -1).astype(int)
            
            if return_scores:
                # decision_function returns anomaly scores (lower = more anomalous)
                scores = self.model.decision_function(X_scaled)
                df["anomaly_score"] = -scores  # Invert so higher = more anomalous
        
        elif self.algorithm == "lof":
            predictions = self.model.predict(X_scaled)
            df["is_anomaly_ml"] = (predictions == -1).astype(int)
            
            if return_scores:
                scores = self.model.decision_function(X_scaled)
                df["anomaly_score"] = -scores
        
        elif self.algorithm == "dbscan":
            labels = self.model.fit_predict(X_scaled)
            df["is_anomaly_ml"] = (labels == -1).astype(int)
            df["anomaly_score"] = 0  # DBSCAN doesn't provide scores
        
        # Add statistical detection
        df = self._add_statistical_anomalies(df)
        
        # Combine ML and statistical detection
        df["is_anomaly"] = (
            (df.get("is_anomaly_ml", 0) == 1) |
            (df.get("is_anomaly_zscore", 0) == 1)
        ).astype(int)
        
        # Add severity classification
        if "anomaly_score" in df.columns:
            df["severity"] = df["anomaly_score"].apply(self._classify_severity)
        
        return df
    
    def _add_statistical_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical anomaly detection (Z-score and IQR).
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            DataFrame with statistical anomaly flags
        """
        threshold = self.config.get("zscore_threshold", 3.0)
        
        # Z-score detection on revenue/amount columns
        for col in ["total_revenue", "total_amount"]:
            if col in df.columns and f"{col.split('_')[1]}_mean" in self._stats:
                mean = self._stats[f"{col.split('_')[1]}_mean"]
                std = self._stats[f"{col.split('_')[1]}_std"]
                
                if std > 0:
                    df[f"{col}_zscore"] = (df[col] - mean) / std
                    df["is_anomaly_zscore"] = (
                        df.get("is_anomaly_zscore", 0) |
                        (np.abs(df[f"{col}_zscore"]) > threshold)
                    ).astype(int)
        
        return df
    
    def _classify_severity(self, score: float) -> str:
        """
        Classify anomaly severity based on score.
        
        Args:
            score: Anomaly score
        
        Returns:
            Severity level string
        """
        abs_score = abs(score)
        
        for level, (low, high) in self.SEVERITY_LEVELS.items():
            if low <= abs_score < high:
                return level
        
        return "low"
    
    def detect_single(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect if a single data point is anomalous.
        
        Args:
            data_point: Dictionary with feature values
        
        Returns:
            Detection result
        """
        if not self._fitted:
            raise RuntimeError("Model not trained")
        
        # Create single-row DataFrame
        df = pd.DataFrame([data_point])
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Detect
        result = self.detect(df)
        row = result.iloc[0]
        
        detection = {
            "is_anomaly": bool(row.get("is_anomaly", False)),
            "anomaly_score": float(row.get("anomaly_score", 0)),
            "severity": row.get("severity", "low"),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add to history
        if detection["is_anomaly"]:
            self._add_to_history({**data_point, **detection})
        
        return detection
    
    def _add_to_history(self, anomaly: Dict[str, Any]) -> None:
        """Add anomaly to history."""
        self._anomaly_history.append(anomaly)
        
        # Trim history if needed
        if len(self._anomaly_history) > self._max_history:
            self._anomaly_history = self._anomaly_history[-self._max_history:]
    
    def explain_anomaly(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation for why a data point is anomalous.
        
        Args:
            data_point: Data point to explain
        
        Returns:
            Explanation dictionary
        """
        explanations = []
        contributing_factors = []
        
        # Check each feature for deviation
        for col in self.feature_columns:
            if col in data_point and col in self._stats:
                value = data_point[col]
                mean = self._stats[col]["mean"]
                std = self._stats[col]["std"]
                
                if std > 0:
                    zscore = (value - mean) / std
                    
                    if abs(zscore) > 2:
                        direction = "higher" if zscore > 0 else "lower"
                        deviation_pct = abs((value - mean) / mean * 100) if mean != 0 else 0
                        
                        contributing_factors.append({
                            "feature": col,
                            "value": value,
                            "expected": mean,
                            "zscore": zscore,
                            "direction": direction,
                            "deviation_percentage": deviation_pct,
                        })
                        
                        explanations.append(
                            f"{col} is {abs(deviation_pct):.1f}% {direction} than expected"
                        )
        
        # Sort by deviation magnitude
        contributing_factors.sort(key=lambda x: abs(x["zscore"]), reverse=True)
        
        # Generate summary
        if contributing_factors:
            top_factor = contributing_factors[0]
            summary = f"Primary cause: {top_factor['feature']} is {top_factor['direction']} than normal"
        else:
            summary = "Unusual pattern detected across multiple features"
        
        return {
            "summary": summary,
            "explanations": explanations[:5],  # Top 5 explanations
            "contributing_factors": contributing_factors[:5],
            "recommendation": self._generate_recommendation(contributing_factors),
        }
    
    def _generate_recommendation(self, factors: List[Dict]) -> str:
        """Generate actionable recommendation based on anomaly factors."""
        if not factors:
            return "Monitor this data point for recurring patterns"
        
        top_factor = factors[0]
        feature = top_factor["feature"]
        direction = top_factor["direction"]
        
        recommendations = {
            "total_revenue": {
                "higher": "Investigate source of revenue spike - potential promotion success or data error",
                "lower": "Urgent: Revenue drop detected - check for system issues or market changes",
            },
            "total_amount": {
                "higher": "Large transaction detected - verify for fraud or bulk order",
                "lower": "Unusually small transaction - may indicate pricing error",
            },
            "quantity": {
                "higher": "High volume order - ensure inventory availability",
                "lower": "Low volume despite normal revenue - check pricing",
            },
            "transaction_count": {
                "higher": "Transaction spike - monitor server capacity",
                "lower": "Transaction drop - investigate potential system issues",
            },
        }
        
        if feature in recommendations and direction in recommendations[feature]:
            return recommendations[feature][direction]
        
        return f"Review {feature} for unusual {direction} values"
    
    def get_anomaly_report(
        self,
        df: pd.DataFrame,
        time_column: str = "timestamp"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive anomaly report.
        
        Args:
            df: DataFrame with anomaly detection results
            time_column: Timestamp column name
        
        Returns:
            Report dictionary
        """
        anomalies = df[df.get("is_anomaly", False) == True].copy()
        
        if anomalies.empty:
            return {
                "total_anomalies": 0,
                "message": "No anomalies detected in the dataset",
            }
        
        # Severity breakdown
        severity_counts = anomalies["severity"].value_counts().to_dict() if "severity" in anomalies.columns else {}
        
        # Time analysis
        if time_column in anomalies.columns:
            anomalies[time_column] = pd.to_datetime(anomalies[time_column])
            
            # Anomalies by time period
            anomalies["date"] = anomalies[time_column].dt.date
            daily_counts = anomalies.groupby("date").size().to_dict()
            
            # Recent anomalies (last 24 hours)
            recent_threshold = datetime.now() - timedelta(hours=24)
            recent_anomalies = anomalies[anomalies[time_column] >= recent_threshold]
        else:
            daily_counts = {}
            recent_anomalies = pd.DataFrame()
        
        # Category/Region analysis
        category_breakdown = {}
        region_breakdown = {}
        
        if "category" in anomalies.columns:
            category_breakdown = anomalies["category"].value_counts().to_dict()
        
        if "region" in anomalies.columns:
            region_breakdown = anomalies["region"].value_counts().to_dict()
        
        return {
            "total_anomalies": len(anomalies),
            "anomaly_rate": len(anomalies) / len(df) * 100,
            "severity_breakdown": severity_counts,
            "daily_counts": {str(k): v for k, v in daily_counts.items()},
            "recent_count_24h": len(recent_anomalies),
            "by_category": category_breakdown,
            "by_region": region_breakdown,
            "critical_count": severity_counts.get("critical", 0),
            "high_count": severity_counts.get("high", 0),
            "generated_at": datetime.now().isoformat(),
        }
    
    def get_recent_anomalies(
        self,
        n: int = 10,
        min_severity: str = "low"
    ) -> List[Dict[str, Any]]:
        """
        Get recent anomalies from history.
        
        Args:
            n: Maximum number to return
            min_severity: Minimum severity level to include
        
        Returns:
            List of recent anomalies
        """
        severity_order = ["low", "medium", "high", "critical"]
        min_idx = severity_order.index(min_severity)
        
        filtered = [
            a for a in self._anomaly_history
            if severity_order.index(a.get("severity", "low")) >= min_idx
        ]
        
        return filtered[-n:]
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            path: Save path
        
        Returns:
            Path where saved
        """
        if not self._fitted:
            raise RuntimeError("Cannot save untrained model")
        
        if path is None:
            path = os.path.join(MODELS_DIR, "anomaly_model.pkl")
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "algorithm": self.algorithm,
            "contamination": self.contamination,
            "config": self.config,
            "stats": self._stats,
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Anomaly model saved to {path}")
        
        return path
    
    def load_model(self, path: Optional[str] = None) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to model file
        """
        if path is None:
            path = os.path.join(MODELS_DIR, "anomaly_model.pkl")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        
        model_data = joblib.load(path)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.algorithm = model_data["algorithm"]
        self.contamination = model_data.get("contamination", 0.05)
        self.config = model_data.get("config", self.config)
        self._stats = model_data.get("stats", {})
        self._fitted = True
        
        logger.info(f"Anomaly model loaded from {path}")


class MultiDimensionalAnomalyDetector:
    """
    Detects anomalies across multiple dimensions simultaneously.
    
    Useful for detecting complex patterns like:
    - High volume but low revenue
    - Regional underperformance
    - Category-specific anomalies
    """
    
    def __init__(self):
        """Initialize multi-dimensional detector."""
        self.detectors: Dict[str, AnomalyDetector] = {}
        self._thresholds: Dict[str, Dict] = {}
    
    def add_dimension(
        self,
        name: str,
        feature_columns: List[str],
        algorithm: str = "isolation_forest"
    ) -> None:
        """
        Add a detection dimension.
        
        Args:
            name: Dimension name
            feature_columns: Features for this dimension
            algorithm: Detection algorithm
        """
        self.detectors[name] = AnomalyDetector(algorithm=algorithm)
        self.detectors[name].feature_columns = feature_columns
    
    def train_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all dimension detectors."""
        results = {}
        
        for name, detector in self.detectors.items():
            try:
                result = detector.train(df, detector.feature_columns)
                results[name] = result
            except Exception as e:
                logger.error(f"Error training {name} detector: {e}")
                results[name] = {"error": str(e)}
        
        return results
    
    def detect_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies across all dimensions."""
        df = df.copy()
        
        for name, detector in self.detectors.items():
            try:
                result = detector.detect(df)
                df[f"anomaly_{name}"] = result["is_anomaly"]
                df[f"score_{name}"] = result.get("anomaly_score", 0)
            except Exception as e:
                logger.error(f"Error detecting in {name}: {e}")
        
        # Combined anomaly flag
        anomaly_cols = [c for c in df.columns if c.startswith("anomaly_")]
        if anomaly_cols:
            df["is_multi_anomaly"] = df[anomaly_cols].any(axis=1).astype(int)
            df["anomaly_dimensions"] = df[anomaly_cols].sum(axis=1)
        
        return df
