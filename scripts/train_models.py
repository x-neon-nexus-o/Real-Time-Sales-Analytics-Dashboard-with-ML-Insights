"""
Model Training Script.
Trains the ML models on the generated sample data.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pandas as pd
from datetime import datetime

from config import RAW_DATA_DIR, MODELS_DIR
from data_preprocessing import DataPreprocessor
from ml_forecast import SalesForecastModel
from ml_anomaly import AnomalyDetector
from utils import setup_logger

logger = setup_logger("train_models")


def load_data():
    """Load the generated sample data."""
    data_path = RAW_DATA_DIR / "sales_data.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Sample data not found at {data_path}. "
            "Run 'python scripts/generate_sample_data.py' first."
        )
    
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df)} transactions from {data_path}")
    
    return df


def train_forecast_model(df: pd.DataFrame):
    """Train the sales forecasting model."""
    logger.info("Training forecast model...")
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Aggregate to daily data (includes time, lag, and rolling features)
    daily_df = preprocessor.aggregate_daily(df_clean)
    
    # Drop NaN rows after feature engineering
    daily_df = daily_df.dropna()
    
    logger.info(f"Prepared {len(daily_df)} daily records for training")
    
    # Initialize and train model
    model = SalesForecastModel(algorithm="random_forest")
    
    # Prepare features
    X, y = model.prepare_features(daily_df, target_column="total_revenue")
    
    # Train with cross-validation
    metrics = model.train_with_cv(X, y, n_splits=5)
    
    logger.info(f"Training metrics: MAPE={metrics['mape']:.2f}%, R²={metrics['r2']:.4f}")
    
    # Save model
    model_path = model.save_model(str(MODELS_DIR / "forecast_model.pkl"))
    logger.info(f"Model saved to {model_path}")
    
    # Generate sample forecast
    forecast = model.forecast(daily_df, horizon=7)
    logger.info(f"7-day forecast total: ${forecast['total_forecast']:,.2f}")
    
    return model, metrics


def train_anomaly_model(df: pd.DataFrame):
    """Train the anomaly detection model."""
    logger.info("Training anomaly detection model...")
    
    # Initialize detector
    detector = AnomalyDetector(algorithm="isolation_forest", contamination=0.02)
    
    # Select features for anomaly detection
    feature_columns = ["quantity", "unit_price", "total_amount"]
    
    # Train
    result = detector.train(df, feature_columns=feature_columns)
    logger.info(f"Trained on {result['samples_trained']} samples with {result['features_used']} features")
    
    # Save model
    model_path = detector.save_model(str(MODELS_DIR / "anomaly_model.pkl"))
    logger.info(f"Model saved to {model_path}")
    
    # Test detection
    test_df = detector.detect(df.head(100))
    anomaly_count = test_df["is_anomaly"].sum()
    logger.info(f"Test detection found {anomaly_count} anomalies in first 100 records")
    
    return detector


def main():
    """Main training function."""
    print("\n" + "=" * 60)
    print("  ML Model Training Script")
    print("=" * 60 + "\n")
    
    start_time = datetime.now()
    
    try:
        # Load data
        df = load_data()
        
        # Train forecast model
        forecast_model, forecast_metrics = train_forecast_model(df)
        
        # Train anomaly model
        anomaly_detector = train_anomaly_model(df)
        
        # Summary
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Forecast Model:")
        print(f"    - Algorithm: Random Forest")
        print(f"    - MAPE: {forecast_metrics['mape']:.2f}%")
        print(f"    - R²: {forecast_metrics['r2']:.4f}")
        print(f"    - Model saved to: models/forecast_model.pkl")
        print()
        print(f"  Anomaly Detector:")
        print(f"    - Algorithm: Isolation Forest")
        print(f"    - Contamination: 2%")
        print(f"    - Model saved to: models/anomaly_model.pkl")
        print()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"  Total training time: {elapsed:.1f} seconds")
        print("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\nError: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
