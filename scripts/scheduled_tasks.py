"""
Scheduled Tasks for PythonAnywhere.

This script should be scheduled to run periodically using PythonAnywhere's
Scheduled Tasks feature (available even on free tier).

Usage:
1. Go to PythonAnywhere Dashboard -> Tasks
2. Add a new scheduled task
3. Set the command to:
   /home/yourusername/.virtualenvs/salesenv/bin/python /home/yourusername/Real-Time-Sales-Analytics-Dashboard/scripts/scheduled_tasks.py

4. Set the frequency (e.g., daily, hourly)
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# =============================================================================
# PATH SETUP
# =============================================================================

# Get the project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / 'backend'

# Add to Python path
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# IMPORTS
# =============================================================================

from utils import setup_logger
from config import RAW_DATA_DIR, MODELS_DIR

logger = setup_logger("scheduled_tasks")

# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def refresh_data():
    """
    Refresh data from external sources or generate new simulated data.
    """
    logger.info("Starting data refresh task...")
    
    try:
        from data_ingestion import DataIngestionService
        
        historical_path = RAW_DATA_DIR / "sales_data.csv"
        service = DataIngestionService(
            historical_data_path=str(historical_path) if historical_path.exists() else None,
            enable_streaming=False
        )
        
        # Generate some new transactions
        for _ in range(10):
            transaction = service.generator.generate_transaction()
            service.generator.add_transaction(transaction)
        
        logger.info("Data refresh completed successfully")
        
    except Exception as e:
        logger.error(f"Data refresh failed: {e}")


def update_forecasts():
    """
    Update sales forecasts with latest data.
    """
    logger.info("Starting forecast update task...")
    
    try:
        from data_ingestion import DataIngestionService
        from data_preprocessing import DataPreprocessor
        from ml_forecast import SalesForecastModel
        
        # Load data
        historical_path = RAW_DATA_DIR / "sales_data.csv"
        if not historical_path.exists():
            logger.warning("No historical data found, skipping forecast update")
            return
        
        service = DataIngestionService(
            historical_data_path=str(historical_path),
            enable_streaming=False
        )
        
        df = service.get_data()
        if df.empty:
            logger.warning("Empty dataset, skipping forecast update")
            return
        
        # Preprocess
        preprocessor = DataPreprocessor()
        df = preprocessor.clean_data(df)
        df = preprocessor.engineer_features(df)
        
        # Load or create model
        model = SalesForecastModel(algorithm="random_forest")
        model_path = MODELS_DIR / "forecast_model.pkl"
        
        if model_path.exists():
            model.load_model(str(model_path))
        
        # Retrain with latest data (if enough data)
        if len(df) >= 100:
            X, y = preprocessor.prepare_ml_features(df)
            if X is not None and y is not None:
                model.train(X, y)
                model.save_model(str(model_path))
                logger.info("Forecast model updated and saved")
        
        logger.info("Forecast update completed successfully")
        
    except Exception as e:
        logger.error(f"Forecast update failed: {e}")


def detect_anomalies():
    """
    Run anomaly detection on latest data.
    """
    logger.info("Starting anomaly detection task...")
    
    try:
        from data_ingestion import DataIngestionService
        from data_preprocessing import DataPreprocessor
        from ml_anomaly import AnomalyDetector
        
        # Load data
        historical_path = RAW_DATA_DIR / "sales_data.csv"
        if not historical_path.exists():
            logger.warning("No historical data found, skipping anomaly detection")
            return
        
        service = DataIngestionService(
            historical_data_path=str(historical_path),
            enable_streaming=False
        )
        
        df = service.get_data()
        if df.empty:
            logger.warning("Empty dataset, skipping anomaly detection")
            return
        
        # Preprocess
        preprocessor = DataPreprocessor()
        df = preprocessor.clean_data(df)
        df = preprocessor.engineer_features(df)
        
        # Load or create detector
        detector = AnomalyDetector(algorithm="isolation_forest")
        model_path = MODELS_DIR / "anomaly_model.pkl"
        
        if model_path.exists():
            detector.load_model(str(model_path))
        
        # Run detection
        X, _ = preprocessor.prepare_ml_features(df)
        if X is not None:
            results = detector.detect(X)
            anomaly_count = sum(1 for r in results if r.get('is_anomaly', False))
            logger.info(f"Detected {anomaly_count} anomalies in {len(results)} records")
        
        logger.info("Anomaly detection completed successfully")
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")


def generate_insights():
    """
    Generate and cache insights from latest data.
    """
    logger.info("Starting insights generation task...")
    
    try:
        from data_ingestion import DataIngestionService
        from data_preprocessing import DataPreprocessor
        from insights_engine import InsightsEngine
        from datetime import timedelta
        import pandas as pd
        
        # Load data
        historical_path = RAW_DATA_DIR / "sales_data.csv"
        if not historical_path.exists():
            logger.warning("No historical data found, skipping insights generation")
            return
        
        service = DataIngestionService(
            historical_data_path=str(historical_path),
            enable_streaming=False
        )
        
        df = service.get_data()
        if df.empty:
            logger.warning("Empty dataset, skipping insights generation")
            return
        
        # Preprocess
        preprocessor = DataPreprocessor()
        df = preprocessor.clean_data(df)
        df = preprocessor.engineer_features(df)
        
        # Split into current and historical
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now() - timedelta(days=30)
        current = df[df["timestamp"] >= cutoff]
        historical = df[df["timestamp"] < cutoff]
        
        # Generate insights
        engine = InsightsEngine()
        engine.set_data(current, historical)
        
        insights = engine.generate_all_insights()
        logger.info(f"Generated {len(insights)} insights")
        
        logger.info("Insights generation completed successfully")
        
    except Exception as e:
        logger.error(f"Insights generation failed: {e}")


def cleanup_logs():
    """
    Clean up old log files to save disk space.
    """
    logger.info("Starting log cleanup task...")
    
    try:
        logs_dir = PROJECT_ROOT / 'logs'
        if not logs_dir.exists():
            return
        
        from datetime import timedelta
        import time
        
        cutoff_time = time.time() - (7 * 24 * 60 * 60)  # 7 days ago
        
        deleted_count = 0
        for log_file in logs_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old log files")
        
    except Exception as e:
        logger.error(f"Log cleanup failed: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_tasks():
    """
    Run all scheduled tasks.
    """
    logger.info("=" * 50)
    logger.info(f"Starting scheduled tasks at {datetime.now().isoformat()}")
    logger.info("=" * 50)
    
    tasks = [
        ("Data Refresh", refresh_data),
        ("Forecast Update", update_forecasts),
        ("Anomaly Detection", detect_anomalies),
        ("Insights Generation", generate_insights),
        ("Log Cleanup", cleanup_logs),
    ]
    
    for task_name, task_func in tasks:
        logger.info(f"\n--- Running: {task_name} ---")
        try:
            task_func()
        except Exception as e:
            logger.error(f"Task '{task_name}' failed with error: {e}")
    
    logger.info("=" * 50)
    logger.info(f"All scheduled tasks completed at {datetime.now().isoformat()}")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_all_tasks()
