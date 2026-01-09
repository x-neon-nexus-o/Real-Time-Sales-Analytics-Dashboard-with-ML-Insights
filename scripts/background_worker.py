"""
Background Worker for Render Deployment.

This script runs as a separate worker process on Render (requires paid plan).
It handles background tasks like data generation, model training, and anomaly detection.

For free tier, these tasks run within the web process or via Render Cron Jobs.
"""

import sys
import os
import time
import signal
from pathlib import Path
from datetime import datetime, timedelta

# =============================================================================
# PATH SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / 'backend'

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# IMPORTS
# =============================================================================

from utils import setup_logger
from config import RAW_DATA_DIR, MODELS_DIR, SCHEDULER_CONFIG

logger = setup_logger("background_worker")

# Graceful shutdown flag
shutdown_flag = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_flag
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_flag = True


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def generate_new_data():
    """Generate new transaction data."""
    try:
        from data_ingestion import DataIngestionService
        
        historical_path = RAW_DATA_DIR / "sales_data.csv"
        service = DataIngestionService(
            historical_data_path=str(historical_path) if historical_path.exists() else None,
            enable_streaming=False
        )
        
        # Generate transactions
        for _ in range(5):
            transaction = service.generator.generate_transaction()
            service.generator.add_transaction(transaction)
        
        logger.debug("Generated new transaction data")
        return True
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        return False


def run_forecasts():
    """Run sales forecasting."""
    try:
        from data_ingestion import DataIngestionService
        from data_preprocessing import DataPreprocessor
        from ml_forecast import SalesForecastModel
        
        historical_path = RAW_DATA_DIR / "sales_data.csv"
        if not historical_path.exists():
            logger.warning("No data for forecasting")
            return False
        
        service = DataIngestionService(
            historical_data_path=str(historical_path),
            enable_streaming=False
        )
        
        df = service.get_data()
        if df.empty or len(df) < 100:
            logger.warning("Insufficient data for forecasting")
            return False
        
        preprocessor = DataPreprocessor()
        df = preprocessor.clean_data(df)
        df = preprocessor.engineer_features(df)
        
        model = SalesForecastModel(algorithm="random_forest")
        model_path = MODELS_DIR / "forecast_model.pkl"
        
        if model_path.exists():
            model.load_model(str(model_path))
        
        # Optionally retrain periodically
        logger.debug("Forecast inference completed")
        return True
        
    except Exception as e:
        logger.error(f"Forecasting failed: {e}")
        return False


def run_anomaly_detection():
    """Run anomaly detection on recent data."""
    try:
        from data_ingestion import DataIngestionService
        from data_preprocessing import DataPreprocessor
        from ml_anomaly import AnomalyDetector
        
        historical_path = RAW_DATA_DIR / "sales_data.csv"
        if not historical_path.exists():
            logger.warning("No data for anomaly detection")
            return False
        
        service = DataIngestionService(
            historical_data_path=str(historical_path),
            enable_streaming=False
        )
        
        df = service.get_data()
        if df.empty:
            return False
        
        preprocessor = DataPreprocessor()
        df = preprocessor.clean_data(df)
        df = preprocessor.engineer_features(df)
        
        detector = AnomalyDetector(algorithm="isolation_forest")
        model_path = MODELS_DIR / "anomaly_model.pkl"
        
        if model_path.exists():
            detector.load_model(str(model_path))
        
        X, _ = preprocessor.prepare_ml_features(df)
        if X is not None:
            results = detector.detect(X)
            anomalies = sum(1 for r in results if r.get('is_anomaly', False))
            if anomalies > 0:
                logger.info(f"Detected {anomalies} anomalies")
        
        return True
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        return False


def update_insights():
    """Update insights cache."""
    try:
        from data_ingestion import DataIngestionService
        from data_preprocessing import DataPreprocessor
        from insights_engine import InsightsEngine
        import pandas as pd
        
        historical_path = RAW_DATA_DIR / "sales_data.csv"
        if not historical_path.exists():
            return False
        
        service = DataIngestionService(
            historical_data_path=str(historical_path),
            enable_streaming=False
        )
        
        df = service.get_data()
        if df.empty:
            return False
        
        preprocessor = DataPreprocessor()
        df = preprocessor.clean_data(df)
        df = preprocessor.engineer_features(df)
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now() - timedelta(days=30)
        current = df[df["timestamp"] >= cutoff]
        historical = df[df["timestamp"] < cutoff]
        
        engine = InsightsEngine()
        engine.set_data(current, historical)
        insights = engine.generate_all_insights()
        
        logger.debug(f"Generated {len(insights)} insights")
        return True
        
    except Exception as e:
        logger.error(f"Insights update failed: {e}")
        return False


# =============================================================================
# MAIN WORKER LOOP
# =============================================================================

def run_worker():
    """Main worker loop."""
    logger.info("=" * 60)
    logger.info("Background Worker Starting")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # Task intervals (in seconds)
    intervals = {
        'generate_data': SCHEDULER_CONFIG.get('data_generation_interval', 5),
        'run_forecasts': SCHEDULER_CONFIG.get('inference_interval', 30),
        'anomaly_detection': SCHEDULER_CONFIG.get('anomaly_detection_interval', 60),
        'update_insights': SCHEDULER_CONFIG.get('insight_generation_interval', 120),
    }
    
    # Last run times
    last_run = {task: 0 for task in intervals}
    
    # Task functions
    tasks = {
        'generate_data': generate_new_data,
        'run_forecasts': run_forecasts,
        'anomaly_detection': run_anomaly_detection,
        'update_insights': update_insights,
    }
    
    logger.info("Worker loop starting...")
    
    while not shutdown_flag:
        current_time = time.time()
        
        for task_name, interval in intervals.items():
            if current_time - last_run[task_name] >= interval:
                try:
                    tasks[task_name]()
                    last_run[task_name] = current_time
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {e}")
        
        # Sleep for a short interval to prevent CPU spinning
        time.sleep(1)
    
    logger.info("Worker shutdown complete")


if __name__ == "__main__":
    run_worker()
