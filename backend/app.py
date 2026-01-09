"""
Flask Application Entry Point.
Main application that initializes all services and starts the dashboard.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

from config import (
    API_CONFIG, RAW_DATA_DIR, MODELS_DIR, 
    TEMPLATES_DIR, STATIC_DIR
)
from utils import setup_logger
from data_ingestion import DataIngestionService
from data_preprocessing import DataPreprocessor
from ml_forecast import SalesForecastModel
from ml_anomaly import AnomalyDetector
from insights_engine import InsightsEngine
from scheduler import DashboardScheduler
from api_routes import api_bp

logger = setup_logger("app")


def create_app(config=None):
    """
    Application factory function.
    
    Args:
        config: Optional configuration overrides
    
    Returns:
        Configured Flask application
    """
    # Create Flask app
    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR)
    )
    
    # Configure app
    app.config.update(
        DEBUG=API_CONFIG.get("debug", True),
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production"),
    )
    
    if config:
        app.config.update(config)
    
    # Enable CORS
    CORS(app, origins=API_CONFIG.get("cors_origins", "*"))
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Initialize services
    with app.app_context():
        initialize_services(app)
    
    # Register routes
    register_routes(app)
    
    logger.info("Flask application created successfully")
    
    return app


def initialize_services(app):
    """
    Initialize all backend services.
    
    Args:
        app: Flask application instance
    """
    logger.info("Initializing services...")
    
    # Initialize preprocessor
    app.preprocessor = DataPreprocessor()
    logger.info("DataPreprocessor initialized")
    
    # Initialize data ingestion service
    historical_path = RAW_DATA_DIR / "sales_data.csv"
    if historical_path.exists():
        app.data_service = DataIngestionService(
            historical_data_path=str(historical_path),
            enable_streaming=True,
            streaming_rate=5
        )
        logger.info("DataIngestionService initialized with historical data")
    else:
        app.data_service = DataIngestionService(
            enable_streaming=True,
            streaming_rate=5
        )
        logger.warning("No historical data found, starting with empty dataset")
    
    # Initialize forecast model
    app.forecast_model = SalesForecastModel(algorithm="random_forest")
    forecast_model_path = MODELS_DIR / "forecast_model.pkl"
    if forecast_model_path.exists():
        try:
            app.forecast_model.load_model(str(forecast_model_path))
            logger.info("Forecast model loaded from disk")
        except Exception as e:
            logger.warning(f"Could not load forecast model: {e}")
    
    # Initialize anomaly detector
    app.anomaly_detector = AnomalyDetector(algorithm="isolation_forest")
    anomaly_model_path = MODELS_DIR / "anomaly_model.pkl"
    if anomaly_model_path.exists():
        try:
            app.anomaly_detector.load_model(str(anomaly_model_path))
            logger.info("Anomaly model loaded from disk")
        except Exception as e:
            logger.warning(f"Could not load anomaly model: {e}")
    
    # Initialize insights engine
    app.insights_engine = InsightsEngine()
    logger.info("InsightsEngine initialized")
    
    # Initialize scheduler
    app.scheduler = DashboardScheduler()
    
    # Register scheduled tasks
    def generate_data_task():
        """Generate new transaction."""
        if hasattr(app, 'data_service') and app.data_service:
            transaction = app.data_service.generator.generate_transaction()
            app.data_service.generator.add_transaction(transaction)
    
    def run_inference_task():
        """Run model inference."""
        pass  # Inference runs on-demand via API
    
    def detect_anomalies_task():
        """Run anomaly detection."""
        pass  # Detection runs on-demand via API
    
    def update_insights_task():
        """Update insights."""
        if hasattr(app, 'insights_engine') and hasattr(app, 'data_service'):
            try:
                df = app.data_service.get_data()
                if not df.empty:
                    from datetime import timedelta
                    import pandas as pd
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    cutoff = datetime.now() - timedelta(days=30)
                    current = df[df["timestamp"] >= cutoff]
                    historical = df[df["timestamp"] < cutoff]
                    app.insights_engine.set_data(current, historical)
            except Exception as e:
                logger.error(f"Error updating insights: {e}")
    
    # Register tasks
    app.scheduler.register_task("generate_data", generate_data_task)
    app.scheduler.register_task("run_inference", run_inference_task)
    app.scheduler.register_task("detect_anomalies", detect_anomalies_task)
    app.scheduler.register_task("update_insights", update_insights_task)
    
    logger.info("All services initialized")


def register_routes(app):
    """
    Register application routes.
    
    Args:
        app: Flask application instance
    """
    
    @app.route("/")
    def index():
        """Serve the main dashboard page."""
        return render_template("index.html")
    
    @app.route("/static/<path:filename>")
    def serve_static(filename):
        """Serve static files."""
        return send_from_directory(app.static_folder, filename)
    
    @app.route("/health")
    def health():
        """Simple health check."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return {"error": "Not found", "status": 404}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        logger.error(f"Internal server error: {error}")
        return {"error": "Internal server error", "status": 500}, 500


def start_background_services(app):
    """
    Start background services.
    
    Args:
        app: Flask application instance
    """
    # Start data streaming
    if hasattr(app, 'data_service'):
        app.data_service.start()
        logger.info("Data streaming started")
    
    # Start scheduler
    if hasattr(app, 'scheduler'):
        app.scheduler.setup_default_jobs()
        app.scheduler.start()
        logger.info("Scheduler started")


def stop_background_services(app):
    """
    Stop background services.
    
    Args:
        app: Flask application instance
    """
    # Stop data streaming
    if hasattr(app, 'data_service'):
        app.data_service.stop()
        logger.info("Data streaming stopped")
    
    # Stop scheduler
    if hasattr(app, 'scheduler'):
        app.scheduler.stop()
        logger.info("Scheduler stopped")


def run_app(host=None, port=None, debug=None):
    """
    Run the Flask application.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        debug: Debug mode
    """
    host = host or API_CONFIG.get("host", "0.0.0.0")
    port = port or API_CONFIG.get("port", 5000)
    debug = debug if debug is not None else API_CONFIG.get("debug", True)
    
    app = create_app()
    
    # Start background services
    start_background_services(app)
    
    try:
        logger.info(f"Starting dashboard at http://{host}:{port}")
        print(f"\n{'='*50}")
        print(f"  Sales Analytics Dashboard")
        print(f"  Running at: http://localhost:{port}")
        print(f"  API Base: http://localhost:{port}/api")
        print(f"{'='*50}\n")
        
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=API_CONFIG.get("threaded", True),
            use_reloader=False  # Disable reloader to prevent duplicate background tasks
        )
    finally:
        stop_background_services(app)


# Application instance for WSGI servers
app = create_app()


if __name__ == "__main__":
    run_app()
