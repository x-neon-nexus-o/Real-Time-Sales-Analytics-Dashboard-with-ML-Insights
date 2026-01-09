"""
Configuration settings for the Sales Analytics Dashboard.
Contains all configurable parameters for data generation, ML models, and API settings.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
STREAMING_DATA_DIR = DATA_DIR / "streaming"

# Model directory
MODELS_DIR = BASE_DIR / "models"

# Frontend directories
FRONTEND_DIR = BASE_DIR / "frontend"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, STREAMING_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Dataset settings
DATASET_CONFIG = {
    "num_transactions": 15000,  # Number of historical transactions to generate
    "date_range_days": 365,     # Historical data span in days
    "start_date": datetime.now() - timedelta(days=365),
    "end_date": datetime.now(),
}

# Product categories with weights (probability of selection)
CATEGORIES = {
    "Electronics": {"weight": 0.25, "avg_price": 299.99, "price_std": 150.0},
    "Clothing": {"weight": 0.20, "avg_price": 59.99, "price_std": 30.0},
    "Food & Beverages": {"weight": 0.18, "avg_price": 24.99, "price_std": 15.0},
    "Home & Garden": {"weight": 0.15, "avg_price": 89.99, "price_std": 50.0},
    "Sports & Outdoors": {"weight": 0.12, "avg_price": 79.99, "price_std": 40.0},
    "Books & Media": {"weight": 0.10, "avg_price": 19.99, "price_std": 10.0},
}

# Products per category
PRODUCTS = {
    "Electronics": [
        "Wireless Headphones", "Smartphone", "Laptop", "Tablet", "Smart Watch",
        "Bluetooth Speaker", "Camera", "Gaming Console", "Monitor", "Keyboard"
    ],
    "Clothing": [
        "T-Shirt", "Jeans", "Jacket", "Sneakers", "Dress",
        "Hoodie", "Shorts", "Sweater", "Boots", "Cap"
    ],
    "Food & Beverages": [
        "Organic Coffee", "Green Tea", "Protein Bar", "Energy Drink", "Snack Pack",
        "Gourmet Chocolate", "Vitamin Supplements", "Bottled Water", "Juice", "Nuts"
    ],
    "Home & Garden": [
        "Table Lamp", "Plant Pot", "Throw Pillow", "Wall Art", "Candle Set",
        "Kitchen Organizer", "Bath Towel Set", "Rug", "Curtains", "Vase"
    ],
    "Sports & Outdoors": [
        "Yoga Mat", "Dumbbell Set", "Running Shoes", "Fitness Tracker", "Water Bottle",
        "Backpack", "Tent", "Bicycle Helmet", "Resistance Bands", "Jump Rope"
    ],
    "Books & Media": [
        "Bestseller Novel", "Self-Help Book", "Cookbook", "Magazine Subscription", "Audiobook",
        "E-Book Reader", "Art Book", "Journal", "Calendar", "Educational Course"
    ],
}

# Regions with weights
REGIONS = {
    "North America": {"weight": 0.35, "timezone_offset": -5},
    "Europe": {"weight": 0.28, "timezone_offset": 1},
    "Asia Pacific": {"weight": 0.22, "timezone_offset": 8},
    "Latin America": {"weight": 0.10, "timezone_offset": -3},
    "Middle East & Africa": {"weight": 0.05, "timezone_offset": 3},
}

# Payment methods
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Digital Wallet"]

# =============================================================================
# REAL-TIME DATA GENERATION
# =============================================================================

STREAMING_CONFIG = {
    "generation_interval_seconds": 5,      # Generate new transaction every N seconds
    "buffer_size": 10000,                  # Keep last N transactions in memory
    "batch_size": 1,                       # Transactions per generation cycle
    "anomaly_probability": 0.02,           # 2% chance of generating anomaly
}

# Time-based patterns (multipliers for different hours)
HOURLY_PATTERNS = {
    0: 0.3, 1: 0.2, 2: 0.15, 3: 0.1, 4: 0.1, 5: 0.15,
    6: 0.3, 7: 0.5, 8: 0.7, 9: 0.9, 10: 1.0, 11: 1.1,
    12: 1.2, 13: 1.1, 14: 1.0, 15: 1.0, 16: 1.1, 17: 1.2,
    18: 1.3, 19: 1.4, 20: 1.3, 21: 1.1, 22: 0.8, 23: 0.5,
}

# Day of week patterns (Monday=0, Sunday=6)
DAILY_PATTERNS = {
    0: 0.85,  # Monday
    1: 0.90,  # Tuesday
    2: 0.95,  # Wednesday
    3: 1.00,  # Thursday
    4: 1.15,  # Friday
    5: 1.30,  # Saturday
    6: 1.20,  # Sunday
}

# Monthly patterns (seasonal trends)
MONTHLY_PATTERNS = {
    1: 0.85, 2: 0.80, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.05,
    7: 1.00, 8: 0.95, 9: 1.00, 10: 1.10, 11: 1.30, 12: 1.50,
}

# =============================================================================
# MACHINE LEARNING CONFIGURATION
# =============================================================================

# Forecasting model settings
FORECAST_CONFIG = {
    "algorithm": "random_forest",  # Options: random_forest, gradient_boosting, arima
    "short_term_horizon": 7,       # Days for short-term forecast
    "long_term_horizon": 30,       # Days for long-term forecast
    "confidence_level": 0.95,      # For confidence intervals
    "train_test_split": 0.8,       # 80% training, 20% testing
    "random_state": 42,
    
    # Random Forest parameters
    "rf_params": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    
    # ARIMA parameters (auto-selected if None)
    "arima_params": {
        "order": None,             # Will be auto-selected
        "seasonal_order": None,    # Will be auto-selected
        "trend": "c",
    },
}

# Anomaly detection settings
ANOMALY_CONFIG = {
    "algorithm": "isolation_forest",  # Options: isolation_forest, zscore, iqr
    "contamination": 0.05,            # Expected proportion of outliers
    "zscore_threshold": 3.0,          # Z-score threshold for statistical method
    "iqr_multiplier": 1.5,            # IQR multiplier for box plot method
    
    # Isolation Forest parameters
    "if_params": {
        "n_estimators": 100,
        "contamination": 0.05,
        "random_state": 42,
        "n_jobs": -1,
    },
    
    # Severity thresholds (deviation from mean)
    "severity_thresholds": {
        "low": 2.0,       # 2-3 standard deviations
        "medium": 3.0,    # 3-4 standard deviations
        "high": 4.0,      # 4-5 standard deviations
        "critical": 5.0,  # >5 standard deviations
    },
}

# Feature engineering settings
FEATURE_CONFIG = {
    "lag_periods": [1, 7, 14, 30],              # Lag features in days
    "rolling_windows": [7, 14, 30],             # Rolling statistics windows
    "include_time_features": True,              # Hour, day_of_week, month, etc.
    "include_lag_features": True,               # Previous day sales, etc.
    "include_rolling_features": True,           # Rolling mean, std, etc.
    "include_growth_features": True,            # Growth rates
}

# =============================================================================
# API CONFIGURATION
# =============================================================================

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
    "threaded": True,
    
    # CORS settings
    "cors_origins": "*",
    
    # Rate limiting
    "rate_limit": "100 per minute",
    
    # Response settings
    "default_page_size": 100,
    "max_page_size": 1000,
}

# =============================================================================
# SCHEDULER CONFIGURATION
# =============================================================================

SCHEDULER_CONFIG = {
    "data_generation_interval": 5,      # seconds
    "inference_interval": 30,           # seconds
    "anomaly_detection_interval": 60,   # seconds
    "insight_generation_interval": 120, # seconds
    "model_retrain_hour": 2,            # 2 AM daily
}

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================

DASHBOARD_CONFIG = {
    "refresh_interval_ms": 10000,      # Frontend refresh rate in milliseconds
    "chart_animation_duration": 750,   # Chart animation duration in ms
    "max_chart_points": 100,           # Maximum data points per chart
    "default_date_range_days": 30,     # Default view range
    "theme": "dark",                   # Default theme: dark or light
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "file": BASE_DIR / "logs" / "dashboard.log",
}

# Create logs directory
(BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)
