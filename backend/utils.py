"""
Utility functions for the Sales Analytics Dashboard.
Contains helper functions for data formatting, validation, logging, and common operations.
"""

import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with the specified name and level.
    
    Args:
        name: Logger name (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# Create default logger
logger = setup_logger("utils")


# =============================================================================
# DATE/TIME UTILITIES
# =============================================================================

def parse_datetime(value: Union[str, datetime]) -> datetime:
    """
    Parse a datetime from various formats.
    
    Args:
        value: Datetime string or datetime object
    
    Returns:
        Parsed datetime object
    """
    if isinstance(value, datetime):
        return value
    
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse datetime: {value}")


def format_datetime(dt: datetime, format_type: str = "iso") -> str:
    """
    Format a datetime to string.
    
    Args:
        dt: Datetime object
        format_type: Output format (iso, display, date_only, time_only)
    
    Returns:
        Formatted datetime string
    """
    formats = {
        "iso": "%Y-%m-%dT%H:%M:%SZ",
        "display": "%B %d, %Y %I:%M %p",
        "date_only": "%Y-%m-%d",
        "time_only": "%H:%M:%S",
        "short": "%m/%d/%Y",
    }
    return dt.strftime(formats.get(format_type, formats["iso"]))


def get_date_range(days: int, end_date: Optional[datetime] = None) -> tuple:
    """
    Get start and end dates for a date range.
    
    Args:
        days: Number of days to look back
        end_date: End date (defaults to now)
    
    Returns:
        Tuple of (start_date, end_date)
    """
    end = end_date or datetime.now()
    start = end - timedelta(days=days)
    return start, end


def get_time_periods(start_date: datetime, end_date: datetime, granularity: str = "daily") -> List[datetime]:
    """
    Generate a list of time periods between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        granularity: Period granularity (hourly, daily, weekly, monthly)
    
    Returns:
        List of datetime objects representing period starts
    """
    periods = []
    current = start_date
    
    deltas = {
        "hourly": timedelta(hours=1),
        "daily": timedelta(days=1),
        "weekly": timedelta(weeks=1),
    }
    
    delta = deltas.get(granularity, timedelta(days=1))
    
    while current <= end_date:
        periods.append(current)
        if granularity == "monthly":
            # Handle monthly specially
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        else:
            current += delta
    
    return periods


# =============================================================================
# NUMERIC UTILITIES
# =============================================================================

def round_currency(value: float, decimals: int = 2) -> float:
    """
    Round a currency value using banker's rounding.
    
    Args:
        value: Numeric value to round
        decimals: Number of decimal places
    
    Returns:
        Rounded value
    """
    d = Decimal(str(value))
    return float(d.quantize(Decimal(10) ** -decimals, rounding=ROUND_HALF_UP))


def format_currency(value: float, symbol: str = "$", decimals: int = 2) -> str:
    """
    Format a number as currency.
    
    Args:
        value: Numeric value
        symbol: Currency symbol
        decimals: Decimal places
    
    Returns:
        Formatted currency string
    """
    if abs(value) >= 1_000_000:
        return f"{symbol}{value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{symbol}{value/1_000:.1f}K"
    else:
        return f"{symbol}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1, include_sign: bool = True) -> str:
    """
    Format a number as percentage.
    
    Args:
        value: Numeric value (0.1 = 10%)
        decimals: Decimal places
        include_sign: Include + sign for positive values
    
    Returns:
        Formatted percentage string
    """
    pct = value * 100 if abs(value) < 1 else value
    sign = "+" if include_sign and pct > 0 else ""
    return f"{sign}{pct:.{decimals}f}%"


def calculate_growth_rate(current: float, previous: float) -> float:
    """
    Calculate growth rate between two values.
    
    Args:
        current: Current period value
        previous: Previous period value
    
    Returns:
        Growth rate as decimal (0.1 = 10%)
    """
    if previous == 0:
        return 0.0 if current == 0 else float('inf')
    return (current - previous) / previous


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Top value
        denominator: Bottom value
        default: Value to return if denominator is zero
    
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


# =============================================================================
# STATISTICS UTILITIES
# =============================================================================

def calculate_zscore(value: float, mean: float, std: float) -> float:
    """
    Calculate z-score for a value.
    
    Args:
        value: Data point
        mean: Mean of distribution
        std: Standard deviation
    
    Returns:
        Z-score
    """
    if std == 0:
        return 0.0
    return (value - mean) / std


def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    Calculate a percentile from a list of values.
    
    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)
    
    Returns:
        Percentile value
    """
    return float(np.percentile(values, percentile))


def calculate_iqr(values: List[float]) -> tuple:
    """
    Calculate interquartile range.
    
    Args:
        values: List of numeric values
    
    Returns:
        Tuple of (Q1, Q3, IQR)
    """
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    return float(q1), float(q3), float(iqr)


def detect_outliers_zscore(values: List[float], threshold: float = 3.0) -> List[int]:
    """
    Detect outliers using z-score method.
    
    Args:
        values: List of numeric values
        threshold: Z-score threshold
    
    Returns:
        List of indices of outlier values
    """
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr)
    
    if std == 0:
        return []
    
    zscores = np.abs((arr - mean) / std)
    return list(np.where(zscores > threshold)[0])


def detect_outliers_iqr(values: List[float], multiplier: float = 1.5) -> List[int]:
    """
    Detect outliers using IQR method.
    
    Args:
        values: List of numeric values
        multiplier: IQR multiplier for bounds
    
    Returns:
        List of indices of outlier values
    """
    q1, q3, iqr = calculate_iqr(values)
    lower = q1 - (multiplier * iqr)
    upper = q3 + (multiplier * iqr)
    
    arr = np.array(values)
    return list(np.where((arr < lower) | (arr > upper))[0])


# =============================================================================
# DATA VALIDATION UTILITIES
# =============================================================================

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """
    Validate a DataFrame has required columns and proper data types.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Validation result dict with 'valid' boolean and 'errors' list
    """
    errors = []
    
    # Check for required columns
    missing = set(required_columns) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")
    
    # Check for empty DataFrame
    if df.empty:
        errors.append("DataFrame is empty")
    
    # Check for null values in required columns
    for col in required_columns:
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            errors.append(f"Column '{col}' has {null_count} null values")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "row_count": len(df),
        "column_count": len(df.columns),
    }


def clean_numeric_column(series: pd.Series, fill_method: str = "mean") -> pd.Series:
    """
    Clean a numeric column by handling missing/invalid values.
    
    Args:
        series: Pandas Series to clean
        fill_method: Method to fill missing values (mean, median, zero, ffill)
    
    Returns:
        Cleaned Series
    """
    # Convert to numeric, coercing errors to NaN
    cleaned = pd.to_numeric(series, errors="coerce")
    
    # Fill missing values
    if fill_method == "mean":
        cleaned = cleaned.fillna(cleaned.mean())
    elif fill_method == "median":
        cleaned = cleaned.fillna(cleaned.median())
    elif fill_method == "zero":
        cleaned = cleaned.fillna(0)
    elif fill_method == "ffill":
        cleaned = cleaned.fillna(method="ffill").fillna(method="bfill")
    
    return cleaned


# =============================================================================
# ID GENERATION UTILITIES
# =============================================================================

def generate_transaction_id(prefix: str = "TXN") -> str:
    """
    Generate a unique transaction ID.
    
    Args:
        prefix: ID prefix
    
    Returns:
        Unique transaction ID
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    random_part = hashlib.md5(str(np.random.random()).encode()).hexdigest()[:6].upper()
    return f"{prefix}-{timestamp}-{random_part}"


def generate_product_id(category: str, index: int) -> str:
    """
    Generate a product ID based on category and index.
    
    Args:
        category: Product category
        index: Product index
    
    Returns:
        Product ID
    """
    cat_code = category[:3].upper()
    return f"PROD-{cat_code}-{index:04d}"


def generate_customer_id() -> str:
    """
    Generate a customer ID.
    
    Returns:
        Customer ID
    """
    random_part = hashlib.md5(str(np.random.random()).encode()).hexdigest()[:8].upper()
    return f"CUST-{random_part}"


# =============================================================================
# JSON UTILITIES
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)


def to_json(obj: Any, pretty: bool = False) -> str:
    """
    Convert an object to JSON string.
    
    Args:
        obj: Object to serialize
        pretty: Whether to format with indentation
    
    Returns:
        JSON string
    """
    return json.dumps(obj, cls=NumpyEncoder, indent=2 if pretty else None)


def dataframe_to_records(df: pd.DataFrame) -> List[Dict]:
    """
    Convert DataFrame to list of dictionaries with proper type handling.
    
    Args:
        df: DataFrame to convert
    
    Returns:
        List of record dictionaries
    """
    return json.loads(df.to_json(orient="records", date_format="iso"))


# =============================================================================
# TREND ANALYSIS UTILITIES
# =============================================================================

def get_trend_indicator(current: float, previous: float, threshold: float = 0.01) -> str:
    """
    Get a trend indicator arrow based on value change.
    
    Args:
        current: Current value
        previous: Previous value
        threshold: Minimum change threshold for trend
    
    Returns:
        Trend indicator (↑, ↓, →)
    """
    if previous == 0:
        return "→"
    
    change = (current - previous) / previous
    
    if change > threshold:
        return "↑"
    elif change < -threshold:
        return "↓"
    else:
        return "→"


def classify_change(value: float) -> str:
    """
    Classify a percentage change into categories.
    
    Args:
        value: Percentage change as decimal
    
    Returns:
        Classification string
    """
    pct = abs(value * 100)
    
    if pct < 1:
        return "stable"
    elif pct < 5:
        return "slight"
    elif pct < 15:
        return "moderate"
    elif pct < 30:
        return "significant"
    else:
        return "dramatic"


def get_severity_class(severity: str) -> str:
    """
    Get CSS class for severity level.
    
    Args:
        severity: Severity level (low, medium, high, critical)
    
    Returns:
        CSS class name
    """
    classes = {
        "low": "severity-low",
        "medium": "severity-medium",
        "high": "severity-high",
        "critical": "severity-critical",
    }
    return classes.get(severity.lower(), "severity-low")


# =============================================================================
# AGGREGATION UTILITIES
# =============================================================================

def aggregate_by_period(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    period: str = "D",
    agg_func: str = "sum"
) -> pd.DataFrame:
    """
    Aggregate data by time period.
    
    Args:
        df: DataFrame with datetime column
        date_column: Name of datetime column
        value_column: Name of value column to aggregate
        period: Pandas period string (H=hourly, D=daily, W=weekly, M=monthly)
        agg_func: Aggregation function (sum, mean, count, min, max)
    
    Returns:
        Aggregated DataFrame
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)
    
    agg_funcs = {
        "sum": df[value_column].resample(period).sum(),
        "mean": df[value_column].resample(period).mean(),
        "count": df[value_column].resample(period).count(),
        "min": df[value_column].resample(period).min(),
        "max": df[value_column].resample(period).max(),
    }
    
    result = agg_funcs.get(agg_func, agg_funcs["sum"])
    return result.reset_index()


def group_by_category(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
    agg_func: str = "sum",
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Group data by category and aggregate.
    
    Args:
        df: DataFrame to group
        category_column: Column to group by
        value_column: Column to aggregate
        agg_func: Aggregation function
        top_n: Return only top N categories
    
    Returns:
        Grouped DataFrame
    """
    grouped = df.groupby(category_column)[value_column].agg(agg_func).reset_index()
    grouped = grouped.sort_values(value_column, ascending=False)
    
    if top_n:
        grouped = grouped.head(top_n)
    
    return grouped
