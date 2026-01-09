"""
REST API Routes for the Sales Analytics Dashboard.
Defines all API endpoints for data retrieval, forecasting, and insights.
"""

from datetime import datetime, timedelta
from typing import Optional
from flask import Blueprint, jsonify, request, current_app
import pandas as pd

from utils import setup_logger, dataframe_to_records

logger = setup_logger("api_routes")

# Create Blueprint
api_bp = Blueprint("api", __name__, url_prefix="/api")


def get_app_state():
    """Get application state from current_app."""
    return {
        "data_service": getattr(current_app, "data_service", None),
        "forecast_model": getattr(current_app, "forecast_model", None),
        "anomaly_detector": getattr(current_app, "anomaly_detector", None),
        "insights_engine": getattr(current_app, "insights_engine", None),
        "preprocessor": getattr(current_app, "preprocessor", None),
    }


def create_response(data, status="success", metadata=None):
    """Create standardized API response."""
    response = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "data": data,
    }
    if metadata:
        response["metadata"] = metadata
    return jsonify(response)


def create_error_response(message, status_code=400):
    """Create error response."""
    return jsonify({
        "status": "error",
        "timestamp": datetime.now().isoformat(),
        "error": message,
    }), status_code


# =============================================================================
# HEALTH & STATUS ENDPOINTS
# =============================================================================

@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    state = get_app_state()
    
    return create_response({
        "status": "healthy",
        "services": {
            "data_service": state["data_service"] is not None,
            "forecast_model": state["forecast_model"] is not None,
            "anomaly_detector": state["anomaly_detector"] is not None,
            "insights_engine": state["insights_engine"] is not None,
        }
    })


@api_bp.route("/metrics", methods=["GET"])
def get_metrics():
    """Get system metrics."""
    state = get_app_state()
    
    metrics = {
        "uptime": "running",
        "timestamp": datetime.now().isoformat(),
    }
    
    if state["data_service"]:
        metrics["data_stats"] = state["data_service"].get_stats()
    
    return create_response(metrics)


# =============================================================================
# SALES DATA ENDPOINTS
# =============================================================================

@api_bp.route("/sales/history", methods=["GET"])
def get_sales_history():
    """
    Get historical sales data.
    
    Query params:
        days: Number of days to fetch (default: 30)
        granularity: daily, hourly, weekly (default: daily)
        category: Filter by category
        region: Filter by region
    """
    state = get_app_state()
    
    if not state["data_service"]:
        return create_error_response("Data service not available", 503)
    
    # Parse parameters
    days = request.args.get("days", 30, type=int)
    granularity = request.args.get("granularity", "daily")
    category = request.args.get("category")
    region = request.args.get("region")
    
    try:
        # Get data
        df = state["data_service"].get_data(include_historical=True)
        
        if df.empty:
            return create_response([], metadata={"records_count": 0})
        
        # Filter by date
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]
        
        # Apply filters
        if category:
            df = df[df["category"] == category]
        if region:
            df = df[df["region"] == region]
        
        # Aggregate based on granularity
        if state["preprocessor"]:
            if granularity == "daily":
                result = state["preprocessor"].aggregate_daily(df)
            elif granularity == "hourly":
                result = state["preprocessor"].aggregate_hourly(df)
            else:
                result = df
        else:
            result = df
        
        records = dataframe_to_records(result)
        
        return create_response(records, metadata={
            "records_count": len(records),
            "days": days,
            "granularity": granularity,
        })
        
    except Exception as e:
        logger.error(f"Error fetching sales history: {e}")
        return create_error_response(str(e), 500)


@api_bp.route("/sales/live", methods=["GET"])
def get_live_sales():
    """
    Get real-time sales data.
    
    Query params:
        limit: Number of records (default: 100)
    """
    state = get_app_state()
    
    if not state["data_service"]:
        return create_error_response("Data service not available", 503)
    
    limit = request.args.get("limit", 100, type=int)
    
    try:
        df = state["data_service"].generator.get_latest_data(limit)
        
        if df.empty:
            return create_response([], metadata={"records_count": 0})
        
        records = dataframe_to_records(df)
        
        return create_response(records, metadata={
            "records_count": len(records),
            "streaming": state["data_service"].generator._running,
        })
        
    except Exception as e:
        logger.error(f"Error fetching live sales: {e}")
        return create_error_response(str(e), 500)


@api_bp.route("/sales/transactions", methods=["GET"])
def get_transactions():
    """
    Get paginated transactions.
    
    Query params:
        limit: Page size (default: 100)
        offset: Starting offset (default: 0)
    """
    state = get_app_state()
    
    if not state["data_service"]:
        return create_error_response("Data service not available", 503)
    
    limit = min(request.args.get("limit", 100, type=int), 1000)
    offset = request.args.get("offset", 0, type=int)
    
    try:
        df = state["data_service"].get_data()
        
        if df.empty:
            return create_response([], metadata={"records_count": 0, "total": 0})
        
        # Sort by timestamp descending
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp", ascending=False)
        
        # Paginate
        total = len(df)
        df = df.iloc[offset:offset + limit]
        
        records = dataframe_to_records(df)
        
        return create_response(records, metadata={
            "records_count": len(records),
            "total": total,
            "offset": offset,
            "limit": limit,
        })
        
    except Exception as e:
        logger.error(f"Error fetching transactions: {e}")
        return create_error_response(str(e), 500)


# =============================================================================
# FORECASTING ENDPOINTS
# =============================================================================

@api_bp.route("/forecast", methods=["GET"])
def get_forecast():
    """
    Get sales forecast.
    
    Query params:
        horizon: Forecast days (default: 7)
    """
    state = get_app_state()
    
    if not state["forecast_model"]:
        return create_error_response("Forecast model not available", 503)
    
    if not state["data_service"]:
        return create_error_response("Data service not available", 503)
    
    horizon = request.args.get("horizon", 7, type=int)
    horizon = min(max(horizon, 1), 30)  # Limit to 1-30 days
    
    try:
        # Get historical data
        df = state["data_service"].get_data()
        
        if df.empty or len(df) < 30:
            return create_error_response("Insufficient data for forecasting", 400)
        
        # Preprocess to daily
        if state["preprocessor"]:
            daily_df = state["preprocessor"].aggregate_daily(df)
        else:
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            daily_df = df.groupby("date").agg({
                "total_amount": "sum"
            }).reset_index()
            daily_df.columns = ["date", "total_revenue"]
        
        # Generate forecast
        forecast = state["forecast_model"].forecast(daily_df, horizon=horizon)
        
        return create_response(forecast, metadata={
            "model": state["forecast_model"].algorithm,
            "horizon": horizon,
        })
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        return create_error_response(str(e), 500)


@api_bp.route("/forecast/confidence", methods=["GET"])
def get_forecast_confidence():
    """Get forecast model confidence and metrics."""
    state = get_app_state()
    
    if not state["forecast_model"]:
        return create_error_response("Forecast model not available", 503)
    
    try:
        metrics = state["forecast_model"].metrics
        feature_importance = state["forecast_model"].get_feature_importance(top_n=10)
        
        return create_response({
            "metrics": metrics,
            "feature_importance": feature_importance,
            "model": state["forecast_model"].algorithm,
            "fitted": state["forecast_model"]._fitted,
        })
        
    except Exception as e:
        logger.error(f"Error fetching forecast confidence: {e}")
        return create_error_response(str(e), 500)


# =============================================================================
# ANOMALY DETECTION ENDPOINTS
# =============================================================================

@api_bp.route("/anomalies", methods=["GET"])
def get_anomalies():
    """
    Get detected anomalies.
    
    Query params:
        days: Days to analyze (default: 7)
        severity: Filter by severity (low, medium, high, critical)
    """
    state = get_app_state()
    
    if not state["anomaly_detector"]:
        return create_error_response("Anomaly detector not available", 503)
    
    if not state["data_service"]:
        return create_error_response("Data service not available", 503)
    
    days = request.args.get("days", 7, type=int)
    severity = request.args.get("severity")
    
    try:
        # Get recent data
        df = state["data_service"].get_data()
        
        if df.empty:
            return create_response([], metadata={"records_count": 0})
        
        # Filter by date
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]
        
        if df.empty:
            return create_response([], metadata={"records_count": 0})
        
        # Detect anomalies
        result = state["anomaly_detector"].detect(df)
        anomalies = result[result["is_anomaly"] == True]
        
        # Filter by severity
        if severity and "severity" in anomalies.columns:
            anomalies = anomalies[anomalies["severity"] == severity]
        
        records = dataframe_to_records(anomalies)
        
        return create_response(records, metadata={
            "records_count": len(records),
            "days": days,
            "total_analyzed": len(df),
        })
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return create_error_response(str(e), 500)


@api_bp.route("/anomalies/latest", methods=["GET"])
def get_latest_anomalies():
    """Get the most recent anomalies."""
    state = get_app_state()
    
    if not state["anomaly_detector"]:
        return create_error_response("Anomaly detector not available", 503)
    
    limit = request.args.get("limit", 10, type=int)
    
    try:
        anomalies = state["anomaly_detector"].get_recent_anomalies(n=limit)
        
        return create_response(anomalies, metadata={
            "records_count": len(anomalies),
        })
        
    except Exception as e:
        logger.error(f"Error fetching latest anomalies: {e}")
        return create_error_response(str(e), 500)


@api_bp.route("/anomalies/report", methods=["GET"])
def get_anomaly_report():
    """Get comprehensive anomaly report."""
    state = get_app_state()
    
    if not state["anomaly_detector"]:
        return create_error_response("Anomaly detector not available", 503)
    
    if not state["data_service"]:
        return create_error_response("Data service not available", 503)
    
    days = request.args.get("days", 30, type=int)
    
    try:
        df = state["data_service"].get_data()
        
        if df.empty:
            return create_response({"total_anomalies": 0, "message": "No data available"})
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]
        
        result = state["anomaly_detector"].detect(df)
        report = state["anomaly_detector"].get_anomaly_report(result)
        
        return create_response(report)
        
    except Exception as e:
        logger.error(f"Error generating anomaly report: {e}")
        return create_error_response(str(e), 500)


# =============================================================================
# KPI ENDPOINTS
# =============================================================================

@api_bp.route("/kpis/summary", methods=["GET"])
def get_kpi_summary():
    """Get summary of all KPIs."""
    state = get_app_state()
    
    if not state["data_service"]:
        return create_error_response("Data service not available", 503)
    
    days = request.args.get("days", 30, type=int)
    
    try:
        df = state["data_service"].get_data()
        
        if df.empty:
            return create_response({
                "total_revenue": 0,
                "transaction_count": 0,
                "avg_order_value": 0,
            })
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Current period
        cutoff = datetime.now() - timedelta(days=days)
        current = df[df["timestamp"] >= cutoff]
        
        # Previous period for comparison
        prev_cutoff = cutoff - timedelta(days=days)
        previous = df[(df["timestamp"] >= prev_cutoff) & (df["timestamp"] < cutoff)]
        
        # Calculate current KPIs
        total_revenue = current["total_amount"].sum() if len(current) > 0 else 0
        transaction_count = len(current)
        avg_order_value = total_revenue / transaction_count if transaction_count > 0 else 0
        unique_customers = current["customer_id"].nunique() if "customer_id" in current.columns else 0
        
        # Calculate changes
        prev_revenue = previous["total_amount"].sum() if len(previous) > 0 else 0
        prev_transactions = len(previous)
        
        revenue_change = ((total_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
        transaction_change = ((transaction_count - prev_transactions) / prev_transactions * 100) if prev_transactions > 0 else 0
        
        # Top category
        top_category = None
        if "category" in current.columns and len(current) > 0:
            cat_revenue = current.groupby("category")["total_amount"].sum()
            if not cat_revenue.empty:
                top_category = {
                    "name": cat_revenue.idxmax(),
                    "revenue": float(cat_revenue.max()),
                }
        
        # Anomaly count
        anomaly_count = 0
        if state["anomaly_detector"] and len(current) > 0:
            try:
                result = state["anomaly_detector"].detect(current)
                anomaly_count = result["is_anomaly"].sum()
            except:
                pass
        
        kpis = {
            "total_revenue": round(total_revenue, 2),
            "revenue_change": round(revenue_change, 2),
            "revenue_trend": "up" if revenue_change > 0 else ("down" if revenue_change < 0 else "stable"),
            "transaction_count": transaction_count,
            "transaction_change": round(transaction_change, 2),
            "avg_order_value": round(avg_order_value, 2),
            "unique_customers": unique_customers,
            "top_category": top_category,
            "anomaly_count": int(anomaly_count),
            "period_days": days,
        }
        
        return create_response(kpis)
        
    except Exception as e:
        logger.error(f"Error calculating KPIs: {e}")
        return create_error_response(str(e), 500)


@api_bp.route("/kpis/realtime", methods=["GET"])
def get_realtime_kpis():
    """Get real-time KPIs from streaming data."""
    state = get_app_state()
    
    if not state["data_service"]:
        return create_error_response("Data service not available", 503)
    
    try:
        # Get last hour of data
        df = state["data_service"].generator.get_latest_data(1000)
        
        if df.empty:
            return create_response({
                "revenue_per_minute": 0,
                "transactions_per_minute": 0,
            })
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        last_hour = datetime.now() - timedelta(hours=1)
        df = df[df["timestamp"] >= last_hour]
        
        if df.empty:
            return create_response({
                "revenue_per_minute": 0,
                "transactions_per_minute": 0,
            })
        
        # Calculate per-minute rates
        time_span_minutes = max(1, (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 60)
        
        kpis = {
            "revenue_per_minute": round(df["total_amount"].sum() / time_span_minutes, 2),
            "transactions_per_minute": round(len(df) / time_span_minutes, 2),
            "last_transaction": df["timestamp"].max().isoformat(),
            "sample_size": len(df),
        }
        
        return create_response(kpis)
        
    except Exception as e:
        logger.error(f"Error calculating realtime KPIs: {e}")
        return create_error_response(str(e), 500)


# =============================================================================
# INSIGHTS ENDPOINTS
# =============================================================================

@api_bp.route("/insights", methods=["GET"])
def get_insights():
    """
    Get business insights.
    
    Query params:
        category: Filter by insight category
    """
    state = get_app_state()
    
    if not state["insights_engine"]:
        return create_error_response("Insights engine not available", 503)
    
    if not state["data_service"]:
        return create_error_response("Data service not available", 503)
    
    category = request.args.get("category", "all")
    
    try:
        # Get data
        df = state["data_service"].get_data()
        
        if df.empty:
            return create_response([], metadata={"records_count": 0})
        
        # Split into current and historical
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now() - timedelta(days=30)
        
        current = df[df["timestamp"] >= cutoff]
        historical = df[df["timestamp"] < cutoff]
        
        # Set data and generate insights
        state["insights_engine"].set_data(current, historical)
        
        # Get forecast data if available
        forecast_data = None
        if state["forecast_model"] and state["forecast_model"]._fitted:
            try:
                if state["preprocessor"]:
                    daily_df = state["preprocessor"].aggregate_daily(df)
                    forecast_data = state["forecast_model"].forecast(daily_df, horizon=7)
            except:
                pass
        
        # Get anomaly data
        anomaly_data = None
        if state["anomaly_detector"] and state["anomaly_detector"]._fitted:
            try:
                anomaly_data = state["anomaly_detector"].detect(current)
            except:
                pass
        
        # Generate insights
        insights = state["insights_engine"].generate_all_insights(
            forecast_data=forecast_data,
            anomaly_data=anomaly_data
        )
        
        return create_response(insights, metadata={
            "records_count": len(insights),
            "summary": state["insights_engine"].get_insights_summary(),
        })
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        return create_error_response(str(e), 500)


@api_bp.route("/insights/top", methods=["GET"])
def get_top_insights():
    """Get top N insights by priority."""
    state = get_app_state()
    
    if not state["insights_engine"]:
        return create_error_response("Insights engine not available", 503)
    
    limit = request.args.get("limit", 5, type=int)
    
    try:
        insights = state["insights_engine"].get_top_insights(n=limit)
        
        return create_response(insights, metadata={
            "records_count": len(insights),
        })
        
    except Exception as e:
        logger.error(f"Error fetching top insights: {e}")
        return create_error_response(str(e), 500)


# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================

@api_bp.route("/analytics/by-category", methods=["GET"])
def get_analytics_by_category():
    """Get sales analytics by category."""
    state = get_app_state()
    
    if not state["data_service"]:
        return create_error_response("Data service not available", 503)
    
    days = request.args.get("days", 30, type=int)
    
    try:
        df = state["data_service"].get_data()
        
        if df.empty:
            return create_response([])
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]
        
        if "category" not in df.columns:
            return create_response([])
        
        if state["preprocessor"]:
            result = state["preprocessor"].aggregate_by_category(df)
        else:
            result = df.groupby("category").agg({
                "total_amount": "sum",
                "transaction_id": "count",
            }).reset_index()
            result.columns = ["category", "total_revenue", "transaction_count"]
        
        records = dataframe_to_records(result)
        
        return create_response(records, metadata={
            "records_count": len(records),
            "days": days,
        })
        
    except Exception as e:
        logger.error(f"Error fetching category analytics: {e}")
        return create_error_response(str(e), 500)


@api_bp.route("/analytics/by-region", methods=["GET"])
def get_analytics_by_region():
    """Get sales analytics by region."""
    state = get_app_state()
    
    if not state["data_service"]:
        return create_error_response("Data service not available", 503)
    
    days = request.args.get("days", 30, type=int)
    
    try:
        df = state["data_service"].get_data()
        
        if df.empty:
            return create_response([])
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]
        
        if "region" not in df.columns:
            return create_response([])
        
        if state["preprocessor"]:
            result = state["preprocessor"].aggregate_by_region(df)
        else:
            result = df.groupby("region").agg({
                "total_amount": "sum",
                "transaction_id": "count",
            }).reset_index()
            result.columns = ["region", "total_revenue", "transaction_count"]
        
        records = dataframe_to_records(result)
        
        return create_response(records, metadata={
            "records_count": len(records),
            "days": days,
        })
        
    except Exception as e:
        logger.error(f"Error fetching region analytics: {e}")
        return create_error_response(str(e), 500)


@api_bp.route("/analytics/trends", methods=["GET"])
def get_trends():
    """Get sales trends over time."""
    state = get_app_state()
    
    if not state["data_service"]:
        return create_error_response("Data service not available", 503)
    
    days = request.args.get("days", 30, type=int)
    granularity = request.args.get("granularity", "daily")
    
    try:
        df = state["data_service"].get_data()
        
        if df.empty:
            return create_response([])
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]
        
        if state["preprocessor"]:
            result = state["preprocessor"].get_trend_data(df, period=granularity, last_n_periods=days)
        else:
            df["date"] = df["timestamp"].dt.date
            result = df.groupby("date").agg({
                "total_amount": "sum",
            }).reset_index()
            result.columns = ["date", "total_revenue"]
        
        records = dataframe_to_records(result)
        
        return create_response(records, metadata={
            "records_count": len(records),
            "days": days,
            "granularity": granularity,
        })
        
    except Exception as e:
        logger.error(f"Error fetching trends: {e}")
        return create_error_response(str(e), 500)


# =============================================================================
# DATA MANAGEMENT ENDPOINTS
# =============================================================================

@api_bp.route("/refresh", methods=["POST"])
def refresh_data():
    """Trigger data refresh and model inference."""
    state = get_app_state()
    
    try:
        results = {"refreshed": []}
        
        # Refresh insights if engine available
        if state["insights_engine"] and state["data_service"]:
            df = state["data_service"].get_data()
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                cutoff = datetime.now() - timedelta(days=30)
                current = df[df["timestamp"] >= cutoff]
                historical = df[df["timestamp"] < cutoff]
                state["insights_engine"].set_data(current, historical)
                results["refreshed"].append("insights")
        
        results["timestamp"] = datetime.now().isoformat()
        
        return create_response(results)
        
    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        return create_error_response(str(e), 500)
