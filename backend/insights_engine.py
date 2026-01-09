"""
Automated Insights Engine.
Generates natural language business insights from sales data analysis.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import pandas as pd

from utils import (
    setup_logger, format_currency, format_percentage,
    calculate_growth_rate, get_trend_indicator, classify_change,
    round_currency
)

logger = setup_logger("insights_engine")


class InsightType(Enum):
    """Categories of business insights."""
    PERFORMANCE = "performance"
    PRODUCT = "product"
    REGIONAL = "regional"
    ANOMALY = "anomaly"
    FORECAST = "forecast"
    RECOMMENDATION = "recommendation"
    TREND = "trend"


class InsightPriority(Enum):
    """Priority levels for insights."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5


class Insight:
    """Represents a single business insight."""
    
    def __init__(
        self,
        title: str,
        message: str,
        insight_type: InsightType,
        priority: InsightPriority,
        data: Optional[Dict] = None,
        action: Optional[str] = None
    ):
        """
        Create an insight.
        
        Args:
            title: Short insight title
            message: Detailed insight message
            insight_type: Category of insight
            priority: Importance level
            data: Supporting data
            action: Recommended action
        """
        self.title = title
        self.message = message
        self.insight_type = insight_type
        self.priority = priority
        self.data = data or {}
        self.action = action
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "message": self.message,
            "type": self.insight_type.value,
            "priority": self.priority.value,
            "priority_label": self.priority.name.lower(),
            "data": self.data,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
        }


class InsightsEngine:
    """
    Automated business insights generation engine.
    
    Analyzes sales data to generate actionable insights across:
    - Performance metrics
    - Product/category analysis
    - Regional comparisons
    - Anomaly explanations
    - Forecast interpretations
    - Recommendations
    """
    
    def __init__(
        self,
        current_data: Optional[pd.DataFrame] = None,
        historical_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize the insights engine.
        
        Args:
            current_data: Current period data
            historical_data: Historical data for comparison
        """
        self.current_data = current_data
        self.historical_data = historical_data
        
        # Insight cache
        self._insights: List[Insight] = []
        self._last_generated: Optional[datetime] = None
        
        logger.info("InsightsEngine initialized")
    
    def set_data(
        self,
        current_data: pd.DataFrame,
        historical_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Update the data for analysis.
        
        Args:
            current_data: Current period data
            historical_data: Historical/comparison data
        """
        self.current_data = current_data
        self.historical_data = historical_data
    
    def generate_all_insights(
        self,
        forecast_data: Optional[Dict] = None,
        anomaly_data: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate all categories of insights.
        
        Args:
            forecast_data: Forecast results
            anomaly_data: Anomaly detection results
        
        Returns:
            List of insight dictionaries sorted by priority
        """
        self._insights = []
        
        if self.current_data is None or self.current_data.empty:
            logger.warning("No data available for insight generation")
            return []
        
        # Generate each category
        self._insights.extend(self._performance_insights())
        self._insights.extend(self._product_insights())
        self._insights.extend(self._regional_insights())
        self._insights.extend(self._trend_insights())
        
        if anomaly_data is not None and not anomaly_data.empty:
            self._insights.extend(self._anomaly_insights(anomaly_data))
        
        if forecast_data:
            self._insights.extend(self._forecast_insights(forecast_data))
        
        self._insights.extend(self._recommendation_insights())
        
        # Sort by priority
        self._insights.sort(key=lambda x: x.priority.value)
        self._last_generated = datetime.now()
        
        logger.info(f"Generated {len(self._insights)} insights")
        
        return [i.to_dict() for i in self._insights]
    
    def _performance_insights(self) -> List[Insight]:
        """Generate performance-related insights."""
        insights = []
        df = self.current_data.copy()
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Total revenue calculation
        total_revenue = df["total_amount"].sum() if "total_amount" in df.columns else 0
        transaction_count = len(df)
        avg_order_value = total_revenue / transaction_count if transaction_count > 0 else 0
        
        # Compare with historical if available
        if self.historical_data is not None and not self.historical_data.empty:
            hist_revenue = self.historical_data["total_amount"].sum() if "total_amount" in self.historical_data.columns else 0
            growth_rate = calculate_growth_rate(total_revenue, hist_revenue)
            
            change_class = classify_change(growth_rate)
            trend = get_trend_indicator(total_revenue, hist_revenue)
            
            if abs(growth_rate) > 0.05:  # More than 5% change
                priority = InsightPriority.HIGH if abs(growth_rate) > 0.20 else InsightPriority.MEDIUM
                direction = "up" if growth_rate > 0 else "down"
                
                insights.append(Insight(
                    title=f"Revenue {direction.title()} {format_percentage(abs(growth_rate))}",
                    message=f"Total revenue is {format_currency(total_revenue)}, which is {format_percentage(growth_rate)} compared to the previous period ({format_currency(hist_revenue)}). This represents a {change_class} change.",
                    insight_type=InsightType.PERFORMANCE,
                    priority=priority,
                    data={
                        "current_revenue": total_revenue,
                        "previous_revenue": hist_revenue,
                        "growth_rate": growth_rate,
                        "trend": trend,
                    },
                    action=f"{'Continue strategies driving growth' if growth_rate > 0 else 'Investigate causes of revenue decline'}"
                ))
        
        # Transaction volume insight
        if transaction_count > 0:
            insights.append(Insight(
                title=f"{transaction_count:,} Transactions Processed",
                message=f"Processed {transaction_count:,} transactions with an average order value of {format_currency(avg_order_value)}.",
                insight_type=InsightType.PERFORMANCE,
                priority=InsightPriority.INFO,
                data={
                    "transaction_count": transaction_count,
                    "avg_order_value": avg_order_value,
                    "total_revenue": total_revenue,
                }
            ))
        
        return insights
    
    def _product_insights(self) -> List[Insight]:
        """Generate product and category insights."""
        insights = []
        df = self.current_data.copy()
        
        if "category" not in df.columns:
            return insights
        
        # Category performance
        category_revenue = df.groupby("category")["total_amount"].sum().sort_values(ascending=False)
        total_revenue = category_revenue.sum()
        
        # Top category
        if not category_revenue.empty:
            top_category = category_revenue.index[0]
            top_revenue = category_revenue.iloc[0]
            top_pct = top_revenue / total_revenue * 100 if total_revenue > 0 else 0
            
            insights.append(Insight(
                title=f"{top_category} Leading Sales",
                message=f"{top_category} is the top-performing category with {format_currency(top_revenue)} in revenue, representing {top_pct:.1f}% of total sales.",
                insight_type=InsightType.PRODUCT,
                priority=InsightPriority.MEDIUM,
                data={
                    "category": top_category,
                    "revenue": top_revenue,
                    "percentage": top_pct,
                }
            ))
            
            # Compare with historical for category growth
            if self.historical_data is not None and "category" in self.historical_data.columns:
                hist_category = self.historical_data.groupby("category")["total_amount"].sum()
                
                for cat in category_revenue.index[:3]:  # Top 3 categories
                    if cat in hist_category.index:
                        cat_growth = calculate_growth_rate(category_revenue[cat], hist_category[cat])
                        
                        if abs(cat_growth) > 0.15:  # More than 15% change
                            direction = "growth" if cat_growth > 0 else "decline"
                            insights.append(Insight(
                                title=f"{cat} Shows Strong {direction.title()}",
                                message=f"{cat} category revenue has changed by {format_percentage(cat_growth)} compared to the previous period.",
                                insight_type=InsightType.PRODUCT,
                                priority=InsightPriority.MEDIUM if cat_growth > 0 else InsightPriority.HIGH,
                                data={
                                    "category": cat,
                                    "growth_rate": cat_growth,
                                    "current_revenue": category_revenue[cat],
                                    "previous_revenue": hist_category[cat],
                                }
                            ))
        
        # Top products
        if "product_name" in df.columns:
            product_revenue = df.groupby("product_name")["total_amount"].sum().sort_values(ascending=False)
            
            if len(product_revenue) > 0:
                top_products = product_revenue.head(3)
                insights.append(Insight(
                    title="Top Selling Products",
                    message=f"Your best-selling products are: {', '.join(top_products.index[:3])}.",
                    insight_type=InsightType.PRODUCT,
                    priority=InsightPriority.INFO,
                    data={
                        "top_products": [
                            {"name": name, "revenue": rev}
                            for name, rev in top_products.items()
                        ]
                    }
                ))
        
        return insights
    
    def _regional_insights(self) -> List[Insight]:
        """Generate regional performance insights."""
        insights = []
        df = self.current_data.copy()
        
        if "region" not in df.columns:
            return insights
        
        # Regional performance
        region_revenue = df.groupby("region")["total_amount"].sum().sort_values(ascending=False)
        total_revenue = region_revenue.sum()
        
        if region_revenue.empty:
            return insights
        
        # Top region
        top_region = region_revenue.index[0]
        top_revenue = region_revenue.iloc[0]
        top_pct = top_revenue / total_revenue * 100 if total_revenue > 0 else 0
        
        insights.append(Insight(
            title=f"{top_region} Leads Regionally",
            message=f"{top_region} is the highest-performing region with {format_currency(top_revenue)} ({top_pct:.1f}% of total).",
            insight_type=InsightType.REGIONAL,
            priority=InsightPriority.INFO,
            data={
                "region": top_region,
                "revenue": top_revenue,
                "percentage": top_pct,
            }
        ))
        
        # Underperforming regions
        if self.historical_data is not None and "region" in self.historical_data.columns:
            hist_region = self.historical_data.groupby("region")["total_amount"].sum()
            
            for region in region_revenue.index:
                if region in hist_region.index:
                    region_growth = calculate_growth_rate(region_revenue[region], hist_region[region])
                    
                    if region_growth < -0.15:  # Down more than 15%
                        insights.append(Insight(
                            title=f"{region} Underperforming",
                            message=f"{region} region is down {format_percentage(abs(region_growth))} compared to the previous period. Immediate attention recommended.",
                            insight_type=InsightType.REGIONAL,
                            priority=InsightPriority.HIGH,
                            data={
                                "region": region,
                                "growth_rate": region_growth,
                                "current_revenue": region_revenue[region],
                                "previous_revenue": hist_region[region],
                            },
                            action=f"Investigate factors affecting {region} sales and consider regional promotions."
                        ))
        
        return insights
    
    def _trend_insights(self) -> List[Insight]:
        """Generate trend-based insights."""
        insights = []
        df = self.current_data.copy()
        
        if "timestamp" not in df.columns:
            return insights
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        
        # Daily trend
        daily_revenue = df.groupby("date")["total_amount"].sum()
        
        if len(daily_revenue) >= 7:
            # Calculate 7-day trend
            recent_7d = daily_revenue.tail(7)
            prev_7d = daily_revenue.iloc[-14:-7] if len(daily_revenue) >= 14 else None
            
            if prev_7d is not None:
                recent_total = recent_7d.sum()
                prev_total = prev_7d.sum()
                week_growth = calculate_growth_rate(recent_total, prev_total)
                
                if abs(week_growth) > 0.10:
                    direction = "upward" if week_growth > 0 else "downward"
                    insights.append(Insight(
                        title=f"Weekly Trend: {direction.title()}",
                        message=f"Sales over the last 7 days are {format_percentage(week_growth)} compared to the previous week, indicating an {direction} trend.",
                        insight_type=InsightType.TREND,
                        priority=InsightPriority.MEDIUM,
                        data={
                            "recent_7d_total": recent_total,
                            "previous_7d_total": prev_total,
                            "growth_rate": week_growth,
                        }
                    ))
            
            # Identify best/worst days
            best_day = daily_revenue.idxmax()
            worst_day = daily_revenue.idxmin()
            
            insights.append(Insight(
                title="Daily Performance Variance",
                message=f"Best day: {best_day} ({format_currency(daily_revenue.max())}). Worst day: {worst_day} ({format_currency(daily_revenue.min())}).",
                insight_type=InsightType.TREND,
                priority=InsightPriority.INFO,
                data={
                    "best_day": str(best_day),
                    "best_revenue": daily_revenue.max(),
                    "worst_day": str(worst_day),
                    "worst_revenue": daily_revenue.min(),
                }
            ))
        
        return insights
    
    def _anomaly_insights(self, anomaly_data: pd.DataFrame) -> List[Insight]:
        """Generate insights from detected anomalies."""
        insights = []
        
        anomalies = anomaly_data[anomaly_data.get("is_anomaly", False) == True]
        
        if anomalies.empty:
            return insights
        
        # Count by severity
        if "severity" in anomalies.columns:
            severity_counts = anomalies["severity"].value_counts()
            
            critical_count = severity_counts.get("critical", 0)
            high_count = severity_counts.get("high", 0)
            
            if critical_count > 0:
                insights.append(Insight(
                    title=f"{critical_count} Critical Anomalies Detected",
                    message=f"There are {critical_count} critical anomalies requiring immediate attention. These represent significant deviations from expected patterns.",
                    insight_type=InsightType.ANOMALY,
                    priority=InsightPriority.CRITICAL,
                    data={
                        "critical_count": critical_count,
                        "high_count": high_count,
                        "total_anomalies": len(anomalies),
                    },
                    action="Review critical anomalies immediately and investigate root causes."
                ))
            elif high_count > 0:
                insights.append(Insight(
                    title=f"{high_count} High-Priority Anomalies",
                    message=f"Detected {high_count} high-priority anomalies that should be reviewed.",
                    insight_type=InsightType.ANOMALY,
                    priority=InsightPriority.HIGH,
                    data={
                        "high_count": high_count,
                        "total_anomalies": len(anomalies),
                    }
                ))
        
        # Regional anomalies
        if "region" in anomalies.columns:
            region_anomalies = anomalies["region"].value_counts()
            
            for region, count in region_anomalies.head(2).items():
                if count >= 3:
                    insights.append(Insight(
                        title=f"Multiple Anomalies in {region}",
                        message=f"{region} region has {count} detected anomalies, which may indicate a regional issue.",
                        insight_type=InsightType.ANOMALY,
                        priority=InsightPriority.HIGH,
                        data={
                            "region": region,
                            "anomaly_count": count,
                        },
                        action=f"Investigate operational issues in {region} region."
                    ))
        
        return insights
    
    def _forecast_insights(self, forecast_data: Dict) -> List[Insight]:
        """Generate insights from forecast data."""
        insights = []
        
        if not forecast_data:
            return insights
        
        values = forecast_data.get("values", [])
        total_forecast = forecast_data.get("total_forecast", sum(values) if values else 0)
        horizon = forecast_data.get("horizon_days", len(values))
        
        if values:
            avg_daily = np.mean(values)
            
            # Compare with current average
            if self.current_data is not None and "timestamp" in self.current_data.columns:
                df = self.current_data.copy()
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["date"] = df["timestamp"].dt.date
                
                current_daily_avg = df.groupby("date")["total_amount"].sum().mean()
                
                if current_daily_avg > 0:
                    forecast_growth = calculate_growth_rate(avg_daily, current_daily_avg)
                    
                    direction = "increase" if forecast_growth > 0 else "decrease"
                    priority = InsightPriority.MEDIUM if abs(forecast_growth) < 0.15 else InsightPriority.HIGH
                    
                    insights.append(Insight(
                        title=f"Forecast: {format_percentage(forecast_growth)} {direction.title()} Expected",
                        message=f"Our ML model predicts a {direction} of {format_percentage(abs(forecast_growth))} over the next {horizon} days. Projected total: {format_currency(total_forecast)}.",
                        insight_type=InsightType.FORECAST,
                        priority=priority,
                        data={
                            "forecast_total": total_forecast,
                            "forecast_daily_avg": avg_daily,
                            "current_daily_avg": current_daily_avg,
                            "growth_rate": forecast_growth,
                            "horizon_days": horizon,
                        },
                        action=f"{'Prepare for increased demand' if forecast_growth > 0 else 'Consider promotional activities to boost sales'}"
                    ))
            else:
                insights.append(Insight(
                    title=f"{horizon}-Day Revenue Forecast",
                    message=f"Projected revenue for the next {horizon} days: {format_currency(total_forecast)} (avg {format_currency(avg_daily)}/day).",
                    insight_type=InsightType.FORECAST,
                    priority=InsightPriority.MEDIUM,
                    data={
                        "forecast_total": total_forecast,
                        "forecast_daily_avg": avg_daily,
                        "horizon_days": horizon,
                    }
                ))
        
        return insights
    
    def _recommendation_insights(self) -> List[Insight]:
        """Generate actionable recommendations."""
        insights = []
        df = self.current_data
        
        if df is None or df.empty:
            return insights
        
        # Weekend vs weekday performance
        if "timestamp" in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
            
            weekend_revenue = df[df["is_weekend"]]["total_amount"].sum()
            weekday_revenue = df[~df["is_weekend"]]["total_amount"].sum()
            
            weekend_count = df["is_weekend"].sum()
            weekday_count = len(df) - weekend_count
            
            if weekend_count > 0 and weekday_count > 0:
                weekend_avg = weekend_revenue / max(1, len(df[df["is_weekend"]]["timestamp"].dt.date.unique()))
                weekday_avg = weekday_revenue / max(1, len(df[~df["is_weekend"]]["timestamp"].dt.date.unique()))
                
                if weekend_avg > weekday_avg * 1.3:
                    insights.append(Insight(
                        title="Weekend Sales Outperformance",
                        message=f"Weekend sales average {format_currency(weekend_avg)}/day vs {format_currency(weekday_avg)}/day on weekdays. Consider weekend-focused campaigns.",
                        insight_type=InsightType.RECOMMENDATION,
                        priority=InsightPriority.LOW,
                        data={
                            "weekend_avg": weekend_avg,
                            "weekday_avg": weekday_avg,
                        },
                        action="Increase weekend staffing and promotional activities."
                    ))
                elif weekday_avg > weekend_avg * 1.3:
                    insights.append(Insight(
                        title="Weekday Sales Dominance",
                        message=f"Weekday sales are stronger ({format_currency(weekday_avg)}/day) than weekends ({format_currency(weekend_avg)}/day).",
                        insight_type=InsightType.RECOMMENDATION,
                        priority=InsightPriority.LOW,
                        data={
                            "weekend_avg": weekend_avg,
                            "weekday_avg": weekday_avg,
                        },
                        action="Consider weekend promotions to boost off-peak sales."
                    ))
        
        return insights
    
    def get_top_insights(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top N insights by priority.
        
        Args:
            n: Number of insights to return
        
        Returns:
            List of top insight dictionaries
        """
        sorted_insights = sorted(self._insights, key=lambda x: x.priority.value)
        return [i.to_dict() for i in sorted_insights[:n]]
    
    def get_insights_by_type(self, insight_type: InsightType) -> List[Dict[str, Any]]:
        """
        Get insights filtered by type.
        
        Args:
            insight_type: Type to filter by
        
        Returns:
            List of matching insights
        """
        filtered = [i for i in self._insights if i.insight_type == insight_type]
        return [i.to_dict() for i in filtered]
    
    def get_critical_insights(self) -> List[Dict[str, Any]]:
        """Get only critical and high priority insights."""
        critical_priorities = {InsightPriority.CRITICAL, InsightPriority.HIGH}
        filtered = [i for i in self._insights if i.priority in critical_priorities]
        return [i.to_dict() for i in filtered]
    
    def get_insights_summary(self) -> Dict[str, Any]:
        """Get summary of all insights."""
        by_type = {}
        by_priority = {}
        
        for insight in self._insights:
            type_key = insight.insight_type.value
            priority_key = insight.priority.name.lower()
            
            by_type[type_key] = by_type.get(type_key, 0) + 1
            by_priority[priority_key] = by_priority.get(priority_key, 0) + 1
        
        return {
            "total_insights": len(self._insights),
            "by_type": by_type,
            "by_priority": by_priority,
            "last_generated": self._last_generated.isoformat() if self._last_generated else None,
            "has_critical": by_priority.get("critical", 0) > 0,
            "has_high": by_priority.get("high", 0) > 0,
        }
