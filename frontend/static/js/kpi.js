/**
 * KPI Calculation and Display Module
 * Handles all KPI-related operations and UI updates
 */

/**
 * KPI Manager Class
 */
class KPIManager {
    constructor() {
        this.elements = {
            totalRevenue: document.getElementById('totalRevenue'),
            revenueChange: document.getElementById('revenueChange'),
            totalTransactions: document.getElementById('totalTransactions'),
            transactionChange: document.getElementById('transactionChange'),
            avgOrderValue: document.getElementById('avgOrderValue'),
            aovChange: document.getElementById('aovChange'),
            uniqueCustomers: document.getElementById('uniqueCustomers'),
            customerChange: document.getElementById('customerChange'),
            topCategory: document.getElementById('topCategory'),
            topCategoryRevenue: document.getElementById('topCategoryRevenue'),
            anomalyCount: document.getElementById('anomalyCount'),
            anomalySeverity: document.getElementById('anomalySeverity')
        };

        this.previousKPIs = {};
    }

    /**
     * Update all KPIs from API response
     */
    updateKPIs(kpiData) {
        if (!kpiData) return;

        // Total Revenue
        this.updateKPICard(
            this.elements.totalRevenue,
            this.elements.revenueChange,
            this.formatCurrency(kpiData.total_revenue || 0),
            kpiData.revenue_change || 0,
            kpiData.revenue_trend
        );

        // Transactions
        this.updateKPICard(
            this.elements.totalTransactions,
            this.elements.transactionChange,
            this.formatNumber(kpiData.transaction_count || 0),
            kpiData.transaction_change || 0
        );

        // Average Order Value
        this.updateKPICard(
            this.elements.avgOrderValue,
            this.elements.aovChange,
            this.formatCurrency(kpiData.avg_order_value || 0),
            0 // AOV change not provided in basic KPI response
        );

        // Unique Customers
        if (this.elements.uniqueCustomers) {
            this.elements.uniqueCustomers.textContent = this.formatNumber(kpiData.unique_customers || 0);
        }

        // Top Category
        if (kpiData.top_category) {
            if (this.elements.topCategory) {
                this.elements.topCategory.textContent = kpiData.top_category.name || '-';
            }
            if (this.elements.topCategoryRevenue) {
                this.elements.topCategoryRevenue.textContent = this.formatCurrency(kpiData.top_category.revenue || 0);
            }
        }

        // Anomaly Count
        if (this.elements.anomalyCount) {
            const count = kpiData.anomaly_count || 0;
            this.elements.anomalyCount.textContent = count;

            // Add animation if count increases
            if (count > (this.previousKPIs.anomaly_count || 0)) {
                this.animateKPICard(document.getElementById('kpi-anomalies'));
            }
        }

        if (this.elements.anomalySeverity) {
            const count = kpiData.anomaly_count || 0;
            if (count === 0) {
                this.elements.anomalySeverity.textContent = 'No alerts';
            } else if (count === 1) {
                this.elements.anomalySeverity.textContent = '1 alert detected';
            } else {
                this.elements.anomalySeverity.textContent = `${count} alerts detected`;
            }
        }

        // Store for comparison
        this.previousKPIs = { ...kpiData };
    }

    /**
     * Update a single KPI card
     */
    updateKPICard(valueElement, changeElement, value, changePercent, trend = null) {
        if (valueElement) {
            const oldValue = valueElement.textContent;
            valueElement.textContent = value;

            // Animate if value changed
            if (oldValue !== value) {
                this.animateValue(valueElement);
            }
        }

        if (changeElement) {
            const isPositive = changePercent > 0;
            const isNegative = changePercent < 0;
            const isNeutral = changePercent === 0;

            // Update class
            changeElement.classList.remove('positive', 'negative', 'neutral');
            if (isPositive) {
                changeElement.classList.add('positive');
            } else if (isNegative) {
                changeElement.classList.add('negative');
            } else {
                changeElement.classList.add('neutral');
            }

            // Update content
            const icon = isPositive ? 'fa-arrow-up' : (isNegative ? 'fa-arrow-down' : 'fa-minus');
            const displayPercent = Math.abs(changePercent).toFixed(1);

            changeElement.innerHTML = `<i class="fas ${icon}"></i> ${displayPercent}%`;
        }
    }

    /**
     * Animate value change
     */
    animateValue(element) {
        element.style.transform = 'scale(1.05)';
        element.style.transition = 'transform 0.3s ease';

        setTimeout(() => {
            element.style.transform = 'scale(1)';
        }, 300);
    }

    /**
     * Animate KPI card
     */
    animateKPICard(cardElement) {
        if (!cardElement) return;

        cardElement.style.animation = 'none';
        cardElement.offsetHeight; // Trigger reflow
        cardElement.style.animation = 'highlightCard 0.5s ease';
    }

    /**
     * Calculate KPIs from transaction data
     */
    calculateFromTransactions(transactions, previousTransactions = []) {
        if (!transactions || transactions.length === 0) {
            return {
                total_revenue: 0,
                transaction_count: 0,
                avg_order_value: 0,
                unique_customers: 0,
                anomaly_count: 0
            };
        }

        const totalRevenue = transactions.reduce((sum, t) => sum + (t.total_amount || 0), 0);
        const transactionCount = transactions.length;
        const avgOrderValue = totalRevenue / transactionCount;

        const uniqueCustomers = new Set(
            transactions.map(t => t.customer_id).filter(Boolean)
        ).size;

        const anomalyCount = transactions.filter(t => t.is_anomaly).length;

        // Calculate changes if previous data available
        let revenueChange = 0;
        let transactionChange = 0;

        if (previousTransactions && previousTransactions.length > 0) {
            const prevRevenue = previousTransactions.reduce((sum, t) => sum + (t.total_amount || 0), 0);
            const prevCount = previousTransactions.length;

            if (prevRevenue > 0) {
                revenueChange = ((totalRevenue - prevRevenue) / prevRevenue) * 100;
            }
            if (prevCount > 0) {
                transactionChange = ((transactionCount - prevCount) / prevCount) * 100;
            }
        }

        // Find top category
        const categoryRevenue = {};
        transactions.forEach(t => {
            if (t.category) {
                categoryRevenue[t.category] = (categoryRevenue[t.category] || 0) + (t.total_amount || 0);
            }
        });

        let topCategory = null;
        let maxRevenue = 0;
        Object.entries(categoryRevenue).forEach(([category, revenue]) => {
            if (revenue > maxRevenue) {
                maxRevenue = revenue;
                topCategory = { name: category, revenue };
            }
        });

        return {
            total_revenue: totalRevenue,
            revenue_change: revenueChange,
            revenue_trend: revenueChange > 0 ? 'up' : (revenueChange < 0 ? 'down' : 'stable'),
            transaction_count: transactionCount,
            transaction_change: transactionChange,
            avg_order_value: avgOrderValue,
            unique_customers: uniqueCustomers,
            top_category: topCategory,
            anomaly_count: anomalyCount
        };
    }

    /**
     * Format currency value
     */
    formatCurrency(value) {
        if (value >= 1000000) {
            return '$' + (value / 1000000).toFixed(2) + 'M';
        } else if (value >= 1000) {
            return '$' + (value / 1000).toFixed(1) + 'K';
        }
        return '$' + value.toFixed(2);
    }

    /**
     * Format number with commas
     */
    formatNumber(value) {
        return value.toLocaleString('en-US');
    }

    /**
     * Format percentage
     */
    formatPercentage(value, includeSign = true) {
        const sign = includeSign && value > 0 ? '+' : '';
        return `${sign}${value.toFixed(1)}%`;
    }

    /**
     * Get trend indicator
     */
    getTrendIndicator(current, previous) {
        if (previous === 0) return '→';
        const change = (current - previous) / previous;
        if (change > 0.01) return '↑';
        if (change < -0.01) return '↓';
        return '→';
    }

    /**
     * Update real-time KPIs
     */
    updateRealtimeKPIs(realtimeData) {
        // This can be used for per-minute metrics
        if (!realtimeData) return;

        // Could add additional real-time KPI elements here
        console.log('Realtime KPIs:', realtimeData);
    }
}

// Export for use in main.js
window.KPIManager = KPIManager;

// Add CSS animation for card highlight
const style = document.createElement('style');
style.textContent = `
    @keyframes highlightCard {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        50% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
`;
document.head.appendChild(style);
