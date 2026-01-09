/**
 * Chart.js Configurations and Chart Management
 * Handles all chart creation, updates, and interactions
 */

// Chart.js default configuration
Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
Chart.defaults.color = getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim() || '#64748b';
Chart.defaults.borderColor = getComputedStyle(document.documentElement).getPropertyValue('--border-color').trim() || '#e2e8f0';

/**
 * Chart Manager Class
 * Manages all dashboard charts
 */
class ChartManager {
    constructor() {
        this.charts = {};
        this.colors = {
            primary: '#3b82f6',
            secondary: '#8b5cf6',
            success: '#10b981',
            warning: '#f59e0b',
            danger: '#ef4444',
            info: '#06b6d4',
            categories: [
                '#3b82f6', '#10b981', '#f59e0b', 
                '#ef4444', '#8b5cf6', '#06b6d4',
                '#ec4899', '#84cc16', '#f97316'
            ]
        };
    }

    /**
     * Initialize all charts
     */
    initializeCharts() {
        this.createSalesTrendChart();
        this.createCategoryChart();
        this.createRegionChart();
        this.createAnomalyChart();
        this.createHourlyChart();
        console.log('All charts initialized');
    }

    /**
     * Get common chart options
     */
    getCommonOptions(showLegend = true) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 750,
                easing: 'easeInOutQuart'
            },
            plugins: {
                legend: {
                    display: showLegend,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                        font: {
                            size: 11
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleFont: {
                        size: 13,
                        weight: '600'
                    },
                    bodyFont: {
                        size: 12
                    },
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: true
                }
            }
        };
    }

    /**
     * Create Sales Trend Chart with Forecast
     */
    createSalesTrendChart() {
        const ctx = document.getElementById('salesTrendChart');
        if (!ctx) return;

        this.charts.salesTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Actual Revenue',
                        data: [],
                        borderColor: this.colors.primary,
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true,
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 3,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'Forecast',
                        data: [],
                        borderColor: this.colors.secondary,
                        backgroundColor: 'transparent',
                        borderDash: [5, 5],
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 3,
                        pointStyle: 'triangle',
                        hidden: true
                    },
                    {
                        label: 'Confidence Interval',
                        data: [],
                        borderColor: 'transparent',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        hidden: true
                    }
                ]
            },
            options: {
                ...this.getCommonOptions(),
                scales: {
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            maxTicksLimit: 10
                        }
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            callback: (value) => '$' + this.formatNumber(value)
                        }
                    }
                },
                plugins: {
                    ...this.getCommonOptions().plugins,
                    tooltip: {
                        ...this.getCommonOptions().plugins.tooltip,
                        callbacks: {
                            label: (context) => {
                                const label = context.dataset.label || '';
                                const value = context.parsed.y;
                                return `${label}: $${this.formatNumber(value)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Create Category Performance Chart
     */
    createCategoryChart() {
        const ctx = document.getElementById('categoryChart');
        if (!ctx) return;

        this.charts.category = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Revenue',
                    data: [],
                    backgroundColor: this.colors.categories,
                    borderRadius: 6,
                    borderSkipped: false
                }]
            },
            options: {
                ...this.getCommonOptions(false),
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            callback: (value) => '$' + this.formatNumber(value)
                        }
                    },
                    y: {
                        grid: {
                            display: false
                        }
                    }
                },
                plugins: {
                    ...this.getCommonOptions(false).plugins,
                    tooltip: {
                        ...this.getCommonOptions().plugins.tooltip,
                        callbacks: {
                            label: (context) => {
                                return `Revenue: $${this.formatNumber(context.parsed.x)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Create Regional Distribution Chart
     */
    createRegionChart() {
        const ctx = document.getElementById('regionChart');
        if (!ctx) return;

        this.charts.region = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: this.colors.categories,
                    borderWidth: 0,
                    hoverOffset: 10
                }]
            },
            options: {
                ...this.getCommonOptions(),
                cutout: '60%',
                plugins: {
                    ...this.getCommonOptions().plugins,
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 15,
                            usePointStyle: true,
                            font: {
                                size: 11
                            }
                        }
                    },
                    tooltip: {
                        ...this.getCommonOptions().plugins.tooltip,
                        callbacks: {
                            label: (context) => {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.parsed / total) * 100).toFixed(1);
                                return `$${this.formatNumber(context.parsed)} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Create Anomaly Detection Chart
     */
    createAnomalyChart() {
        const ctx = document.getElementById('anomalyChart');
        if (!ctx) return;

        this.charts.anomaly = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Normal',
                        data: [],
                        backgroundColor: 'rgba(16, 185, 129, 0.6)',
                        pointRadius: 6,
                        pointHoverRadius: 8
                    },
                    {
                        label: 'Anomaly',
                        data: [],
                        backgroundColor: 'rgba(239, 68, 68, 0.8)',
                        pointRadius: 10,
                        pointHoverRadius: 12,
                        pointStyle: 'triangle'
                    }
                ]
            },
            options: {
                ...this.getCommonOptions(),
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Time Index'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Revenue'
                        },
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            callback: (value) => '$' + this.formatNumber(value)
                        }
                    }
                },
                plugins: {
                    ...this.getCommonOptions().plugins,
                    tooltip: {
                        ...this.getCommonOptions().plugins.tooltip,
                        callbacks: {
                            label: (context) => {
                                const datasetLabel = context.dataset.label;
                                const value = context.parsed.y;
                                return `${datasetLabel}: $${this.formatNumber(value)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Create Hourly Pattern Chart
     */
    createHourlyChart() {
        const ctx = document.getElementById('hourlyChart');
        if (!ctx) return;

        this.charts.hourly = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Array.from({length: 24}, (_, i) => `${i}:00`),
                datasets: [{
                    label: 'Transactions',
                    data: new Array(24).fill(0),
                    backgroundColor: 'rgba(59, 130, 246, 0.7)',
                    borderRadius: 4
                }]
            },
            options: {
                ...this.getCommonOptions(false),
                scales: {
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            maxTicksLimit: 12
                        }
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    }
                }
            }
        });
    }

    /**
     * Update Sales Trend Chart
     */
    updateSalesTrend(trendData, forecastData = null) {
        if (!this.charts.salesTrend || !trendData) return;

        const labels = trendData.map(d => this.formatDate(d.date || d.datetime));
        const values = trendData.map(d => d.total_revenue || d.total_amount || 0);

        this.charts.salesTrend.data.labels = labels;
        this.charts.salesTrend.data.datasets[0].data = values;

        // Update forecast if provided
        if (forecastData && forecastData.dates) {
            const allLabels = [...labels, ...forecastData.dates];
            const forecastValues = [...new Array(values.length).fill(null), ...forecastData.values];
            
            this.charts.salesTrend.data.labels = allLabels;
            this.charts.salesTrend.data.datasets[1].data = forecastValues;
            
            // Confidence interval
            if (forecastData.confidence_upper) {
                const upperBound = [...new Array(values.length).fill(null), ...forecastData.confidence_upper];
                this.charts.salesTrend.data.datasets[2].data = upperBound;
            }
        }

        this.charts.salesTrend.update('none');
    }

    /**
     * Toggle forecast visibility
     */
    toggleForecast(show) {
        if (!this.charts.salesTrend) return;
        
        this.charts.salesTrend.data.datasets[1].hidden = !show;
        this.charts.salesTrend.data.datasets[2].hidden = !show;
        this.charts.salesTrend.update();
    }

    /**
     * Update Category Chart
     */
    updateCategoryChart(categoryData) {
        if (!this.charts.category || !categoryData) return;

        // Sort by revenue descending and take top 6
        const sorted = [...categoryData].sort((a, b) => b.total_revenue - a.total_revenue).slice(0, 6);
        
        this.charts.category.data.labels = sorted.map(d => d.category);
        this.charts.category.data.datasets[0].data = sorted.map(d => d.total_revenue);
        this.charts.category.update('none');
    }

    /**
     * Update Region Chart
     */
    updateRegionChart(regionData) {
        if (!this.charts.region || !regionData) return;

        this.charts.region.data.labels = regionData.map(d => d.region);
        this.charts.region.data.datasets[0].data = regionData.map(d => d.total_revenue);
        this.charts.region.update('none');
    }

    /**
     * Update Anomaly Chart
     */
    updateAnomalyChart(transactions) {
        if (!this.charts.anomaly || !transactions) return;

        const normal = [];
        const anomalies = [];

        transactions.forEach((t, index) => {
            const point = {
                x: index,
                y: t.total_amount || 0
            };

            if (t.is_anomaly) {
                anomalies.push(point);
            } else {
                normal.push(point);
            }
        });

        this.charts.anomaly.data.datasets[0].data = normal;
        this.charts.anomaly.data.datasets[1].data = anomalies;
        this.charts.anomaly.update('none');
    }

    /**
     * Update Hourly Pattern Chart
     */
    updateHourlyChart(transactions) {
        if (!this.charts.hourly || !transactions) return;

        const hourCounts = new Array(24).fill(0);

        transactions.forEach(t => {
            if (t.timestamp) {
                const date = new Date(t.timestamp);
                const hour = date.getHours();
                hourCounts[hour]++;
            }
        });

        this.charts.hourly.data.datasets[0].data = hourCounts;
        this.charts.hourly.update('none');
    }

    /**
     * Format number for display
     */
    formatNumber(value) {
        if (value >= 1000000) {
            return (value / 1000000).toFixed(1) + 'M';
        } else if (value >= 1000) {
            return (value / 1000).toFixed(1) + 'K';
        }
        return value.toFixed(0);
    }

    /**
     * Format date for display
     */
    formatDate(dateStr) {
        if (!dateStr) return '';
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }

    /**
     * Update chart theme
     */
    updateTheme(isDark) {
        const textColor = isDark ? '#cbd5e1' : '#64748b';
        const gridColor = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)';

        Chart.defaults.color = textColor;
        Chart.defaults.borderColor = isDark ? '#334155' : '#e2e8f0';

        // Update all charts
        Object.values(this.charts).forEach(chart => {
            if (chart.options.scales) {
                Object.values(chart.options.scales).forEach(scale => {
                    if (scale.grid) {
                        scale.grid.color = gridColor;
                    }
                });
            }
            chart.update('none');
        });
    }

    /**
     * Destroy all charts
     */
    destroy() {
        Object.values(this.charts).forEach(chart => {
            if (chart) chart.destroy();
        });
        this.charts = {};
    }
}

// Export for use in main.js
window.ChartManager = ChartManager;
