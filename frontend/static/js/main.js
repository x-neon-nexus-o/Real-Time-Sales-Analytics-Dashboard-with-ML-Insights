/**
 * Main Dashboard Controller
 * Handles API communication, data updates, and UI interactions
 */

/**
 * Sales Dashboard Class
 */
class SalesDashboard {
    constructor() {
        this.apiBase = window.location.origin + '/api';
        this.refreshInterval = 10000; // 10 seconds
        this.refreshTimer = null;
        this.isLoading = false;
        this.showForecast = false;

        // Managers
        this.chartManager = new ChartManager();
        this.kpiManager = new KPIManager();

        // State
        this.currentFilters = {
            days: 30,
            region: '',
            category: ''
        };

        // Cached data
        this.cachedData = {
            kpis: null,
            trends: null,
            categories: null,
            regions: null,
            insights: null,
            anomalies: null,
            transactions: null,
            forecast: null
        };

        // Initialize
        this.init();
    }

    /**
     * Initialize the dashboard
     */
    async init() {
        console.log('Initializing Sales Analytics Dashboard...');

        // Show loading overlay
        this.showLoading(true);

        try {
            // Initialize charts
            this.chartManager.initializeCharts();

            // Set up event listeners
            this.setupEventListeners();

            // Load initial data
            await this.loadAllData();

            // Start auto-refresh
            this.startAutoRefresh();

            // Hide loading overlay
            this.showLoading(false);

            // Update last updated time
            this.updateLastUpdatedTime();

            console.log('Dashboard initialized successfully');
        } catch (error) {
            console.error('Error initializing dashboard:', error);
            this.showLoading(false);
            this.showToast('Error loading dashboard', 'error');
        }
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Filter changes
        document.getElementById('dateRange')?.addEventListener('change', (e) => {
            this.currentFilters.days = parseInt(e.target.value);
            this.loadAllData();
        });

        document.getElementById('regionFilter')?.addEventListener('change', (e) => {
            this.currentFilters.region = e.target.value;
            this.loadAllData();
        });

        document.getElementById('categoryFilter')?.addEventListener('change', (e) => {
            this.currentFilters.category = e.target.value;
            this.loadAllData();
        });

        // Refresh button
        document.getElementById('refreshBtn')?.addEventListener('click', () => {
            this.loadAllData();
        });

        // Export button
        document.getElementById('exportBtn')?.addEventListener('click', () => {
            this.exportData();
        });

        // Theme toggle
        document.getElementById('themeToggle')?.addEventListener('click', () => {
            this.toggleTheme();
        });

        // Forecast toggle
        document.getElementById('forecastToggle')?.addEventListener('click', (e) => {
            this.showForecast = !this.showForecast;
            this.chartManager.toggleForecast(this.showForecast);
            e.target.innerHTML = this.showForecast
                ? '<i class="fas fa-magic"></i> Hide Forecast'
                : '<i class="fas fa-magic"></i> Show Forecast';

            if (this.showForecast && !this.cachedData.forecast) {
                this.loadForecast();
            }
        });
    }

    /**
     * Load all dashboard data
     */
    async loadAllData() {
        if (this.isLoading) return;
        this.isLoading = true;

        try {
            // Load data in parallel
            const [kpis, trends, categories, regions, insights, transactions] = await Promise.all([
                this.fetchKPIs(),
                this.fetchTrends(),
                this.fetchCategoryData(),
                this.fetchRegionData(),
                this.fetchInsights(),
                this.fetchLiveTransactions()
            ]);

            // Update UI components
            this.kpiManager.updateKPIs(kpis);
            this.chartManager.updateSalesTrend(trends, this.cachedData.forecast);
            this.chartManager.updateCategoryChart(categories);
            this.chartManager.updateRegionChart(regions);
            this.chartManager.updateAnomalyChart(transactions);
            this.chartManager.updateHourlyChart(transactions);

            this.updateInsightsPanel(insights);
            this.updateTransactionsTable(transactions);
            this.updateAnomalyAlerts(transactions);

            // Update connection status
            this.updateConnectionStatus(true);
            this.updateLastUpdatedTime();

        } catch (error) {
            console.error('Error loading data:', error);
            this.updateConnectionStatus(false);
            this.showToast('Error loading data', 'error');
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Fetch KPIs from API
     */
    async fetchKPIs() {
        const response = await fetch(
            `${this.apiBase}/kpis/summary?days=${this.currentFilters.days}`
        );
        const result = await response.json();
        this.cachedData.kpis = result.data;
        return result.data;
    }

    /**
     * Fetch trends data
     */
    async fetchTrends() {
        const params = new URLSearchParams({
            days: this.currentFilters.days,
            granularity: 'daily'
        });

        if (this.currentFilters.region) {
            params.append('region', this.currentFilters.region);
        }
        if (this.currentFilters.category) {
            params.append('category', this.currentFilters.category);
        }

        const response = await fetch(`${this.apiBase}/analytics/trends?${params}`);
        const result = await response.json();
        this.cachedData.trends = result.data;
        return result.data;
    }

    /**
     * Fetch category data
     */
    async fetchCategoryData() {
        const response = await fetch(
            `${this.apiBase}/analytics/by-category?days=${this.currentFilters.days}`
        );
        const result = await response.json();
        this.cachedData.categories = result.data;
        return result.data;
    }

    /**
     * Fetch region data
     */
    async fetchRegionData() {
        const response = await fetch(
            `${this.apiBase}/analytics/by-region?days=${this.currentFilters.days}`
        );
        const result = await response.json();
        this.cachedData.regions = result.data;
        return result.data;
    }

    /**
     * Fetch insights
     */
    async fetchInsights() {
        try {
            const response = await fetch(`${this.apiBase}/insights`);
            const result = await response.json();
            this.cachedData.insights = result.data;
            return result.data;
        } catch (error) {
            console.warn('Could not fetch insights:', error);
            return [];
        }
    }

    /**
     * Fetch live transactions
     */
    async fetchLiveTransactions() {
        const response = await fetch(`${this.apiBase}/sales/live?limit=100`);
        const result = await response.json();
        this.cachedData.transactions = result.data;
        return result.data;
    }

    /**
     * Fetch forecast data
     */
    async loadForecast() {
        try {
            const response = await fetch(`${this.apiBase}/forecast?horizon=7`);
            const result = await response.json();

            if (result.status === 'success') {
                this.cachedData.forecast = result.data;
                this.chartManager.updateSalesTrend(this.cachedData.trends, result.data);
                this.showToast('Forecast loaded successfully', 'success');
            }
        } catch (error) {
            console.error('Error loading forecast:', error);
            this.showToast('Error loading forecast', 'error');
        }
    }

    /**
     * Update insights panel
     */
    updateInsightsPanel(insights) {
        const container = document.getElementById('insightsList');
        const countBadge = document.getElementById('insightCount');

        if (!container) return;

        if (!insights || insights.length === 0) {
            container.innerHTML = `
                <div class="insight-placeholder">
                    <i class="fas fa-check-circle"></i>
                    <span>No new insights available</span>
                </div>
            `;
            if (countBadge) countBadge.textContent = '0';
            return;
        }

        if (countBadge) countBadge.textContent = insights.length;

        const html = insights.slice(0, 10).map(insight => `
            <div class="insight-item">
                <div class="insight-icon ${insight.type || 'performance'}">
                    <i class="fas ${this.getInsightIcon(insight.type)}"></i>
                </div>
                <div class="insight-content">
                    <div class="insight-title">${this.escapeHtml(insight.title)}</div>
                    <div class="insight-message">${this.escapeHtml(insight.message)}</div>
                    ${insight.action ? `<div class="insight-action">${this.escapeHtml(insight.action)}</div>` : ''}
                </div>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    /**
     * Update anomaly alerts panel
     */
    updateAnomalyAlerts(transactions) {
        const container = document.getElementById('alertsList');
        const countBadge = document.getElementById('alertCount');

        if (!container) return;

        const anomalies = transactions?.filter(t => t.is_anomaly) || [];

        if (anomalies.length === 0) {
            container.innerHTML = `
                <div class="alert-placeholder">
                    <i class="fas fa-check-circle"></i>
                    <span>No anomalies detected</span>
                </div>
            `;
            if (countBadge) countBadge.textContent = '0';
            return;
        }

        if (countBadge) countBadge.textContent = anomalies.length;

        const html = anomalies.slice(0, 10).map(anomaly => `
            <div class="alert-item">
                <div class="alert-severity ${anomaly.severity || 'medium'}"></div>
                <div class="alert-content">
                    <div class="alert-title">${this.escapeHtml(anomaly.product_name || 'Unknown Product')}</div>
                    <div class="alert-message">
                        Unusual ${anomaly.category || 'transaction'} in ${anomaly.region || 'unknown region'} - 
                        $${(anomaly.total_amount || 0).toFixed(2)}
                    </div>
                    <div class="alert-time">${this.formatTime(anomaly.timestamp)}</div>
                </div>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    /**
     * Update transactions table
     */
    updateTransactionsTable(transactions) {
        const tbody = document.getElementById('transactionsBody');
        if (!tbody || !transactions) return;

        const html = transactions.slice(0, 20).map((t, index) => `
            <tr class="${index < 3 ? 'new-row' : ''}">
                <td>${this.formatTime(t.timestamp)}</td>
                <td><code>${this.escapeHtml(t.transaction_id?.substring(0, 15) || '-')}...</code></td>
                <td>${this.escapeHtml(t.product_name || '-')}</td>
                <td>${this.escapeHtml(t.category || '-')}</td>
                <td>${this.escapeHtml(t.region || '-')}</td>
                <td>${t.quantity || 0}</td>
                <td>$${(t.total_amount || 0).toFixed(2)}</td>
                <td>
                    <span class="status-badge ${t.is_anomaly ? 'anomaly' : 'normal'}">
                        ${t.is_anomaly ? 'Anomaly' : 'Normal'}
                    </span>
                </td>
            </tr>
        `).join('');

        tbody.innerHTML = html;
    }

    /**
     * Start auto-refresh
     */
    startAutoRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
        }
        this.refreshTimer = setInterval(() => {
            this.loadAllData();
        }, this.refreshInterval);
    }

    /**
     * Stop auto-refresh
     */
    stopAutoRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    }

    /**
     * Toggle theme
     */
    toggleTheme() {
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        html.setAttribute('data-theme', newTheme);

        const themeIcon = document.querySelector('#themeToggle i');
        if (themeIcon) {
            themeIcon.className = newTheme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
        }

        // Update charts for new theme
        this.chartManager.updateTheme(newTheme === 'dark');

        // Save preference
        localStorage.setItem('dashboard-theme', newTheme);
    }

    /**
     * Export data to CSV
     */
    exportData() {
        const transactions = this.cachedData.transactions || [];

        if (transactions.length === 0) {
            this.showToast('No data to export', 'warning');
            return;
        }

        // Create CSV content
        const headers = ['timestamp', 'transaction_id', 'product_name', 'category', 'region', 'quantity', 'unit_price', 'total_amount'];
        const csvContent = [
            headers.join(','),
            ...transactions.map(t => headers.map(h => `"${t[h] || ''}"`).join(','))
        ].join('\n');

        // Create download
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `sales_data_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);

        this.showToast('Data exported successfully', 'success');
    }

    /**
     * Show/hide loading overlay
     */
    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            if (show) {
                overlay.classList.remove('hidden');
            } else {
                overlay.classList.add('hidden');
            }
        }
    }

    /**
     * Update connection status
     */
    updateConnectionStatus(isOnline) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');

        if (statusDot) {
            statusDot.classList.toggle('online', isOnline);
            statusDot.classList.toggle('offline', !isOnline);
        }

        if (statusText) {
            statusText.textContent = isOnline ? 'Live' : 'Offline';
            statusText.style.color = isOnline ? 'var(--accent-success)' : 'var(--accent-danger)';
        }
    }

    /**
     * Update last updated time
     */
    updateLastUpdatedTime() {
        const element = document.getElementById('lastUpdateTime');
        if (element) {
            element.textContent = new Date().toLocaleTimeString();
        }
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        const toastId = 'toast-' + Date.now();
        const bgClass = {
            success: 'bg-success',
            error: 'bg-danger',
            warning: 'bg-warning',
            info: 'bg-info'
        }[type] || 'bg-info';

        const toastHtml = `
            <div id="${toastId}" class="toast" role="alert">
                <div class="toast-header ${bgClass} text-white">
                    <strong class="me-auto">Dashboard</strong>
                    <small>just now</small>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">
                    ${this.escapeHtml(message)}
                </div>
            </div>
        `;

        container.insertAdjacentHTML('beforeend', toastHtml);

        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, { autohide: true, delay: 3000 });
        toast.show();

        // Remove after hidden
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    }

    /**
     * Get insight icon based on type
     */
    getInsightIcon(type) {
        const icons = {
            performance: 'fa-chart-line',
            product: 'fa-box',
            regional: 'fa-globe',
            anomaly: 'fa-exclamation-triangle',
            forecast: 'fa-magic',
            trend: 'fa-trending-up',
            recommendation: 'fa-lightbulb'
        };
        return icons[type] || 'fa-info-circle';
    }

    /**
     * Format timestamp for display
     */
    formatTime(timestamp) {
        if (!timestamp) return '-';
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        });
    }

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Destroy dashboard
     */
    destroy() {
        this.stopAutoRefresh();
        this.chartManager.destroy();
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    // Load saved theme
    const savedTheme = localStorage.getItem('dashboard-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);

    const themeIcon = document.querySelector('#themeToggle i');
    if (themeIcon) {
        themeIcon.className = savedTheme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
    }

    // Initialize dashboard
    window.dashboard = new SalesDashboard();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (window.dashboard) {
        if (document.hidden) {
            window.dashboard.stopAutoRefresh();
        } else {
            window.dashboard.startAutoRefresh();
            window.dashboard.loadAllData();
        }
    }
});

// Export for external use
window.SalesDashboard = SalesDashboard;
