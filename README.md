# Real-Time Sales Analytics Dashboard with ML Insights

A production-ready, enterprise-grade sales analytics platform featuring ML-powered forecasting, real-time anomaly detection, and automated business insights.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

### Core Analytics
- **Real-Time Data Streaming**: Continuous sales data simulation with realistic patterns
- **Interactive Dashboard**: Modern, responsive web interface with Chart.js visualizations
- **KPI Tracking**: Live metrics with trend indicators and historical comparisons

### Machine Learning
- **Sales Forecasting**: 7-30 day predictions using Random Forest and ARIMA ensemble
- **Anomaly Detection**: Real-time outlier detection using Isolation Forest and statistical methods
- **Automated Insights**: Natural language business recommendations

### Visualizations
- 📈 Sales trend with forecast overlay (Line Chart)
- 📊 Category performance analysis (Bar Chart)
- 🌍 Regional revenue distribution (Doughnut Chart)
- ⚠️ Anomaly timeline (Scatter Chart)

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | Flask, Python 3.9+ |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn, statsmodels |
| Statistics | scipy |
| Scheduling | APScheduler |
| Frontend | HTML5, CSS3, JavaScript ES6+ |
| Charts | Chart.js |
| UI Framework | Bootstrap 5 |
| Icons | Font Awesome |

## 📁 Project Structure

```
sales_analytics_dashboard/
├── data/
│   ├── raw/                    # Original sales data
│   ├── processed/              # Cleaned and engineered features
│   └── streaming/              # Real-time data buffer
├── models/                     # Trained ML models (.pkl)
├── backend/
│   ├── app.py                  # Flask application
│   ├── config.py               # Configuration settings
│   ├── data_ingestion.py       # Real-time data streaming
│   ├── data_preprocessing.py   # Data cleaning & features
│   ├── ml_forecast.py          # Forecasting models
│   ├── ml_anomaly.py           # Anomaly detection
│   ├── insights_engine.py      # Auto insight generation
│   ├── api_routes.py           # REST API endpoints
│   ├── utils.py                # Helper functions
│   └── scheduler.py            # Background tasks
├── frontend/
│   ├── templates/
│   │   └── index.html          # Dashboard UI
│   └── static/
│       ├── css/dashboard.css   # Styling
│       └── js/                 # JavaScript modules
├── scripts/
│   ├── generate_sample_data.py # Data generation
│   ├── train_models.py         # Model training
│   └── run_dashboard.py        # Application launcher
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks
├── requirements.txt
├── main.py                     # Entry point
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sales-analytics-dashboard.git
   cd sales-analytics-dashboard
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate sample data**
   ```bash
   python scripts/generate_sample_data.py --num-transactions 15000
   ```

5. **Train ML models**
   ```bash
   python scripts/train_models.py
   ```

6. **Start the dashboard**
   ```bash
   python main.py
   ```

7. **Open in browser**
   ```
   http://localhost:5000
   ```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sales/history` | GET | Historical sales data |
| `/api/sales/live` | GET | Real-time transactions |
| `/api/forecast` | GET | ML predictions |
| `/api/anomalies` | GET | Detected anomalies |
| `/api/kpis/summary` | GET | KPI metrics |
| `/api/insights` | GET | Business insights |
| `/api/analytics/by-category` | GET | Category breakdown |
| `/api/analytics/by-region` | GET | Regional analysis |
| `/api/health` | GET | System health check |

### Example API Response

```json
{
  "status": "success",
  "timestamp": "2026-01-07T14:30:00Z",
  "data": {
    "total_revenue": 1250000.00,
    "transactions": 15000,
    "avg_order_value": 83.33,
    "growth_rate": 0.15
  },
  "metadata": {
    "records_count": 100,
    "computation_time_ms": 45
  }
}
```

## ⚙️ Configuration

Key settings in `backend/config.py`:

```python
# Data generation
STREAMING_CONFIG = {
    "generation_interval_seconds": 5,
    "buffer_size": 10000,
}

# ML Models
FORECAST_CONFIG = {
    "algorithm": "random_forest",
    "short_term_horizon": 7,
    "long_term_horizon": 30,
}

# Anomaly Detection
ANOMALY_CONFIG = {
    "algorithm": "isolation_forest",
    "contamination": 0.05,
    "zscore_threshold": 3.0,
}
```

## 📈 Machine Learning Models

### Sales Forecasting
- **Random Forest Regressor**: Primary model with 200 estimators
- **ARIMA**: Time series baseline for trend analysis
- **Metrics**: RMSE, MAE, MAPE, R², Direction Accuracy

### Anomaly Detection
- **Isolation Forest**: Unsupervised outlier detection
- **Z-Score**: Statistical threshold-based detection
- **Severity Levels**: Low, Medium, High, Critical

## 🎨 Dashboard Features

- **Dark/Light Theme**: Toggle between themes
- **Real-Time Updates**: Auto-refresh every 10 seconds
- **Interactive Filters**: Date range, region, category
- **Export Options**: Download data as CSV
- **Responsive Design**: Mobile-friendly layout

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_ml_models.py -v

# Run with coverage
python -m pytest tests/ --cov=backend --cov-report=html
```

## 📝 Development

### Adding New Features

1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for functions/classes
- Keep functions small and focused

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/) for ML algorithms
- [Chart.js](https://www.chartjs.org/) for visualizations
- [Bootstrap](https://getbootstrap.com/) for UI components
- [Font Awesome](https://fontawesome.com/) for icons

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/sales-analytics-dashboard](https://github.com/yourusername/sales-analytics-dashboard)

---

⭐ Star this repository if you find it helpful!
