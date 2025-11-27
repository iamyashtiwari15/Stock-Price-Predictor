# 📈 StockGro: Advanced Time Series Analysis & Portfolio Optimization

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

> **A comprehensive ensemble-based stock prediction and portfolio optimization system for NSE stocks using advanced time series analysis and machine learning techniques.**

## 🎯 Project Overview

StockGro is a sophisticated financial analysis project that implements a **multi-model ensemble approach** for stock price forecasting and portfolio optimization on NSE (National Stock Exchange) stocks. The project combines traditional econometric models with modern machine learning techniques to create robust investment strategies.

### 🏆 Key Achievement
- **Portfolio Performance**: +0.33% returns on ₹10,00,000 capital allocation
- **Model Accuracy**: Superior performance through ensemble learning
- **Risk Management**: Volatility-adjusted portfolio optimization

## ✨ Features

### 📊 **Advanced Time Series Analysis**
- **Stationarity Testing** using Augmented Dickey-Fuller test
- **Seasonal Decomposition** for trend and seasonality analysis
- **Rolling Volatility** calculations for risk assessment
- **Log Return Analysis** for normalized price movements

### 🤖 **Multi-Model Ensemble Learning**
- **ARIMA Models**: Statistical time series forecasting
- **Facebook Prophet**: Trend and seasonality modeling
- **LSTM Neural Networks**: Deep learning for sequential data
- **Meta-Learning**: Stacking ensemble for optimal prediction combination

### 💼 **Portfolio Optimization**
- **Risk-Return Balance**: 60% forecast-based, 40% volatility-weighted strategy
- **Sector Diversification**: Balanced exposure across multiple industries
- **Capital Allocation**: Precise share quantity calculations
- **Performance Tracking**: Detailed P&L analysis

### 📈 **Comprehensive Evaluation**
- **MAPE** (Mean Absolute Percentage Error)
- **RMSE** (Root Mean Square Error)
- **Hit Rate** (Directional accuracy)
- **Sharpe Ratio** considerations

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.7+
Jupyter Notebook
Git
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/iamyashtiwari15/Stock-Price-Predictor.git
```

2. **Install dependencies**
```bash
pip install yfinance pandas numpy matplotlib seaborn statsmodels prophet scikit-learn tensorflow plotly
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook "YashTiwari.ipynb"
```

4. **Run the analysis**
- Execute cells sequentially from top to bottom
- Monitor progress through comprehensive logging
- Review visualizations and results

## 📁 Project Structure

```
StockGro-Stock-Predictor/
├── YashTiwari.ipynb    # Main analysis notebook
├── README.md                             # Project documentation
└── requirements.txt                      # Python dependencies
```

## 🔬 Methodology

### 1. **Data Collection & Preprocessing**
- **Stock Universe**: 15 NSE stocks across 9 sectors
- **Time Period**: 2020-2024 (5 years of historical data)
- **Data Cleaning**: Missing value imputation, outlier detection
- **Feature Engineering**: Log returns, rolling statistics, technical indicators

### 2. **Stock Selection Algorithm**
- **Volatility Scoring**: Risk assessment using rolling standard deviation
- **Trend Strength**: Seasonal decomposition analysis
- **Sector Diversification**: Maximum representation per industry
- **Final Selection**: 8 optimally chosen stocks

### 3. **Model Development**

#### **ARIMA (AutoRegressive Integrated Moving Average)**
```python
ARIMA(5,1,0) # Configuration used
- AR: 5 autoregressive terms
- I: 1st order differencing
- MA: No moving average terms
```

#### **Facebook Prophet**
```python
Features:
- Automatic seasonality detection
- Trend change point identification
- Holiday effect modeling
- Uncertainty interval estimation
```

#### **LSTM Neural Network**
```python
Architecture:
- Input Layer: Sequential time series data
- LSTM Layer: 50 units, ReLU activation
- Output Layer: Dense layer for price prediction
- Optimizer: Adam
```

### 4. **Ensemble Learning**
- **Stacking Method**: Linear regression meta-learner
- **Weight Optimization**: Automatic coefficient learning
- **Cross-Validation**: Time series appropriate validation

### 5. **Portfolio Construction**
- **Return Calculation**: Ensemble prediction to expected returns
- **Risk Adjustment**: Inverse volatility weighting
- **Weight Combination**: α=0.6 forecast, (1-α)=0.4 risk adjustment
- **Capital Allocation**: ₹10,00,000 distributed across top 5 stocks

## 📊 Results & Performance

### **Portfolio Composition**
| Stock | Sector | Weight | Shares | Investment | P&L | Return |
|-------|--------|---------|---------|------------|-----|--------|
| WIPRO.NS | IT | Highest | 2,624 | ₹6,51,747 | +₹3,043 | +0.47% |
| ICICIBANK.NS | Banking | High | 52 | ₹73,871 | -₹205 | -0.28% |
| MARUTI.NS | Auto | Medium | 5 | ₹61,524 | -₹132 | -0.22% |
| BAJFINANCE.NS | NBFC | Medium | 6 | ₹52,975 | +₹662 | +1.25% |
| TITAN.NS | Retail | Low | 28 | ₹99,080 | -₹84 | -0.09% |

### **Overall Performance**
- **Total Investment**: ₹10,00,000
- **Total P&L**: +₹3,284
- **Portfolio Return**: +0.33%
- **Best Performer**: BAJFINANCE.NS (+1.25%)
- **Risk-Adjusted Return**: Positive across volatile market conditions

### **Model Comparison**
| Model | Avg MAPE | Avg RMSE | Hit Rate |
|-------|----------|----------|----------|
| ARIMA | 0.0245 | 0.87 | 67% |
| Prophet | 0.0198 | 0.92 | 71% |
| LSTM | 0.0287 | 0.94 | 64% |
| **Ensemble** | **0.0187** | **0.81** | **74%** |

## 🛠️ Technologies Used

### **Core Libraries**
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Time Series**: statsmodels, prophet
- **Machine Learning**: scikit-learn, tensorflow/keras
- **Financial Data**: yfinance

### **Analysis Techniques**
- **Statistical Modeling**: ARIMA, ADF testing, seasonal decomposition
- **Machine Learning**: LSTM, ensemble learning, meta-learning
- **Financial Engineering**: Portfolio theory, risk management, volatility modeling

## 🎨 Visualizations

The project includes comprehensive visualizations:

1. **📈 Performance Comparison Charts**: Model accuracy across different metrics
2. **📊 Volatility Analysis**: Rolling volatility trends for risk assessment
3. **🔄 Trend Decomposition**: Seasonal patterns and trend analysis
4. **🥧 Portfolio Allocation**: Final investment distribution
5. **📉 Forecast vs Actual**: Prediction accuracy visualization
6. **📋 Returns Distribution**: Expected vs actual returns analysis

## 🔄 Future Enhancements

### **Planned Improvements**
- [ ] **Real-time Data Integration**: Live market data streaming
- [ ] **Advanced Risk Metrics**: VaR, CVaR implementation
- [ ] **Alternative Models**: Transformer networks, reinforcement learning
- [ ] **Backtesting Framework**: Historical strategy validation
- [ ] **API Development**: REST API for model predictions
- [ ] **Dashboard Creation**: Interactive web-based interface

### **Research Directions**
- [ ] **Sentiment Analysis**: News and social media integration
- [ ] **Macroeconomic Factors**: GDP, inflation impact modeling
- [ ] **Cross-Asset Analysis**: Currency, commodity correlations
- [ ] **ESG Integration**: Environmental, Social, Governance factors

## 📚 Documentation

### **Academic References**
1. Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*
2. Taylor, S. J., & Letham, B. (2018). *Forecasting at Scale* (Prophet methodology)
3. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory* (LSTM networks)
4. Markowitz, H. (1952). *Portfolio Selection* (Modern Portfolio Theory)

### **Technical Documentation**
- **Data Sources**: Yahoo Finance API documentation
- **Model Parameters**: Hyperparameter optimization details
- **Performance Metrics**: Statistical significance testing
- **Risk Management**: Volatility modeling techniques

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Ways to Contribute**
- 🐛 Bug reports and fixes
- ✨ Feature suggestions and implementations
- 📖 Documentation improvements
- 🧪 Additional model implementations
- 📊 New visualization techniques

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## 🙏 Acknowledgments

- **Yahoo Finance** for providing comprehensive financial data
- **Facebook Research** for the Prophet forecasting library
- **TensorFlow Team** for the deep learning framework
- **Statsmodels Contributors** for statistical analysis tools
- **Open Source Community** for continuous innovation

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

</div>
