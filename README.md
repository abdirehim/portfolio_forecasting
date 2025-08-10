# Portfolio Forecasting System

A comprehensive financial analysis platform that combines time series forecasting with Modern Portfolio Theory to optimize investment strategies for GMF Investments.

## Overview

This system processes historical financial data for three key assets (TSLA, BND, SPY) through a pipeline that includes:

- Data extraction from YFinance API
- Comprehensive exploratory data analysis
- Time series model development (ARIMA/SARIMA and LSTM)
- Portfolio optimization using the Efficient Frontier
- Strategy backtesting against benchmarks

## Features

- **Data Processing**: Automated data fetching, cleaning, and preprocessing
- **Time Series Forecasting**: ARIMA/SARIMA and LSTM models with automated parameter optimization
- **Portfolio Optimization**: Modern Portfolio Theory implementation using PyPortfolioOpt
- **Risk Analysis**: Value at Risk, Sharpe Ratio, and comprehensive risk metrics
- **Backtesting**: Strategy validation against benchmark portfolios
- **Visualization**: Interactive charts and comprehensive reporting

## Project Structure

```
portfolio_forecasting/
├── src/                    # Main source code
│   ├── data/              # Data access and preprocessing
│   ├── analysis/          # EDA and feature engineering
│   ├── models/            # Forecasting models
│   ├── portfolio/         # Portfolio optimization
│   ├── utils/             # Utilities and configuration
│   └── visualization/     # Plotting and reporting
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Test suite
├── config/                # Configuration files
├── data/                  # Data storage
└── requirements.txt       # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gmf-investments/portfolio-forecasting.git
cd portfolio-forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### 1. Configuration

The system uses YAML configuration files in the `config/` directory:
- `settings.yaml`: Application settings (data sources, preprocessing, visualization)
- `model_params.yaml`: Model hyperparameters and optimization settings

### 2. Data Fetching and Preprocessing

```python
from src.data.yfinance_client import YFinanceClient
from src.data.preprocessor import DataPreprocessor

# Fetch data
client = YFinanceClient()
data = client.fetch_data(['TSLA', 'BND', 'SPY'], '2015-07-01', '2025-07-31')

# Preprocess data
preprocessor = DataPreprocessor()
clean_data = preprocessor.preprocess_data(data)
```

### 3. Exploratory Data Analysis

```python
from src.analysis.eda_engine import EDAEngine

eda = EDAEngine()
report = eda.perform_eda(clean_data)
```

### 4. Time Series Forecasting

```python
from src.models.arima_forecaster import ARIMAForecaster
from src.models.lstm_forecaster import LSTMForecaster
from src.models.model_evaluator import ModelEvaluator

# Train models
arima_model = ARIMAForecaster()
lstm_model = LSTMForecaster()

arima_model.train(data['TSLA'])
lstm_model.train(data['TSLA'])

# Evaluate and compare
evaluator = ModelEvaluator()
results = evaluator.evaluate_models([arima_model, lstm_model], test_data)
```

### 5. Portfolio Optimization

```python
from src.portfolio.optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer()
optimal_portfolio = optimizer.optimize_portfolio(expected_returns, cov_matrix)
```

### 6. Backtesting

```python
from src.portfolio.backtester import Backtester

backtester = Backtester()
results = backtester.backtest_strategy(optimal_weights, '2024-08-01', '2025-07-31')
```

## Jupyter Notebooks

The `notebooks/` directory contains interactive analysis notebooks:

1. `01_data_exploration.ipynb` - Data fetching and EDA
2. `02_model_development.ipynb` - Model training and comparison
3. `03_forecasting.ipynb` - Future predictions and analysis
4. `04_portfolio_optimization.ipynb` - Portfolio optimization
5. `05_backtesting.ipynb` - Strategy validation

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

## Configuration

### Data Configuration
- **Symbols**: TSLA, BND, SPY
- **Date Range**: July 1, 2015 to July 31, 2025
- **Train/Test Split**: 2015-2023 (train), 2024-2025 (test)
- **Backtest Period**: August 1, 2024 to July 31, 2025

### Model Parameters
- **ARIMA**: Automated parameter optimization with seasonal components
- **LSTM**: 60-day sequence length, 2-layer architecture with dropout
- **Evaluation**: MAE, RMSE, MAPE metrics with 90% confidence intervals

### Portfolio Settings
- **Benchmark**: 60% SPY, 40% BND
- **Optimization**: Maximum Sharpe Ratio and Minimum Volatility
- **Risk-free Rate**: 2% annual

## Key Dependencies

- **Data**: pandas, numpy, yfinance
- **Statistical Models**: statsmodels, pmdarima
- **Deep Learning**: tensorflow, keras
- **Portfolio Optimization**: PyPortfolioOpt
- **Visualization**: matplotlib, seaborn, plotly

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, contact the GMF Investments team at analyst@gmf-investments.com.

## Acknowledgments

- Modern Portfolio Theory: Markowitz, H. (1952). "Portfolio Selection." Journal of Finance
- YFinance API for financial data
- PyPortfolioOpt library for portfolio optimization
- TensorFlow/Keras for deep learning implementations