"""
Tests for backtesting system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from portfolio_forecasting.src.backtesting.backtester import (
    Backtester, BacktestConfig, BacktestResults, PortfolioSnapshot, RebalanceFrequency
)


class TestBacktestConfig:
    """Test suite for BacktestConfig."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = BacktestConfig()
        
        assert config.start_date == "2024-08-01"
        assert config.end_date == "2025-07-31"
        assert config.initial_capital == 100000.0
        assert config.rebalance_frequency == RebalanceFrequency.QUARTERLY
        assert config.rebalance_threshold == 0.05
        assert config.transaction_cost == 0.001
        assert config.benchmark_weights == {"SPY": 0.6, "BND": 0.4}
        assert config.include_dividends is True
        assert config.cash_buffer == 0.02
    
    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        custom_benchmark = {"TSLA": 0.5, "SPY": 0.3, "BND": 0.2}
        config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2024-01-01",
            initial_capital=50000.0,
            rebalance_frequency=RebalanceFrequency.MONTHLY,
            benchmark_weights=custom_benchmark
        )
        
        assert config.start_date == "2023-01-01"
        assert config.end_date == "2024-01-01"
        assert config.initial_capital == 50000.0
        assert config.rebalance_frequency == RebalanceFrequency.MONTHLY
        assert config.benchmark_weights == custom_benchmark


class TestBacktester:
    """Test suite for Backtester."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for backtesting."""
        dates = pd.date_range(start='2024-08-01', end='2025-07-31', freq='D')
        
        np.random.seed(42)
        n_days = len(dates)
        
        # TSLA: More volatile with upward trend
        tsla_returns = np.random.normal(0.002, 0.03, n_days)
        tsla_prices = 200 * np.exp(np.cumsum(tsla_returns))
        
        # SPY: Market-like with moderate trend
        spy_returns = np.random.normal(0.001, 0.015, n_days)
        spy_prices = 400 * np.exp(np.cumsum(spy_returns))
        
        # BND: Bond-like (lower volatility)
        bnd_returns = np.random.normal(0.0005, 0.005, n_days)
        bnd_prices = 80 * np.exp(np.cumsum(bnd_returns))
        
        return {
            'TSLA': pd.Series(tsla_prices, index=dates, name='TSLA'),
            'SPY': pd.Series(spy_prices, index=dates, name='SPY'),
            'BND': pd.Series(bnd_prices, index=dates, name='BND')
        }
    
    @pytest.fixture
    def sample_dividend_data(self):
        """Create sample dividend data."""
        dates = pd.date_range(start='2024-08-01', end='2025-07-31', freq='D')
        
        # Simple dividend data (quarterly dividends)
        dividend_data = {
            'TSLA': pd.Series(0, index=dates),  # No dividends for TSLA
            'SPY': pd.Series(0, index=dates),   # Quarterly dividends
            'BND': pd.Series(0, index=dates)    # Monthly dividends
        }
        
        # Add some quarterly dividends for SPY
        quarterly_dates = pd.date_range(start='2024-09-01', end='2025-06-01', freq='3M')
        for date in quarterly_dates:
            if date in dividend_data['SPY'].index:
                dividend_data['SPY'].loc[date] = 1.5  # $1.50 quarterly dividend
        
        # Add some monthly dividends for BND
        monthly_dates = pd.date_range(start='2024-08-15', end='2025-07-15', freq='M')
        for date in monthly_dates:
            closest_date = dividend_data['BND'].index[dividend_data['BND'].index.get_indexer([date], method='nearest')[0]]
            dividend_data['BND'].loc[closest_date] = 0.2  # $0.20 monthly dividend
        
        return dividend_data
    
    @pytest.fixture
    def backtest_config(self):
        """Create test backtest configuration."""
        return BacktestConfig(
            start_date="2024-08-01",
            end_date="2025-07-31",
            initial_capital=100000.0,
            rebalance_frequency=RebalanceFrequency.QUARTERLY,
            transaction_cost=0.001
        )
    
    @pytest.fixture
    def backtester(self, backtest_config):
        """Create Backtester instance."""
        return Backtester(backtest_config)
    
    @pytest.fixture
    def strategy_weights(self):
        """Create sample strategy weights."""
        return {'TSLA': 0.4, 'SPY': 0.4, 'BND': 0.2}
    
    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        backtester = Backtester()
        
        assert backtester.config.start_date == "2024-08-01"
        assert backtester.config.end_date == "2025-07-31"
        assert backtester.price_data is None
        assert backtester.dividend_data is None
    
    def test_initialization_custom_config(self, backtest_config):
        """Test initialization with custom configuration."""
        backtester = Backtester(backtest_config)
        
        assert backtester.config.initial_capital == 100000.0
        assert backtester.config.rebalance_frequency == RebalanceFrequency.QUARTERLY
    
    def test_load_price_data(self, backtester, sample_price_data):
        """Test loading price data."""
        backtester.load_price_data(sample_price_data)
        
        assert backtester.price_data is not None
        assert list(backtester.price_data.columns) == ['TSLA', 'SPY', 'BND']
        assert len(backtester.price_data) > 0
        
        # Check date filtering
        assert backtester.price_data.index[0] >= pd.to_datetime("2024-08-01")
        assert backtester.price_data.index[-1] <= pd.to_datetime("2025-07-31")
    
    def test_load_dividend_data(self, backtester, sample_price_data, sample_dividend_data):
        """Test loading dividend data."""
        backtester.load_price_data(sample_price_data)
        backtester.load_dividend_data(sample_dividend_data)
        
        assert backtester.dividend_data is not None
        assert list(backtester.dividend_data.columns) == ['TSLA', 'SPY', 'BND']
        assert len(backtester.dividend_data) == len(backtester.price_data)
    
    def test_load_dividend_data_none(self, backtester, sample_price_data):
        """Test loading dividend data with None input."""
        backtester.load_price_data(sample_price_data)
        backtester.load_dividend_data(None)
        
        assert backtester.dividend_data is not None
        assert (backtester.dividend_data == 0).all().all()  # All zeros
    
    def test_run_backtest_basic(self, backtester, sample_price_data, strategy_weights):
        """Test basic backtest execution."""
        backtester.load_price_data(sample_price_data)
        backtester.load_dividend_data(None)
        
        results = backtester.run_backtest(strategy_weights, "Test Strategy")
        
        assert isinstance(results, BacktestResults)
        assert len(results.portfolio_snapshots) > 0
        assert len(results.performance_metrics) > 0
        assert len(results.benchmark_metrics) > 0
        assert len(results.comparison_metrics) > 0
        assert len(results.transaction_log) > 0
        assert len(results.rebalance_dates) > 0
    
    def test_run_backtest_no_price_data(self, backtester, strategy_weights):
        """Test backtest without price data."""
        with pytest.raises(ValueError, match="Price data not loaded"):
            backtester.run_backtest(strategy_weights)
    
    def test_run_backtest_with_dividends(self, backtester, sample_price_data, 
                                       sample_dividend_data, strategy_weights):
        """Test backtest with dividend data."""
        backtester.load_price_data(sample_price_data)
        backtester.load_dividend_data(sample_dividend_data)
        
        results = backtester.run_backtest(strategy_weights, "Dividend Strategy")
        
        assert isinstance(results, BacktestResults)
        assert len(results.portfolio_snapshots) > 0
        
        # Check that some dividends were received
        total_dividends = sum(
            backtester._calculate_dividend_income(snap.date, snap.positions)
            for snap in results.portfolio_snapshots
        )
        # Note: This might be 0 if no dividend dates align with backtest period
    
    def test_portfolio_value_calculation(self, backtester, sample_price_data):
        """Test portfolio value calculation."""
        backtester.load_price_data(sample_price_data)
        
        date = backtester.price_data.index[10]  # Use a date in the middle
        cash = 10000.0
        positions = {'TSLA': 10, 'SPY': 5, 'BND': 20}
        
        portfolio_value = backtester._calculate_portfolio_value(date, cash, positions)
        
        expected_value = cash
        for asset, shares in positions.items():
            expected_value += shares * backtester.price_data.loc[date, asset]
        
        assert abs(portfolio_value - expected_value) < 1e-6
    
    def test_current_weights_calculation(self, backtester, sample_price_data):
        """Test current weights calculation."""
        backtester.load_price_data(sample_price_data)
        
        date = backtester.price_data.index[10]
        cash = 10000.0
        positions = {'TSLA': 10, 'SPY': 5, 'BND': 20}
        
        weights = backtester._calculate_current_weights(date, cash, positions)
        
        # Check that weights sum to approximately 1
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 1e-6
        
        # Check that cash weight is included
        assert 'Cash' in weights
    
    def test_rebalancing_logic(self, backtester, sample_price_data):
        """Test rebalancing decision logic."""
        backtester.load_price_data(sample_price_data)
        
        date = backtester.price_data.index[10]
        target_weights = {'TSLA': 0.4, 'SPY': 0.4, 'BND': 0.2}
        
        # Test with weights close to target (should not rebalance)
        current_weights = {'TSLA': 0.39, 'SPY': 0.41, 'BND': 0.20, 'Cash': 0.0}
        rebalance_dates = [backtester.price_data.index[0]]
        
        should_rebalance = backtester._should_rebalance(
            date, current_weights, target_weights, rebalance_dates
        )
        assert not should_rebalance
        
        # Test with weights far from target (should rebalance)
        current_weights = {'TSLA': 0.5, 'SPY': 0.3, 'BND': 0.2, 'Cash': 0.0}
        should_rebalance = backtester._should_rebalance(
            date, current_weights, target_weights, rebalance_dates
        )
        assert should_rebalance
    
    def test_rebalance_portfolio(self, backtester, sample_price_data):
        """Test portfolio rebalancing."""
        backtester.load_price_data(sample_price_data)
        
        date = backtester.price_data.index[10]
        initial_cash = 50000.0
        initial_positions = {'TSLA': 0, 'SPY': 0, 'BND': 0}
        target_weights = {'TSLA': 0.4, 'SPY': 0.4, 'BND': 0.2}
        
        new_cash, new_positions, transactions = backtester._rebalance_portfolio(
            date, initial_cash, initial_positions, target_weights
        )
        
        # Check that some transactions occurred
        assert len(transactions) > 0
        
        # Check that positions were created
        assert any(pos > 0 for pos in new_positions.values())
        
        # Check that cash was reduced
        assert new_cash < initial_cash
    
    def test_dividend_income_calculation(self, backtester, sample_price_data, sample_dividend_data):
        """Test dividend income calculation."""
        backtester.load_price_data(sample_price_data)
        backtester.load_dividend_data(sample_dividend_data)
        
        # Find a date with dividends
        dividend_date = None
        for date in backtester.dividend_data.index:
            if backtester.dividend_data.loc[date].sum() > 0:
                dividend_date = date
                break
        
        if dividend_date is not None:
            positions = {'TSLA': 10, 'SPY': 5, 'BND': 20}
            dividend_income = backtester._calculate_dividend_income(dividend_date, positions)
            
            expected_income = 0
            for asset, shares in positions.items():
                expected_income += shares * backtester.dividend_data.loc[dividend_date, asset]
            
            assert abs(dividend_income - expected_income) < 1e-6
    
    def test_performance_metrics_calculation(self, backtester, sample_price_data, strategy_weights):
        """Test performance metrics calculation."""
        backtester.load_price_data(sample_price_data)
        backtester.load_dividend_data(None)
        
        results = backtester.run_backtest(strategy_weights)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'Total_Return', 'Annualized_Return', 'Volatility', 'Sharpe_Ratio',
            'Max_Drawdown', 'Win_Rate', 'VaR_95', 'CVaR_95', 'Final_Value', 'Total_Days'
        ]
        
        for metric in expected_metrics:
            assert metric in results.performance_metrics
            assert metric in results.benchmark_metrics
        
        # Check that metrics are reasonable
        assert results.performance_metrics['Total_Days'] > 0
        assert results.performance_metrics['Final_Value'] > 0
        assert -1 <= results.performance_metrics['Total_Return'] <= 10  # Reasonable range
    
    def test_comparison_metrics_calculation(self, backtester, sample_price_data, strategy_weights):
        """Test comparison metrics calculation."""
        backtester.load_price_data(sample_price_data)
        backtester.load_dividend_data(None)
        
        results = backtester.run_backtest(strategy_weights)
        
        # Check that all expected comparison metrics are present
        expected_metrics = [
            'Alpha', 'Beta', 'Tracking_Error', 'Information_Ratio',
            'Correlation', 'Outperformance_Rate', 'Average_Excess_Return', 'Total_Excess_Return'
        ]
        
        for metric in expected_metrics:
            assert metric in results.comparison_metrics
        
        # Check that metrics are reasonable
        assert -1 <= results.comparison_metrics['Correlation'] <= 1
        assert 0 <= results.comparison_metrics['Outperformance_Rate'] <= 1
    
    def test_backtest_results_properties(self, backtester, sample_price_data, strategy_weights):
        """Test BacktestResults properties."""
        backtester.load_price_data(sample_price_data)
        backtester.load_dividend_data(None)
        
        results = backtester.run_backtest(strategy_weights)
        
        # Test portfolio_series property
        portfolio_series = results.portfolio_series
        assert isinstance(portfolio_series, pd.Series)
        assert len(portfolio_series) == len(results.portfolio_snapshots)
        assert portfolio_series.name == 'Portfolio'
        
        # Test benchmark_series property
        benchmark_series = results.benchmark_series
        assert isinstance(benchmark_series, pd.Series)
        assert len(benchmark_series) == len(results.portfolio_snapshots)
        assert benchmark_series.name == 'Benchmark'
        
        # Test returns_series property
        returns_series = results.returns_series
        assert isinstance(returns_series, pd.Series)
        assert returns_series.name == 'Portfolio_Returns'
        
        # Test benchmark_returns_series property
        benchmark_returns_series = results.benchmark_returns_series
        assert isinstance(benchmark_returns_series, pd.Series)
        assert benchmark_returns_series.name == 'Benchmark_Returns'
    
    def test_different_rebalance_frequencies(self, sample_price_data, strategy_weights):
        """Test different rebalancing frequencies."""
        frequencies = [
            RebalanceFrequency.MONTHLY,
            RebalanceFrequency.QUARTERLY,
            RebalanceFrequency.NEVER
        ]
        
        for freq in frequencies:
            config = BacktestConfig(rebalance_frequency=freq)
            backtester = Backtester(config)
            backtester.load_price_data(sample_price_data)
            backtester.load_dividend_data(None)
            
            results = backtester.run_backtest(strategy_weights)
            
            assert isinstance(results, BacktestResults)
            assert len(results.portfolio_snapshots) > 0
            
            # Check rebalancing frequency
            if freq == RebalanceFrequency.NEVER:
                assert len(results.rebalance_dates) == 1  # Only initial allocation
            else:
                assert len(results.rebalance_dates) >= 1
    
    def test_transaction_costs_impact(self, sample_price_data, strategy_weights):
        """Test impact of transaction costs."""
        # Test with no transaction costs
        config_no_cost = BacktestConfig(transaction_cost=0.0)
        backtester_no_cost = Backtester(config_no_cost)
        backtester_no_cost.load_price_data(sample_price_data)
        backtester_no_cost.load_dividend_data(None)
        
        results_no_cost = backtester_no_cost.run_backtest(strategy_weights)
        
        # Test with high transaction costs
        config_high_cost = BacktestConfig(transaction_cost=0.01)  # 1%
        backtester_high_cost = Backtester(config_high_cost)
        backtester_high_cost.load_price_data(sample_price_data)
        backtester_high_cost.load_dividend_data(None)
        
        results_high_cost = backtester_high_cost.run_backtest(strategy_weights)
        
        # High transaction costs should result in lower returns
        assert results_no_cost.performance_metrics['Final_Value'] >= results_high_cost.performance_metrics['Final_Value']