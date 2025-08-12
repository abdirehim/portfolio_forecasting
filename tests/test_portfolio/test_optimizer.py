"""
Tests for portfolio optimization system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from portfolio_forecasting.src.portfolio.optimizer import (
    PortfolioOptimizer, OptimizationConfig, PortfolioMetrics, EfficientFrontierData
)
from portfolio_forecasting.src.forecasting.forecast_generator import ForecastOutput


class TestOptimizationConfig:
    """Test suite for OptimizationConfig."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = OptimizationConfig()
        
        assert config.risk_free_rate == 0.02
        assert config.target_return is None
        assert config.target_volatility is None
        assert config.weight_bounds == (0.0, 1.0)
        assert config.sector_constraints is None
        assert config.optimization_method == "max_sharpe"
        assert config.gamma == 0
        assert config.market_neutral is False
        assert config.l2_reg == 0.01
    
    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        config = OptimizationConfig(
            risk_free_rate=0.03,
            target_return=0.10,
            weight_bounds=(0.05, 0.40),
            optimization_method="min_volatility"
        )
        
        assert config.risk_free_rate == 0.03
        assert config.target_return == 0.10
        assert config.weight_bounds == (0.05, 0.40)
        assert config.optimization_method == "min_volatility"


class TestPortfolioOptimizer:
    """Test suite for PortfolioOptimizer."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        
        # Create correlated price series
        np.random.seed(42)
        n_days = len(dates)
        
        # TSLA: More volatile
        tsla_returns = np.random.normal(0.001, 0.03, n_days)
        tsla_prices = 100 * np.exp(np.cumsum(tsla_returns))
        
        # SPY: Market-like
        spy_returns = np.random.normal(0.0008, 0.015, n_days)
        spy_prices = 300 * np.exp(np.cumsum(spy_returns))
        
        # BND: Bond-like (lower volatility)
        bnd_returns = np.random.normal(0.0003, 0.005, n_days)
        bnd_prices = 80 * np.exp(np.cumsum(bnd_returns))
        
        return {
            'TSLA': pd.Series(tsla_prices, index=dates, name='TSLA'),
            'SPY': pd.Series(spy_prices, index=dates, name='SPY'),
            'BND': pd.Series(bnd_prices, index=dates, name='BND')
        }
    
    @pytest.fixture
    def optimizer_config(self):
        """Create test optimization configuration."""
        return OptimizationConfig(
            risk_free_rate=0.02,
            optimization_method="max_sharpe"
        )
    
    @pytest.fixture
    def portfolio_optimizer(self, optimizer_config):
        """Create PortfolioOptimizer instance."""
        return PortfolioOptimizer(optimizer_config)
    
    @pytest.fixture
    def mock_forecast_data(self):
        """Create mock forecast data."""
        mock_forecast = Mock(spec=ForecastOutput)
        
        # Create mock predictions
        dates = pd.date_range(start='2025-01-01', periods=365, freq='D')
        predictions = pd.Series(np.linspace(200, 250, 365), index=dates)
        mock_forecast.predictions = predictions
        
        return {'TSLA': mock_forecast}
    
    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        optimizer = PortfolioOptimizer()
        
        assert optimizer.config.risk_free_rate == 0.02
        assert optimizer.assets == []
        assert optimizer.price_data is None
        assert optimizer.expected_returns is None
        assert optimizer.covariance_matrix is None
    
    def test_initialization_custom_config(self, optimizer_config):
        """Test initialization with custom configuration."""
        optimizer = PortfolioOptimizer(optimizer_config)
        
        assert optimizer.config.risk_free_rate == 0.02
        assert optimizer.config.optimization_method == "max_sharpe"
    
    def test_set_assets(self, portfolio_optimizer):
        """Test setting assets for optimization."""
        assets = ['TSLA', 'SPY', 'BND']
        portfolio_optimizer.set_assets(assets)
        
        assert portfolio_optimizer.assets == assets
    
    def test_load_price_data(self, portfolio_optimizer, sample_price_data):
        """Test loading price data."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        
        assert portfolio_optimizer.price_data is not None
        assert list(portfolio_optimizer.price_data.columns) == ['TSLA', 'SPY', 'BND']
        assert len(portfolio_optimizer.price_data) > 0
    
    def test_load_price_data_missing_assets(self, portfolio_optimizer, sample_price_data):
        """Test loading price data with missing assets."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND', 'MISSING'])
        
        with pytest.raises(ValueError, match="Missing price data for assets"):
            portfolio_optimizer.load_price_data(sample_price_data)
    
    def test_calculate_expected_returns_historical(self, portfolio_optimizer, sample_price_data):
        """Test calculating expected returns using historical method."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        
        expected_returns = portfolio_optimizer.calculate_expected_returns(method="historical")
        
        assert isinstance(expected_returns, pd.Series)
        assert len(expected_returns) == 3
        assert all(asset in expected_returns.index for asset in ['TSLA', 'SPY', 'BND'])
        assert portfolio_optimizer.expected_returns is not None
    
    def test_calculate_expected_returns_forecast(self, portfolio_optimizer, sample_price_data, mock_forecast_data):
        """Test calculating expected returns using forecast method."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        
        expected_returns = portfolio_optimizer.calculate_expected_returns(
            forecast_data=mock_forecast_data, 
            method="forecast"
        )
        
        assert isinstance(expected_returns, pd.Series)
        assert len(expected_returns) == 3
        # TSLA should have forecast-based return, others historical
        assert 'TSLA' in expected_returns.index
    
    def test_calculate_expected_returns_mixed(self, portfolio_optimizer, sample_price_data, mock_forecast_data):
        """Test calculating expected returns using mixed method."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        
        expected_returns = portfolio_optimizer.calculate_expected_returns(
            forecast_data=mock_forecast_data, 
            method="mixed"
        )
        
        assert isinstance(expected_returns, pd.Series)
        assert len(expected_returns) == 3
    
    def test_calculate_expected_returns_invalid_method(self, portfolio_optimizer, sample_price_data):
        """Test calculating expected returns with invalid method."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        
        with pytest.raises(ValueError, match="Invalid method"):
            portfolio_optimizer.calculate_expected_returns(method="invalid")
    
    def test_calculate_covariance_matrix_sample(self, portfolio_optimizer, sample_price_data):
        """Test calculating covariance matrix using sample method."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        
        cov_matrix = portfolio_optimizer.calculate_covariance_matrix(method="sample")
        
        assert isinstance(cov_matrix, pd.DataFrame)
        assert cov_matrix.shape == (3, 3)
        assert list(cov_matrix.index) == ['TSLA', 'SPY', 'BND']
        assert list(cov_matrix.columns) == ['TSLA', 'SPY', 'BND']
        assert portfolio_optimizer.covariance_matrix is not None
    
    def test_calculate_covariance_matrix_invalid_method(self, portfolio_optimizer, sample_price_data):
        """Test calculating covariance matrix with invalid method."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        
        with pytest.raises(ValueError, match="Invalid covariance method"):
            portfolio_optimizer.calculate_covariance_matrix(method="invalid")
    
    def test_optimize_portfolio_max_sharpe(self, portfolio_optimizer, sample_price_data):
        """Test portfolio optimization using max Sharpe method."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        portfolio_optimizer.calculate_expected_returns()
        portfolio_optimizer.calculate_covariance_matrix()
        
        portfolio_metrics = portfolio_optimizer.optimize_portfolio(method="max_sharpe")
        
        assert isinstance(portfolio_metrics, PortfolioMetrics)
        assert portfolio_metrics.expected_return > 0
        assert portfolio_metrics.volatility > 0
        assert portfolio_metrics.sharpe_ratio > 0
        assert len(portfolio_metrics.weights) == 3
        assert abs(sum(portfolio_metrics.weights.values()) - 1.0) < 1e-6  # Weights sum to 1
    
    def test_optimize_portfolio_min_volatility(self, portfolio_optimizer, sample_price_data):
        """Test portfolio optimization using min volatility method."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        portfolio_optimizer.calculate_expected_returns()
        portfolio_optimizer.calculate_covariance_matrix()
        
        portfolio_metrics = portfolio_optimizer.optimize_portfolio(method="min_volatility")
        
        assert isinstance(portfolio_metrics, PortfolioMetrics)
        assert portfolio_metrics.volatility > 0
        assert len(portfolio_metrics.weights) == 3
    
    def test_optimize_portfolio_efficient_return(self, portfolio_optimizer, sample_price_data):
        """Test portfolio optimization using efficient return method."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        portfolio_optimizer.calculate_expected_returns()
        portfolio_optimizer.calculate_covariance_matrix()
        
        target_return = 0.08
        portfolio_metrics = portfolio_optimizer.optimize_portfolio(
            method="efficient_return", 
            target_return=target_return
        )
        
        assert isinstance(portfolio_metrics, PortfolioMetrics)
        assert abs(portfolio_metrics.expected_return - target_return) < 0.01  # Close to target
    
    def test_optimize_portfolio_no_data(self, portfolio_optimizer):
        """Test portfolio optimization without required data."""
        with pytest.raises(ValueError, match="Expected returns and covariance matrix must be calculated first"):
            portfolio_optimizer.optimize_portfolio()
    
    def test_generate_efficient_frontier(self, portfolio_optimizer, sample_price_data):
        """Test efficient frontier generation."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        portfolio_optimizer.calculate_expected_returns()
        portfolio_optimizer.calculate_covariance_matrix()
        
        frontier_data = portfolio_optimizer.generate_efficient_frontier(num_portfolios=20)
        
        assert isinstance(frontier_data, EfficientFrontierData)
        assert len(frontier_data.returns) > 0
        assert len(frontier_data.volatilities) > 0
        assert len(frontier_data.sharpe_ratios) > 0
        assert isinstance(frontier_data.max_sharpe_portfolio, PortfolioMetrics)
        assert isinstance(frontier_data.min_volatility_portfolio, PortfolioMetrics)
        assert len(frontier_data.frontier_weights) > 0
    
    def test_generate_efficient_frontier_no_data(self, portfolio_optimizer):
        """Test efficient frontier generation without required data."""
        with pytest.raises(ValueError, match="Expected returns and covariance matrix must be calculated first"):
            portfolio_optimizer.generate_efficient_frontier()
    
    @patch('portfolio_forecasting.src.portfolio.optimizer.DiscreteAllocation')
    @patch('portfolio_forecasting.src.portfolio.optimizer.get_latest_prices')
    def test_calculate_discrete_allocation(self, mock_get_prices, mock_discrete_alloc, 
                                         portfolio_optimizer, sample_price_data):
        """Test discrete allocation calculation."""
        # Setup
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        
        # Mock portfolio metrics
        portfolio_metrics = PortfolioMetrics(
            expected_return=0.10,
            volatility=0.15,
            sharpe_ratio=0.67,
            weights={'TSLA': 0.4, 'SPY': 0.4, 'BND': 0.2},
            portfolio_value=1.0,
            diversification_ratio=1.2
        )
        
        # Mock discrete allocation
        mock_da_instance = Mock()
        mock_da_instance.lp_portfolio.return_value = ({'TSLA': 2, 'SPY': 1, 'BND': 5}, 100.0)
        mock_discrete_alloc.return_value = mock_da_instance
        
        mock_get_prices.return_value = {'TSLA': 200, 'SPY': 400, 'BND': 80}
        
        # Test
        allocation = portfolio_optimizer.calculate_discrete_allocation(portfolio_metrics, 10000)
        
        assert isinstance(allocation, dict)
        assert 'TSLA' in allocation
        mock_discrete_alloc.assert_called_once()
    
    def test_visualize_efficient_frontier(self, portfolio_optimizer, sample_price_data):
        """Test efficient frontier visualization."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        portfolio_optimizer.calculate_expected_returns()
        portfolio_optimizer.calculate_covariance_matrix()
        
        frontier_data = portfolio_optimizer.generate_efficient_frontier(num_portfolios=10)
        fig = portfolio_optimizer.visualize_efficient_frontier(frontier_data)
        
        assert fig is not None
        assert len(fig.axes) == 1
    
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_efficient_frontier_with_save(self, mock_savefig, 
                                                   portfolio_optimizer, sample_price_data):
        """Test efficient frontier visualization with saving."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        portfolio_optimizer.calculate_expected_returns()
        portfolio_optimizer.calculate_covariance_matrix()
        
        frontier_data = portfolio_optimizer.generate_efficient_frontier(num_portfolios=10)
        save_path = "/test/path/frontier.png"
        fig = portfolio_optimizer.visualize_efficient_frontier(frontier_data, save_path=save_path)
        
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
    
    def test_diversification_ratio_calculation(self, portfolio_optimizer, sample_price_data):
        """Test diversification ratio calculation."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        portfolio_optimizer.calculate_covariance_matrix()
        
        weights = {'TSLA': 0.4, 'SPY': 0.4, 'BND': 0.2}
        div_ratio = portfolio_optimizer._calculate_diversification_ratio(weights)
        
        assert isinstance(div_ratio, float)
        assert div_ratio >= 1.0  # Should be >= 1 for diversified portfolio
    
    def test_portfolio_metrics_structure(self, portfolio_optimizer, sample_price_data):
        """Test the structure of portfolio metrics."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        portfolio_optimizer.calculate_expected_returns()
        portfolio_optimizer.calculate_covariance_matrix()
        
        portfolio_metrics = portfolio_optimizer.optimize_portfolio()
        
        # Verify structure
        assert hasattr(portfolio_metrics, 'expected_return')
        assert hasattr(portfolio_metrics, 'volatility')
        assert hasattr(portfolio_metrics, 'sharpe_ratio')
        assert hasattr(portfolio_metrics, 'weights')
        assert hasattr(portfolio_metrics, 'portfolio_value')
        assert hasattr(portfolio_metrics, 'diversification_ratio')
        
        # Verify types
        assert isinstance(portfolio_metrics.expected_return, float)
        assert isinstance(portfolio_metrics.volatility, float)
        assert isinstance(portfolio_metrics.sharpe_ratio, float)
        assert isinstance(portfolio_metrics.weights, dict)
        assert isinstance(portfolio_metrics.diversification_ratio, float)
    
    def test_weight_bounds_constraint(self, sample_price_data):
        """Test portfolio optimization with weight bounds."""
        config = OptimizationConfig(weight_bounds=(0.1, 0.5))  # Min 10%, Max 50%
        optimizer = PortfolioOptimizer(config)
        
        optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        optimizer.load_price_data(sample_price_data)
        optimizer.calculate_expected_returns()
        optimizer.calculate_covariance_matrix()
        
        portfolio_metrics = optimizer.optimize_portfolio()
        
        # Check that all weights are within bounds
        for weight in portfolio_metrics.weights.values():
            assert weight >= 0.1 - 1e-6  # Allow small numerical errors
            assert weight <= 0.5 + 1e-6
    
    def test_different_optimization_methods(self, portfolio_optimizer, sample_price_data):
        """Test different optimization methods produce different results."""
        portfolio_optimizer.set_assets(['TSLA', 'SPY', 'BND'])
        portfolio_optimizer.load_price_data(sample_price_data)
        portfolio_optimizer.calculate_expected_returns()
        portfolio_optimizer.calculate_covariance_matrix()
        
        max_sharpe = portfolio_optimizer.optimize_portfolio(method="max_sharpe")
        min_vol = portfolio_optimizer.optimize_portfolio(method="min_volatility")
        
        # Results should be different
        assert max_sharpe.weights != min_vol.weights
        assert max_sharpe.volatility >= min_vol.volatility  # Min vol should have lower volatility