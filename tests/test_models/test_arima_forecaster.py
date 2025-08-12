"""
Tests for ARIMA forecaster implementation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from portfolio_forecasting.src.models.arima_forecaster import ARIMAForecaster
from portfolio_forecasting.src.models.base_forecaster import ForecastResult, ModelDiagnostics


class TestARIMAForecaster:
    """Test suite for ARIMA forecaster."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        # Create a simple trend + noise time series
        trend = np.linspace(100, 200, len(dates))
        noise = np.random.normal(0, 5, len(dates))
        values = trend + noise
        return pd.Series(values, index=dates, name='test_series')
    
    @pytest.fixture
    def arima_forecaster(self):
        """Create ARIMA forecaster instance."""
        return ARIMAForecaster(name="TestARIMA", auto_optimize=True)
    
    def test_initialization(self, arima_forecaster):
        """Test ARIMA forecaster initialization."""
        assert arima_forecaster.name == "TestARIMA"
        assert arima_forecaster.auto_optimize is True
        assert arima_forecaster.is_fitted is False
        assert arima_forecaster.model is None
    
    def test_fit_with_valid_data(self, arima_forecaster, sample_data):
        """Test fitting ARIMA model with valid data."""
        result = arima_forecaster.fit(sample_data)
        
        assert result is arima_forecaster  # Should return self
        assert arima_forecaster.is_fitted is True
        assert arima_forecaster.model is not None
        assert arima_forecaster.order is not None
        assert len(arima_forecaster.order) == 3
    
    def test_fit_with_invalid_data(self, arima_forecaster):
        """Test fitting with invalid data raises appropriate errors."""
        # Test with non-Series data
        with pytest.raises(ValueError, match="Data must be a pandas Series"):
            arima_forecaster.fit([1, 2, 3, 4, 5])
        
        # Test with non-datetime index
        invalid_series = pd.Series([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Data must have a DatetimeIndex"):
            arima_forecaster.fit(invalid_series)
        
        # Test with insufficient data
        dates = pd.date_range(start='2020-01-01', periods=5, freq='D')
        short_series = pd.Series([1, 2, 3, 4, 5], index=dates)
        with pytest.raises(ValueError, match="Insufficient data"):
            arima_forecaster.fit(short_series)
    
    def test_forecast_before_fitting(self, arima_forecaster):
        """Test that forecasting before fitting raises error."""
        with pytest.raises(ValueError, match="model must be fitted"):
            arima_forecaster.forecast(10)
    
    def test_forecast_after_fitting(self, arima_forecaster, sample_data):
        """Test forecasting after fitting model."""
        arima_forecaster.fit(sample_data)
        
        periods = 30
        result = arima_forecaster.forecast(periods)
        
        assert isinstance(result, ForecastResult)
        assert len(result.predictions) == periods
        assert len(result.confidence_intervals) == periods
        assert 'lower' in result.confidence_intervals.columns
        assert 'upper' in result.confidence_intervals.columns
        assert result.forecast_horizon == periods
        assert result.model_type.startswith("ARIMA")
        assert isinstance(result.model_metrics, dict)
    
    def test_forecast_with_invalid_periods(self, arima_forecaster, sample_data):
        """Test forecasting with invalid periods."""
        arima_forecaster.fit(sample_data)
        
        with pytest.raises(ValueError, match="Number of periods must be positive"):
            arima_forecaster.forecast(0)
        
        with pytest.raises(ValueError, match="Number of periods must be positive"):
            arima_forecaster.forecast(-5)
    
    def test_predict_method(self, arima_forecaster, sample_data):
        """Test the predict method returns point forecasts."""
        arima_forecaster.fit(sample_data)
        
        periods = 10
        predictions = arima_forecaster.predict(periods)
        
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == periods
        assert predictions.name == 'forecast'
    
    def test_get_diagnostics(self, arima_forecaster, sample_data):
        """Test model diagnostics generation."""
        arima_forecaster.fit(sample_data)
        
        diagnostics = arima_forecaster.get_diagnostics()
        
        assert isinstance(diagnostics, ModelDiagnostics)
        assert len(diagnostics.residuals) > 0
        assert len(diagnostics.fitted_values) > 0
        assert diagnostics.aic is not None
        assert diagnostics.bic is not None
    
    def test_evaluate_method(self, arima_forecaster, sample_data):
        """Test model evaluation on test data."""
        # Split data for training and testing
        train_data = sample_data[:-30]
        test_data = sample_data[-30:]
        
        arima_forecaster.fit(train_data)
        metrics = arima_forecaster.evaluate(test_data)
        
        assert isinstance(metrics, dict)
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'MAPE' in metrics
        assert 'n_predictions' in metrics
        assert metrics['MAE'] >= 0
        assert metrics['RMSE'] >= 0
        assert metrics['MAPE'] >= 0
    
    def test_manual_order_specification(self, sample_data):
        """Test manual ARIMA order specification."""
        forecaster = ARIMAForecaster(name="ManualARIMA", auto_optimize=False)
        
        # Fit with manual order
        forecaster.fit(sample_data, order=(2, 1, 1))
        
        assert forecaster.order == (2, 1, 1)
        assert forecaster.is_fitted is True
    
    def test_seasonal_arima(self, sample_data):
        """Test seasonal ARIMA functionality."""
        forecaster = ARIMAForecaster(
            name="SeasonalARIMA",
            seasonal=True,
            m=7,  # Weekly seasonality
            auto_optimize=True
        )
        
        forecaster.fit(sample_data)
        
        assert forecaster.is_fitted is True
        assert forecaster.seasonal_order is not None
        assert len(forecaster.seasonal_order) == 4
    
    def test_stationarity_handling(self, arima_forecaster):
        """Test stationarity detection and handling."""
        # Create non-stationary data (random walk)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        values = np.cumsum(np.random.normal(0, 1, len(dates))) + 100
        non_stationary_data = pd.Series(values, index=dates)
        
        arima_forecaster.fit(non_stationary_data)
        
        assert arima_forecaster.is_fitted is True
        assert arima_forecaster.order[1] > 0  # Should have differencing
    
    def test_repr_method(self, arima_forecaster, sample_data):
        """Test string representation of forecaster."""
        # Before fitting
        repr_before = repr(arima_forecaster)
        assert "not fitted" in repr_before
        assert "TestARIMA" in repr_before
        
        # After fitting
        arima_forecaster.fit(sample_data)
        repr_after = repr(arima_forecaster)
        assert "fitted" in repr_after
        assert "TestARIMA" in repr_after