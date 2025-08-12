"""
Tests for forecast generation system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from portfolio_forecasting.src.forecasting.forecast_generator import (
    ForecastGenerator, ForecastConfig, ForecastOutput
)
from portfolio_forecasting.src.models.base_forecaster import BaseForecastor, ForecastResult


class MockForecaster(BaseForecastor):
    """Mock forecaster for testing."""
    
    def __init__(self, name: str, performance_score: float = 1.0):
        super().__init__(name)
        self.performance_score = performance_score
        self.fit_called = False
        
    def fit(self, data: pd.Series, **kwargs):
        self.fit_called = True
        self.is_fitted = True
        self.training_data = data
        return self
        
    def forecast(self, periods: int, confidence_level: float = 0.95):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        # Create mock forecast
        last_date = self.training_data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        # Generate simple trend + noise
        base_value = self.training_data.iloc[-1]
        trend = np.linspace(0, periods * 0.1, periods)
        noise = np.random.normal(0, 1, periods)
        predictions = base_value + trend + noise
        
        predictions_series = pd.Series(predictions, index=forecast_dates, name='forecast')
        
        # Simple confidence intervals
        margin = 2.0 * (1 - confidence_level + 0.5)
        confidence_df = pd.DataFrame({
            'lower': predictions_series - margin,
            'upper': predictions_series + margin
        }, index=forecast_dates)
        
        return ForecastResult(
            predictions=predictions_series,
            confidence_intervals=confidence_df,
            model_metrics={'RMSE': self.performance_score, 'MAE': self.performance_score * 0.8},
            forecast_horizon=periods,
            model_type=self.name,
            training_end_date=last_date,
            forecast_dates=forecast_dates
        )
    
    def get_diagnostics(self):
        from portfolio_forecasting.src.models.base_forecaster import ModelDiagnostics
        return ModelDiagnostics(
            residuals=pd.Series([0.1, -0.2, 0.3]),
            fitted_values=pd.Series([100, 101, 102])
        )


class TestForecastConfig:
    """Test suite for ForecastConfig."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = ForecastConfig()
        
        assert config.forecast_horizon_months == 12
        assert config.confidence_levels == [0.80, 0.95]
        assert config.model_selection_metric == "RMSE"
        assert config.train_test_split_date == "2024-01-01"
        assert config.visualization_style == "seaborn"
        assert config.include_model_comparison is True
    
    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        config = ForecastConfig(
            forecast_horizon_months=6,
            confidence_levels=[0.90],
            model_selection_metric="MAE",
            visualization_style="matplotlib"
        )
        
        assert config.forecast_horizon_months == 6
        assert config.confidence_levels == [0.90]
        assert config.model_selection_metric == "MAE"
        assert config.visualization_style == "matplotlib"


class TestForecastGenerator:
    """Test suite for ForecastGenerator."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        trend = np.linspace(100, 150, len(dates))
        noise = np.random.normal(0, 2, len(dates))
        values = trend + noise
        return pd.Series(values, index=dates, name='test_series')
    
    @pytest.fixture
    def forecast_config(self):
        """Create test forecast configuration."""
        return ForecastConfig(
            forecast_horizon_months=6,
            confidence_levels=[0.80, 0.95],
            model_selection_metric="RMSE"
        )
    
    @pytest.fixture
    def forecast_generator(self, forecast_config):
        """Create ForecastGenerator instance."""
        return ForecastGenerator(forecast_config)
    
    @pytest.fixture
    def mock_models(self):
        """Create mock forecasting models."""
        return [
            MockForecaster("MockARIMA", performance_score=1.5),
            MockForecaster("MockLSTM", performance_score=1.2),  # Better performance
            MockForecaster("MockBaseline", performance_score=2.0)
        ]
    
    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        generator = ForecastGenerator()
        
        assert generator.config.forecast_horizon_months == 12
        assert generator.model_evaluator is None
        assert generator.best_model is None
        assert len(generator.available_models) == 0
    
    def test_initialization_custom_config(self, forecast_config):
        """Test initialization with custom configuration."""
        generator = ForecastGenerator(forecast_config)
        
        assert generator.config.forecast_horizon_months == 6
        assert generator.config.confidence_levels == [0.80, 0.95]
    
    def test_setup_models_arima_only(self, forecast_generator):
        """Test setting up ARIMA model only."""
        forecast_generator.setup_models(include_arima=True, include_lstm=False)
        
        assert len(forecast_generator.available_models) == 1
        assert forecast_generator.available_models[0].name == "ARIMA_Optimized"
        assert forecast_generator.model_evaluator is not None
    
    def test_setup_models_lstm_only(self, forecast_generator):
        """Test setting up LSTM model only."""
        forecast_generator.setup_models(include_arima=False, include_lstm=True)
        
        assert len(forecast_generator.available_models) == 1
        assert forecast_generator.available_models[0].name == "LSTM_Optimized"
    
    def test_setup_models_both(self, forecast_generator):
        """Test setting up both models."""
        forecast_generator.setup_models(include_arima=True, include_lstm=True)
        
        assert len(forecast_generator.available_models) == 2
        model_names = [model.name for model in forecast_generator.available_models]
        assert "ARIMA_Optimized" in model_names
        assert "LSTM_Optimized" in model_names
    
    def test_setup_models_with_custom_params(self, forecast_generator):
        """Test setting up models with custom parameters."""
        arima_params = {'max_p': 3, 'seasonal': True}
        lstm_params = {'sequence_length': 30, 'epochs': 50}
        
        forecast_generator.setup_models(
            include_arima=True,
            include_lstm=True,
            arima_params=arima_params,
            lstm_params=lstm_params
        )
        
        assert len(forecast_generator.available_models) == 2
    
    @patch('portfolio_forecasting.src.forecasting.forecast_generator.ModelEvaluator')
    def test_select_best_model(self, mock_evaluator_class, forecast_generator, sample_data, mock_models):
        """Test best model selection."""
        # Setup mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.get_best_model.return_value = mock_models[1]  # Return MockLSTM
        mock_evaluator_class.return_value = mock_evaluator
        
        # Setup models
        forecast_generator.available_models = mock_models
        forecast_generator.model_evaluator = mock_evaluator
        
        # Select best model
        best_model = forecast_generator.select_best_model(sample_data)
        
        assert best_model == mock_models[1]
        assert forecast_generator.best_model == mock_models[1]
        mock_evaluator.get_best_model.assert_called_once()
    
    def test_select_best_model_no_models(self, forecast_generator, sample_data):
        """Test model selection with no available models."""
        with pytest.raises(ValueError, match="No models available"):
            forecast_generator.select_best_model(sample_data)
    
    def test_select_best_model_no_evaluator(self, forecast_generator, sample_data, mock_models):
        """Test model selection with no evaluator."""
        forecast_generator.available_models = mock_models
        
        with pytest.raises(ValueError, match="Model evaluator not initialized"):
            forecast_generator.select_best_model(sample_data)
    
    def test_generate_forecast_auto_select(self, forecast_generator, sample_data, mock_models):
        """Test forecast generation with automatic model selection."""
        # Setup
        forecast_generator.available_models = mock_models
        forecast_generator.model_evaluator = Mock()
        forecast_generator.model_evaluator.get_best_model.return_value = mock_models[1]
        
        # Generate forecast
        forecast_output = forecast_generator.generate_forecast(sample_data, auto_select_model=True)
        
        assert isinstance(forecast_output, ForecastOutput)
        assert forecast_output.model_used == "MockLSTM"
        assert len(forecast_output.predictions) == 6 * 30  # 6 months * 30 days
        assert len(forecast_output.confidence_intervals) == 2  # Two confidence levels
        assert 0.80 in forecast_output.confidence_intervals
        assert 0.95 in forecast_output.confidence_intervals
    
    def test_generate_forecast_with_selected_model(self, forecast_generator, sample_data, mock_models):
        """Test forecast generation with pre-selected model."""
        # Pre-select model
        forecast_generator.best_model = mock_models[0]
        
        # Generate forecast
        forecast_output = forecast_generator.generate_forecast(sample_data, auto_select_model=False)
        
        assert isinstance(forecast_output, ForecastOutput)
        assert forecast_output.model_used == "MockARIMA"
        assert isinstance(forecast_output.uncertainty_metrics, dict)
        assert isinstance(forecast_output.forecast_summary, dict)
    
    def test_generate_forecast_no_model(self, forecast_generator, sample_data):
        """Test forecast generation with no model selected."""
        with pytest.raises(ValueError, match="No model selected"):
            forecast_generator.generate_forecast(sample_data, auto_select_model=False)
    
    def test_visualize_forecast(self, forecast_generator, sample_data, mock_models):
        """Test forecast visualization."""
        # Setup
        forecast_generator.best_model = mock_models[0]
        forecast_output = forecast_generator.generate_forecast(sample_data)
        
        # Create visualization
        fig = forecast_generator.visualize_forecast(sample_data, forecast_output)
        
        assert fig is not None
        assert len(fig.axes) == 2  # Two subplots
    
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_forecast_with_save(self, mock_savefig, forecast_generator, sample_data, mock_models):
        """Test forecast visualization with saving."""
        # Setup
        forecast_generator.best_model = mock_models[0]
        forecast_output = forecast_generator.generate_forecast(sample_data)
        
        # Create visualization with save
        save_path = "/test/path/forecast.png"
        fig = forecast_generator.visualize_forecast(sample_data, forecast_output, save_path=save_path)
        
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
    
    def test_calculate_uncertainty_metrics(self, forecast_generator, sample_data, mock_models):
        """Test uncertainty metrics calculation."""
        # Setup
        forecast_generator.best_model = mock_models[0]
        forecast_output = forecast_generator.generate_forecast(sample_data)
        
        # Check uncertainty metrics
        uncertainty_metrics = forecast_output.uncertainty_metrics
        
        assert 'avg_ci_width_80' in uncertainty_metrics
        assert 'avg_ci_width_95' in uncertainty_metrics
        assert 'relative_ci_width_80' in uncertainty_metrics
        assert 'relative_ci_width_95' in uncertainty_metrics
        assert 'forecast_volatility' in uncertainty_metrics
        assert 'trend_consistency_pct' in uncertainty_metrics
    
    def test_generate_forecast_summary(self, forecast_generator, sample_data, mock_models):
        """Test forecast summary generation."""
        # Setup
        forecast_generator.best_model = mock_models[0]
        forecast_output = forecast_generator.generate_forecast(sample_data)
        
        # Check forecast summary
        summary = forecast_output.forecast_summary
        
        expected_keys = [
            'forecast_start', 'forecast_end', 'forecast_periods',
            'last_historical_value', 'first_forecast_value', 'last_forecast_value',
            'immediate_change_pct', 'total_forecast_change_pct', 'trend_direction',
            'historical_volatility', 'forecast_volatility', 'volatility_change',
            'confidence_assessment'
        ]
        
        for key in expected_keys:
            assert key in summary
    
    def test_forecast_output_structure(self, forecast_generator, sample_data, mock_models):
        """Test the structure of forecast output."""
        # Setup
        forecast_generator.best_model = mock_models[0]
        forecast_output = forecast_generator.generate_forecast(sample_data)
        
        # Verify structure
        assert isinstance(forecast_output.predictions, pd.Series)
        assert isinstance(forecast_output.confidence_intervals, dict)
        assert isinstance(forecast_output.model_used, str)
        assert isinstance(forecast_output.model_metrics, dict)
        assert isinstance(forecast_output.forecast_config, ForecastConfig)
        assert isinstance(forecast_output.generation_timestamp, pd.Timestamp)
        assert isinstance(forecast_output.uncertainty_metrics, dict)
        assert isinstance(forecast_output.forecast_summary, dict)
    
    def test_different_forecast_horizons(self, sample_data, mock_models):
        """Test different forecast horizons."""
        horizons = [3, 6, 12, 18]
        
        for horizon in horizons:
            config = ForecastConfig(forecast_horizon_months=horizon)
            generator = ForecastGenerator(config)
            generator.best_model = mock_models[0]
            
            forecast_output = generator.generate_forecast(sample_data)
            
            expected_periods = horizon * 30
            assert len(forecast_output.predictions) == expected_periods
            assert forecast_output.forecast_summary['forecast_periods'] == expected_periods
    
    def test_different_confidence_levels(self, sample_data, mock_models):
        """Test different confidence levels."""
        confidence_levels = [[0.90], [0.80, 0.95], [0.68, 0.95, 0.99]]
        
        for levels in confidence_levels:
            config = ForecastConfig(confidence_levels=levels)
            generator = ForecastGenerator(config)
            generator.best_model = mock_models[0]
            
            forecast_output = generator.generate_forecast(sample_data)
            
            assert len(forecast_output.confidence_intervals) == len(levels)
            for level in levels:
                assert level in forecast_output.confidence_intervals