"""
Tests for model evaluation and comparison system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from portfolio_forecasting.src.models.model_evaluator import ModelEvaluator, ModelPerformance, ComparisonResult
from portfolio_forecasting.src.models.base_forecaster import BaseForecastor, ForecastResult


class MockForecaster(BaseForecastor):
    """Mock forecaster for testing purposes."""
    
    def __init__(self, name: str, prediction_values: list):
        super().__init__(name)
        self.prediction_values = prediction_values
        self.fit_called = False
        
    def fit(self, data: pd.Series, **kwargs):
        self.fit_called = True
        self.is_fitted = True
        self.training_data = data
        return self
        
    def forecast(self, periods: int, confidence_level: float = 0.95):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        # Create mock predictions
        last_date = self.training_data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        # Use provided prediction values or generate simple ones
        if len(self.prediction_values) >= periods:
            predictions = self.prediction_values[:periods]
        else:
            predictions = np.random.normal(100, 10, periods)
            
        predictions_series = pd.Series(predictions, index=forecast_dates, name='forecast')
        
        # Simple confidence intervals
        confidence_df = pd.DataFrame({
            'lower': predictions_series - 5,
            'upper': predictions_series + 5
        }, index=forecast_dates)
        
        return ForecastResult(
            predictions=predictions_series,
            confidence_intervals=confidence_df,
            model_metrics={'test_metric': 1.0},
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


class TestModelEvaluator:
    """Test suite for ModelEvaluator."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data spanning train and test periods."""
        dates = pd.date_range(start='2015-01-01', end='2025-12-31', freq='D')
        # Create trend + noise
        trend = np.linspace(100, 200, len(dates))
        noise = np.random.normal(0, 5, len(dates))
        values = trend + noise
        return pd.Series(values, index=dates, name='test_series')
    
    @pytest.fixture
    def evaluator(self):
        """Create ModelEvaluator instance."""
        return ModelEvaluator(
            train_start='2015-01-01',
            train_end='2023-12-31',
            test_start='2024-01-01',
            test_end='2025-12-31',
            primary_metric='RMSE'
        )
    
    @pytest.fixture
    def mock_models(self):
        """Create mock forecasting models."""
        return [
            MockForecaster("Model_A", [105, 106, 107, 108, 109]),
            MockForecaster("Model_B", [103, 104, 105, 106, 107]),
            MockForecaster("Model_C", [107, 108, 109, 110, 111])
        ]
    
    def test_initialization(self, evaluator):
        """Test ModelEvaluator initialization."""
        assert evaluator.train_start == pd.to_datetime('2015-01-01')
        assert evaluator.train_end == pd.to_datetime('2023-12-31')
        assert evaluator.test_start == pd.to_datetime('2024-01-01')
        assert evaluator.test_end == pd.to_datetime('2025-12-31')
        assert evaluator.primary_metric == 'RMSE'
    
    def test_initialization_invalid_periods(self):
        """Test initialization with invalid date periods."""
        with pytest.raises(ValueError, match="Training period must end before test period starts"):
            ModelEvaluator(
                train_start='2020-01-01',
                train_end='2024-12-31',
                test_start='2024-01-01',
                test_end='2025-12-31'
            )
    
    def test_split_data(self, evaluator, sample_data):
        """Test chronological data splitting."""
        train_data, test_data = evaluator.split_data(sample_data)
        
        assert isinstance(train_data, pd.Series)
        assert isinstance(test_data, pd.Series)
        assert len(train_data) > 0
        assert len(test_data) > 0
        assert train_data.index.max() <= evaluator.train_end
        assert test_data.index.min() >= evaluator.test_start
    
    def test_split_data_invalid_index(self, evaluator):
        """Test data splitting with invalid index type."""
        invalid_data = pd.Series([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError, match="Data must have a DatetimeIndex"):
            evaluator.split_data(invalid_data)
    
    def test_split_data_no_training_data(self):
        """Test data splitting when no training data is available."""
        evaluator = ModelEvaluator(
            train_start='2030-01-01',
            train_end='2030-12-31',
            test_start='2031-01-01',
            test_end='2031-12-31'
        )
        
        dates = pd.date_range(start='2020-01-01', end='2025-12-31', freq='D')
        data = pd.Series(np.random.randn(len(dates)), index=dates)
        
        with pytest.raises(ValueError, match="No training data found"):
            evaluator.split_data(data)
    
    def test_evaluate_model(self, evaluator, sample_data, mock_models):
        """Test single model evaluation."""
        model = mock_models[0]
        
        performance = evaluator.evaluate_model(model, sample_data, forecast_periods=30)
        
        assert isinstance(performance, ModelPerformance)
        assert performance.model_name == "Model_A"
        assert model.fit_called is True
        assert 'MAE' in performance.metrics
        assert 'RMSE' in performance.metrics
        assert 'MAPE' in performance.metrics
        assert performance.training_time >= 0
        assert performance.prediction_time >= 0
        assert len(performance.predictions) == 30
    
    def test_compare_models(self, evaluator, sample_data, mock_models):
        """Test multiple model comparison."""
        comparison_result = evaluator.compare_models(mock_models, sample_data, forecast_periods=10)
        
        assert isinstance(comparison_result, ComparisonResult)
        assert len(comparison_result.performances) == 3
        assert comparison_result.best_model in ["Model_A", "Model_B", "Model_C"]
        assert len(comparison_result.ranking) == 3
        assert isinstance(comparison_result.summary_metrics, pd.DataFrame)
        assert len(comparison_result.summary_metrics) == 3
    
    def test_compare_models_with_failures(self, evaluator, sample_data):
        """Test model comparison when some models fail."""
        # Create models where one will fail
        failing_model = Mock(spec=BaseForecastor)
        failing_model.name = "FailingModel"
        failing_model.fit.side_effect = Exception("Model fitting failed")
        
        working_model = mock_models[0]
        models = [failing_model, working_model]
        
        comparison_result = evaluator.compare_models(models, sample_data)
        
        # Should only have results for the working model
        assert len(comparison_result.performances) == 1
        assert comparison_result.performances[0].model_name == working_model.name
    
    def test_compare_models_all_fail(self, evaluator, sample_data):
        """Test model comparison when all models fail."""
        failing_models = [Mock(spec=BaseForecastor) for _ in range(2)]
        for i, model in enumerate(failing_models):
            model.name = f"FailingModel_{i}"
            model.fit.side_effect = Exception("Model fitting failed")
        
        with pytest.raises(ValueError, match="No models could be evaluated successfully"):
            evaluator.compare_models(failing_models, sample_data)
    
    def test_get_best_model(self, evaluator, sample_data, mock_models):
        """Test best model selection."""
        best_model = evaluator.get_best_model(mock_models, sample_data)
        
        assert isinstance(best_model, BaseForecastor)
        assert best_model.name in ["Model_A", "Model_B", "Model_C"]
    
    def test_visualize_comparison(self, evaluator, sample_data, mock_models):
        """Test visualization generation."""
        comparison_result = evaluator.compare_models(mock_models, sample_data, forecast_periods=5)
        
        plots = evaluator.visualize_comparison(comparison_result)
        
        assert isinstance(plots, dict)
        expected_plots = ['performance_comparison', 'predictions_comparison', 
                         'model_ranking', 'timing_comparison']
        
        for plot_name in expected_plots:
            assert plot_name in plots
    
    @patch('matplotlib.pyplot.savefig')
    @patch('os.makedirs')
    def test_visualize_comparison_with_save(self, mock_makedirs, mock_savefig, 
                                          evaluator, sample_data, mock_models):
        """Test visualization with saving to file."""
        comparison_result = evaluator.compare_models(mock_models, sample_data, forecast_periods=5)
        
        save_path = "/test/path"
        plots = evaluator.visualize_comparison(comparison_result, save_path=save_path)
        
        # Check that directory creation was called
        mock_makedirs.assert_called_once_with(save_path, exist_ok=True)
        
        # Check that savefig was called for each plot
        assert mock_savefig.call_count == len(plots)
    
    def test_calculate_comprehensive_metrics(self, evaluator):
        """Test comprehensive metrics calculation."""
        actual = pd.Series([100, 101, 102, 103, 104])
        predicted = pd.Series([99, 102, 101, 104, 103])
        
        metrics = evaluator._calculate_comprehensive_metrics(actual, predicted)
        
        expected_metrics = ['MAE', 'RMSE', 'MAPE', 'MSE', 'ME', 'SMAPE', 'Correlation', 'n_predictions']
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        assert metrics['n_predictions'] == 5
        assert metrics['MAE'] >= 0
        assert metrics['RMSE'] >= 0
        assert metrics['MSE'] >= 0
    
    def test_calculate_metrics_with_zeros(self, evaluator):
        """Test metrics calculation with zero values (MAPE edge case)."""
        actual = pd.Series([0, 1, 2, 0, 4])
        predicted = pd.Series([1, 1, 2, 1, 4])
        
        metrics = evaluator._calculate_comprehensive_metrics(actual, predicted)
        
        # Should handle division by zero gracefully
        assert 'MAPE' in metrics
        assert np.isfinite(metrics['MAPE']) or np.isinf(metrics['MAPE'])
    
    def test_different_primary_metrics(self, sample_data, mock_models):
        """Test evaluator with different primary metrics."""
        metrics_to_test = ['MAE', 'RMSE', 'MAPE']
        
        for metric in metrics_to_test:
            evaluator = ModelEvaluator(primary_metric=metric)
            comparison_result = evaluator.compare_models(mock_models, sample_data, forecast_periods=5)
            
            # Check that ranking is based on the specified metric
            best_performance = next(p for p in comparison_result.performances 
                                  if p.model_name == comparison_result.best_model)
            
            # Verify the best model has the lowest score for the primary metric
            all_scores = [p.metrics[metric] for p in comparison_result.performances]
            assert best_performance.metrics[metric] == min(all_scores)