"""
Forecast generation system with confidence intervals and uncertainty quantification.

This module provides a comprehensive forecasting service that generates 6-12 month
forecasts using the best performing model, with confidence intervals and visualization.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from ..models import BaseForecastor, ModelEvaluator, ARIMAForecaster, LSTMForecaster
from ..models.base_forecaster import ForecastResult

logger = logging.getLogger(__name__)


@dataclass
class ForecastConfig:
    """Configuration for forecast generation."""
    
    forecast_horizon_months: int = 12
    confidence_levels: List[float] = None
    model_selection_metric: str = "RMSE"
    train_test_split_date: str = "2024-01-01"
    visualization_style: str = "seaborn"
    include_model_comparison: bool = True
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.80, 0.95]


@dataclass
class ForecastOutput:
    """Container for comprehensive forecast results."""
    
    predictions: pd.Series
    confidence_intervals: Dict[float, pd.DataFrame]  # confidence_level -> DataFrame with lower/upper
    model_used: str
    model_metrics: Dict[str, float]
    forecast_config: ForecastConfig
    generation_timestamp: pd.Timestamp
    uncertainty_metrics: Dict[str, float]
    forecast_summary: Dict[str, Union[str, float]]


class ForecastGenerator:
    """
    Comprehensive forecast generation system.
    
    This class provides methods to generate 6-12 month forecasts using the best
    performing model, with confidence intervals, uncertainty quantification,
    and comprehensive visualization capabilities.
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        """
        Initialize forecast generator.
        
        Args:
            config: Forecast configuration settings
        """
        self.config = config or ForecastConfig()
        self.model_evaluator = None
        self.best_model = None
        self.available_models = []
        
        # Set up plotting style
        if self.config.visualization_style == "seaborn":
            sns.set_style("whitegrid")
            plt.style.use("seaborn-v0_8")
        
        logger.info(f"ForecastGenerator initialized with {self.config.forecast_horizon_months}-month horizon")
    
    def setup_models(self, 
                    include_arima: bool = True,
                    include_lstm: bool = True,
                    arima_params: Optional[Dict] = None,
                    lstm_params: Optional[Dict] = None) -> None:
        """
        Set up forecasting models for comparison and selection.
        
        Args:
            include_arima: Whether to include ARIMA model
            include_lstm: Whether to include LSTM model
            arima_params: Parameters for ARIMA model
            lstm_params: Parameters for LSTM model
        """
        self.available_models = []
        
        if include_arima:
            arima_config = arima_params or {}
            arima_model = ARIMAForecaster(
                name="ARIMA_Optimized",
                auto_optimize=arima_config.get('auto_optimize', True),
                max_p=arima_config.get('max_p', 5),
                max_d=arima_config.get('max_d', 2),
                max_q=arima_config.get('max_q', 5),
                seasonal=arima_config.get('seasonal', False)
            )
            self.available_models.append(arima_model)
            logger.info("Added ARIMA model to available models")
        
        if include_lstm:
            lstm_config = lstm_params or {}
            lstm_model = LSTMForecaster(
                name="LSTM_Optimized",
                sequence_length=lstm_config.get('sequence_length', 60),
                lstm_units=lstm_config.get('lstm_units', [50, 50]),
                dropout_rate=lstm_config.get('dropout_rate', 0.2),
                epochs=lstm_config.get('epochs', 100),
                batch_size=lstm_config.get('batch_size', 32)
            )
            self.available_models.append(lstm_model)
            logger.info("Added LSTM model to available models")
        
        # Set up model evaluator
        self.model_evaluator = ModelEvaluator(
            train_start="2015-01-01",
            train_end="2023-12-31",
            test_start=self.config.train_test_split_date,
            test_end="2025-12-31",
            primary_metric=self.config.model_selection_metric
        )
        
        logger.info(f"Set up {len(self.available_models)} models for evaluation")
    
    def select_best_model(self, data: pd.Series) -> BaseForecastor:
        """
        Select the best performing model based on evaluation metrics.
        
        Args:
            data: Historical time series data
            
        Returns:
            Best performing forecasting model
        """
        if not self.available_models:
            raise ValueError("No models available. Call setup_models() first.")
        
        if not self.model_evaluator:
            raise ValueError("Model evaluator not initialized. Call setup_models() first.")
        
        logger.info("Starting model selection process")
        
        # Compare models and select the best one
        self.best_model = self.model_evaluator.get_best_model(
            self.available_models, 
            data,
            forecast_periods=30  # Use 30 days for evaluation
        )
        
        logger.info(f"Selected best model: {self.best_model.name}")
        return self.best_model
    
    def generate_forecast(self, 
                         data: pd.Series,
                         auto_select_model: bool = True) -> ForecastOutput:
        """
        Generate comprehensive forecast with confidence intervals.
        
        Args:
            data: Historical time series data
            auto_select_model: Whether to automatically select best model
            
        Returns:
            ForecastOutput containing predictions and analysis
        """
        logger.info(f"Generating {self.config.forecast_horizon_months}-month forecast")
        
        # Model selection
        if auto_select_model or self.best_model is None:
            self.select_best_model(data)
        
        if self.best_model is None:
            raise ValueError("No model selected. Call select_best_model() or set auto_select_model=True")
        
        # Calculate forecast periods (approximate days in months)
        forecast_periods = self.config.forecast_horizon_months * 30
        
        # Generate forecasts for different confidence levels
        confidence_intervals = {}
        base_forecast = None
        
        for confidence_level in self.config.confidence_levels:
            forecast_result = self.best_model.forecast(
                periods=forecast_periods,
                confidence_level=confidence_level
            )
            
            if base_forecast is None:
                base_forecast = forecast_result
            
            confidence_intervals[confidence_level] = forecast_result.confidence_intervals
        
        # Calculate uncertainty metrics
        uncertainty_metrics = self._calculate_uncertainty_metrics(
            base_forecast, confidence_intervals
        )
        
        # Generate forecast summary
        forecast_summary = self._generate_forecast_summary(
            data, base_forecast, uncertainty_metrics
        )
        
        # Create output
        output = ForecastOutput(
            predictions=base_forecast.predictions,
            confidence_intervals=confidence_intervals,
            model_used=self.best_model.name,
            model_metrics=base_forecast.model_metrics,
            forecast_config=self.config,
            generation_timestamp=pd.Timestamp.now(),
            uncertainty_metrics=uncertainty_metrics,
            forecast_summary=forecast_summary
        )
        
        logger.info(f"Forecast generated successfully using {self.best_model.name}")
        return output
    
    def visualize_forecast(self, 
                          data: pd.Series,
                          forecast_output: ForecastOutput,
                          save_path: Optional[str] = None,
                          show_historical_periods: int = 365) -> plt.Figure:
        """
        Create comprehensive forecast visualization.
        
        Args:
            data: Historical time series data
            forecast_output: Forecast results to visualize
            save_path: Optional path to save the plot
            show_historical_periods: Number of historical days to show
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Prepare historical data for visualization
        historical_data = data.tail(show_historical_periods)
        
        # Plot 1: Main forecast with confidence intervals
        ax1.plot(historical_data.index, historical_data.values, 
                label='Historical Data', linewidth=2, color='black', alpha=0.8)
        
        # Plot predictions
        ax1.plot(forecast_output.predictions.index, forecast_output.predictions.values,
                label=f'Forecast ({forecast_output.model_used})', 
                linewidth=2, color='red', alpha=0.9)
        
        # Plot confidence intervals
        colors = ['lightblue', 'lightcoral']
        alphas = [0.3, 0.2]
        
        for i, (confidence_level, ci_data) in enumerate(forecast_output.confidence_intervals.items()):
            color = colors[i % len(colors)]
            alpha = alphas[i % len(alphas)]
            
            ax1.fill_between(
                ci_data.index,
                ci_data['lower'],
                ci_data['upper'],
                alpha=alpha,
                color=color,
                label=f'{int(confidence_level*100)}% Confidence Interval'
            )
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price/Value')
        ax1.set_title(f'{self.config.forecast_horizon_months}-Month Forecast with Confidence Intervals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Forecast analysis
        forecast_values = forecast_output.predictions.values
        
        # Calculate rolling statistics for forecast
        forecast_df = pd.DataFrame({
            'Forecast': forecast_values,
            'Rolling_Mean_30': pd.Series(forecast_values).rolling(30).mean(),
            'Rolling_Std_30': pd.Series(forecast_values).rolling(30).std()
        }, index=forecast_output.predictions.index)
        
        ax2.plot(forecast_df.index, forecast_df['Forecast'], 
                label='Forecast', linewidth=2, color='red')
        ax2.plot(forecast_df.index, forecast_df['Rolling_Mean_30'], 
                label='30-Day Rolling Mean', linewidth=1, color='blue', alpha=0.7)
        
        # Add uncertainty bands
        ax2.fill_between(
            forecast_df.index,
            forecast_df['Rolling_Mean_30'] - forecast_df['Rolling_Std_30'],
            forecast_df['Rolling_Mean_30'] + forecast_df['Rolling_Std_30'],
            alpha=0.2, color='blue', label='±1 Std Dev'
        )
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price/Value')
        ax2.set_title('Forecast Trend Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Add summary text
        summary_text = self._create_summary_text(forecast_output)
        fig.text(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast visualization saved to {save_path}")
        
        return fig
    
    def _calculate_uncertainty_metrics(self, 
                                     forecast_result: ForecastResult,
                                     confidence_intervals: Dict[float, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate uncertainty and reliability metrics.
        
        Args:
            forecast_result: Base forecast result
            confidence_intervals: Confidence intervals for different levels
            
        Returns:
            Dictionary of uncertainty metrics
        """
        metrics = {}
        
        # Calculate average confidence interval width
        for confidence_level, ci_data in confidence_intervals.items():
            ci_width = (ci_data['upper'] - ci_data['lower']).mean()
            metrics[f'avg_ci_width_{int(confidence_level*100)}'] = ci_width
            
            # Calculate relative confidence interval width
            avg_prediction = forecast_result.predictions.mean()
            relative_width = (ci_width / avg_prediction) * 100 if avg_prediction != 0 else np.inf
            metrics[f'relative_ci_width_{int(confidence_level*100)}'] = relative_width
        
        # Calculate forecast volatility
        forecast_volatility = forecast_result.predictions.std()
        metrics['forecast_volatility'] = forecast_volatility
        
        # Calculate trend consistency
        forecast_diff = forecast_result.predictions.diff().dropna()
        trend_consistency = (forecast_diff > 0).sum() / len(forecast_diff) * 100
        metrics['trend_consistency_pct'] = trend_consistency
        
        return metrics
    
    def _generate_forecast_summary(self, 
                                 historical_data: pd.Series,
                                 forecast_result: ForecastResult,
                                 uncertainty_metrics: Dict[str, float]) -> Dict[str, Union[str, float]]:
        """
        Generate comprehensive forecast summary.
        
        Args:
            historical_data: Historical time series data
            forecast_result: Forecast results
            uncertainty_metrics: Calculated uncertainty metrics
            
        Returns:
            Dictionary containing forecast summary
        """
        summary = {}
        
        # Basic forecast statistics
        summary['forecast_start'] = forecast_result.predictions.index[0].strftime('%Y-%m-%d')
        summary['forecast_end'] = forecast_result.predictions.index[-1].strftime('%Y-%m-%d')
        summary['forecast_periods'] = len(forecast_result.predictions)
        
        # Price/value analysis
        last_historical_value = historical_data.iloc[-1]
        first_forecast_value = forecast_result.predictions.iloc[0]
        last_forecast_value = forecast_result.predictions.iloc[-1]
        
        summary['last_historical_value'] = last_historical_value
        summary['first_forecast_value'] = first_forecast_value
        summary['last_forecast_value'] = last_forecast_value
        
        # Calculate changes
        immediate_change = ((first_forecast_value - last_historical_value) / last_historical_value) * 100
        total_forecast_change = ((last_forecast_value - first_forecast_value) / first_forecast_value) * 100
        
        summary['immediate_change_pct'] = immediate_change
        summary['total_forecast_change_pct'] = total_forecast_change
        
        # Trend analysis
        if total_forecast_change > 5:
            summary['trend_direction'] = "Upward"
        elif total_forecast_change < -5:
            summary['trend_direction'] = "Downward"
        else:
            summary['trend_direction'] = "Stable"
        
        # Volatility comparison
        historical_volatility = historical_data.tail(252).std()  # Last year volatility
        forecast_volatility = uncertainty_metrics.get('forecast_volatility', 0)
        
        summary['historical_volatility'] = historical_volatility
        summary['forecast_volatility'] = forecast_volatility
        summary['volatility_change'] = forecast_volatility - historical_volatility
        
        # Confidence assessment
        avg_ci_width_95 = uncertainty_metrics.get('avg_ci_width_95', 0)
        relative_ci_width_95 = uncertainty_metrics.get('relative_ci_width_95', 0)
        
        if relative_ci_width_95 < 10:
            summary['confidence_assessment'] = "High"
        elif relative_ci_width_95 < 25:
            summary['confidence_assessment'] = "Medium"
        else:
            summary['confidence_assessment'] = "Low"
        
        return summary
    
    def _create_summary_text(self, forecast_output: ForecastOutput) -> str:
        """
        Create formatted summary text for visualization.
        
        Args:
            forecast_output: Forecast results
            
        Returns:
            Formatted summary string
        """
        summary = forecast_output.forecast_summary
        
        text = f"""Forecast Summary:
Model: {forecast_output.model_used}
Period: {summary['forecast_start']} to {summary['forecast_end']}
Trend: {summary['trend_direction']} ({summary['total_forecast_change_pct']:.1f}%)
Confidence: {summary['confidence_assessment']}
Generated: {forecast_output.generation_timestamp.strftime('%Y-%m-%d %H:%M')}"""
        
        return text