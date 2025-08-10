"""Configuration management for the portfolio forecasting system."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data configuration settings."""
    symbols: list
    start_date: str
    end_date: str
    train_end_date: str
    test_start_date: str
    backtest_start_date: str
    backtest_end_date: str


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    enabled: bool
    cache_dir: str
    expiry_hours: int


@dataclass
class PreprocessingConfig:
    """Data preprocessing configuration."""
    handle_missing: str
    outlier_threshold: float
    scaling_method: str


@dataclass
class RiskConfig:
    """Risk analysis configuration."""
    confidence_level: float
    rolling_window: int


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    figure_size: list
    style: str
    color_palette: str
    save_plots: bool
    plot_dir: str


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    format: str
    file: str


class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._settings = None
        self._model_params = None
        
    def load_settings(self) -> Dict[str, Any]:
        """Load application settings from YAML file.
        
        Returns:
            Dictionary containing application settings
        """
        if self._settings is None:
            settings_path = self.config_dir / "settings.yaml"
            with open(settings_path, 'r') as file:
                self._settings = yaml.safe_load(file)
        return self._settings
    
    def load_model_params(self) -> Dict[str, Any]:
        """Load model parameters from YAML file.
        
        Returns:
            Dictionary containing model parameters
        """
        if self._model_params is None:
            params_path = self.config_dir / "model_params.yaml"
            with open(params_path, 'r') as file:
                self._model_params = yaml.safe_load(file)
        return self._model_params
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration object.
        
        Returns:
            DataConfig object with data settings
        """
        settings = self.load_settings()
        return DataConfig(**settings['data'])
    
    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration object.
        
        Returns:
            CacheConfig object with cache settings
        """
        settings = self.load_settings()
        return CacheConfig(**settings['cache'])
    
    def get_preprocessing_config(self) -> PreprocessingConfig:
        """Get preprocessing configuration object.
        
        Returns:
            PreprocessingConfig object with preprocessing settings
        """
        settings = self.load_settings()
        return PreprocessingConfig(**settings['preprocessing'])
    
    def get_risk_config(self) -> RiskConfig:
        """Get risk analysis configuration object.
        
        Returns:
            RiskConfig object with risk settings
        """
        settings = self.load_settings()
        return RiskConfig(**settings['risk'])
    
    def get_visualization_config(self) -> VisualizationConfig:
        """Get visualization configuration object.
        
        Returns:
            VisualizationConfig object with visualization settings
        """
        settings = self.load_settings()
        return VisualizationConfig(**settings['visualization'])
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration object.
        
        Returns:
            LoggingConfig object with logging settings
        """
        settings = self.load_settings()
        return LoggingConfig(**settings['logging'])
    
    def get_arima_params(self) -> Dict[str, Any]:
        """Get ARIMA model parameters.
        
        Returns:
            Dictionary containing ARIMA parameters
        """
        params = self.load_model_params()
        return params['arima']
    
    def get_lstm_params(self) -> Dict[str, Any]:
        """Get LSTM model parameters.
        
        Returns:
            Dictionary containing LSTM parameters
        """
        params = self.load_model_params()
        return params['lstm']
    
    def get_evaluation_params(self) -> Dict[str, Any]:
        """Get model evaluation parameters.
        
        Returns:
            Dictionary containing evaluation parameters
        """
        params = self.load_model_params()
        return params['evaluation']
    
    def get_portfolio_params(self) -> Dict[str, Any]:
        """Get portfolio optimization parameters.
        
        Returns:
            Dictionary containing portfolio parameters
        """
        params = self.load_model_params()
        return params['portfolio']
    
    def get_backtesting_params(self) -> Dict[str, Any]:
        """Get backtesting parameters.
        
        Returns:
            Dictionary containing backtesting parameters
        """
        params = self.load_model_params()
        return params['backtesting']


# Global configuration instance
config_manager = ConfigManager()