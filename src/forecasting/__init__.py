"""
Forecasting and prediction system package.

This package provides comprehensive forecasting capabilities including forecast generation,
analysis, and insights for portfolio forecasting applications.
"""

from .forecast_generator import ForecastGenerator, ForecastConfig, ForecastOutput
from .forecast_analyzer import (
    ForecastAnalyzer, ForecastInsights, TrendAnalysis, AnomalyDetection,
    ConfidenceAssessment, MarketOpportunity, TrendType, RiskLevel
)

__all__ = [
    'ForecastGenerator',
    'ForecastConfig',
    'ForecastOutput',
    'ForecastAnalyzer',
    'ForecastInsights',
    'TrendAnalysis',
    'AnomalyDetection',
    'ConfidenceAssessment',
    'MarketOpportunity',
    'TrendType',
    'RiskLevel'
]