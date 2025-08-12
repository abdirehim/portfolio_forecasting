"""
Portfolio optimization and management system package.

This package provides comprehensive portfolio optimization and recommendation
capabilities using Modern Portfolio Theory and advanced forecasting integration.
"""

from .optimizer import (
    PortfolioOptimizer, OptimizationConfig, PortfolioMetrics, 
    EfficientFrontierData
)
from .recommender import (
    PortfolioRecommender, RecommendationCriteria, PortfolioRecommendation,
    RecommendationReport, RecommendationType, RiskProfile
)

__all__ = [
    'PortfolioOptimizer',
    'OptimizationConfig',
    'PortfolioMetrics',
    'EfficientFrontierData',
    'PortfolioRecommender',
    'RecommendationCriteria',
    'PortfolioRecommendation',
    'RecommendationReport',
    'RecommendationType',
    'RiskProfile'
]