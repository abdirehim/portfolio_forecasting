"""
Backtesting and performance analysis system package.

This package provides comprehensive backtesting capabilities for portfolio
strategies with detailed performance analysis and benchmark comparisons.
"""

from .backtester import (
    Backtester, BacktestConfig, BacktestResults, PortfolioSnapshot,
    RebalanceFrequency
)
from .performance_analyzer import (
    PerformanceAnalyzer, PerformanceComparison, RiskAnalysis
)

__all__ = [
    'Backtester',
    'BacktestConfig',
    'BacktestResults',
    'PortfolioSnapshot',
    'RebalanceFrequency',
    'PerformanceAnalyzer',
    'PerformanceComparison',
    'RiskAnalysis'
]