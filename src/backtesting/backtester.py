"""
Strategy backtesting system for portfolio performance simulation.

This module provides comprehensive backtesting capabilities for simulating
portfolio performance over specified periods with benchmark comparisons.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from ..portfolio import PortfolioMetrics

logger = logging.getLogger(__name__)


class RebalanceFrequency(Enum):
    """Enumeration for rebalancing frequencies."""
    DAILY = "Daily"
    WEEKLY = "Weekly"
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"
    ANNUALLY = "Annually"
    NEVER = "Never"


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    
    start_date: str = "2024-08-01"
    end_date: str = "2025-07-31"
    initial_capital: float = 100000.0
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.QUARTERLY
    rebalance_threshold: float = 0.05  # 5% drift threshold
    transaction_cost: float = 0.001  # 0.1% transaction cost
    benchmark_weights: Dict[str, float] = None
    include_dividends: bool = True
    cash_buffer: float = 0.02  # 2% cash buffer
    
    def __post_init__(self):
        if self.benchmark_weights is None:
            self.benchmark_weights = {"SPY": 0.6, "BND": 0.4}


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""
    
    date: pd.Timestamp
    portfolio_value: float
    cash: float
    positions: Dict[str, int]  # Asset -> number of shares
    weights: Dict[str, float]  # Asset -> weight
    daily_return: float
    cumulative_return: float
    benchmark_value: float
    benchmark_return: float
    benchmark_cumulative_return: float


@dataclass
class BacktestResults:
    """Container for comprehensive backtesting results."""
    
    portfolio_snapshots: List[PortfolioSnapshot]
    performance_metrics: Dict[str, float]
    benchmark_metrics: Dict[str, float]
    comparison_metrics: Dict[str, float]
    transaction_log: List[Dict[str, Any]]
    rebalance_dates: List[pd.Timestamp]
    config: BacktestConfig
    
    @property
    def portfolio_series(self) -> pd.Series:
        """Get portfolio value time series."""
        return pd.Series(
            [snap.portfolio_value for snap in self.portfolio_snapshots],
            index=[snap.date for snap in self.portfolio_snapshots],
            name='Portfolio'
        )
    
    @property
    def benchmark_series(self) -> pd.Series:
        """Get benchmark value time series."""
        return pd.Series(
            [snap.benchmark_value for snap in self.portfolio_snapshots],
            index=[snap.date for snap in self.portfolio_snapshots],
            name='Benchmark'
        )
    
    @property
    def returns_series(self) -> pd.Series:
        """Get portfolio returns time series."""
        return pd.Series(
            [snap.daily_return for snap in self.portfolio_snapshots],
            index=[snap.date for snap in self.portfolio_snapshots],
            name='Portfolio_Returns'
        )
    
    @property
    def benchmark_returns_series(self) -> pd.Series:
        """Get benchmark returns time series."""
        return pd.Series(
            [snap.benchmark_return for snap in self.portfolio_snapshots],
            index=[snap.date for snap in self.portfolio_snapshots],
            name='Benchmark_Returns'
        )


class Backtester:
    """
    Comprehensive strategy backtesting system.
    
    This class simulates portfolio performance over specified periods,
    including rebalancing, transaction costs, and benchmark comparisons.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtester.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        self.price_data = None
        self.dividend_data = None
        
        logger.info(f"Backtester initialized for period {self.config.start_date} to {self.config.end_date}")
    
    def load_price_data(self, price_data: Dict[str, pd.Series]) -> None:
        """
        Load historical price data for backtesting.
        
        Args:
            price_data: Dictionary mapping asset symbols to price series
        """
        self.price_data = pd.DataFrame(price_data)
        
        # Filter to backtest period
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        self.price_data = self.price_data[
            (self.price_data.index >= start_date) & 
            (self.price_data.index <= end_date)
        ].dropna()
        
        logger.info(f"Loaded price data: {len(self.price_data)} observations for {len(self.price_data.columns)} assets")
    
    def load_dividend_data(self, dividend_data: Optional[Dict[str, pd.Series]] = None) -> None:
        """
        Load dividend data for backtesting.
        
        Args:
            dividend_data: Dictionary mapping asset symbols to dividend series
        """
        if dividend_data is not None:
            self.dividend_data = pd.DataFrame(dividend_data)
            
            # Filter to backtest period
            start_date = pd.to_datetime(self.config.start_date)
            end_date = pd.to_datetime(self.config.end_date)
            
            self.dividend_data = self.dividend_data[
                (self.dividend_data.index >= start_date) & 
                (self.dividend_data.index <= end_date)
            ].fillna(0)
            
            logger.info(f"Loaded dividend data for {len(self.dividend_data.columns)} assets")
        else:
            # Create empty dividend data
            self.dividend_data = pd.DataFrame(
                0, 
                index=self.price_data.index, 
                columns=self.price_data.columns
            )
    
    def run_backtest(self, 
                    strategy_weights: Dict[str, float],
                    strategy_name: str = "Strategy") -> BacktestResults:
        """
        Run comprehensive backtest simulation.
        
        Args:
            strategy_weights: Target portfolio weights
            strategy_name: Name for the strategy
            
        Returns:
            BacktestResults containing comprehensive results
        """
        if self.price_data is None:
            raise ValueError("Price data not loaded. Call load_price_data() first.")
        
        logger.info(f"Running backtest for {strategy_name}")
        
        # Initialize portfolio state
        portfolio_snapshots = []
        transaction_log = []
        rebalance_dates = []
        
        # Initial setup
        current_cash = self.config.initial_capital
        current_positions = {asset: 0 for asset in strategy_weights.keys()}
        
        # Get benchmark assets
        benchmark_assets = list(self.config.benchmark_weights.keys())
        benchmark_positions = {asset: 0 for asset in benchmark_assets}
        benchmark_cash = self.config.initial_capital
        
        # Initial allocation
        first_date = self.price_data.index[0]
        current_cash, current_positions, transactions = self._rebalance_portfolio(
            first_date, current_cash, current_positions, strategy_weights
        )
        transaction_log.extend(transactions)
        rebalance_dates.append(first_date)
        
        # Initial benchmark allocation
        benchmark_cash, benchmark_positions, _ = self._rebalance_portfolio(
            first_date, benchmark_cash, benchmark_positions, self.config.benchmark_weights
        )
        
        # Track initial values
        initial_portfolio_value = self._calculate_portfolio_value(
            first_date, current_cash, current_positions
        )
        initial_benchmark_value = self._calculate_portfolio_value(
            first_date, benchmark_cash, benchmark_positions
        )
        
        # Simulate each day
        for i, date in enumerate(self.price_data.index):
            # Calculate current portfolio value
            portfolio_value = self._calculate_portfolio_value(date, current_cash, current_positions)
            benchmark_value = self._calculate_portfolio_value(date, benchmark_cash, benchmark_positions)
            
            # Add dividends if applicable
            if self.config.include_dividends and self.dividend_data is not None:
                dividend_income = self._calculate_dividend_income(date, current_positions)
                current_cash += dividend_income
                portfolio_value += dividend_income
                
                benchmark_dividend_income = self._calculate_dividend_income(date, benchmark_positions)
                benchmark_cash += benchmark_dividend_income
                benchmark_value += benchmark_dividend_income
            
            # Calculate returns
            if i == 0:
                daily_return = 0.0
                cumulative_return = 0.0
                benchmark_return = 0.0
                benchmark_cumulative_return = 0.0
            else:
                prev_value = portfolio_snapshots[-1].portfolio_value
                daily_return = (portfolio_value - prev_value) / prev_value
                cumulative_return = (portfolio_value - initial_portfolio_value) / initial_portfolio_value
                
                prev_benchmark_value = portfolio_snapshots[-1].benchmark_value
                benchmark_return = (benchmark_value - prev_benchmark_value) / prev_benchmark_value
                benchmark_cumulative_return = (benchmark_value - initial_benchmark_value) / initial_benchmark_value
            
            # Calculate current weights
            current_weights = self._calculate_current_weights(date, current_cash, current_positions)
            
            # Check if rebalancing is needed
            if self._should_rebalance(date, current_weights, strategy_weights, rebalance_dates):
                current_cash, current_positions, transactions = self._rebalance_portfolio(
                    date, current_cash, current_positions, strategy_weights
                )
                transaction_log.extend(transactions)
                rebalance_dates.append(date)
                
                # Recalculate portfolio value after rebalancing
                portfolio_value = self._calculate_portfolio_value(date, current_cash, current_positions)
                current_weights = self._calculate_current_weights(date, current_cash, current_positions)
            
            # Create snapshot
            snapshot = PortfolioSnapshot(
                date=date,
                portfolio_value=portfolio_value,
                cash=current_cash,
                positions=current_positions.copy(),
                weights=current_weights,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                benchmark_value=benchmark_value,
                benchmark_return=benchmark_return,
                benchmark_cumulative_return=benchmark_cumulative_return
            )
            portfolio_snapshots.append(snapshot)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(portfolio_snapshots, "portfolio")
        benchmark_metrics = self._calculate_performance_metrics(portfolio_snapshots, "benchmark")
        comparison_metrics = self._calculate_comparison_metrics(portfolio_snapshots)
        
        results = BacktestResults(
            portfolio_snapshots=portfolio_snapshots,
            performance_metrics=performance_metrics,
            benchmark_metrics=benchmark_metrics,
            comparison_metrics=comparison_metrics,
            transaction_log=transaction_log,
            rebalance_dates=rebalance_dates,
            config=self.config
        )
        
        logger.info(f"Backtest completed. Portfolio return: {performance_metrics['Total_Return']:.2%}, "
                   f"Benchmark return: {benchmark_metrics['Total_Return']:.2%}")
        
        return results
    
    def _calculate_portfolio_value(self, 
                                 date: pd.Timestamp,
                                 cash: float,
                                 positions: Dict[str, int]) -> float:
        """Calculate total portfolio value on a given date."""
        total_value = cash
        
        for asset, shares in positions.items():
            if shares > 0 and asset in self.price_data.columns:
                price = self.price_data.loc[date, asset]
                total_value += shares * price
        
        return total_value
    
    def _calculate_current_weights(self, 
                                 date: pd.Timestamp,
                                 cash: float,
                                 positions: Dict[str, int]) -> Dict[str, float]:
        """Calculate current portfolio weights."""
        total_value = self._calculate_portfolio_value(date, cash, positions)
        weights = {}
        
        for asset, shares in positions.items():
            if asset in self.price_data.columns:
                price = self.price_data.loc[date, asset]
                asset_value = shares * price
                weights[asset] = asset_value / total_value if total_value > 0 else 0
        
        # Add cash weight
        weights['Cash'] = cash / total_value if total_value > 0 else 0
        
        return weights
    
    def _should_rebalance(self, 
                        date: pd.Timestamp,
                        current_weights: Dict[str, float],
                        target_weights: Dict[str, float],
                        rebalance_dates: List[pd.Timestamp]) -> bool:
        """Determine if portfolio should be rebalanced."""
        
        # Check frequency-based rebalancing
        if self.config.rebalance_frequency != RebalanceFrequency.NEVER:
            if not rebalance_dates:
                return True  # First rebalance
            
            last_rebalance = rebalance_dates[-1]
            days_since_rebalance = (date - last_rebalance).days
            
            if self.config.rebalance_frequency == RebalanceFrequency.DAILY and days_since_rebalance >= 1:
                return True
            elif self.config.rebalance_frequency == RebalanceFrequency.WEEKLY and days_since_rebalance >= 7:
                return True
            elif self.config.rebalance_frequency == RebalanceFrequency.MONTHLY and days_since_rebalance >= 30:
                return True
            elif self.config.rebalance_frequency == RebalanceFrequency.QUARTERLY and days_since_rebalance >= 90:
                return True
            elif self.config.rebalance_frequency == RebalanceFrequency.ANNUALLY and days_since_rebalance >= 365:
                return True
        
        # Check threshold-based rebalancing
        for asset, target_weight in target_weights.items():
            current_weight = current_weights.get(asset, 0)
            if abs(current_weight - target_weight) > self.config.rebalance_threshold:
                return True
        
        return False
    
    def _rebalance_portfolio(self, 
                           date: pd.Timestamp,
                           cash: float,
                           positions: Dict[str, int],
                           target_weights: Dict[str, float]) -> Tuple[float, Dict[str, int], List[Dict]]:
        """Rebalance portfolio to target weights."""
        
        # Calculate current portfolio value
        total_value = self._calculate_portfolio_value(date, cash, positions)
        
        # Calculate target values for each asset
        target_values = {asset: weight * total_value for asset, weight in target_weights.items()}
        
        transactions = []
        new_positions = positions.copy()
        new_cash = cash
        
        # Sell positions that are overweight
        for asset, target_value in target_values.items():
            if asset not in self.price_data.columns:
                continue
                
            current_shares = positions.get(asset, 0)
            price = self.price_data.loc[date, asset]
            current_value = current_shares * price
            
            if current_value > target_value:
                # Sell excess shares
                excess_value = current_value - target_value
                shares_to_sell = int(excess_value / price)
                
                if shares_to_sell > 0:
                    sale_proceeds = shares_to_sell * price * (1 - self.config.transaction_cost)
                    new_cash += sale_proceeds
                    new_positions[asset] -= shares_to_sell
                    
                    transactions.append({
                        'date': date,
                        'asset': asset,
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': price,
                        'value': shares_to_sell * price,
                        'cost': shares_to_sell * price * self.config.transaction_cost
                    })
        
        # Buy positions that are underweight
        for asset, target_value in target_values.items():
            if asset not in self.price_data.columns:
                continue
                
            current_shares = new_positions.get(asset, 0)
            price = self.price_data.loc[date, asset]
            current_value = current_shares * price
            
            if current_value < target_value:
                # Buy additional shares
                deficit_value = target_value - current_value
                shares_to_buy = int(deficit_value / (price * (1 + self.config.transaction_cost)))
                
                if shares_to_buy > 0:
                    purchase_cost = shares_to_buy * price * (1 + self.config.transaction_cost)
                    
                    if new_cash >= purchase_cost:
                        new_cash -= purchase_cost
                        new_positions[asset] += shares_to_buy
                        
                        transactions.append({
                            'date': date,
                            'asset': asset,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': price,
                            'value': shares_to_buy * price,
                            'cost': shares_to_buy * price * self.config.transaction_cost
                        })
        
        return new_cash, new_positions, transactions
    
    def _calculate_dividend_income(self, 
                                 date: pd.Timestamp,
                                 positions: Dict[str, int]) -> float:
        """Calculate dividend income for the date."""
        total_dividends = 0.0
        
        if self.dividend_data is not None:
            for asset, shares in positions.items():
                if asset in self.dividend_data.columns and shares > 0:
                    dividend_per_share = self.dividend_data.loc[date, asset]
                    total_dividends += shares * dividend_per_share
        
        return total_dividends
    
    def _calculate_performance_metrics(self, 
                                     snapshots: List[PortfolioSnapshot],
                                     portfolio_type: str) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        
        if portfolio_type == "portfolio":
            values = [snap.portfolio_value for snap in snapshots]
            returns = [snap.daily_return for snap in snapshots[1:]]  # Skip first day
        else:  # benchmark
            values = [snap.benchmark_value for snap in snapshots]
            returns = [snap.benchmark_return for snap in snapshots[1:]]
        
        if not returns:
            return {}
        
        returns_series = pd.Series(returns)
        
        # Basic metrics
        total_return = (values[-1] - values[0]) / values[0]
        annualized_return = (1 + total_return) ** (252 / len(values)) - 1
        
        # Risk metrics
        volatility = returns_series.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (annualized_return - self.config.benchmark_weights.get('risk_free_rate', 0.02)) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_values = pd.Series(values)
        rolling_max = cumulative_values.expanding().max()
        drawdowns = (cumulative_values - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Additional metrics
        positive_days = (returns_series > 0).sum()
        win_rate = positive_days / len(returns_series)
        
        # Value at Risk (95%)
        var_95 = returns_series.quantile(0.05)
        
        # Conditional Value at Risk (95%)
        cvar_95 = returns_series[returns_series <= var_95].mean()
        
        return {
            'Total_Return': total_return,
            'Annualized_Return': annualized_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'VaR_95': var_95,
            'CVaR_95': cvar_95,
            'Final_Value': values[-1],
            'Total_Days': len(values)
        }
    
    def _calculate_comparison_metrics(self, snapshots: List[PortfolioSnapshot]) -> Dict[str, float]:
        """Calculate comparison metrics between strategy and benchmark."""
        
        portfolio_returns = [snap.daily_return for snap in snapshots[1:]]
        benchmark_returns = [snap.benchmark_return for snap in snapshots[1:]]
        
        if not portfolio_returns or not benchmark_returns:
            return {}
        
        portfolio_series = pd.Series(portfolio_returns)
        benchmark_series = pd.Series(benchmark_returns)
        
        # Excess returns
        excess_returns = portfolio_series - benchmark_series
        
        # Tracking error
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # Information ratio
        information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
        
        # Beta
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
        
        # Alpha (annualized)
        portfolio_annual_return = portfolio_series.mean() * 252
        benchmark_annual_return = benchmark_series.mean() * 252
        risk_free_rate = 0.02  # Assume 2%
        alpha = portfolio_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
        
        # Correlation
        correlation = portfolio_series.corr(benchmark_series)
        
        # Outperformance metrics
        outperformance_days = (excess_returns > 0).sum()
        outperformance_rate = outperformance_days / len(excess_returns)
        
        return {
            'Alpha': alpha,
            'Beta': beta,
            'Tracking_Error': tracking_error,
            'Information_Ratio': information_ratio,
            'Correlation': correlation,
            'Outperformance_Rate': outperformance_rate,
            'Average_Excess_Return': excess_returns.mean(),
            'Total_Excess_Return': excess_returns.sum()
        }