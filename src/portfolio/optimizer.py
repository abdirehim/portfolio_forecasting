"""
Portfolio optimization system using Modern Portfolio Theory.

This module provides comprehensive portfolio optimization capabilities using the
PyPortfolioOpt library, including expected returns calculation, covariance matrix
computation, and Efficient Frontier generation.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings

# PyPortfolioOpt imports
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.plotting import plot_efficient_frontier
import cvxpy as cp

from ..forecasting import ForecastOutput

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""
    
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    weight_bounds: Tuple[float, float] = (0.0, 1.0)  # Min and max weights
    sector_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    optimization_method: str = "max_sharpe"  # "max_sharpe", "min_volatility", "efficient_return"
    gamma: float = 0  # Risk aversion parameter for utility maximization
    market_neutral: bool = False
    l2_reg: float = 0.01  # L2 regularization parameter


@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics."""
    
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: Dict[str, float]
    portfolio_value: float
    diversification_ratio: float
    max_drawdown: Optional[float] = None
    var_95: Optional[float] = None  # Value at Risk at 95% confidence
    cvar_95: Optional[float] = None  # Conditional Value at Risk


@dataclass
class EfficientFrontierData:
    """Container for Efficient Frontier data."""
    
    returns: np.ndarray
    volatilities: np.ndarray
    sharpe_ratios: np.ndarray
    max_sharpe_portfolio: PortfolioMetrics
    min_volatility_portfolio: PortfolioMetrics
    frontier_weights: List[Dict[str, float]]


class PortfolioOptimizer:
    """
    Comprehensive portfolio optimization system using Modern Portfolio Theory.
    
    This class provides methods for calculating expected returns using forecasts,
    computing covariance matrices, generating the Efficient Frontier, and
    identifying optimal portfolios based on various criteria.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize portfolio optimizer.
        
        Args:
            config: Optimization configuration settings
        """
        self.config = config or OptimizationConfig()
        self.assets = []
        self.price_data = None
        self.expected_returns = None
        self.covariance_matrix = None
        self.efficient_frontier = None
        
        logger.info("PortfolioOptimizer initialized")
    
    def set_assets(self, assets: List[str]) -> None:
        """
        Set the assets for portfolio optimization.
        
        Args:
            assets: List of asset symbols (e.g., ['TSLA', 'SPY', 'BND'])
        """
        self.assets = assets
        logger.info(f"Set assets for optimization: {assets}")
    
    def load_price_data(self, price_data: Dict[str, pd.Series]) -> None:
        """
        Load historical price data for assets.
        
        Args:
            price_data: Dictionary mapping asset symbols to price series
        """
        # Convert to DataFrame
        self.price_data = pd.DataFrame(price_data)
        
        # Ensure all assets are present
        missing_assets = set(self.assets) - set(self.price_data.columns)
        if missing_assets:
            raise ValueError(f"Missing price data for assets: {missing_assets}")
        
        # Align data and handle missing values
        self.price_data = self.price_data[self.assets].dropna()
        
        logger.info(f"Loaded price data: {len(self.price_data)} observations for {len(self.assets)} assets")
    
    def calculate_expected_returns(self, 
                                 forecast_data: Optional[Dict[str, ForecastOutput]] = None,
                                 method: str = "mixed") -> pd.Series:
        """
        Calculate expected returns using forecasts and historical data.
        
        Args:
            forecast_data: Dictionary mapping asset symbols to forecast outputs
            method: Method for calculating returns ("historical", "forecast", "mixed")
            
        Returns:
            Series of expected annual returns for each asset
        """
        if self.price_data is None:
            raise ValueError("Price data not loaded. Call load_price_data() first.")
        
        expected_returns_dict = {}
        
        for asset in self.assets:
            if method == "historical":
                # Use historical mean returns
                returns = self.price_data[asset].pct_change().dropna()
                annual_return = returns.mean() * 252  # Annualize daily returns
                
            elif method == "forecast" and forecast_data and asset in forecast_data:
                # Use forecast-based returns
                forecast = forecast_data[asset]
                current_price = self.price_data[asset].iloc[-1]
                forecast_price = forecast.predictions.iloc[-1]
                
                # Calculate annualized return from forecast
                forecast_horizon_years = len(forecast.predictions) / 252
                total_return = (forecast_price - current_price) / current_price
                annual_return = (1 + total_return) ** (1 / forecast_horizon_years) - 1
                
            elif method == "mixed":
                # Combine forecast and historical data
                if forecast_data and asset in forecast_data:
                    # Use forecast for this asset
                    forecast = forecast_data[asset]
                    current_price = self.price_data[asset].iloc[-1]
                    forecast_price = forecast.predictions.iloc[-1]
                    
                    forecast_horizon_years = len(forecast.predictions) / 252
                    total_return = (forecast_price - current_price) / current_price
                    forecast_return = (1 + total_return) ** (1 / forecast_horizon_years) - 1
                    
                    # Get historical return for comparison
                    returns = self.price_data[asset].pct_change().dropna()
                    historical_return = returns.mean() * 252
                    
                    # Weight forecast more heavily but include historical context
                    annual_return = 0.7 * forecast_return + 0.3 * historical_return
                else:
                    # Fall back to historical for assets without forecasts
                    returns = self.price_data[asset].pct_change().dropna()
                    annual_return = returns.mean() * 252
            
            else:
                raise ValueError(f"Invalid method: {method}")
            
            expected_returns_dict[asset] = annual_return
        
        self.expected_returns = pd.Series(expected_returns_dict)
        
        logger.info(f"Calculated expected returns using {method} method")
        logger.info(f"Expected returns: {self.expected_returns.to_dict()}")
        
        return self.expected_returns
    
    def calculate_covariance_matrix(self, method: str = "sample") -> pd.DataFrame:
        """
        Calculate covariance matrix based on historical daily returns.
        
        Args:
            method: Method for covariance estimation ("sample", "semicovariance", "exp_cov")
            
        Returns:
            Covariance matrix DataFrame
        """
        if self.price_data is None:
            raise ValueError("Price data not loaded. Call load_price_data() first.")
        
        # Calculate daily returns
        returns = self.price_data.pct_change().dropna()
        
        if method == "sample":
            # Standard sample covariance matrix
            self.covariance_matrix = risk_models.sample_cov(self.price_data, frequency=252)
            
        elif method == "semicovariance":
            # Semicovariance (downside risk)
            self.covariance_matrix = risk_models.semicovariance(self.price_data, frequency=252)
            
        elif method == "exp_cov":
            # Exponentially weighted covariance
            self.covariance_matrix = risk_models.exp_cov(self.price_data, frequency=252)
            
        else:
            raise ValueError(f"Invalid covariance method: {method}")
        
        logger.info(f"Calculated covariance matrix using {method} method")
        
        return self.covariance_matrix
    
    def generate_efficient_frontier(self, 
                                  num_portfolios: int = 100) -> EfficientFrontierData:
        """
        Generate the Efficient Frontier and identify key portfolios.
        
        Args:
            num_portfolios: Number of portfolios to generate along the frontier
            
        Returns:
            EfficientFrontierData containing frontier data and key portfolios
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Expected returns and covariance matrix must be calculated first")
        
        # Create EfficientFrontier object
        ef = EfficientFrontier(self.expected_returns, self.covariance_matrix, 
                              weight_bounds=self.config.weight_bounds)
        
        # Generate frontier
        returns_range = np.linspace(
            self.expected_returns.min() * 1.1,
            self.expected_returns.max() * 0.9,
            num_portfolios
        )
        
        frontier_returns = []
        frontier_volatilities = []
        frontier_sharpe_ratios = []
        frontier_weights = []
        
        for target_return in returns_range:
            try:
                # Create a fresh EfficientFrontier object for each optimization
                ef_temp = EfficientFrontier(self.expected_returns, self.covariance_matrix,
                                          weight_bounds=self.config.weight_bounds)
                
                # Optimize for target return
                weights = ef_temp.efficient_return(target_return, market_neutral=self.config.market_neutral)
                
                # Calculate portfolio metrics
                ret, vol, sharpe = ef_temp.portfolio_performance(
                    risk_free_rate=self.config.risk_free_rate, verbose=False
                )
                
                frontier_returns.append(ret)
                frontier_volatilities.append(vol)
                frontier_sharpe_ratios.append(sharpe)
                frontier_weights.append(dict(weights))
                
            except Exception as e:
                logger.warning(f"Failed to optimize for return {target_return:.4f}: {str(e)}")
                continue
        
        # Find maximum Sharpe ratio portfolio
        max_sharpe_ef = EfficientFrontier(self.expected_returns, self.covariance_matrix,
                                        weight_bounds=self.config.weight_bounds)
        max_sharpe_weights = max_sharpe_ef.max_sharpe(risk_free_rate=self.config.risk_free_rate)
        max_sharpe_ret, max_sharpe_vol, max_sharpe_ratio = max_sharpe_ef.portfolio_performance(
            risk_free_rate=self.config.risk_free_rate, verbose=False
        )
        
        max_sharpe_portfolio = PortfolioMetrics(
            expected_return=max_sharpe_ret,
            volatility=max_sharpe_vol,
            sharpe_ratio=max_sharpe_ratio,
            weights=dict(max_sharpe_weights),
            portfolio_value=1.0,  # Normalized
            diversification_ratio=self._calculate_diversification_ratio(max_sharpe_weights)
        )
        
        # Find minimum volatility portfolio
        min_vol_ef = EfficientFrontier(self.expected_returns, self.covariance_matrix,
                                     weight_bounds=self.config.weight_bounds)
        min_vol_weights = min_vol_ef.min_volatility()
        min_vol_ret, min_vol_vol, min_vol_sharpe = min_vol_ef.portfolio_performance(
            risk_free_rate=self.config.risk_free_rate, verbose=False
        )
        
        min_volatility_portfolio = PortfolioMetrics(
            expected_return=min_vol_ret,
            volatility=min_vol_vol,
            sharpe_ratio=min_vol_sharpe,
            weights=dict(min_vol_weights),
            portfolio_value=1.0,  # Normalized
            diversification_ratio=self._calculate_diversification_ratio(min_vol_weights)
        )
        
        # Store the efficient frontier for later use
        self.efficient_frontier = EfficientFrontier(self.expected_returns, self.covariance_matrix,
                                                  weight_bounds=self.config.weight_bounds)
        
        frontier_data = EfficientFrontierData(
            returns=np.array(frontier_returns),
            volatilities=np.array(frontier_volatilities),
            sharpe_ratios=np.array(frontier_sharpe_ratios),
            max_sharpe_portfolio=max_sharpe_portfolio,
            min_volatility_portfolio=min_volatility_portfolio,
            frontier_weights=frontier_weights
        )
        
        logger.info("Generated Efficient Frontier")
        logger.info(f"Max Sharpe Portfolio - Return: {max_sharpe_ret:.4f}, Vol: {max_sharpe_vol:.4f}, Sharpe: {max_sharpe_ratio:.4f}")
        logger.info(f"Min Vol Portfolio - Return: {min_vol_ret:.4f}, Vol: {min_vol_vol:.4f}, Sharpe: {min_vol_sharpe:.4f}")
        
        return frontier_data
    
    def optimize_portfolio(self, 
                          method: Optional[str] = None,
                          **kwargs) -> PortfolioMetrics:
        """
        Optimize portfolio based on specified method.
        
        Args:
            method: Optimization method (overrides config if provided)
            **kwargs: Additional parameters for optimization
            
        Returns:
            PortfolioMetrics for the optimized portfolio
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Expected returns and covariance matrix must be calculated first")
        
        optimization_method = method or self.config.optimization_method
        
        # Create EfficientFrontier object
        ef = EfficientFrontier(self.expected_returns, self.covariance_matrix,
                              weight_bounds=self.config.weight_bounds)
        
        # Apply sector constraints if specified
        if self.config.sector_constraints:
            for sector, (min_weight, max_weight) in self.config.sector_constraints.items():
                # This would require sector mapping - simplified for now
                pass
        
        # Optimize based on method
        if optimization_method == "max_sharpe":
            weights = ef.max_sharpe(risk_free_rate=self.config.risk_free_rate)
            
        elif optimization_method == "min_volatility":
            weights = ef.min_volatility()
            
        elif optimization_method == "efficient_return":
            target_return = kwargs.get('target_return', self.config.target_return)
            if target_return is None:
                raise ValueError("target_return must be specified for efficient_return method")
            weights = ef.efficient_return(target_return)
            
        elif optimization_method == "efficient_risk":
            target_volatility = kwargs.get('target_volatility', self.config.target_volatility)
            if target_volatility is None:
                raise ValueError("target_volatility must be specified for efficient_risk method")
            weights = ef.efficient_risk(target_volatility)
            
        else:
            raise ValueError(f"Invalid optimization method: {optimization_method}")
        
        # Calculate portfolio performance
        ret, vol, sharpe = ef.portfolio_performance(
            risk_free_rate=self.config.risk_free_rate, verbose=False
        )
        
        # Calculate additional metrics
        diversification_ratio = self._calculate_diversification_ratio(weights)
        
        portfolio_metrics = PortfolioMetrics(
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            weights=dict(weights),
            portfolio_value=1.0,  # Normalized
            diversification_ratio=diversification_ratio
        )
        
        logger.info(f"Optimized portfolio using {optimization_method} method")
        logger.info(f"Portfolio metrics - Return: {ret:.4f}, Vol: {vol:.4f}, Sharpe: {sharpe:.4f}")
        
        return portfolio_metrics
    
    def calculate_discrete_allocation(self, 
                                    portfolio_metrics: PortfolioMetrics,
                                    total_portfolio_value: float,
                                    latest_prices: Optional[Dict[str, float]] = None) -> Dict[str, int]:
        """
        Calculate discrete allocation of shares for a given portfolio value.
        
        Args:
            portfolio_metrics: Portfolio metrics with optimal weights
            total_portfolio_value: Total value to invest
            latest_prices: Latest prices for assets (if None, uses last price from data)
            
        Returns:
            Dictionary mapping asset symbols to number of shares
        """
        if latest_prices is None:
            if self.price_data is None:
                raise ValueError("Price data not available and latest_prices not provided")
            latest_prices = get_latest_prices(self.price_data)
        
        # Create discrete allocation
        da = DiscreteAllocation(
            portfolio_metrics.weights,
            latest_prices,
            total_portfolio_value=total_portfolio_value
        )
        
        allocation, leftover = da.lp_portfolio()
        
        logger.info(f"Discrete allocation for ${total_portfolio_value:,.2f}")
        logger.info(f"Allocation: {allocation}")
        logger.info(f"Leftover cash: ${leftover:.2f}")
        
        return allocation
    
    def visualize_efficient_frontier(self, 
                                   frontier_data: EfficientFrontierData,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of the Efficient Frontier.
        
        Args:
            frontier_data: Efficient frontier data
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot efficient frontier
        ax.plot(frontier_data.volatilities, frontier_data.returns, 
               'b-', linewidth=2, label='Efficient Frontier')
        
        # Plot maximum Sharpe ratio portfolio
        max_sharpe = frontier_data.max_sharpe_portfolio
        ax.scatter(max_sharpe.volatility, max_sharpe.expected_return,
                  marker='*', s=300, c='red', label=f'Max Sharpe Ratio ({max_sharpe.sharpe_ratio:.3f})')
        
        # Plot minimum volatility portfolio
        min_vol = frontier_data.min_volatility_portfolio
        ax.scatter(min_vol.volatility, min_vol.expected_return,
                  marker='*', s=300, c='green', label=f'Min Volatility ({min_vol.volatility:.3f})')
        
        # Plot individual assets
        for asset in self.assets:
            asset_return = self.expected_returns[asset]
            asset_vol = np.sqrt(self.covariance_matrix.loc[asset, asset])
            ax.scatter(asset_vol, asset_return, marker='o', s=100, alpha=0.7, label=asset)
        
        # Formatting
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Efficient Frontier')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add risk-free rate line if applicable
        if self.config.risk_free_rate > 0:
            max_vol = max(frontier_data.volatilities.max(), 
                         max([np.sqrt(self.covariance_matrix.loc[asset, asset]) for asset in self.assets]))
            ax.plot([0, max_vol], [self.config.risk_free_rate, self.config.risk_free_rate], 
                   'k--', alpha=0.5, label=f'Risk-free rate ({self.config.risk_free_rate:.1%})')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Efficient frontier plot saved to {save_path}")
        
        return fig
    
    def _calculate_diversification_ratio(self, weights: Dict[str, float]) -> float:
        """
        Calculate the diversification ratio of a portfolio.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Diversification ratio
        """
        if self.covariance_matrix is None:
            return 0.0
        
        # Convert weights to array
        weight_array = np.array([weights.get(asset, 0.0) for asset in self.assets])
        
        # Calculate weighted average volatility
        individual_vols = np.sqrt(np.diag(self.covariance_matrix))
        weighted_avg_vol = np.sum(weight_array * individual_vols)
        
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weight_array, np.dot(self.covariance_matrix, weight_array)))
        
        # Diversification ratio
        if portfolio_vol > 0:
            return weighted_avg_vol / portfolio_vol
        else:
            return 1.0