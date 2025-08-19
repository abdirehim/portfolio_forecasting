"""Exploratory Data Analysis engine for financial data."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from statsmodels.tsa.stattools import adfuller
import warnings
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class EDAReport:
    """Container for EDA results."""
    summary_stats: Dict[str, Any]
    correlation_matrix: pd.DataFrame
    stationarity_tests: Dict[str, Dict[str, Any]]
    outlier_analysis: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    volatility_analysis: Dict[str, Any]
    plots_generated: List[str]


class EDAEngine:
    """Comprehensive exploratory data analysis for financial data."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = "seaborn-v0_8"):
        """Initialize EDA engine.
        
        Args:
            figsize: Default figure size for plots
            style: Matplotlib style
        """
        self.figsize = figsize
        self.style = style
        
        # Set plotting style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
            logger.warning(f"Style {style} not available, using default")
        
        # Color palette for different assets
        self.colors = {
            'TSLA': '#E31E24',  # Tesla red
            'BND': '#1f77b4',   # Blue
            'SPY': '#2ca02c'    # Green
        }
        
        logger.info("EDAEngine initialized")
    
    def plot_price_trends(self, data: pd.DataFrame):
        """Plot price trends for all symbols."""
        self._plot_price_trends(data, "./")
        
    def plot_returns_distribution(self, data: pd.DataFrame):
        """Plot returns distribution analysis."""
        return_data = self._calculate_returns(data)
        self._plot_returns(return_data, "./")
        
    def plot_volatility_analysis(self, data: pd.DataFrame):
        """Plot volatility analysis."""
        return_data = self._calculate_returns(data)
        self._plot_volatility(return_data, "./")
    
    def perform_eda(self, data: pd.DataFrame, save_plots: bool = True, plot_dir: str = "data/plots") -> EDAReport:
        """Perform comprehensive EDA.
        
        Args:
            data: DataFrame with financial data
            save_plots: Whether to save plots to files
            plot_dir: Directory to save plots
            
        Returns:
            EDAReport with analysis results
        """
        logger.info(f"Starting EDA for {len(data)} rows")
        
        # Create plots directory if saving
        if save_plots:
            import os
            os.makedirs(plot_dir, exist_ok=True)
        
        plots_generated = []
        
        # 1. Summary statistics
        summary_stats = self._generate_summary_stats(data)
        
        # 2. Price trend analysis
        if save_plots:
            self._plot_price_trends(data, plot_dir)
            plots_generated.append("price_trends.png")
        
        # 3. Return analysis
        return_data = self._calculate_returns(data)
        if save_plots:
            self._plot_returns(return_data, plot_dir)
            plots_generated.append("returns_analysis.png")
        
        # 4. Volatility analysis
        volatility_analysis = self._analyze_volatility(return_data)
        if save_plots:
            self._plot_volatility(return_data, plot_dir)
            plots_generated.append("volatility_analysis.png")
        
        # 5. Correlation analysis
        correlation_matrix = self._calculate_correlations(return_data)
        if save_plots:
            self._plot_correlation_matrix(correlation_matrix, plot_dir)
            plots_generated.append("correlation_matrix.png")
        
        # 6. Stationarity tests
        stationarity_tests = self._test_stationarity(data, return_data)
        
        # 7. Outlier analysis
        outlier_analysis = self._analyze_outliers(return_data)
        if save_plots:
            self._plot_outliers(return_data, plot_dir)
            plots_generated.append("outlier_analysis.png")
        
        # 8. Trend analysis
        trend_analysis = self._analyze_trends(data)
        if save_plots:
            self._plot_rolling_statistics(data, plot_dir)
            plots_generated.append("rolling_statistics.png")
        
        # 9. Distribution analysis
        if save_plots:
            self._plot_distributions(return_data, plot_dir)
            plots_generated.append("return_distributions.png")
        
        # 10. Volume analysis
        if 'Volume' in data.columns and save_plots:
            self._plot_volume_analysis(data, plot_dir)
            plots_generated.append("volume_analysis.png")
        
        logger.info(f"EDA completed. Generated {len(plots_generated)} plots")
        
        return EDAReport(
            summary_stats=summary_stats,
            correlation_matrix=correlation_matrix,
            stationarity_tests=stationarity_tests,
            outlier_analysis=outlier_analysis,
            trend_analysis=trend_analysis,
            volatility_analysis=volatility_analysis,
            plots_generated=plots_generated
        )
    
    def _generate_summary_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        logger.debug("Generating summary statistics")
        
        summary = {
            'data_shape': data.shape,
            'date_range': {},
            'symbols': [],
            'missing_values': data.isnull().sum().to_dict(),
            'price_statistics': {},
            'volume_statistics': {}
        }
        
        # Date range
        if 'Date' in data.columns:
            summary['date_range'] = {
                'start': data['Date'].min().strftime('%Y-%m-%d'),
                'end': data['Date'].max().strftime('%Y-%m-%d'),
                'total_days': (data['Date'].max() - data['Date'].min()).days
            }
        
        # Symbols
        if 'Symbol' in data.columns:
            summary['symbols'] = sorted(data['Symbol'].unique().tolist())
        
        # Price statistics by symbol
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        if 'Symbol' in data.columns:
            for symbol in data['Symbol'].unique():
                symbol_data = data[data['Symbol'] == symbol]
                summary['price_statistics'][symbol] = {}
                
                for col in price_cols:
                    if col in symbol_data.columns:
                        summary['price_statistics'][symbol][col] = {
                            'mean': float(symbol_data[col].mean()),
                            'std': float(symbol_data[col].std()),
                            'min': float(symbol_data[col].min()),
                            'max': float(symbol_data[col].max()),
                            'median': float(symbol_data[col].median())
                        }
        
        # Volume statistics
        if 'Volume' in data.columns:
            if 'Symbol' in data.columns:
                for symbol in data['Symbol'].unique():
                    symbol_data = data[data['Symbol'] == symbol]
                    summary['volume_statistics'][symbol] = {
                        'mean': float(symbol_data['Volume'].mean()),
                        'std': float(symbol_data['Volume'].std()),
                        'median': float(symbol_data['Volume'].median())
                    }
            else:
                summary['volume_statistics']['overall'] = {
                    'mean': float(data['Volume'].mean()),
                    'std': float(data['Volume'].std()),
                    'median': float(data['Volume'].median())
                }
        
        return summary
    
    def _calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns for analysis."""
        return_data = data.copy()
        
        if 'Symbol' in data.columns:
            for symbol in data['Symbol'].unique():
                mask = data['Symbol'] == symbol
                symbol_data = data[mask].sort_values('Date')
                
                if 'Adj Close' in symbol_data.columns:
                    returns = symbol_data['Adj Close'].pct_change()
                    return_data.loc[mask, 'Returns'] = returns.values
        else:
            if 'Adj Close' in data.columns:
                return_data['Returns'] = data['Adj Close'].pct_change()
        
        return return_data
    
    def _plot_price_trends(self, data: pd.DataFrame, plot_dir: str):
        """Plot price trends for all symbols with enhanced clarity and labeling."""
        logger.debug("Plotting price trends")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Portfolio Asset Price Analysis (2015-2025)', fontsize=18, fontweight='bold', y=0.98)
        
        if 'Symbol' in data.columns:
            # Plot 1: Closing prices with clear asset identification
            ax1 = axes[0, 0]
            for symbol in sorted(data['Symbol'].unique()):
                symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
                if 'Adj Close' in symbol_data.columns:
                    ax1.plot(symbol_data['Date'], symbol_data['Adj Close'], 
                            label=f'{symbol} (${symbol_data["Adj Close"].iloc[-1]:.2f})', 
                            color=self.colors.get(symbol, None), linewidth=2.5)
            
            ax1.set_title('Asset Price Evolution', fontsize=14, fontweight='bold', pad=15)
            ax1.set_xlabel('Time Period', fontsize=12)
            ax1.set_ylabel('Adjusted Close Price (USD)', fontsize=12)
            ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Normalized performance comparison
            ax2 = axes[0, 1]
            for symbol in sorted(data['Symbol'].unique()):
                symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
                if 'Adj Close' in symbol_data.columns:
                    normalized = (symbol_data['Adj Close'] / symbol_data['Adj Close'].iloc[0]) * 100
                    total_return = (normalized.iloc[-1] - 100)
                    ax2.plot(symbol_data['Date'], normalized, 
                            label=f'{symbol} ({total_return:+.1f}%)', 
                            color=self.colors.get(symbol, None), linewidth=2.5)
            
            ax2.axhline(y=100, color='black', linestyle='-', alpha=0.5, linewidth=1)
            ax2.set_title('Cumulative Performance Comparison', fontsize=14, fontweight='bold', pad=15)
            ax2.set_xlabel('Time Period', fontsize=12)
            ax2.set_ylabel('Normalized Price (Base = 100)', fontsize=12)
            ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Daily volatility (price ranges)
            ax3 = axes[1, 0]
            for symbol in sorted(data['Symbol'].unique()):
                symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
                if 'High' in symbol_data.columns and 'Low' in symbol_data.columns:
                    price_range_pct = ((symbol_data['High'] - symbol_data['Low']) / symbol_data['Close']) * 100
                    avg_range = price_range_pct.mean()
                    ax3.plot(symbol_data['Date'], price_range_pct.rolling(30).mean(), 
                            label=f'{symbol} (Avg: {avg_range:.2f}%)', 
                            color=self.colors.get(symbol, None), linewidth=2, alpha=0.8)
            
            ax3.set_title('Daily Volatility Patterns (30-Day MA)', fontsize=14, fontweight='bold', pad=15)
            ax3.set_xlabel('Time Period', fontsize=12)
            ax3.set_ylabel('Intraday Range (%)', fontsize=12)
            ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            ax3.grid(True, alpha=0.3, linestyle='--')
            ax3.tick_params(axis='x', rotation=45)
            
            # Plot 4: Trading activity analysis
            ax4 = axes[1, 1]
            if 'Volume' in data.columns:
                for symbol in sorted(data['Symbol'].unique()):
                    symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
                    volume_ma = symbol_data['Volume'].rolling(30).mean()
                    avg_volume = symbol_data['Volume'].mean()
                    ax4.plot(symbol_data['Date'], volume_ma / 1e6, 
                            label=f'{symbol} (Avg: {avg_volume/1e6:.1f}M)', 
                            color=self.colors.get(symbol, None), linewidth=2, alpha=0.8)
                
                ax4.set_title('Trading Activity Trends (30-Day MA)', fontsize=14, fontweight='bold', pad=15)
                ax4.set_xlabel('Time Period', fontsize=12)
                ax4.set_ylabel('Volume (Millions)', fontsize=12)
                ax4.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
                ax4.grid(True, alpha=0.3, linestyle='--')
                ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(f"{plot_dir}/price_trends.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_returns(self, data: pd.DataFrame, plot_dir: str):
        """Plot return analysis with enhanced statistical insights."""
        logger.debug("Plotting returns analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Portfolio Return Analysis & Risk Assessment', fontsize=18, fontweight='bold', y=0.98)
        
        if 'Symbol' in data.columns and 'Returns' in data.columns:
            # Plot 1: Daily returns with volatility bands
            ax1 = axes[0, 0]
            for symbol in sorted(data['Symbol'].unique()):
                symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
                returns = symbol_data['Returns'].dropna()
                if len(returns) > 0:
                    # Calculate rolling volatility for bands
                    rolling_vol = returns.rolling(30).std()
                    dates = symbol_data['Date'].iloc[1:len(returns)+1]
                    
                    ax1.plot(dates, returns * 100, 
                            label=f'{symbol} (σ={returns.std()*100:.2f}%)', 
                            color=self.colors.get(symbol, None), alpha=0.7, linewidth=1.5)
            
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            ax1.set_title('Daily Return Patterns', fontsize=14, fontweight='bold', pad=15)
            ax1.set_xlabel('Time Period', fontsize=12)
            ax1.set_ylabel('Daily Return (%)', fontsize=12)
            ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Cumulative wealth growth
            ax2 = axes[0, 1]
            for symbol in sorted(data['Symbol'].unique()):
                symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
                returns = symbol_data['Returns'].dropna()
                if len(returns) > 0:
                    cumulative = (1 + returns).cumprod()
                    total_return = (cumulative.iloc[-1] - 1) * 100
                    cagr = ((cumulative.iloc[-1]) ** (252/len(returns)) - 1) * 100
                    dates = symbol_data['Date'].iloc[1:len(cumulative)+1]
                    
                    ax2.plot(dates, cumulative, 
                            label=f'{symbol} (CAGR: {cagr:.1f}%)', 
                            color=self.colors.get(symbol, None), linewidth=3)
            
            ax2.axhline(y=1, color='black', linestyle='-', alpha=0.5, linewidth=1)
            ax2.set_title('Wealth Growth Comparison ($1 Initial Investment)', fontsize=14, fontweight='bold', pad=15)
            ax2.set_xlabel('Time Period', fontsize=12)
            ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Risk-return distribution analysis
            ax3 = axes[1, 0]
            for symbol in sorted(data['Symbol'].unique()):
                symbol_data = data[data['Symbol'] == symbol]
                returns = symbol_data['Returns'].dropna() * 100
                if len(returns) > 0:
                    # Calculate statistics for legend
                    mean_ret = returns.mean()
                    std_ret = returns.std()
                    skewness = returns.skew()
                    
                    ax3.hist(returns, bins=50, alpha=0.6, 
                            label=f'{symbol} (μ={mean_ret:.3f}%, σ={std_ret:.2f}%)', 
                            color=self.colors.get(symbol, None), density=True, edgecolor='white')
            
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            ax3.set_title('Return Distribution Analysis', fontsize=14, fontweight='bold', pad=15)
            ax3.set_xlabel('Daily Return (%)', fontsize=12)
            ax3.set_ylabel('Probability Density', fontsize=12)
            ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            ax3.grid(True, alpha=0.3, linestyle='--')
            
            # Plot 4: Risk-adjusted performance (Rolling Sharpe)
            ax4 = axes[1, 1]
            for symbol in sorted(data['Symbol'].unique()):
                symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
                returns = symbol_data['Returns'].dropna()
                if len(returns) > 60:
                    rolling_sharpe = (returns.rolling(60).mean() / returns.rolling(60).std()) * np.sqrt(252)
                    avg_sharpe = rolling_sharpe.mean()
                    dates = symbol_data['Date'].iloc[60:60+len(rolling_sharpe)]
                    
                    ax4.plot(dates, rolling_sharpe, 
                            label=f'{symbol} (Avg: {avg_sharpe:.2f})', 
                            color=self.colors.get(symbol, None), linewidth=2.5)
            
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Good Performance')
            ax4.set_title('Risk-Adjusted Performance (60-Day Rolling Sharpe)', fontsize=14, fontweight='bold', pad=15)
            ax4.set_xlabel('Time Period', fontsize=12)
            ax4.set_ylabel('Sharpe Ratio', fontsize=12)
            ax4.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            ax4.grid(True, alpha=0.3, linestyle='--')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(f"{plot_dir}/returns_analysis.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility patterns."""
        logger.debug("Analyzing volatility")
        
        volatility_analysis = {}
        
        if 'Symbol' in data.columns and 'Returns' in data.columns:
            for symbol in data['Symbol'].unique():
                symbol_data = data[data['Symbol'] == symbol]
                returns = symbol_data['Returns'].dropna()
                
                if len(returns) > 30:
                    # Calculate various volatility measures
                    volatility_analysis[symbol] = {
                        'daily_volatility': float(returns.std()),
                        'annualized_volatility': float(returns.std() * np.sqrt(252)),
                        'rolling_vol_30d': returns.rolling(30).std().iloc[-1] * np.sqrt(252) if len(returns) > 30 else None,
                        'vol_of_vol': float(returns.rolling(30).std().std()) if len(returns) > 60 else None,
                        'max_daily_return': float(returns.max()),
                        'min_daily_return': float(returns.min()),
                        'skewness': float(returns.skew()),
                        'kurtosis': float(returns.kurtosis())
                    }
        
        return volatility_analysis
    
    def _plot_volatility(self, data: pd.DataFrame, plot_dir: str):
        """Plot comprehensive volatility analysis with risk assessment context."""
        logger.debug("Plotting volatility analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Portfolio Risk & Volatility Assessment', fontsize=18, fontweight='bold', y=0.98)
        
        if 'Symbol' in data.columns and 'Returns' in data.columns:
            # Plot 1: Rolling volatility with risk bands
            ax1 = axes[0, 0]
            for symbol in sorted(data['Symbol'].unique()):
                symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
                returns = symbol_data['Returns'].dropna()
                if len(returns) > 30:
                    rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
                    avg_vol = rolling_vol.mean()
                    dates = symbol_data['Date'].iloc[30:30+len(rolling_vol)]
                    
                    ax1.plot(dates, rolling_vol, 
                            label=f'{symbol} (Avg: {avg_vol:.1f}%)', 
                            color=self.colors.get(symbol, None), linewidth=2.5)
            
            # Add risk level bands
            ax1.axhline(y=15, color='green', linestyle='--', alpha=0.7, label='Low Risk (15%)')
            ax1.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Medium Risk (25%)')
            ax1.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='High Risk (40%)')
            
            ax1.set_title('Rolling Volatility Trends (30-Day Annualized)', fontsize=14, fontweight='bold', pad=15)
            ax1.set_xlabel('Time Period', fontsize=12)
            ax1.set_ylabel('Volatility (%)', fontsize=12)
            ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Risk profile comparison
            ax2 = axes[0, 1]
            vol_data = []
            labels = []
            colors = []
            for symbol in sorted(data['Symbol'].unique()):
                symbol_data = data[data['Symbol'] == symbol]
                returns = symbol_data['Returns'].dropna()
                if len(returns) > 30:
                    rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
                    vol_data.append(rolling_vol.dropna())
                    labels.append(symbol)
                    colors.append(self.colors.get(symbol, 'blue'))
            
            if vol_data:
                bp = ax2.boxplot(vol_data, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax2.set_title('Volatility Distribution & Risk Profiles', fontsize=14, fontweight='bold', pad=15)
                ax2.set_ylabel('Annualized Volatility (%)', fontsize=12)
                ax2.grid(True, alpha=0.3, linestyle='--')
                
                # Add risk level reference lines
                ax2.axhline(y=15, color='green', linestyle='--', alpha=0.5)
                ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.5)
                ax2.axhline(y=40, color='red', linestyle='--', alpha=0.5)
            
            # Plot 3: Risk-Return Efficiency Frontier
            ax3 = axes[1, 0]
            risk_return_data = []
            for symbol in sorted(data['Symbol'].unique()):
                symbol_data = data[data['Symbol'] == symbol]
                returns = symbol_data['Returns'].dropna()
                if len(returns) > 252:
                    annual_return = returns.mean() * 252 * 100
                    annual_vol = returns.std() * np.sqrt(252) * 100
                    sharpe = (annual_return - 2) / annual_vol if annual_vol > 0 else 0
                    
                    ax3.scatter(annual_vol, annual_return, 
                              label=f'{symbol} (Sharpe: {sharpe:.2f})', 
                              color=self.colors.get(symbol, None), s=150, alpha=0.8, edgecolors='black')
                    
                    # Add asset labels
                    ax3.annotate(symbol, (annual_vol, annual_return), 
                               xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
                    
                    risk_return_data.append((annual_vol, annual_return, symbol))
            
            # Add efficient frontier reference line
            if len(risk_return_data) >= 2:
                vols = [x[0] for x in risk_return_data]
                rets = [x[1] for x in risk_return_data]
                ax3.plot(vols, rets, '--', alpha=0.5, color='gray', label='Asset Connection')
            
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_title('Risk-Return Efficiency Analysis', fontsize=14, fontweight='bold', pad=15)
            ax3.set_xlabel('Risk (Annualized Volatility %)', fontsize=12)
            ax3.set_ylabel('Expected Return (% per year)', fontsize=12)
            ax3.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
            ax3.grid(True, alpha=0.3, linestyle='--')
            
            # Plot 4: Volatility clustering and regime analysis
            ax4 = axes[1, 1]
            for symbol in sorted(data['Symbol'].unique()):
                symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
                returns = symbol_data['Returns'].dropna()
                if len(returns) > 0:
                    # Use squared returns to show volatility clustering
                    squared_returns = (returns * 100) ** 2
                    volatility_ma = squared_returns.rolling(20).mean()
                    dates = symbol_data['Date'].iloc[1:len(volatility_ma)+1]
                    
                    ax4.plot(dates, volatility_ma, 
                            label=f'{symbol} Volatility²', 
                            color=self.colors.get(symbol, None), linewidth=2, alpha=0.8)
            
            ax4.set_title('Volatility Clustering Patterns (20-Day MA of Squared Returns)', fontsize=14, fontweight='bold', pad=15)
            ax4.set_xlabel('Time Period', fontsize=12)
            ax4.set_ylabel('Volatility² (%²)', fontsize=12)
            ax4.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            ax4.grid(True, alpha=0.3, linestyle='--')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(f"{plot_dir}/volatility_analysis.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _calculate_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix."""
        logger.debug("Calculating correlations")
        
        if 'Symbol' in data.columns and 'Returns' in data.columns:
            # Pivot returns data
            pivot_data = data.pivot(index='Date', columns='Symbol', values='Returns')
            correlation_matrix = pivot_data.corr()
            return correlation_matrix
        
        return pd.DataFrame()
    
    def _plot_correlation_matrix(self, correlation_matrix: pd.DataFrame, plot_dir: str):
        """Plot correlation matrix."""
        logger.debug("Plotting correlation matrix")
        
        if not correlation_matrix.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
            
            ax.set_title('Return Correlation Matrix', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _test_stationarity(self, data: pd.DataFrame, return_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Test stationarity using Augmented Dickey-Fuller test."""
        logger.debug("Testing stationarity")
        
        stationarity_tests = {}
        
        if 'Symbol' in data.columns:
            for symbol in data['Symbol'].unique():
                stationarity_tests[symbol] = {}
                
                # Test price levels
                symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
                if 'Adj Close' in symbol_data.columns:
                    prices = symbol_data['Adj Close'].dropna()
                    if len(prices) > 10:
                        adf_result = adfuller(prices)
                        stationarity_tests[symbol]['prices'] = {
                            'adf_statistic': adf_result[0],
                            'p_value': adf_result[1],
                            'critical_values': adf_result[4],
                            'is_stationary': adf_result[1] < 0.05
                        }
                
                # Test returns
                symbol_return_data = return_data[return_data['Symbol'] == symbol]
                if 'Returns' in symbol_return_data.columns:
                    returns = symbol_return_data['Returns'].dropna()
                    if len(returns) > 10:
                        adf_result = adfuller(returns)
                        stationarity_tests[symbol]['returns'] = {
                            'adf_statistic': adf_result[0],
                            'p_value': adf_result[1],
                            'critical_values': adf_result[4],
                            'is_stationary': adf_result[1] < 0.05
                        }
        
        return stationarity_tests
    
    def _analyze_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in returns."""
        logger.debug("Analyzing outliers")
        
        outlier_analysis = {}
        
        if 'Symbol' in data.columns and 'Returns' in data.columns:
            for symbol in data['Symbol'].unique():
                symbol_data = data[data['Symbol'] == symbol]
                returns = symbol_data['Returns'].dropna()
                
                if len(returns) > 0:
                    # Calculate outlier thresholds (3 standard deviations)
                    mean_return = returns.mean()
                    std_return = returns.std()
                    
                    upper_threshold = mean_return + 3 * std_return
                    lower_threshold = mean_return - 3 * std_return
                    
                    outliers = returns[(returns > upper_threshold) | (returns < lower_threshold)]
                    
                    outlier_analysis[symbol] = {
                        'total_outliers': len(outliers),
                        'outlier_percentage': len(outliers) / len(returns) * 100,
                        'extreme_positive': returns.nlargest(5).tolist(),
                        'extreme_negative': returns.nsmallest(5).tolist(),
                        'upper_threshold': upper_threshold,
                        'lower_threshold': lower_threshold
                    }
        
        return outlier_analysis
    
    def _plot_outliers(self, data: pd.DataFrame, plot_dir: str):
        """Plot outlier analysis."""
        logger.debug("Plotting outlier analysis")
        
        if 'Symbol' in data.columns and 'Returns' in data.columns:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Outlier Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Box plots
            ax1 = axes[0]
            return_data = []
            labels = []
            for symbol in data['Symbol'].unique():
                symbol_data = data[data['Symbol'] == symbol]
                returns = symbol_data['Returns'].dropna()
                if len(returns) > 0:
                    return_data.append(returns)
                    labels.append(symbol)
            
            if return_data:
                ax1.boxplot(return_data, labels=labels)
                ax1.set_title('Return Distribution Box Plots')
                ax1.set_ylabel('Return')
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Q-Q plots
            ax2 = axes[1]
            from scipy import stats as scipy_stats
            
            for i, symbol in enumerate(data['Symbol'].unique()):
                symbol_data = data[data['Symbol'] == symbol]
                returns = symbol_data['Returns'].dropna()
                if len(returns) > 10:
                    scipy_stats.probplot(returns, dist="norm", plot=ax2)
                    break  # Just plot one for space
            
            ax2.set_title('Q-Q Plot (Normal Distribution)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/outlier_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trends."""
        logger.debug("Analyzing trends")
        
        trend_analysis = {}
        
        if 'Symbol' in data.columns:
            for symbol in data['Symbol'].unique():
                symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
                
                if 'Adj Close' in symbol_data.columns and len(symbol_data) > 100:
                    prices = symbol_data['Adj Close']
                    
                    # Linear trend
                    x = np.arange(len(prices))
                    slope, intercept = np.polyfit(x, prices, 1)
                    
                    # Moving averages
                    ma_20 = prices.rolling(20).mean()
                    ma_50 = prices.rolling(50).mean()
                    
                    trend_analysis[symbol] = {
                        'linear_trend_slope': slope,
                        'trend_direction': 'upward' if slope > 0 else 'downward',
                        'price_change_total': float((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]),
                        'above_ma20_pct': float((prices > ma_20).sum() / len(prices) * 100),
                        'above_ma50_pct': float((prices > ma_50).sum() / len(prices) * 100)
                    }
        
        return trend_analysis
    
    def _plot_rolling_statistics(self, data: pd.DataFrame, plot_dir: str):
        """Plot rolling statistics."""
        logger.debug("Plotting rolling statistics")
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Rolling Statistics', fontsize=16, fontweight='bold')
        
        if 'Symbol' in data.columns:
            # Plot 1: Rolling means
            ax1 = axes[0]
            for symbol in data['Symbol'].unique():
                symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
                if 'Adj Close' in symbol_data.columns:
                    prices = symbol_data['Adj Close']
                    ma_20 = prices.rolling(20).mean()
                    ma_50 = prices.rolling(50).mean()
                    
                    ax1.plot(symbol_data['Date'], prices, 
                            label=f'{symbol} Price', color=self.colors.get(symbol, None), alpha=0.7)
                    ax1.plot(symbol_data['Date'], ma_20, 
                            label=f'{symbol} MA20', color=self.colors.get(symbol, None), linestyle='--')
                    ax1.plot(symbol_data['Date'], ma_50, 
                            label=f'{symbol} MA50', color=self.colors.get(symbol, None), linestyle=':')
            
            ax1.set_title('Price with Moving Averages')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Rolling standard deviation
            ax2 = axes[1]
            for symbol in data['Symbol'].unique():
                symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
                if 'Returns' in symbol_data.columns:
                    returns = symbol_data['Returns'].dropna()
                    if len(returns) > 30:
                        rolling_std = returns.rolling(30).std()
                        ax2.plot(symbol_data['Date'].iloc[1:len(rolling_std)+1], rolling_std, 
                                label=symbol, color=self.colors.get(symbol, None), linewidth=2)
            
            ax2.set_title('Rolling Standard Deviation (30-day)')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Standard Deviation')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/rolling_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distributions(self, data: pd.DataFrame, plot_dir: str):
        """Plot return distributions."""
        logger.debug("Plotting distributions")
        
        if 'Symbol' in data.columns and 'Returns' in data.columns:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Return Distributions', fontsize=16, fontweight='bold')
            
            symbols = data['Symbol'].unique()
            
            for i, symbol in enumerate(symbols):
                if i < 3:  # Limit to 3 symbols
                    symbol_data = data[data['Symbol'] == symbol]
                    returns = symbol_data['Returns'].dropna()
                    
                    if len(returns) > 0:
                        ax = axes[i]
                        
                        # Histogram
                        ax.hist(returns, bins=50, alpha=0.7, density=True, 
                               color=self.colors.get(symbol, None))
                        
                        # Normal distribution overlay
                        mu, sigma = returns.mean(), returns.std()
                        x = np.linspace(returns.min(), returns.max(), 100)
                        normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                        ax.plot(x, normal_dist, 'r--', linewidth=2, label='Normal')
                        
                        ax.set_title(f'{symbol} Return Distribution')
                        ax.set_xlabel('Return')
                        ax.set_ylabel('Density')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/return_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_volume_analysis(self, data: pd.DataFrame, plot_dir: str):
        """Plot volume analysis."""
        logger.debug("Plotting volume analysis")
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Volume Analysis', fontsize=16, fontweight='bold')
        
        if 'Symbol' in data.columns and 'Volume' in data.columns:
            # Plot 1: Volume over time
            ax1 = axes[0]
            for symbol in data['Symbol'].unique():
                symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
                ax1.plot(symbol_data['Date'], symbol_data['Volume'], 
                        label=symbol, color=self.colors.get(symbol, None), alpha=0.7)
            
            ax1.set_title('Trading Volume Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Volume')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Volume vs Price Change
            ax2 = axes[1]
            for symbol in data['Symbol'].unique():
                symbol_data = data[data['Symbol'] == symbol]
                if 'Returns' in symbol_data.columns:
                    returns = symbol_data['Returns'].dropna()
                    volumes = symbol_data['Volume'].iloc[1:len(returns)+1]  # Align with returns
                    
                    if len(returns) > 0 and len(volumes) > 0:
                        ax2.scatter(volumes, returns, alpha=0.5, 
                                  label=symbol, color=self.colors.get(symbol, None))
            
            ax2.set_title('Volume vs Returns')
            ax2.set_xlabel('Volume')
            ax2.set_ylabel('Return')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/volume_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_eda_summary(self, report: EDAReport):
        """Print a summary of EDA results."""
        print("=" * 60)
        print("EXPLORATORY DATA ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Basic info
        print(f"\nData Shape: {report.summary_stats['data_shape']}")
        print(f"Symbols: {', '.join(report.summary_stats['symbols'])}")
        print(f"Date Range: {report.summary_stats['date_range']['start']} to {report.summary_stats['date_range']['end']}")
        
        # Stationarity results
        print("\nSTATIONARITY TESTS (ADF Test):")
        for symbol, tests in report.stationarity_tests.items():
            print(f"\n{symbol}:")
            if 'prices' in tests:
                is_stationary = "Stationary" if tests['prices']['is_stationary'] else "Non-stationary"
                print(f"  Prices: {is_stationary} (p-value: {tests['prices']['p_value']:.4f})")
            if 'returns' in tests:
                is_stationary = "Stationary" if tests['returns']['is_stationary'] else "Non-stationary"
                print(f"  Returns: {is_stationary} (p-value: {tests['returns']['p_value']:.4f})")
        
        # Volatility summary
        print("\nVOLATILITY ANALYSIS:")
        for symbol, vol_data in report.volatility_analysis.items():
            print(f"\n{symbol}:")
            print(f"  Annualized Volatility: {vol_data['annualized_volatility']:.2%}")
            print(f"  Max Daily Return: {vol_data['max_daily_return']:.2%}")
            print(f"  Min Daily Return: {vol_data['min_daily_return']:.2%}")
        
        # Correlation summary
        if not report.correlation_matrix.empty:
            print("\nCORRELATION MATRIX:")
            print(report.correlation_matrix.round(3))
        
        print(f"\nGenerated {len(report.plots_generated)} plots:")
        for plot in report.plots_generated:
            print(f"  - {plot}")
        
        print("=" * 60)