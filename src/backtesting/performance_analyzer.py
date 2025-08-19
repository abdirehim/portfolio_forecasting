"""
Performance analysis and comparison system for backtesting results.

This module provides comprehensive performance analysis capabilities including
cumulative returns visualization, Sharpe ratio calculations, and strategy
effectiveness analysis.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from .backtester import BacktestResults, PortfolioSnapshot

logger = logging.getLogger(__name__)


@dataclass
class PerformanceComparison:
    """Container for performance comparison results."""
    
    strategy_metrics: Dict[str, float]
    benchmark_metrics: Dict[str, float]
    relative_metrics: Dict[str, float]
    risk_adjusted_metrics: Dict[str, float]
    drawdown_analysis: Dict[str, Any]
    rolling_metrics: pd.DataFrame
    performance_attribution: Dict[str, float]
    conclusion: str


@dataclass
class RiskAnalysis:
    """Container for risk analysis results."""
    
    volatility_analysis: Dict[str, float]
    drawdown_analysis: Dict[str, float]
    var_analysis: Dict[str, float]
    correlation_analysis: Dict[str, float]
    risk_contribution: Dict[str, float]
    risk_summary: str


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis and comparison system.
    
    This class analyzes backtesting results to provide detailed performance
    comparisons, risk analysis, and strategy effectiveness conclusions.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        logger.info("PerformanceAnalyzer initialized")
    
    def analyze_performance(self, 
                          backtest_results: BacktestResults,
                          strategy_name: str = "Strategy") -> PerformanceComparison:
        """
        Perform comprehensive performance analysis.
        
        Args:
            backtest_results: Results from backtesting
            strategy_name: Name of the strategy being analyzed
            
        Returns:
            PerformanceComparison containing detailed analysis
        """
        logger.info(f"Analyzing performance for {strategy_name}")
        
        # Extract basic metrics
        strategy_metrics = backtest_results.performance_metrics
        benchmark_metrics = backtest_results.benchmark_metrics
        relative_metrics = backtest_results.comparison_metrics
        
        # Calculate risk-adjusted metrics
        risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(backtest_results)
        
        # Perform drawdown analysis
        drawdown_analysis = self._analyze_drawdowns(backtest_results)
        
        # Calculate rolling metrics
        rolling_metrics = self._calculate_rolling_metrics(backtest_results)
        
        # Performance attribution
        performance_attribution = self._calculate_performance_attribution(backtest_results)
        
        # Generate conclusion
        conclusion = self._generate_performance_conclusion(
            strategy_metrics, benchmark_metrics, relative_metrics, strategy_name
        )
        
        comparison = PerformanceComparison(
            strategy_metrics=strategy_metrics,
            benchmark_metrics=benchmark_metrics,
            relative_metrics=relative_metrics,
            risk_adjusted_metrics=risk_adjusted_metrics,
            drawdown_analysis=drawdown_analysis,
            rolling_metrics=rolling_metrics,
            performance_attribution=performance_attribution,
            conclusion=conclusion
        )
        
        logger.info("Performance analysis completed")
        return comparison
    
    def analyze_risk(self, backtest_results: BacktestResults) -> RiskAnalysis:
        """
        Perform comprehensive risk analysis.
        
        Args:
            backtest_results: Results from backtesting
            
        Returns:
            RiskAnalysis containing detailed risk metrics
        """
        logger.info("Performing risk analysis")
        
        # Volatility analysis
        volatility_analysis = self._analyze_volatility(backtest_results)
        
        # Drawdown analysis
        drawdown_analysis = self._analyze_risk_drawdowns(backtest_results)
        
        # Value at Risk analysis
        var_analysis = self._analyze_var(backtest_results)
        
        # Correlation analysis
        correlation_analysis = self._analyze_correlations(backtest_results)
        
        # Risk contribution analysis
        risk_contribution = self._analyze_risk_contribution(backtest_results)
        
        # Generate risk summary
        risk_summary = self._generate_risk_summary(
            volatility_analysis, drawdown_analysis, var_analysis
        )
        
        risk_analysis = RiskAnalysis(
            volatility_analysis=volatility_analysis,
            drawdown_analysis=drawdown_analysis,
            var_analysis=var_analysis,
            correlation_analysis=correlation_analysis,
            risk_contribution=risk_contribution,
            risk_summary=risk_summary
        )
        
        logger.info("Risk analysis completed")
        return risk_analysis
    
    def create_performance_report(self, 
                                comparison: PerformanceComparison,
                                risk_analysis: RiskAnalysis,
                                strategy_name: str = "Strategy") -> str:
        """
        Create comprehensive performance report.
        
        Args:
            comparison: Performance comparison results
            risk_analysis: Risk analysis results
            strategy_name: Name of the strategy
            
        Returns:
            Formatted performance report string
        """
        report = f"""
PORTFOLIO PERFORMANCE ANALYSIS REPORT
Strategy: {strategy_name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*60}
PERFORMANCE SUMMARY
{'='*60}

Strategy Performance:
- Total Return: {comparison.strategy_metrics['Total_Return']:.2%}
- Annualized Return: {comparison.strategy_metrics['Annualized_Return']:.2%}
- Volatility: {comparison.strategy_metrics['Volatility']:.2%}
- Sharpe Ratio: {comparison.strategy_metrics['Sharpe_Ratio']:.3f}
- Maximum Drawdown: {comparison.strategy_metrics['Max_Drawdown']:.2%}

Benchmark Performance:
- Total Return: {comparison.benchmark_metrics['Total_Return']:.2%}
- Annualized Return: {comparison.benchmark_metrics['Annualized_Return']:.2%}
- Volatility: {comparison.benchmark_metrics['Volatility']:.2%}
- Sharpe Ratio: {comparison.benchmark_metrics['Sharpe_Ratio']:.3f}
- Maximum Drawdown: {comparison.benchmark_metrics['Max_Drawdown']:.2%}

{'='*60}
RELATIVE PERFORMANCE
{'='*60}

- Alpha: {comparison.relative_metrics['Alpha']:.2%}
- Beta: {comparison.relative_metrics['Beta']:.3f}
- Information Ratio: {comparison.relative_metrics['Information_Ratio']:.3f}
- Tracking Error: {comparison.relative_metrics['Tracking_Error']:.2%}
- Outperformance Rate: {comparison.relative_metrics['Outperformance_Rate']:.1%}

{'='*60}
RISK ANALYSIS
{'='*60}

{risk_analysis.risk_summary}

Value at Risk (95%): {risk_analysis.var_analysis['VaR_95']:.2%}
Conditional VaR (95%): {risk_analysis.var_analysis['CVaR_95']:.2%}

{'='*60}
CONCLUSION
{'='*60}

{comparison.conclusion}

{'='*60}
"""
        
        return report.strip()
    
    def visualize_performance(self, 
                            backtest_results: BacktestResults,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive performance visualization with enhanced clarity and insights.
        
        Args:
            backtest_results: Backtesting results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Portfolio Performance Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        # Plot 1: Wealth growth comparison with key metrics
        ax1 = axes[0, 0]
        portfolio_series = backtest_results.portfolio_series
        benchmark_series = backtest_results.benchmark_series
        
        # Normalize to starting value for comparison
        portfolio_normalized = portfolio_series / portfolio_series.iloc[0]
        benchmark_normalized = benchmark_series / benchmark_series.iloc[0]
        
        # Calculate key metrics for labels
        portfolio_total_return = (portfolio_normalized.iloc[-1] - 1) * 100
        benchmark_total_return = (benchmark_normalized.iloc[-1] - 1) * 100
        outperformance = portfolio_total_return - benchmark_total_return
        
        ax1.plot(portfolio_normalized.index, portfolio_normalized.values, 
                label=f'Strategy ({portfolio_total_return:+.1f}%)', 
                linewidth=3, color='#2E86AB', alpha=0.9)
        ax1.plot(benchmark_normalized.index, benchmark_normalized.values, 
                label=f'Benchmark ({benchmark_total_return:+.1f}%)', 
                linewidth=3, color='#A23B72', alpha=0.8)
        
        # Add performance annotation
        ax1.text(0.02, 0.98, f'Outperformance: {outperformance:+.1f}%', 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top')
        
        ax1.axhline(y=1, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax1.set_title('Cumulative Wealth Growth ($1 Initial Investment)', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Time Period', fontsize=12)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Risk-adjusted performance evolution
        ax2 = axes[0, 1]
        rolling_metrics = self._calculate_rolling_metrics(backtest_results, window=60)
        
        if not rolling_metrics.empty:
            strategy_avg_sharpe = rolling_metrics['Portfolio_Sharpe'].mean()
            benchmark_avg_sharpe = rolling_metrics['Benchmark_Sharpe'].mean()
            
            ax2.plot(rolling_metrics.index, rolling_metrics['Portfolio_Sharpe'], 
                    label=f'Strategy (Avg: {strategy_avg_sharpe:.2f})', 
                    linewidth=3, color='#2E86AB')
            ax2.plot(rolling_metrics.index, rolling_metrics['Benchmark_Sharpe'], 
                    label=f'Benchmark (Avg: {benchmark_avg_sharpe:.2f})', 
                    linewidth=3, color='#A23B72', alpha=0.8)
            
            # Add performance threshold lines
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            ax2.axhline(y=1, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Good Performance')
            ax2.axhline(y=2, color='darkgreen', linestyle='--', alpha=0.7, linewidth=1, label='Excellent Performance')
        
        ax2.set_title('Risk-Adjusted Performance Evolution (60-Day Rolling Sharpe)', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Time Period', fontsize=12)
        ax2.set_ylabel('Sharpe Ratio', fontsize=12)
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Drawdown comparison with recovery analysis
        ax3 = axes[1, 0]
        drawdowns = self._calculate_drawdown_series(backtest_results.portfolio_series) * 100
        benchmark_drawdowns = self._calculate_drawdown_series(backtest_results.benchmark_series) * 100
        
        # Calculate max drawdowns for labels
        max_strategy_dd = drawdowns.min()
        max_benchmark_dd = benchmark_drawdowns.min()
        
        ax3.fill_between(drawdowns.index, drawdowns.values, 0, 
                        alpha=0.4, color='#2E86AB', 
                        label=f'Strategy (Max: {max_strategy_dd:.1f}%)')
        ax3.fill_between(benchmark_drawdowns.index, benchmark_drawdowns.values, 0, 
                        alpha=0.4, color='#A23B72', 
                        label=f'Benchmark (Max: {max_benchmark_dd:.1f}%)')
        
        # Add drawdown severity reference lines
        ax3.axhline(y=-5, color='orange', linestyle='--', alpha=0.7, linewidth=1, label='Moderate Risk (-5%)')
        ax3.axhline(y=-10, color='red', linestyle='--', alpha=0.7, linewidth=1, label='High Risk (-10%)')
        
        ax3.set_title('Drawdown Analysis & Risk Assessment', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Time Period', fontsize=12)
        ax3.set_ylabel('Drawdown (%)', fontsize=12)
        ax3.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Return distribution comparison with statistics
        ax4 = axes[1, 1]
        monthly_returns = self._calculate_monthly_returns(backtest_results)
        
        if not monthly_returns.empty:
            # Calculate statistics for annotations
            strategy_mean = monthly_returns['Portfolio'].mean()
            strategy_std = monthly_returns['Portfolio'].std()
            benchmark_mean = monthly_returns['Benchmark'].mean()
            benchmark_std = monthly_returns['Benchmark'].std()
            
            ax4.hist(monthly_returns['Portfolio'], bins=25, alpha=0.7, 
                    label=f'Strategy (μ={strategy_mean:.1f}%, σ={strategy_std:.1f}%)', 
                    color='#2E86AB', density=True, edgecolor='white')
            ax4.hist(monthly_returns['Benchmark'], bins=25, alpha=0.7, 
                    label=f'Benchmark (μ={benchmark_mean:.1f}%, σ={benchmark_std:.1f}%)', 
                    color='#A23B72', density=True, edgecolor='white')
            
            # Add mean lines
            ax4.axvline(x=strategy_mean, color='#2E86AB', linestyle='--', linewidth=2, alpha=0.8)
            ax4.axvline(x=benchmark_mean, color='#A23B72', linestyle='--', linewidth=2, alpha=0.8)
        
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax4.set_title('Monthly Return Distribution Analysis', fontsize=14, fontweight='bold', pad=15)
        ax4.set_xlabel('Monthly Return (%)', fontsize=12)
        ax4.set_ylabel('Probability Density', fontsize=12)
        ax4.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout(pad=3.0)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Performance visualization saved to {save_path}")
        
        return fig
    
    def _calculate_risk_adjusted_metrics(self, backtest_results: BacktestResults) -> Dict[str, float]:
        """Calculate additional risk-adjusted performance metrics."""
        
        portfolio_returns = backtest_results.returns_series.dropna()
        benchmark_returns = backtest_results.benchmark_returns_series.dropna()
        
        # Sortino ratio (using downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (portfolio_returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        annual_return = backtest_results.performance_metrics['Annualized_Return']
        max_drawdown = abs(backtest_results.performance_metrics['Max_Drawdown'])
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Treynor ratio
        beta = backtest_results.comparison_metrics['Beta']
        risk_free_rate = 0.02
        treynor_ratio = (annual_return - risk_free_rate) / beta if beta != 0 else 0
        
        return {
            'Sortino_Ratio': sortino_ratio,
            'Calmar_Ratio': calmar_ratio,
            'Treynor_Ratio': treynor_ratio
        }
    
    def _analyze_drawdowns(self, backtest_results: BacktestResults) -> Dict[str, Any]:
        """Analyze drawdown characteristics."""
        
        portfolio_series = backtest_results.portfolio_series
        drawdowns = self._calculate_drawdown_series(portfolio_series)
        
        # Find drawdown periods
        in_drawdown = drawdowns < 0
        drawdown_periods = []
        
        start_idx = None
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                end_idx = i - 1
                period_drawdown = drawdowns.iloc[start_idx:end_idx+1]
                drawdown_periods.append({
                    'start': drawdowns.index[start_idx],
                    'end': drawdowns.index[end_idx],
                    'duration': end_idx - start_idx + 1,
                    'max_drawdown': period_drawdown.min(),
                    'recovery_time': None  # Could be calculated if needed
                })
                start_idx = None
        
        # Summary statistics
        if drawdown_periods:
            avg_duration = np.mean([dd['duration'] for dd in drawdown_periods])
            avg_drawdown = np.mean([dd['max_drawdown'] for dd in drawdown_periods])
            max_duration = max([dd['duration'] for dd in drawdown_periods])
        else:
            avg_duration = 0
            avg_drawdown = 0
            max_duration = 0
        
        return {
            'drawdown_periods': drawdown_periods,
            'num_drawdown_periods': len(drawdown_periods),
            'avg_drawdown_duration': avg_duration,
            'avg_drawdown_magnitude': avg_drawdown,
            'max_drawdown_duration': max_duration,
            'time_in_drawdown': (drawdowns < 0).sum() / len(drawdowns)
        }
    
    def _calculate_rolling_metrics(self, 
                                 backtest_results: BacktestResults,
                                 window: int = 30) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        
        portfolio_returns = backtest_results.returns_series
        benchmark_returns = backtest_results.benchmark_returns_series
        
        # Rolling Sharpe ratios
        portfolio_rolling_sharpe = (
            portfolio_returns.rolling(window).mean() * np.sqrt(252) /
            (portfolio_returns.rolling(window).std() * np.sqrt(252))
        )
        
        benchmark_rolling_sharpe = (
            benchmark_returns.rolling(window).mean() * np.sqrt(252) /
            (benchmark_returns.rolling(window).std() * np.sqrt(252))
        )
        
        # Rolling volatility
        portfolio_rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
        benchmark_rolling_vol = benchmark_returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling correlation
        rolling_correlation = portfolio_returns.rolling(window).corr(benchmark_returns)
        
        rolling_metrics = pd.DataFrame({
            'Portfolio_Sharpe': portfolio_rolling_sharpe,
            'Benchmark_Sharpe': benchmark_rolling_sharpe,
            'Portfolio_Volatility': portfolio_rolling_vol,
            'Benchmark_Volatility': benchmark_rolling_vol,
            'Correlation': rolling_correlation
        })
        
        return rolling_metrics.dropna()
    
    def _calculate_performance_attribution(self, backtest_results: BacktestResults) -> Dict[str, float]:
        """Calculate performance attribution analysis."""
        
        # This is a simplified attribution analysis
        # In practice, this would be more sophisticated
        
        total_return = backtest_results.performance_metrics['Total_Return']
        benchmark_return = backtest_results.benchmark_metrics['Total_Return']
        
        # Excess return
        excess_return = total_return - benchmark_return
        
        # Attribution components (simplified)
        alpha_contribution = backtest_results.comparison_metrics['Alpha'] / 252 * len(backtest_results.portfolio_snapshots)
        beta_contribution = (backtest_results.comparison_metrics['Beta'] - 1) * benchmark_return
        
        return {
            'Total_Excess_Return': excess_return,
            'Alpha_Contribution': alpha_contribution,
            'Beta_Contribution': beta_contribution,
            'Selection_Effect': excess_return - alpha_contribution - beta_contribution
        }
    
    def _generate_performance_conclusion(self, 
                                       strategy_metrics: Dict[str, float],
                                       benchmark_metrics: Dict[str, float],
                                       relative_metrics: Dict[str, float],
                                       strategy_name: str) -> str:
        """Generate performance conclusion."""
        
        strategy_return = strategy_metrics['Total_Return']
        benchmark_return = benchmark_metrics['Total_Return']
        strategy_sharpe = strategy_metrics['Sharpe_Ratio']
        benchmark_sharpe = benchmark_metrics['Sharpe_Ratio']
        alpha = relative_metrics['Alpha']
        
        conclusion_parts = []
        
        # Return comparison
        if strategy_return > benchmark_return:
            outperformance = strategy_return - benchmark_return
            conclusion_parts.append(
                f"{strategy_name} outperformed the benchmark by {outperformance:.2%} "
                f"({strategy_return:.2%} vs {benchmark_return:.2%})."
            )
        else:
            underperformance = benchmark_return - strategy_return
            conclusion_parts.append(
                f"{strategy_name} underperformed the benchmark by {underperformance:.2%} "
                f"({strategy_return:.2%} vs {benchmark_return:.2%})."
            )
        
        # Risk-adjusted performance
        if strategy_sharpe > benchmark_sharpe:
            conclusion_parts.append(
                f"On a risk-adjusted basis, {strategy_name} performed better with a Sharpe ratio of "
                f"{strategy_sharpe:.3f} compared to the benchmark's {benchmark_sharpe:.3f}."
            )
        else:
            conclusion_parts.append(
                f"On a risk-adjusted basis, {strategy_name} performed worse with a Sharpe ratio of "
                f"{strategy_sharpe:.3f} compared to the benchmark's {benchmark_sharpe:.3f}."
            )
        
        # Alpha assessment
        if alpha > 0.02:  # 2% alpha threshold
            conclusion_parts.append(
                f"The strategy generated significant alpha of {alpha:.2%}, indicating strong active management value."
            )
        elif alpha > 0:
            conclusion_parts.append(
                f"The strategy generated modest alpha of {alpha:.2%}."
            )
        else:
            conclusion_parts.append(
                f"The strategy generated negative alpha of {alpha:.2%}, suggesting it did not add value over passive management."
            )
        
        # Overall assessment
        if strategy_return > benchmark_return and strategy_sharpe > benchmark_sharpe:
            conclusion_parts.append("Overall, the strategy demonstrates effective active management.")
        elif strategy_return > benchmark_return:
            conclusion_parts.append("The strategy achieved higher returns but with increased risk.")
        elif strategy_sharpe > benchmark_sharpe:
            conclusion_parts.append("While returns were lower, the strategy achieved better risk-adjusted performance.")
        else:
            conclusion_parts.append("The strategy did not demonstrate clear advantages over the benchmark.")
        
        return " ".join(conclusion_parts)
    
    def _analyze_volatility(self, backtest_results: BacktestResults) -> Dict[str, float]:
        """Analyze volatility characteristics."""
        
        portfolio_returns = backtest_results.returns_series.dropna()
        
        # Basic volatility metrics
        daily_vol = portfolio_returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Upside/downside volatility
        upside_returns = portfolio_returns[portfolio_returns > 0]
        downside_returns = portfolio_returns[portfolio_returns < 0]
        
        upside_vol = upside_returns.std() * np.sqrt(252) if len(upside_returns) > 0 else 0
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        return {
            'Daily_Volatility': daily_vol,
            'Annual_Volatility': annual_vol,
            'Upside_Volatility': upside_vol,
            'Downside_Volatility': downside_vol,
            'Volatility_Ratio': upside_vol / downside_vol if downside_vol > 0 else 0
        }
    
    def _analyze_risk_drawdowns(self, backtest_results: BacktestResults) -> Dict[str, float]:
        """Analyze drawdown risk characteristics."""
        
        portfolio_series = backtest_results.portfolio_series
        drawdowns = self._calculate_drawdown_series(portfolio_series)
        
        return {
            'Max_Drawdown': drawdowns.min(),
            'Average_Drawdown': drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0,
            'Drawdown_Volatility': drawdowns.std(),
            'Time_Underwater': (drawdowns < 0).sum() / len(drawdowns)
        }
    
    def _analyze_var(self, backtest_results: BacktestResults) -> Dict[str, float]:
        """Analyze Value at Risk metrics."""
        
        portfolio_returns = backtest_results.returns_series.dropna()
        
        # Historical VaR
        var_95 = portfolio_returns.quantile(0.05)
        var_99 = portfolio_returns.quantile(0.01)
        
        # Conditional VaR
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        return {
            'VaR_95': var_95,
            'VaR_99': var_99,
            'CVaR_95': cvar_95,
            'CVaR_99': cvar_99
        }
    
    def _analyze_correlations(self, backtest_results: BacktestResults) -> Dict[str, float]:
        """Analyze correlation characteristics."""
        
        portfolio_returns = backtest_results.returns_series.dropna()
        benchmark_returns = backtest_results.benchmark_returns_series.dropna()
        
        # Overall correlation
        correlation = portfolio_returns.corr(benchmark_returns)
        
        # Rolling correlation statistics
        rolling_corr = portfolio_returns.rolling(30).corr(benchmark_returns).dropna()
        
        return {
            'Overall_Correlation': correlation,
            'Average_Rolling_Correlation': rolling_corr.mean(),
            'Correlation_Volatility': rolling_corr.std(),
            'Min_Correlation': rolling_corr.min(),
            'Max_Correlation': rolling_corr.max()
        }
    
    def _analyze_risk_contribution(self, backtest_results: BacktestResults) -> Dict[str, float]:
        """Analyze risk contribution by component."""
        
        # This is a simplified risk contribution analysis
        # In practice, this would require position-level data
        
        portfolio_vol = backtest_results.performance_metrics['Volatility']
        benchmark_vol = backtest_results.benchmark_metrics['Volatility']
        
        return {
            'Total_Risk': portfolio_vol,
            'Systematic_Risk': benchmark_vol * backtest_results.comparison_metrics['Beta'],
            'Idiosyncratic_Risk': np.sqrt(max(0, portfolio_vol**2 - (benchmark_vol * backtest_results.comparison_metrics['Beta'])**2))
        }
    
    def _generate_risk_summary(self, 
                             volatility_analysis: Dict[str, float],
                             drawdown_analysis: Dict[str, float],
                             var_analysis: Dict[str, float]) -> str:
        """Generate risk analysis summary."""
        
        vol = volatility_analysis['Annual_Volatility']
        max_dd = abs(drawdown_analysis['Max_Drawdown'])
        var_95 = abs(var_analysis['VaR_95'])
        
        risk_level = "Low" if vol < 0.15 else "Medium" if vol < 0.25 else "High"
        
        summary = f"""
Risk Level: {risk_level}
The portfolio exhibits {vol:.1%} annual volatility with a maximum drawdown of {max_dd:.1%}.
Daily Value at Risk (95% confidence) is {var_95:.1%}, indicating potential daily losses.
"""
        
        return summary.strip()
    
    def _calculate_drawdown_series(self, price_series: pd.Series) -> pd.Series:
        """Calculate drawdown series from price series."""
        rolling_max = price_series.expanding().max()
        drawdowns = (price_series - rolling_max) / rolling_max
        return drawdowns
    
    def _calculate_monthly_returns(self, backtest_results: BacktestResults) -> pd.DataFrame:
        """Calculate monthly returns for distribution analysis."""
        
        portfolio_series = backtest_results.portfolio_series
        benchmark_series = backtest_results.benchmark_series
        
        # Resample to monthly
        portfolio_monthly = portfolio_series.resample('M').last().pct_change().dropna() * 100
        benchmark_monthly = benchmark_series.resample('M').last().pct_change().dropna() * 100
        
        return pd.DataFrame({
            'Portfolio': portfolio_monthly,
            'Benchmark': benchmark_monthly
        })