"""
Portfolio recommendation system.

This module provides comprehensive portfolio recommendation capabilities including
portfolio selection logic, performance analysis, and recommendation report generation.
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

from .optimizer import PortfolioOptimizer, PortfolioMetrics, EfficientFrontierData, OptimizationConfig
from ..forecasting import ForecastOutput

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Enumeration for recommendation types."""
    CONSERVATIVE = "Conservative"
    BALANCED = "Balanced"
    AGGRESSIVE = "Aggressive"
    INCOME_FOCUSED = "Income Focused"
    GROWTH_FOCUSED = "Growth Focused"


class RiskProfile(Enum):
    """Enumeration for risk profiles."""
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"


@dataclass
class RecommendationCriteria:
    """Criteria for portfolio recommendation."""
    
    risk_tolerance: RiskProfile = RiskProfile.MODERATE
    return_target: Optional[float] = None
    volatility_limit: Optional[float] = None
    min_sharpe_ratio: float = 0.5
    max_concentration: float = 0.6  # Maximum weight in single asset
    prefer_diversification: bool = True
    income_requirement: Optional[float] = None
    time_horizon_years: int = 5


@dataclass
class PortfolioRecommendation:
    """Container for portfolio recommendation."""
    
    recommendation_type: RecommendationType
    portfolio_metrics: PortfolioMetrics
    justification: str
    risk_assessment: str
    expected_performance: Dict[str, float]
    allocation_breakdown: Dict[str, Dict[str, Union[float, str]]]
    key_benefits: List[str]
    key_risks: List[str]
    suitability_score: float  # 0-100 scale
    alternative_options: List['PortfolioRecommendation'] = None


@dataclass
class RecommendationReport:
    """Comprehensive recommendation report."""
    
    primary_recommendation: PortfolioRecommendation
    alternative_recommendations: List[PortfolioRecommendation]
    market_outlook: str
    risk_warnings: List[str]
    implementation_guidance: List[str]
    monitoring_recommendations: List[str]
    report_summary: str
    generation_timestamp: pd.Timestamp


class PortfolioRecommender:
    """
    Comprehensive portfolio recommendation system.
    
    This class provides methods to identify optimal portfolios, generate
    recommendations with justifications, and create comprehensive reports
    for different investor profiles and objectives.
    """
    
    def __init__(self, optimizer: PortfolioOptimizer):
        """
        Initialize portfolio recommender.
        
        Args:
            optimizer: Configured portfolio optimizer
        """
        self.optimizer = optimizer
        self.efficient_frontier_data = None
        self.forecast_data = None
        
        logger.info("PortfolioRecommender initialized")
    
    def set_forecast_data(self, forecast_data: Dict[str, ForecastOutput]) -> None:
        """
        Set forecast data for enhanced recommendations.
        
        Args:
            forecast_data: Dictionary mapping asset symbols to forecast outputs
        """
        self.forecast_data = forecast_data
        logger.info(f"Set forecast data for {len(forecast_data)} assets")
    
    def generate_recommendations(self, 
                               criteria: RecommendationCriteria,
                               num_alternatives: int = 2) -> RecommendationReport:
        """
        Generate comprehensive portfolio recommendations.
        
        Args:
            criteria: Recommendation criteria and constraints
            num_alternatives: Number of alternative recommendations to generate
            
        Returns:
            RecommendationReport containing primary and alternative recommendations
        """
        logger.info("Generating portfolio recommendations")
        
        # Generate efficient frontier if not already done
        if self.efficient_frontier_data is None:
            self.efficient_frontier_data = self.optimizer.generate_efficient_frontier()
        
        # Generate primary recommendation
        primary_recommendation = self._generate_primary_recommendation(criteria)
        
        # Generate alternative recommendations
        alternative_recommendations = self._generate_alternative_recommendations(
            criteria, num_alternatives
        )
        
        # Generate market outlook
        market_outlook = self._generate_market_outlook()
        
        # Generate risk warnings
        risk_warnings = self._generate_risk_warnings(primary_recommendation, criteria)
        
        # Generate implementation guidance
        implementation_guidance = self._generate_implementation_guidance(primary_recommendation)
        
        # Generate monitoring recommendations
        monitoring_recommendations = self._generate_monitoring_recommendations()
        
        # Generate report summary
        report_summary = self._generate_report_summary(
            primary_recommendation, alternative_recommendations, criteria
        )
        
        report = RecommendationReport(
            primary_recommendation=primary_recommendation,
            alternative_recommendations=alternative_recommendations,
            market_outlook=market_outlook,
            risk_warnings=risk_warnings,
            implementation_guidance=implementation_guidance,
            monitoring_recommendations=monitoring_recommendations,
            report_summary=report_summary,
            generation_timestamp=pd.Timestamp.now()
        )
        
        logger.info("Portfolio recommendations generated successfully")
        return report
    
    def _generate_primary_recommendation(self, 
                                       criteria: RecommendationCriteria) -> PortfolioRecommendation:
        """
        Generate the primary portfolio recommendation.
        
        Args:
            criteria: Recommendation criteria
            
        Returns:
            Primary PortfolioRecommendation
        """
        # Determine optimization approach based on criteria
        if criteria.risk_tolerance == RiskProfile.LOW:
            # Conservative approach - minimize volatility
            portfolio_metrics = self.optimizer.optimize_portfolio(method="min_volatility")
            recommendation_type = RecommendationType.CONSERVATIVE
            
        elif criteria.return_target is not None:
            # Target return approach
            portfolio_metrics = self.optimizer.optimize_portfolio(
                method="efficient_return", 
                target_return=criteria.return_target
            )
            recommendation_type = RecommendationType.BALANCED
            
        else:
            # Default to maximum Sharpe ratio
            portfolio_metrics = self.optimizer.optimize_portfolio(method="max_sharpe")
            
            # Determine recommendation type based on portfolio characteristics
            if portfolio_metrics.volatility < 0.15:
                recommendation_type = RecommendationType.CONSERVATIVE
            elif portfolio_metrics.volatility > 0.25:
                recommendation_type = RecommendationType.AGGRESSIVE
            else:
                recommendation_type = RecommendationType.BALANCED
        
        # Check concentration limits
        max_weight = max(portfolio_metrics.weights.values())
        if max_weight > criteria.max_concentration:
            # Reoptimize with concentration constraint
            adjusted_config = OptimizationConfig(
                weight_bounds=(0.0, criteria.max_concentration),
                optimization_method="max_sharpe"
            )
            adjusted_optimizer = PortfolioOptimizer(adjusted_config)
            adjusted_optimizer.assets = self.optimizer.assets
            adjusted_optimizer.price_data = self.optimizer.price_data
            adjusted_optimizer.expected_returns = self.optimizer.expected_returns
            adjusted_optimizer.covariance_matrix = self.optimizer.covariance_matrix
            
            portfolio_metrics = adjusted_optimizer.optimize_portfolio()
        
        # Generate justification
        justification = self._generate_justification(portfolio_metrics, criteria, recommendation_type)
        
        # Generate risk assessment
        risk_assessment = self._generate_risk_assessment(portfolio_metrics, criteria)
        
        # Calculate expected performance
        expected_performance = self._calculate_expected_performance(portfolio_metrics, criteria)
        
        # Create allocation breakdown
        allocation_breakdown = self._create_allocation_breakdown(portfolio_metrics)
        
        # Identify key benefits and risks
        key_benefits = self._identify_key_benefits(portfolio_metrics, recommendation_type)
        key_risks = self._identify_key_risks(portfolio_metrics, criteria)
        
        # Calculate suitability score
        suitability_score = self._calculate_suitability_score(portfolio_metrics, criteria)
        
        recommendation = PortfolioRecommendation(
            recommendation_type=recommendation_type,
            portfolio_metrics=portfolio_metrics,
            justification=justification,
            risk_assessment=risk_assessment,
            expected_performance=expected_performance,
            allocation_breakdown=allocation_breakdown,
            key_benefits=key_benefits,
            key_risks=key_risks,
            suitability_score=suitability_score
        )
        
        return recommendation
    
    def _generate_alternative_recommendations(self, 
                                            criteria: RecommendationCriteria,
                                            num_alternatives: int) -> List[PortfolioRecommendation]:
        """
        Generate alternative portfolio recommendations.
        
        Args:
            criteria: Recommendation criteria
            num_alternatives: Number of alternatives to generate
            
        Returns:
            List of alternative recommendations
        """
        alternatives = []
        
        # Alternative 1: Minimum volatility portfolio
        if num_alternatives >= 1:
            min_vol_metrics = self.efficient_frontier_data.min_volatility_portfolio
            
            alternative = PortfolioRecommendation(
                recommendation_type=RecommendationType.CONSERVATIVE,
                portfolio_metrics=min_vol_metrics,
                justification="Minimum risk portfolio for capital preservation",
                risk_assessment="Lowest volatility option with reduced return expectations",
                expected_performance=self._calculate_expected_performance(min_vol_metrics, criteria),
                allocation_breakdown=self._create_allocation_breakdown(min_vol_metrics),
                key_benefits=["Lowest volatility", "Capital preservation", "Stable returns"],
                key_risks=["Lower expected returns", "Inflation risk"],
                suitability_score=self._calculate_suitability_score(min_vol_metrics, criteria) * 0.9
            )
            alternatives.append(alternative)
        
        # Alternative 2: Maximum Sharpe ratio portfolio
        if num_alternatives >= 2:
            max_sharpe_metrics = self.efficient_frontier_data.max_sharpe_portfolio
            
            alternative = PortfolioRecommendation(
                recommendation_type=RecommendationType.BALANCED,
                portfolio_metrics=max_sharpe_metrics,
                justification="Optimal risk-adjusted return portfolio",
                risk_assessment="Balanced risk-return profile with good diversification",
                expected_performance=self._calculate_expected_performance(max_sharpe_metrics, criteria),
                allocation_breakdown=self._create_allocation_breakdown(max_sharpe_metrics),
                key_benefits=["Optimal risk-adjusted returns", "Good diversification", "Efficient allocation"],
                key_risks=["Market volatility", "Correlation risk"],
                suitability_score=self._calculate_suitability_score(max_sharpe_metrics, criteria) * 0.95
            )
            alternatives.append(alternative)
        
        return alternatives
    
    def _generate_justification(self, 
                              portfolio_metrics: PortfolioMetrics,
                              criteria: RecommendationCriteria,
                              recommendation_type: RecommendationType) -> str:
        """Generate justification for the recommendation."""
        
        justification_parts = []
        
        # Risk-return justification
        justification_parts.append(
            f"This {recommendation_type.value.lower()} portfolio offers an expected annual return of "
            f"{portfolio_metrics.expected_return:.1%} with a volatility of {portfolio_metrics.volatility:.1%}, "
            f"resulting in a Sharpe ratio of {portfolio_metrics.sharpe_ratio:.2f}."
        )
        
        # Diversification justification
        if portfolio_metrics.diversification_ratio > 1.2:
            justification_parts.append(
                f"The portfolio benefits from good diversification (ratio: {portfolio_metrics.diversification_ratio:.2f}), "
                "which helps reduce overall risk through asset correlation benefits."
            )
        
        # Asset allocation justification
        max_weight_asset = max(portfolio_metrics.weights.items(), key=lambda x: x[1])
        justification_parts.append(
            f"The largest allocation is {max_weight_asset[1]:.1%} to {max_weight_asset[0]}, "
            "maintaining appropriate concentration limits while optimizing returns."
        )
        
        # Forecast-based justification
        if self.forecast_data:
            forecast_assets = set(self.forecast_data.keys()) & set(portfolio_metrics.weights.keys())
            if forecast_assets:
                justification_parts.append(
                    f"The allocation incorporates forward-looking forecasts for {len(forecast_assets)} assets, "
                    "enhancing the expected return estimates beyond historical averages."
                )
        
        return " ".join(justification_parts)
    
    def _generate_risk_assessment(self, 
                                portfolio_metrics: PortfolioMetrics,
                                criteria: RecommendationCriteria) -> str:
        """Generate risk assessment for the portfolio."""
        
        risk_parts = []
        
        # Volatility assessment
        if portfolio_metrics.volatility < 0.15:
            risk_level = "low"
        elif portfolio_metrics.volatility < 0.25:
            risk_level = "moderate"
        else:
            risk_level = "high"
        
        risk_parts.append(
            f"The portfolio exhibits {risk_level} risk with an expected volatility of "
            f"{portfolio_metrics.volatility:.1%} annually."
        )
        
        # Concentration risk
        max_weight = max(portfolio_metrics.weights.values())
        if max_weight > 0.5:
            risk_parts.append(
                f"There is moderate concentration risk with {max_weight:.1%} allocated to a single asset."
            )
        elif max_weight < 0.4:
            risk_parts.append("Concentration risk is well-managed with no single asset dominating the portfolio.")
        
        # Market risk
        risk_parts.append(
            "The portfolio is subject to general market risk, and performance will vary with market conditions."
        )
        
        return " ".join(risk_parts)
    
    def _calculate_expected_performance(self, 
                                      portfolio_metrics: PortfolioMetrics,
                                      criteria: RecommendationCriteria) -> Dict[str, float]:
        """Calculate expected performance metrics."""
        
        performance = {
            'annual_return': portfolio_metrics.expected_return,
            'annual_volatility': portfolio_metrics.volatility,
            'sharpe_ratio': portfolio_metrics.sharpe_ratio,
            'diversification_ratio': portfolio_metrics.diversification_ratio
        }
        
        # Calculate time horizon projections
        if criteria.time_horizon_years > 1:
            # Compound annual growth
            total_expected_return = (1 + portfolio_metrics.expected_return) ** criteria.time_horizon_years - 1
            performance[f'{criteria.time_horizon_years}_year_total_return'] = total_expected_return
            
            # Probability of positive returns (simplified normal approximation)
            annual_return = portfolio_metrics.expected_return
            annual_vol = portfolio_metrics.volatility
            z_score = annual_return / annual_vol if annual_vol > 0 else 0
            prob_positive = 0.5 + 0.5 * np.sign(z_score) * min(abs(z_score) / 2, 0.5)
            performance['probability_positive_return'] = prob_positive
        
        return performance
    
    def _create_allocation_breakdown(self, 
                                   portfolio_metrics: PortfolioMetrics) -> Dict[str, Dict[str, Union[float, str]]]:
        """Create detailed allocation breakdown."""
        
        breakdown = {}
        
        for asset, weight in portfolio_metrics.weights.items():
            if weight > 0.001:  # Only include meaningful allocations
                asset_info = {
                    'weight': weight,
                    'weight_pct': f"{weight:.1%}",
                    'expected_contribution': weight * self.optimizer.expected_returns[asset] if self.optimizer.expected_returns is not None else 0
                }
                
                # Add asset classification
                if asset in ['SPY', 'VTI', 'ITOT']:
                    asset_info['asset_class'] = 'US Equity'
                elif asset in ['BND', 'AGG', 'TLT']:
                    asset_info['asset_class'] = 'Fixed Income'
                elif asset == 'TSLA':
                    asset_info['asset_class'] = 'Individual Stock'
                else:
                    asset_info['asset_class'] = 'Other'
                
                breakdown[asset] = asset_info
        
        return breakdown
    
    def _identify_key_benefits(self, 
                             portfolio_metrics: PortfolioMetrics,
                             recommendation_type: RecommendationType) -> List[str]:
        """Identify key benefits of the portfolio."""
        
        benefits = []
        
        # Sharpe ratio benefits
        if portfolio_metrics.sharpe_ratio > 1.0:
            benefits.append("Excellent risk-adjusted returns (Sharpe ratio > 1.0)")
        elif portfolio_metrics.sharpe_ratio > 0.5:
            benefits.append("Good risk-adjusted returns")
        
        # Diversification benefits
        if portfolio_metrics.diversification_ratio > 1.3:
            benefits.append("Strong diversification benefits")
        elif portfolio_metrics.diversification_ratio > 1.1:
            benefits.append("Moderate diversification benefits")
        
        # Return benefits
        if portfolio_metrics.expected_return > 0.12:
            benefits.append("High expected returns")
        elif portfolio_metrics.expected_return > 0.08:
            benefits.append("Solid expected returns")
        
        # Risk benefits
        if portfolio_metrics.volatility < 0.15:
            benefits.append("Low volatility for capital preservation")
        
        # Type-specific benefits
        if recommendation_type == RecommendationType.CONSERVATIVE:
            benefits.append("Capital preservation focus")
        elif recommendation_type == RecommendationType.AGGRESSIVE:
            benefits.append("Growth potential")
        
        return benefits
    
    def _identify_key_risks(self, 
                          portfolio_metrics: PortfolioMetrics,
                          criteria: RecommendationCriteria) -> List[str]:
        """Identify key risks of the portfolio."""
        
        risks = []
        
        # Volatility risks
        if portfolio_metrics.volatility > 0.25:
            risks.append("High volatility may result in significant short-term losses")
        elif portfolio_metrics.volatility > 0.20:
            risks.append("Moderate volatility with potential for periodic losses")
        
        # Concentration risks
        max_weight = max(portfolio_metrics.weights.values())
        if max_weight > 0.6:
            risks.append("High concentration in single asset increases specific risk")
        elif max_weight > 0.4:
            risks.append("Moderate concentration risk")
        
        # Market risks
        risks.append("Subject to general market risk and economic conditions")
        
        # Forecast risks
        if self.forecast_data:
            risks.append("Forecast-based allocations subject to prediction uncertainty")
        
        # Inflation risk
        if portfolio_metrics.expected_return < 0.04:
            risks.append("Low expected returns may not keep pace with inflation")
        
        return risks
    
    def _calculate_suitability_score(self, 
                                   portfolio_metrics: PortfolioMetrics,
                                   criteria: RecommendationCriteria) -> float:
        """Calculate suitability score (0-100) based on criteria match."""
        
        score = 100.0
        
        # Risk tolerance match
        portfolio_risk_level = self._classify_portfolio_risk(portfolio_metrics.volatility)
        if portfolio_risk_level != criteria.risk_tolerance:
            if abs(list(RiskProfile).index(portfolio_risk_level) - list(RiskProfile).index(criteria.risk_tolerance)) == 1:
                score -= 10  # One level difference
            else:
                score -= 25  # Two level difference
        
        # Return target match
        if criteria.return_target is not None:
            return_diff = abs(portfolio_metrics.expected_return - criteria.return_target)
            score -= min(return_diff * 200, 20)  # Penalize return mismatches
        
        # Volatility limit match
        if criteria.volatility_limit is not None and portfolio_metrics.volatility > criteria.volatility_limit:
            excess_vol = portfolio_metrics.volatility - criteria.volatility_limit
            score -= min(excess_vol * 100, 30)
        
        # Sharpe ratio requirement
        if portfolio_metrics.sharpe_ratio < criteria.min_sharpe_ratio:
            sharpe_deficit = criteria.min_sharpe_ratio - portfolio_metrics.sharpe_ratio
            score -= min(sharpe_deficit * 50, 25)
        
        # Concentration limit
        max_weight = max(portfolio_metrics.weights.values())
        if max_weight > criteria.max_concentration:
            excess_concentration = max_weight - criteria.max_concentration
            score -= min(excess_concentration * 50, 15)
        
        return max(score, 0.0)
    
    def _classify_portfolio_risk(self, volatility: float) -> RiskProfile:
        """Classify portfolio risk based on volatility."""
        if volatility < 0.15:
            return RiskProfile.LOW
        elif volatility < 0.25:
            return RiskProfile.MODERATE
        else:
            return RiskProfile.HIGH
    
    def _generate_market_outlook(self) -> str:
        """Generate market outlook based on forecasts."""
        if not self.forecast_data:
            return "Market outlook based on historical analysis suggests continued long-term growth with periodic volatility."
        
        # Analyze forecast trends
        positive_forecasts = 0
        total_forecasts = len(self.forecast_data)
        
        for asset, forecast in self.forecast_data.items():
            if hasattr(forecast, 'forecast_summary') and 'total_forecast_change_pct' in forecast.forecast_summary:
                if forecast.forecast_summary['total_forecast_change_pct'] > 0:
                    positive_forecasts += 1
        
        if positive_forecasts / total_forecasts > 0.6:
            outlook = "Market outlook is generally positive based on current forecasts, with most assets showing upward trends."
        elif positive_forecasts / total_forecasts < 0.4:
            outlook = "Market outlook shows mixed signals with some downward pressure expected in the forecast period."
        else:
            outlook = "Market outlook is neutral with balanced upward and downward expectations across assets."
        
        return outlook
    
    def _generate_risk_warnings(self, 
                              recommendation: PortfolioRecommendation,
                              criteria: RecommendationCriteria) -> List[str]:
        """Generate risk warnings."""
        warnings = []
        
        # High volatility warning
        if recommendation.portfolio_metrics.volatility > 0.25:
            warnings.append("High volatility portfolio: Expect significant short-term fluctuations in value")
        
        # Concentration warning
        max_weight = max(recommendation.portfolio_metrics.weights.values())
        if max_weight > 0.5:
            warnings.append(f"Concentration risk: {max_weight:.1%} allocation to single asset increases specific risk")
        
        # Low diversification warning
        if recommendation.portfolio_metrics.diversification_ratio < 1.1:
            warnings.append("Limited diversification benefits: Portfolio may not reduce risk as effectively")
        
        # Forecast uncertainty warning
        if self.forecast_data:
            warnings.append("Forecast-based allocations: Future performance may differ significantly from predictions")
        
        return warnings
    
    def _generate_implementation_guidance(self, 
                                        recommendation: PortfolioRecommendation) -> List[str]:
        """Generate implementation guidance."""
        guidance = []
        
        # Rebalancing guidance
        guidance.append("Rebalance portfolio quarterly or when allocations drift more than 5% from targets")
        
        # Dollar-cost averaging
        guidance.append("Consider dollar-cost averaging for large initial investments to reduce timing risk")
        
        # Tax considerations
        guidance.append("Implement in tax-advantaged accounts when possible to minimize tax impact")
        
        # Monitoring guidance
        guidance.append("Monitor portfolio performance against benchmarks and adjust if underperforming consistently")
        
        # Asset-specific guidance
        for asset, weight in recommendation.portfolio_metrics.weights.items():
            if weight > 0.3:
                guidance.append(f"Large {asset} allocation: Monitor company/sector-specific news and developments")
        
        return guidance
    
    def _generate_monitoring_recommendations(self) -> List[str]:
        """Generate monitoring recommendations."""
        recommendations = [
            "Review portfolio performance monthly and compare to relevant benchmarks",
            "Reassess risk tolerance and investment objectives annually",
            "Monitor correlation changes between assets, especially during market stress",
            "Track forecast accuracy and adjust methodology if systematic biases emerge",
            "Consider tactical adjustments during extreme market conditions",
            "Maintain emergency cash reserves outside of investment portfolio"
        ]
        
        return recommendations
    
    def _generate_report_summary(self, 
                               primary: PortfolioRecommendation,
                               alternatives: List[PortfolioRecommendation],
                               criteria: RecommendationCriteria) -> str:
        """Generate comprehensive report summary."""
        
        summary = f"""
PORTFOLIO RECOMMENDATION SUMMARY

Primary Recommendation: {primary.recommendation_type.value}
- Expected Return: {primary.portfolio_metrics.expected_return:.1%}
- Volatility: {primary.portfolio_metrics.volatility:.1%}
- Sharpe Ratio: {primary.portfolio_metrics.sharpe_ratio:.2f}
- Suitability Score: {primary.suitability_score:.0f}/100

Key Allocations:
"""
        
        # Add top 3 allocations
        sorted_weights = sorted(primary.portfolio_metrics.weights.items(), 
                              key=lambda x: x[1], reverse=True)[:3]
        for asset, weight in sorted_weights:
            summary += f"- {asset}: {weight:.1%}\n"
        
        summary += f"\nAlternative Options: {len(alternatives)} alternatives provided"
        summary += f"\nRisk Profile Match: {criteria.risk_tolerance.value}"
        
        return summary.strip()