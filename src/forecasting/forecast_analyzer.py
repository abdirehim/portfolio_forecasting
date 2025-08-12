"""
Forecast analysis and insights engine.

This module provides comprehensive analysis of forecast results including trend analysis,
anomaly detection, confidence assessment, and market opportunity identification.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from .forecast_generator import ForecastOutput

logger = logging.getLogger(__name__)


class TrendType(Enum):
    """Enumeration for trend types."""
    STRONG_UPWARD = "Strong Upward"
    MODERATE_UPWARD = "Moderate Upward"
    STABLE = "Stable"
    MODERATE_DOWNWARD = "Moderate Downward"
    STRONG_DOWNWARD = "Strong Downward"


class RiskLevel(Enum):
    """Enumeration for risk levels."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"


@dataclass
class TrendAnalysis:
    """Container for trend analysis results."""
    
    trend_type: TrendType
    trend_strength: float  # 0-100 scale
    trend_consistency: float  # 0-100 scale
    trend_duration_months: float
    key_turning_points: List[pd.Timestamp]
    trend_description: str


@dataclass
class AnomalyDetection:
    """Container for anomaly detection results."""
    
    anomalies_detected: List[Tuple[pd.Timestamp, float, str]]  # (date, value, description)
    anomaly_count: int
    anomaly_severity: RiskLevel
    anomaly_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]  # (start, end) of anomalous periods
    anomaly_description: str


@dataclass
class ConfidenceAssessment:
    """Container for confidence assessment results."""
    
    overall_confidence: RiskLevel
    confidence_score: float  # 0-100 scale
    confidence_factors: Dict[str, float]
    reliability_metrics: Dict[str, float]
    confidence_evolution: pd.Series  # Confidence over time
    assessment_description: str


@dataclass
class MarketOpportunity:
    """Container for market opportunity analysis."""
    
    opportunity_type: str  # "Buy", "Sell", "Hold", "Wait"
    opportunity_strength: float  # 0-100 scale
    optimal_entry_points: List[Tuple[pd.Timestamp, float, str]]  # (date, price, reason)
    risk_factors: List[str]
    potential_return: float
    time_horizon: str
    opportunity_description: str


@dataclass
class ForecastInsights:
    """Container for comprehensive forecast insights."""
    
    trend_analysis: TrendAnalysis
    anomaly_detection: AnomalyDetection
    confidence_assessment: ConfidenceAssessment
    market_opportunities: List[MarketOpportunity]
    risk_assessment: Dict[str, Any]
    key_insights: List[str]
    actionable_recommendations: List[str]
    report_summary: str


class ForecastAnalyzer:
    """
    Comprehensive forecast analysis and insights engine.
    
    This class analyzes forecast results to identify trends, detect anomalies,
    assess confidence levels, and generate actionable market insights.
    """
    
    def __init__(self, 
                 trend_threshold: float = 5.0,
                 anomaly_threshold: float = 2.0,
                 confidence_threshold: float = 0.8):
        """
        Initialize forecast analyzer.
        
        Args:
            trend_threshold: Minimum percentage change to consider significant trend
            anomaly_threshold: Standard deviation threshold for anomaly detection
            confidence_threshold: Minimum confidence level for reliable forecasts
        """
        self.trend_threshold = trend_threshold
        self.anomaly_threshold = anomaly_threshold
        self.confidence_threshold = confidence_threshold
        
        logger.info("ForecastAnalyzer initialized")
    
    def analyze_forecast(self, 
                        forecast_output: ForecastOutput,
                        historical_data: pd.Series) -> ForecastInsights:
        """
        Perform comprehensive analysis of forecast results.
        
        Args:
            forecast_output: Forecast results to analyze
            historical_data: Historical time series data for context
            
        Returns:
            ForecastInsights containing comprehensive analysis
        """
        logger.info("Starting comprehensive forecast analysis")
        
        # Perform individual analyses
        trend_analysis = self._analyze_trends(forecast_output, historical_data)
        anomaly_detection = self._detect_anomalies(forecast_output, historical_data)
        confidence_assessment = self._assess_confidence(forecast_output)
        market_opportunities = self._identify_opportunities(
            forecast_output, historical_data, trend_analysis, confidence_assessment
        )
        risk_assessment = self._assess_risks(
            forecast_output, trend_analysis, anomaly_detection, confidence_assessment
        )
        
        # Generate insights and recommendations
        key_insights = self._generate_key_insights(
            trend_analysis, anomaly_detection, confidence_assessment, market_opportunities
        )
        actionable_recommendations = self._generate_recommendations(
            trend_analysis, market_opportunities, risk_assessment
        )
        report_summary = self._generate_report_summary(
            trend_analysis, confidence_assessment, market_opportunities
        )
        
        insights = ForecastInsights(
            trend_analysis=trend_analysis,
            anomaly_detection=anomaly_detection,
            confidence_assessment=confidence_assessment,
            market_opportunities=market_opportunities,
            risk_assessment=risk_assessment,
            key_insights=key_insights,
            actionable_recommendations=actionable_recommendations,
            report_summary=report_summary
        )
        
        logger.info("Forecast analysis completed")
        return insights
    
    def _analyze_trends(self, 
                       forecast_output: ForecastOutput,
                       historical_data: pd.Series) -> TrendAnalysis:
        """
        Analyze long-term trends in the forecast.
        
        Args:
            forecast_output: Forecast results
            historical_data: Historical data for context
            
        Returns:
            TrendAnalysis results
        """
        predictions = forecast_output.predictions
        
        # Calculate overall trend
        start_value = predictions.iloc[0]
        end_value = predictions.iloc[-1]
        total_change_pct = ((end_value - start_value) / start_value) * 100
        
        # Determine trend type
        if total_change_pct > 15:
            trend_type = TrendType.STRONG_UPWARD
        elif total_change_pct > 5:
            trend_type = TrendType.MODERATE_UPWARD
        elif total_change_pct > -5:
            trend_type = TrendType.STABLE
        elif total_change_pct > -15:
            trend_type = TrendType.MODERATE_DOWNWARD
        else:
            trend_type = TrendType.STRONG_DOWNWARD
        
        # Calculate trend strength (0-100)
        trend_strength = min(abs(total_change_pct) * 2, 100)
        
        # Calculate trend consistency
        daily_changes = predictions.pct_change().dropna()
        if trend_type in [TrendType.STRONG_UPWARD, TrendType.MODERATE_UPWARD]:
            consistency = (daily_changes > 0).sum() / len(daily_changes) * 100
        elif trend_type in [TrendType.STRONG_DOWNWARD, TrendType.MODERATE_DOWNWARD]:
            consistency = (daily_changes < 0).sum() / len(daily_changes) * 100
        else:
            consistency = 100 - (abs(daily_changes) > 0.01).sum() / len(daily_changes) * 100
        
        # Find turning points
        turning_points = self._find_turning_points(predictions)
        
        # Calculate trend duration
        trend_duration_months = len(predictions) / 30  # Approximate months
        
        # Generate description
        trend_description = self._generate_trend_description(
            trend_type, total_change_pct, consistency, trend_duration_months
        )
        
        return TrendAnalysis(
            trend_type=trend_type,
            trend_strength=trend_strength,
            trend_consistency=consistency,
            trend_duration_months=trend_duration_months,
            key_turning_points=turning_points,
            trend_description=trend_description
        )
    
    def _detect_anomalies(self, 
                         forecast_output: ForecastOutput,
                         historical_data: pd.Series) -> AnomalyDetection:
        """
        Detect anomalies and unusual patterns in the forecast.
        
        Args:
            forecast_output: Forecast results
            historical_data: Historical data for baseline
            
        Returns:
            AnomalyDetection results
        """
        predictions = forecast_output.predictions
        
        # Calculate rolling statistics
        rolling_mean = predictions.rolling(window=30).mean()
        rolling_std = predictions.rolling(window=30).std()
        
        # Detect statistical anomalies
        anomalies = []
        for i, (date, value) in enumerate(predictions.items()):
            if i < 30:  # Skip initial period without enough rolling data
                continue
                
            z_score = abs((value - rolling_mean.iloc[i]) / rolling_std.iloc[i])
            
            if z_score > self.anomaly_threshold:
                if value > rolling_mean.iloc[i]:
                    description = f"Unusually high value (Z-score: {z_score:.2f})"
                else:
                    description = f"Unusually low value (Z-score: {z_score:.2f})"
                anomalies.append((date, value, description))
        
        # Detect volatility anomalies
        daily_returns = predictions.pct_change().dropna()
        volatility_threshold = daily_returns.std() * 3
        
        for i, (date, return_val) in enumerate(daily_returns.items()):
            if abs(return_val) > volatility_threshold:
                description = f"High volatility event ({return_val*100:.1f}% change)"
                anomalies.append((date, predictions[date], description))
        
        # Group anomalies into periods
        anomaly_periods = self._group_anomaly_periods(anomalies)
        
        # Assess severity
        anomaly_count = len(anomalies)
        if anomaly_count == 0:
            severity = RiskLevel.LOW
        elif anomaly_count < 5:
            severity = RiskLevel.MEDIUM
        elif anomaly_count < 10:
            severity = RiskLevel.HIGH
        else:
            severity = RiskLevel.VERY_HIGH
        
        # Generate description
        anomaly_description = self._generate_anomaly_description(anomalies, severity)
        
        return AnomalyDetection(
            anomalies_detected=anomalies,
            anomaly_count=anomaly_count,
            anomaly_severity=severity,
            anomaly_periods=anomaly_periods,
            anomaly_description=anomaly_description
        )
    
    def _assess_confidence(self, forecast_output: ForecastOutput) -> ConfidenceAssessment:
        """
        Assess confidence and reliability of the forecast.
        
        Args:
            forecast_output: Forecast results
            
        Returns:
            ConfidenceAssessment results
        """
        # Extract confidence interval data
        ci_95 = forecast_output.confidence_intervals.get(0.95)
        if ci_95 is None:
            # Use the highest available confidence level
            ci_95 = list(forecast_output.confidence_intervals.values())[0]
        
        # Calculate confidence factors
        confidence_factors = {}
        
        # 1. Confidence interval width (narrower = more confident)
        ci_width = (ci_95['upper'] - ci_95['lower']).mean()
        avg_prediction = forecast_output.predictions.mean()
        relative_ci_width = (ci_width / avg_prediction) * 100 if avg_prediction != 0 else 100
        
        if relative_ci_width < 10:
            confidence_factors['interval_width'] = 90
        elif relative_ci_width < 20:
            confidence_factors['interval_width'] = 70
        elif relative_ci_width < 30:
            confidence_factors['interval_width'] = 50
        else:
            confidence_factors['interval_width'] = 30
        
        # 2. Model performance metrics
        model_metrics = forecast_output.model_metrics
        if 'RMSE' in model_metrics:
            rmse_score = max(0, 100 - model_metrics['RMSE'])
            confidence_factors['model_performance'] = min(rmse_score, 100)
        else:
            confidence_factors['model_performance'] = 70  # Default
        
        # 3. Forecast consistency
        forecast_volatility = forecast_output.predictions.std()
        if forecast_volatility < avg_prediction * 0.1:
            confidence_factors['consistency'] = 90
        elif forecast_volatility < avg_prediction * 0.2:
            confidence_factors['consistency'] = 70
        else:
            confidence_factors['consistency'] = 50
        
        # 4. Uncertainty metrics
        uncertainty_metrics = forecast_output.uncertainty_metrics
        trend_consistency = uncertainty_metrics.get('trend_consistency_pct', 50)
        confidence_factors['trend_consistency'] = trend_consistency
        
        # Calculate overall confidence score
        confidence_score = np.mean(list(confidence_factors.values()))
        
        # Determine overall confidence level
        if confidence_score >= 80:
            overall_confidence = RiskLevel.LOW  # Low risk = High confidence
        elif confidence_score >= 60:
            overall_confidence = RiskLevel.MEDIUM
        elif confidence_score >= 40:
            overall_confidence = RiskLevel.HIGH
        else:
            overall_confidence = RiskLevel.VERY_HIGH
        
        # Calculate confidence evolution over time
        confidence_evolution = self._calculate_confidence_evolution(forecast_output)
        
        # Generate reliability metrics
        reliability_metrics = {
            'confidence_score': confidence_score,
            'relative_ci_width': relative_ci_width,
            'forecast_volatility': forecast_volatility,
            'model_reliability': confidence_factors['model_performance']
        }
        
        # Generate description
        assessment_description = self._generate_confidence_description(
            overall_confidence, confidence_score, confidence_factors
        )
        
        return ConfidenceAssessment(
            overall_confidence=overall_confidence,
            confidence_score=confidence_score,
            confidence_factors=confidence_factors,
            reliability_metrics=reliability_metrics,
            confidence_evolution=confidence_evolution,
            assessment_description=assessment_description
        )
    
    def _identify_opportunities(self, 
                              forecast_output: ForecastOutput,
                              historical_data: pd.Series,
                              trend_analysis: TrendAnalysis,
                              confidence_assessment: ConfidenceAssessment) -> List[MarketOpportunity]:
        """
        Identify market opportunities based on forecast analysis.
        
        Args:
            forecast_output: Forecast results
            historical_data: Historical data
            trend_analysis: Trend analysis results
            confidence_assessment: Confidence assessment results
            
        Returns:
            List of MarketOpportunity objects
        """
        opportunities = []
        predictions = forecast_output.predictions
        current_price = historical_data.iloc[-1]
        
        # Opportunity 1: Trend-based opportunities
        if trend_analysis.trend_type in [TrendType.STRONG_UPWARD, TrendType.MODERATE_UPWARD]:
            if confidence_assessment.confidence_score > 60:
                # Buy opportunity
                entry_points = self._find_optimal_entry_points(predictions, "buy")
                potential_return = ((predictions.iloc[-1] - current_price) / current_price) * 100
                
                opportunity = MarketOpportunity(
                    opportunity_type="Buy",
                    opportunity_strength=min(trend_analysis.trend_strength, confidence_assessment.confidence_score),
                    optimal_entry_points=entry_points,
                    risk_factors=self._identify_risk_factors(forecast_output, trend_analysis),
                    potential_return=potential_return,
                    time_horizon=f"{trend_analysis.trend_duration_months:.1f} months",
                    opportunity_description=f"Strong {trend_analysis.trend_type.value.lower()} trend with {confidence_assessment.confidence_score:.0f}% confidence"
                )
                opportunities.append(opportunity)
        
        elif trend_analysis.trend_type in [TrendType.STRONG_DOWNWARD, TrendType.MODERATE_DOWNWARD]:
            if confidence_assessment.confidence_score > 60:
                # Sell/Short opportunity
                entry_points = self._find_optimal_entry_points(predictions, "sell")
                potential_return = ((current_price - predictions.iloc[-1]) / current_price) * 100
                
                opportunity = MarketOpportunity(
                    opportunity_type="Sell",
                    opportunity_strength=min(trend_analysis.trend_strength, confidence_assessment.confidence_score),
                    optimal_entry_points=entry_points,
                    risk_factors=self._identify_risk_factors(forecast_output, trend_analysis),
                    potential_return=potential_return,
                    time_horizon=f"{trend_analysis.trend_duration_months:.1f} months",
                    opportunity_description=f"Strong {trend_analysis.trend_type.value.lower()} trend with {confidence_assessment.confidence_score:.0f}% confidence"
                )
                opportunities.append(opportunity)
        
        # Opportunity 2: Volatility-based opportunities
        forecast_volatility = predictions.std()
        historical_volatility = historical_data.tail(252).std()
        
        if forecast_volatility > historical_volatility * 1.5:
            # High volatility opportunity
            entry_points = self._find_volatility_entry_points(predictions)
            
            opportunity = MarketOpportunity(
                opportunity_type="Volatility Trading",
                opportunity_strength=min(70, confidence_assessment.confidence_score),
                optimal_entry_points=entry_points,
                risk_factors=["High volatility", "Increased uncertainty"],
                potential_return=15.0,  # Estimated volatility trading return
                time_horizon="Short-term (1-3 months)",
                opportunity_description="Increased volatility presents trading opportunities"
            )
            opportunities.append(opportunity)
        
        # Opportunity 3: Mean reversion opportunities
        if trend_analysis.trend_type == TrendType.STABLE:
            mean_price = predictions.mean()
            current_deviation = abs(current_price - mean_price) / mean_price * 100
            
            if current_deviation > 10:
                entry_points = [(predictions.index[0], mean_price, "Mean reversion target")]
                
                opportunity = MarketOpportunity(
                    opportunity_type="Mean Reversion",
                    opportunity_strength=confidence_assessment.confidence_score * 0.8,
                    optimal_entry_points=entry_points,
                    risk_factors=["Market regime change", "Extended deviation"],
                    potential_return=current_deviation,
                    time_horizon="Medium-term (3-6 months)",
                    opportunity_description=f"Price deviation of {current_deviation:.1f}% from forecast mean"
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def _assess_risks(self, 
                     forecast_output: ForecastOutput,
                     trend_analysis: TrendAnalysis,
                     anomaly_detection: AnomalyDetection,
                     confidence_assessment: ConfidenceAssessment) -> Dict[str, Any]:
        """
        Assess various risk factors in the forecast.
        
        Args:
            forecast_output: Forecast results
            trend_analysis: Trend analysis results
            anomaly_detection: Anomaly detection results
            confidence_assessment: Confidence assessment results
            
        Returns:
            Dictionary containing risk assessment
        """
        risk_assessment = {}
        
        # Overall risk level
        risk_factors = [
            confidence_assessment.overall_confidence.value,
            anomaly_detection.anomaly_severity.value,
        ]
        
        # Model risk
        model_confidence = confidence_assessment.confidence_score
        if model_confidence < 50:
            risk_assessment['model_risk'] = RiskLevel.VERY_HIGH
        elif model_confidence < 70:
            risk_assessment['model_risk'] = RiskLevel.HIGH
        else:
            risk_assessment['model_risk'] = RiskLevel.MEDIUM
        
        # Volatility risk
        forecast_volatility = forecast_output.uncertainty_metrics.get('forecast_volatility', 0)
        if forecast_volatility > 20:
            risk_assessment['volatility_risk'] = RiskLevel.VERY_HIGH
        elif forecast_volatility > 15:
            risk_assessment['volatility_risk'] = RiskLevel.HIGH
        elif forecast_volatility > 10:
            risk_assessment['volatility_risk'] = RiskLevel.MEDIUM
        else:
            risk_assessment['volatility_risk'] = RiskLevel.LOW
        
        # Trend risk
        if trend_analysis.trend_consistency < 60:
            risk_assessment['trend_risk'] = RiskLevel.HIGH
        elif trend_analysis.trend_consistency < 80:
            risk_assessment['trend_risk'] = RiskLevel.MEDIUM
        else:
            risk_assessment['trend_risk'] = RiskLevel.LOW
        
        # Anomaly risk
        risk_assessment['anomaly_risk'] = anomaly_detection.anomaly_severity
        
        # Calculate overall risk score
        risk_scores = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.VERY_HIGH: 4
        }
        
        avg_risk_score = np.mean([
            risk_scores[risk_assessment['model_risk']],
            risk_scores[risk_assessment['volatility_risk']],
            risk_scores[risk_assessment['trend_risk']],
            risk_scores[risk_assessment['anomaly_risk']]
        ])
        
        if avg_risk_score <= 1.5:
            risk_assessment['overall_risk'] = RiskLevel.LOW
        elif avg_risk_score <= 2.5:
            risk_assessment['overall_risk'] = RiskLevel.MEDIUM
        elif avg_risk_score <= 3.5:
            risk_assessment['overall_risk'] = RiskLevel.HIGH
        else:
            risk_assessment['overall_risk'] = RiskLevel.VERY_HIGH
        
        return risk_assessment
    
    def _generate_key_insights(self, 
                              trend_analysis: TrendAnalysis,
                              anomaly_detection: AnomalyDetection,
                              confidence_assessment: ConfidenceAssessment,
                              market_opportunities: List[MarketOpportunity]) -> List[str]:
        """Generate key insights from the analysis."""
        insights = []
        
        # Trend insights
        insights.append(f"Forecast shows {trend_analysis.trend_type.value.lower()} trend with {trend_analysis.trend_consistency:.0f}% consistency")
        
        # Confidence insights
        insights.append(f"Model confidence is {confidence_assessment.overall_confidence.value.lower()} with {confidence_assessment.confidence_score:.0f}% reliability score")
        
        # Anomaly insights
        if anomaly_detection.anomaly_count > 0:
            insights.append(f"Detected {anomaly_detection.anomaly_count} anomalies with {anomaly_detection.anomaly_severity.value.lower()} severity")
        else:
            insights.append("No significant anomalies detected in forecast period")
        
        # Opportunity insights
        if market_opportunities:
            best_opportunity = max(market_opportunities, key=lambda x: x.opportunity_strength)
            insights.append(f"Best opportunity: {best_opportunity.opportunity_type} with {best_opportunity.opportunity_strength:.0f}% strength")
        
        return insights
    
    def _generate_recommendations(self, 
                                 trend_analysis: TrendAnalysis,
                                 market_opportunities: List[MarketOpportunity],
                                 risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Risk-based recommendations
        overall_risk = risk_assessment.get('overall_risk', RiskLevel.MEDIUM)
        
        if overall_risk == RiskLevel.LOW:
            recommendations.append("Low risk environment - consider increasing position sizes")
        elif overall_risk == RiskLevel.VERY_HIGH:
            recommendations.append("High risk environment - consider reducing exposure or hedging")
        
        # Opportunity-based recommendations
        for opportunity in market_opportunities:
            if opportunity.opportunity_strength > 70:
                recommendations.append(f"Strong {opportunity.opportunity_type.lower()} signal - consider action within {opportunity.time_horizon}")
        
        # Trend-based recommendations
        if trend_analysis.trend_consistency > 80:
            recommendations.append("High trend consistency - suitable for trend-following strategies")
        elif trend_analysis.trend_consistency < 60:
            recommendations.append("Low trend consistency - consider mean-reversion strategies")
        
        return recommendations
    
    def _generate_report_summary(self, 
                                trend_analysis: TrendAnalysis,
                                confidence_assessment: ConfidenceAssessment,
                                market_opportunities: List[MarketOpportunity]) -> str:
        """Generate comprehensive report summary."""
        
        summary = f"""
FORECAST ANALYSIS SUMMARY

Trend Analysis:
- Direction: {trend_analysis.trend_type.value}
- Strength: {trend_analysis.trend_strength:.0f}/100
- Consistency: {trend_analysis.trend_consistency:.0f}%

Confidence Assessment:
- Overall Confidence: {confidence_assessment.overall_confidence.value}
- Reliability Score: {confidence_assessment.confidence_score:.0f}/100

Market Opportunities:
- {len(market_opportunities)} opportunities identified
"""
        
        if market_opportunities:
            best_opp = max(market_opportunities, key=lambda x: x.opportunity_strength)
            summary += f"- Best Opportunity: {best_opp.opportunity_type} ({best_opp.opportunity_strength:.0f}% strength)"
        
        return summary.strip()
    
    # Helper methods
    def _find_turning_points(self, predictions: pd.Series) -> List[pd.Timestamp]:
        """Find significant turning points in the forecast."""
        turning_points = []
        
        # Calculate rolling derivatives
        rolling_change = predictions.rolling(window=7).mean().diff()
        
        # Find sign changes
        for i in range(1, len(rolling_change) - 1):
            if pd.isna(rolling_change.iloc[i]):
                continue
                
            prev_change = rolling_change.iloc[i-1]
            curr_change = rolling_change.iloc[i]
            next_change = rolling_change.iloc[i+1]
            
            # Check for sign change with significance
            if (prev_change > 0 and next_change < 0) or (prev_change < 0 and next_change > 0):
                if abs(curr_change) > predictions.std() * 0.1:  # Significant change
                    turning_points.append(predictions.index[i])
        
        return turning_points[:5]  # Limit to top 5 turning points
    
    def _group_anomaly_periods(self, anomalies: List[Tuple]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Group consecutive anomalies into periods."""
        if not anomalies:
            return []
        
        periods = []
        current_start = anomalies[0][0]
        current_end = anomalies[0][0]
        
        for i in range(1, len(anomalies)):
            date = anomalies[i][0]
            if (date - current_end).days <= 7:  # Within a week
                current_end = date
            else:
                periods.append((current_start, current_end))
                current_start = date
                current_end = date
        
        periods.append((current_start, current_end))
        return periods
    
    def _calculate_confidence_evolution(self, forecast_output: ForecastOutput) -> pd.Series:
        """Calculate how confidence evolves over the forecast period."""
        predictions = forecast_output.predictions
        ci_95 = list(forecast_output.confidence_intervals.values())[0]
        
        # Calculate relative confidence interval width over time
        ci_width = ci_95['upper'] - ci_95['lower']
        relative_width = ci_width / predictions * 100
        
        # Convert to confidence score (inverse of relative width)
        confidence_evolution = 100 - np.clip(relative_width, 0, 100)
        
        return confidence_evolution
    
    def _find_optimal_entry_points(self, predictions: pd.Series, action: str) -> List[Tuple[pd.Timestamp, float, str]]:
        """Find optimal entry points for trading actions."""
        entry_points = []
        
        if action == "buy":
            # Find local minima
            rolling_min = predictions.rolling(window=30).min()
            for i, (date, price) in enumerate(predictions.items()):
                if i > 30 and price == rolling_min.iloc[i]:
                    entry_points.append((date, price, "Local minimum - good buy point"))
        
        elif action == "sell":
            # Find local maxima
            rolling_max = predictions.rolling(window=30).max()
            for i, (date, price) in enumerate(predictions.items()):
                if i > 30 and price == rolling_max.iloc[i]:
                    entry_points.append((date, price, "Local maximum - good sell point"))
        
        return entry_points[:3]  # Limit to top 3 entry points
    
    def _find_volatility_entry_points(self, predictions: pd.Series) -> List[Tuple[pd.Timestamp, float, str]]:
        """Find entry points for volatility trading."""
        entry_points = []
        
        # Calculate rolling volatility
        rolling_vol = predictions.rolling(window=30).std()
        high_vol_threshold = rolling_vol.quantile(0.8)
        
        for i, (date, vol) in enumerate(rolling_vol.items()):
            if vol > high_vol_threshold:
                price = predictions[date]
                entry_points.append((date, price, f"High volatility period ({vol:.2f})"))
        
        return entry_points[:3]
    
    def _identify_risk_factors(self, forecast_output: ForecastOutput, trend_analysis: TrendAnalysis) -> List[str]:
        """Identify risk factors for opportunities."""
        risk_factors = []
        
        if trend_analysis.trend_consistency < 70:
            risk_factors.append("Low trend consistency")
        
        if forecast_output.uncertainty_metrics.get('forecast_volatility', 0) > 15:
            risk_factors.append("High forecast volatility")
        
        confidence_score = forecast_output.uncertainty_metrics.get('relative_ci_width_95', 0)
        if confidence_score > 25:
            risk_factors.append("Wide confidence intervals")
        
        return risk_factors
    
    def _generate_trend_description(self, trend_type: TrendType, change_pct: float, 
                                   consistency: float, duration: float) -> str:
        """Generate human-readable trend description."""
        return f"{trend_type.value} trend with {change_pct:.1f}% total change over {duration:.1f} months, {consistency:.0f}% consistency"
    
    def _generate_anomaly_description(self, anomalies: List[Tuple], severity: RiskLevel) -> str:
        """Generate human-readable anomaly description."""
        if not anomalies:
            return "No significant anomalies detected"
        
        return f"{len(anomalies)} anomalies detected with {severity.value.lower()} severity level"
    
    def _generate_confidence_description(self, overall_confidence: RiskLevel, 
                                        confidence_score: float, factors: Dict[str, float]) -> str:
        """Generate human-readable confidence description."""
        return f"{overall_confidence.value} confidence level with {confidence_score:.0f}% reliability score based on model performance and forecast consistency"