"""Feature engineering utilities for financial data analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates financial metrics, technical indicators, and derived features."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        logger.info("FeatureEngineer initialized")
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive financial features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional features
        """
        logger.info(f"Engineering features for {len(data)} rows")
        
        # Make a copy to avoid modifying original data
        featured_data = data.copy()
        
        # Process each symbol separately
        if 'Symbol' in data.columns:
            for symbol in data['Symbol'].unique():
                mask = data['Symbol'] == symbol
                symbol_data = data[mask].copy().sort_values('Date')
                
                # Add all features for this symbol
                symbol_features = self._create_symbol_features(symbol_data)
                
                # Update the main dataframe
                for col in symbol_features.columns:
                    if col not in ['Date', 'Symbol']:
                        featured_data.loc[mask, col] = symbol_features[col].values
        else:
            # Process all data together if no symbol column
            featured_data = self._create_symbol_features(featured_data)
        
        logger.info(f"Feature engineering completed. New shape: {featured_data.shape}")
        return featured_data
    
    def _create_symbol_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for a single symbol."""
        featured_data = data.copy()
        
        # Price-based features
        featured_data = self._add_price_features(featured_data)
        
        # Return-based features
        featured_data = self._add_return_features(featured_data)
        
        # Volatility features
        featured_data = self._add_volatility_features(featured_data)
        
        # Technical indicators
        featured_data = self._add_technical_indicators(featured_data)
        
        # Volume features
        featured_data = self._add_volume_features(featured_data)
        
        # Risk metrics
        featured_data = self._add_risk_metrics(featured_data)
        
        return featured_data
    
    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        if 'High' in data.columns and 'Low' in data.columns:
            # Price range
            data['Price_Range'] = data['High'] - data['Low']
            
            # Price range percentage
            if 'Close' in data.columns:
                data['Price_Range_Pct'] = (data['High'] - data['Low']) / data['Close']
        
        if 'Open' in data.columns and 'Close' in data.columns:
            # Intraday return
            data['Intraday_Return'] = (data['Close'] - data['Open']) / data['Open']
            
            # Gap (overnight return)
            data['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        if 'High' in data.columns and 'Low' in data.columns and 'Close' in data.columns:
            # True Range
            data['True_Range'] = np.maximum(
                data['High'] - data['Low'],
                np.maximum(
                    np.abs(data['High'] - data['Close'].shift(1)),
                    np.abs(data['Low'] - data['Close'].shift(1))
                )
            )
        
        return data
    
    def _add_return_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        if 'Adj Close' in data.columns:
            # Simple returns
            data['Returns'] = data['Adj Close'].pct_change()
            
            # Log returns
            data['Log_Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
            
            # Cumulative returns
            data['Cumulative_Returns'] = (1 + data['Returns']).cumprod() - 1
            
            # Forward returns (for prediction targets)
            data['Forward_Return_1d'] = data['Returns'].shift(-1)
            data['Forward_Return_5d'] = data['Adj Close'].pct_change(5).shift(-5)
            data['Forward_Return_10d'] = data['Adj Close'].pct_change(10).shift(-10)
            
            # Rolling returns
            for window in [5, 10, 20, 60]:
                data[f'Return_{window}d'] = data['Adj Close'].pct_change(window)
        
        return data
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        if 'Returns' in data.columns:
            # Rolling volatility (annualized)
            for window in [10, 20, 30, 60]:
                data[f'Volatility_{window}d'] = data['Returns'].rolling(window).std() * np.sqrt(252)
            
            # Realized volatility using high-frequency proxy
            if 'True_Range' in data.columns:
                for window in [10, 20, 30]:
                    data[f'Realized_Vol_{window}d'] = data['True_Range'].rolling(window).mean()
            
            # GARCH-like volatility (exponentially weighted)
            data['EWMA_Volatility'] = data['Returns'].ewm(span=30).std() * np.sqrt(252)
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        if 'Adj Close' in data.columns:
            # Moving averages
            for window in [5, 10, 20, 50, 200]:
                data[f'MA_{window}'] = data['Adj Close'].rolling(window).mean()
                
                # Price relative to moving average
                data[f'Price_to_MA_{window}'] = data['Adj Close'] / data[f'MA_{window}'] - 1
            
            # Exponential moving averages
            for span in [12, 26, 50]:
                data[f'EMA_{span}'] = data['Adj Close'].ewm(span=span).mean()
            
            # MACD
            if 'EMA_12' in data.columns and 'EMA_26' in data.columns:
                data['MACD'] = data['EMA_12'] - data['EMA_26']
                data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
                data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # RSI (Relative Strength Index)
            delta = data['Adj Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            if 'MA_20' in data.columns:
                bb_std = data['Adj Close'].rolling(20).std()
                data['BB_Upper'] = data['MA_20'] + (bb_std * 2)
                data['BB_Lower'] = data['MA_20'] - (bb_std * 2)
                data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
                data['BB_Position'] = (data['Adj Close'] - data['BB_Lower']) / data['BB_Width']
        
        if all(col in data.columns for col in ['High', 'Low', 'Close']):
            # Stochastic Oscillator
            low_14 = data['Low'].rolling(14).min()
            high_14 = data['High'].rolling(14).max()
            data['Stoch_K'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)
            data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()
        
        return data
    
    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        if 'Volume' in data.columns:
            # Volume moving averages
            for window in [10, 20, 50]:
                data[f'Volume_MA_{window}'] = data['Volume'].rolling(window).mean()
                
                # Volume relative to average
                data[f'Volume_Ratio_{window}'] = data['Volume'] / data[f'Volume_MA_{window}']
            
            # Volume trend
            data['Volume_Trend'] = data['Volume'].rolling(10).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan
            )
            
            # On-Balance Volume (OBV)
            if 'Returns' in data.columns:
                obv_direction = np.where(data['Returns'] > 0, 1, 
                                np.where(data['Returns'] < 0, -1, 0))
                data['OBV'] = (data['Volume'] * obv_direction).cumsum()
            
            # Volume-Price Trend (VPT)
            if 'Returns' in data.columns:
                data['VPT'] = (data['Volume'] * data['Returns']).cumsum()
        
        return data
    
    def _add_risk_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add risk-based metrics."""
        if 'Returns' in data.columns:
            # Value at Risk (VaR) - rolling 5% VaR
            for window in [30, 60, 252]:
                data[f'VaR_5pct_{window}d'] = data['Returns'].rolling(window).quantile(0.05)
            
            # Maximum Drawdown (rolling)
            if 'Cumulative_Returns' in data.columns:
                cumulative = 1 + data['Cumulative_Returns']
                for window in [60, 252]:
                    rolling_max = cumulative.rolling(window).max()
                    drawdown = (cumulative - rolling_max) / rolling_max
                    data[f'Max_Drawdown_{window}d'] = drawdown.rolling(window).min()
            
            # Sharpe Ratio (rolling, assuming 2% risk-free rate)
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            for window in [60, 252]:
                excess_returns = data['Returns'] - risk_free_rate
                mean_excess = excess_returns.rolling(window).mean()
                vol = data['Returns'].rolling(window).std()
                data[f'Sharpe_Ratio_{window}d'] = mean_excess / vol * np.sqrt(252)
            
            # Sortino Ratio (downside deviation)
            for window in [60, 252]:
                downside_returns = data['Returns'].where(data['Returns'] < 0, 0)
                downside_vol = downside_returns.rolling(window).std()
                excess_returns = data['Returns'] - risk_free_rate
                mean_excess = excess_returns.rolling(window).mean()
                data[f'Sortino_Ratio_{window}d'] = mean_excess / downside_vol * np.sqrt(252)
        
        return data
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for a return series."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        
        return excess_returns / volatility
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns.dropna(), confidence_level * 100)
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())
    
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market."""
        if len(asset_returns) == 0 or len(market_returns) == 0:
            return 1.0
        
        # Align the series
        aligned_data = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < 2:
            return 1.0
        
        covariance = aligned_data['asset'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        
        if market_variance == 0:
            return 1.0
        
        return covariance / market_variance
    
    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of engineered features."""
        original_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Symbol']
        new_features = [col for col in data.columns if col not in original_cols]
        
        feature_categories = {
            'price_features': [col for col in new_features if any(x in col for x in ['Price_Range', 'Intraday', 'Gap', 'True_Range'])],
            'return_features': [col for col in new_features if 'Return' in col],
            'volatility_features': [col for col in new_features if any(x in col for x in ['Volatility', 'Vol'])],
            'technical_indicators': [col for col in new_features if any(x in col for x in ['MA_', 'EMA_', 'MACD', 'RSI', 'BB_', 'Stoch'])],
            'volume_features': [col for col in new_features if 'Volume' in col or col in ['OBV', 'VPT']],
            'risk_metrics': [col for col in new_features if any(x in col for x in ['VaR', 'Drawdown', 'Sharpe', 'Sortino'])]
        }
        
        summary = {
            'total_features': len(new_features),
            'feature_categories': {k: len(v) for k, v in feature_categories.items()},
            'feature_list': new_features,
            'missing_values': data[new_features].isnull().sum().to_dict()
        }
        
        return summary