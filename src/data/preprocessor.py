"""Data preprocessing pipeline for the portfolio forecasting system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import logging

# Set up logging
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles missing value imputation, data cleaning, and preprocessing."""
    
    def __init__(self, 
                 handle_missing: str = "interpolate",
                 outlier_threshold: float = 3.0,
                 scaling_method: str = "standard"):
        """Initialize data preprocessor."""
        self.handle_missing = handle_missing
        self.outlier_threshold = outlier_threshold
        self.scaling_method = scaling_method
        
        # Initialize scaler
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        self.numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        self.price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        
        logger.info(f"DataPreprocessor initialized")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        logger.info(f"Starting preprocessing for {len(data)} rows")
        
        processed_data = data.copy()
        
        # Convert data types
        processed_data = self._convert_data_types(processed_data)
        
        # Handle missing values
        processed_data = self._handle_missing_values(processed_data)
        
        # Remove duplicates
        processed_data = self._remove_duplicates(processed_data)
        
        # Sort data
        processed_data = self._sort_data(processed_data)
        
        # Handle outliers
        processed_data = self._handle_outliers(processed_data)
        
        # Add derived features
        processed_data = self._add_derived_features(processed_data)
        
        logger.info(f"Preprocessing completed. Shape: {processed_data.shape}")
        return processed_data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data types."""
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
        
        for col in self.numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        if 'Symbol' in data.columns:
            data['Symbol'] = data['Symbol'].astype(str)
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        missing_before = data.isnull().sum().sum()
        
        if missing_before == 0:
            return data
        
        if self.handle_missing == "interpolate":
            if 'Symbol' in data.columns:
                for symbol in data['Symbol'].unique():
                    mask = data['Symbol'] == symbol
                    for col in self.numeric_columns:
                        if col in data.columns:
                            data.loc[mask, col] = data.loc[mask, col].interpolate()
            else:
                for col in self.numeric_columns:
                    if col in data.columns:
                        data[col] = data[col].interpolate()
        
        elif self.handle_missing == "forward_fill":
            if 'Symbol' in data.columns:
                for symbol in data['Symbol'].unique():
                    mask = data['Symbol'] == symbol
                    for col in self.numeric_columns:
                        if col in data.columns:
                            data.loc[mask, col] = data.loc[mask, col].fillna(method='ffill')
            else:
                for col in self.numeric_columns:
                    if col in data.columns:
                        data[col] = data[col].fillna(method='ffill')
        
        elif self.handle_missing == "drop":
            data = data.dropna()
        
        missing_after = data.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")
        
        return data
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates."""
        initial_rows = len(data)
        
        if 'Date' in data.columns and 'Symbol' in data.columns:
            data = data.drop_duplicates(subset=['Date', 'Symbol'], keep='first')
        else:
            data = data.drop_duplicates(keep='first')
        
        removed = initial_rows - len(data)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return data
    
    def _sort_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sort data."""
        sort_columns = []
        if 'Date' in data.columns:
            sort_columns.append('Date')
        if 'Symbol' in data.columns:
            sort_columns.append('Symbol')
        
        if sort_columns:
            data = data.sort_values(sort_columns).reset_index(drop=True)
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers."""
        outliers_detected = 0
        
        if 'Symbol' in data.columns:
            for symbol in data['Symbol'].unique():
                mask = data['Symbol'] == symbol
                symbol_data = data[mask]
                
                for col in self.price_columns:
                    if col in data.columns and len(symbol_data) > 10:
                        values = symbol_data[col].dropna()
                        if len(values) > 0:
                            z_scores = np.abs(stats.zscore(values))
                            outlier_mask = z_scores > self.outlier_threshold
                            
                            if outlier_mask.any():
                                outlier_count = outlier_mask.sum()
                                outliers_detected += outlier_count
                                
                                median_value = values.median()
                                outlier_indices = values[outlier_mask].index
                                data.loc[outlier_indices, col] = median_value
        
        if outliers_detected > 0:
            logger.info(f"Handled {outliers_detected} outliers")
        
        return data
    
    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add derived features."""
        if 'Symbol' in data.columns:
            for symbol in data['Symbol'].unique():
                mask = data['Symbol'] == symbol
                symbol_data = data[mask].copy()
                
                if 'Adj Close' in symbol_data.columns:
                    # Calculate returns
                    returns = symbol_data['Adj Close'].pct_change()
                    data.loc[mask, 'Returns'] = returns
                    
                    # Calculate rolling volatility
                    if len(returns.dropna()) > 30:
                        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
                        data.loc[mask, 'Volatility'] = rolling_vol
                    
                    # Price change
                    price_change = symbol_data['Adj Close'].diff()
                    data.loc[mask, 'Price_Change'] = price_change
                
                # Volume moving average
                if 'Volume' in symbol_data.columns and len(symbol_data) > 20:
                    volume_ma = symbol_data['Volume'].rolling(window=20).mean()
                    data.loc[mask, 'Volume_MA'] = volume_ma
                
                # Price range
                if 'High' in symbol_data.columns and 'Low' in symbol_data.columns:
                    price_range = symbol_data['High'] - symbol_data['Low']
                    data.loc[mask, 'Price_Range'] = price_range
        
        logger.info("Added derived features")
        return data