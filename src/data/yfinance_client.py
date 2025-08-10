"""YFinance client for fetching historical financial data."""

import yfinance as yf
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .cache_manager import CacheManager
from .data_validator import DataValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YFinanceClient:
    """Client for fetching financial data from Yahoo Finance API."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, use_cache: bool = True, cache_expiry_hours: int = 24):
        """Initialize YFinance client.
        
        Args:
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Initial delay between retries in seconds
            use_cache: Whether to use caching for data
            cache_expiry_hours: Hours after which cache expires
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.valid_symbols = ["TSLA", "BND", "SPY"]  # Supported symbols
        self.use_cache = use_cache
        
        # Initialize cache manager and validator
        if self.use_cache:
            self.cache_manager = CacheManager(expiry_hours=cache_expiry_hours)
        else:
            self.cache_manager = None
            
        self.validator = DataValidator()
        
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate asset symbols.
        
        Args:
            symbols: List of asset symbols to validate
            
        Returns:
            List of valid symbols
            
        Raises:
            ValueError: If any symbol is invalid
        """
        invalid_symbols = [s for s in symbols if s not in self.valid_symbols]
        if invalid_symbols:
            raise ValueError(f"Invalid symbols: {invalid_symbols}. "
                           f"Supported symbols: {self.valid_symbols}")
        return symbols
    
    def validate_date_range(self, start_date: str, end_date: str) -> tuple:
        """Validate date range format and logic.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Tuple of validated (start_date, end_date)
            
        Raises:
            ValueError: If dates are invalid or in wrong order
        """
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
        except Exception as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")
        
        if start >= end:
            raise ValueError("Start date must be before end date")
        
        # Check if dates are too far in the future
        today = pd.Timestamp.now()
        if start > today:
            raise ValueError("Start date cannot be in the future")
        
        return start_date, end_date
    
    def fetch_data_with_retry(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """Fetch data for a single symbol with retry mechanism.
        
        Args:
            symbol: Asset symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with historical data
            
        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching data for {symbol} (attempt {attempt + 1})")
                
                # Create ticker object
                ticker = yf.Ticker(symbol)
                
                # Fetch historical data
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    prepost=True
                )
                
                if data.empty:
                    raise ValueError(f"No data returned for {symbol}")
                
                # Add symbol column
                data['Symbol'] = symbol
                
                # Reset index to make Date a column
                data = data.reset_index()
                
                logger.info(f"Successfully fetched {len(data)} records for {symbol}")
                return data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed for {symbol}")
                    raise e
    
    def fetch_data(
        self, 
        symbols: Union[str, List[str]], 
        start_date: str, 
        end_date: str,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """Fetch historical data for multiple assets with caching support.
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            DataFrame with historical data for all symbols
            
        Raises:
            ValueError: If symbols or dates are invalid
        """
        # Convert single symbol to list
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Validate inputs
        symbols = self.validate_symbols(symbols)
        start_date, end_date = self.validate_date_range(start_date, end_date)
        
        logger.info(f"Fetching data for symbols: {symbols}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Try to get cached data first (if caching is enabled and not forcing refresh)
        if self.use_cache and self.cache_manager and not force_refresh:
            cached_data = self.cache_manager.get_cached_data(symbols, start_date, end_date)
            if cached_data is not None:
                # Validate cached data
                validation_result = self.validator.validate_data(cached_data)
                if validation_result.is_valid:
                    logger.info("Using cached data")
                    return cached_data
                else:
                    logger.warning("Cached data failed validation, fetching fresh data")
        
        # Fetch fresh data
        all_data = []
        for symbol in symbols:
            try:
                data = self.fetch_data_with_retry(symbol, start_date, end_date)
                all_data.append(data)
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                # Continue with other symbols instead of failing completely
                continue
        
        if not all_data:
            raise ValueError("Failed to fetch data for any symbols")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort by date and symbol
        combined_data = combined_data.sort_values(['Date', 'Symbol']).reset_index(drop=True)
        
        # Validate the fetched data
        validation_result = self.validator.validate_data(combined_data)
        if not validation_result.is_valid:
            logger.warning("Fetched data failed validation:")
            validation_result.print_report()
            # Continue anyway but log the issues
        
        # Cache the data if caching is enabled
        if self.use_cache and self.cache_manager and validation_result.is_valid:
            self.cache_manager.save_data_to_cache(combined_data, symbols, start_date, end_date)
        
        logger.info(f"Successfully fetched data for {len(symbols)} symbols, "
                   f"total records: {len(combined_data)}")
        
        return combined_data
    
    def get_latest_data(
        self, 
        symbols: Union[str, List[str]], 
        period: str = "1y"
    ) -> pd.DataFrame:
        """Fetch latest data for specified period.
        
        Args:
            symbols: Single symbol or list of symbols
            period: Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with latest historical data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        symbols = self.validate_symbols(symbols)
        
        logger.info(f"Fetching latest {period} data for symbols: {symbols}")
        
        all_data = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, auto_adjust=True)
                
                if not data.empty:
                    data['Symbol'] = symbol
                    data = data.reset_index()
                    all_data.append(data)
                    
            except Exception as e:
                logger.error(f"Failed to fetch latest data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("Failed to fetch latest data for any symbols")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values(['Date', 'Symbol']).reset_index(drop=True)
        
        return combined_data
    
    def get_info(self, symbol: str) -> Dict:
        """Get basic information about a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary with symbol information
        """
        self.validate_symbols([symbol])
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            key_info = {
                'symbol': symbol,
                'longName': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'marketCap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A')
            }
            
            return key_info
            
        except Exception as e:
            logger.error(f"Failed to get info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def check_data_availability(
        self, 
        symbols: Union[str, List[str]], 
        start_date: str, 
        end_date: str
    ) -> Dict[str, bool]:
        """Check data availability for symbols in date range.
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping symbols to availability status
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        symbols = self.validate_symbols(symbols)
        self.validate_date_range(start_date, end_date)
        
        availability = {}
        
        for symbol in symbols:
            try:
                # Try to fetch a small sample
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, period="1d")
                availability[symbol] = not data.empty
                
            except Exception as e:
                logger.warning(f"Could not check availability for {symbol}: {e}")
                availability[symbol] = False
        
        return availability
    
    def clear_cache(self) -> int:
        """Clear all cached data.
        
        Returns:
            Number of cache entries removed
        """
        if self.cache_manager:
            return self.cache_manager.clear_all_cache()
        return 0
    
    def clear_expired_cache(self) -> int:
        """Clear expired cached data.
        
        Returns:
            Number of expired cache entries removed
        """
        if self.cache_manager:
            return self.cache_manager.clear_expired_cache()
        return 0
    
    def get_cache_info(self) -> Dict:
        """Get information about current cache status.
        
        Returns:
            Dictionary with cache information
        """
        if self.cache_manager:
            return self.cache_manager.get_cache_info()
        return {'message': 'Caching is disabled'}
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data using the data validator.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        return self.validator.quick_validate(data)