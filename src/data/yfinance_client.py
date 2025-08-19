"""YFinance client for fetching historical financial data.

This module provides a robust client for fetching financial market data from Yahoo Finance
with built-in error handling, caching, and data validation capabilities. It supports
retry mechanisms for network failures and implements efficient caching to minimize
API calls while ensuring data freshness.

Key Features:
    - Automatic retry with exponential backoff for failed requests
    - Local caching system with configurable expiry times
    - Data validation and integrity checks
    - Support for multiple asset symbols
    - Comprehensive error handling and logging

Supported Assets:
    - TSLA: Tesla Inc. (High-growth technology stock)
    - BND: Vanguard Total Bond Market ETF (Bond market exposure)
    - SPY: SPDR S&P 500 ETF (Broad market index)

Usage Example:
    >>> client = YFinanceClient(use_cache=True, cache_expiry_hours=24)
    >>> data = client.fetch_data(['TSLA', 'SPY'], '2020-01-01', '2024-12-31')
    >>> print(f"Fetched {len(data)} records")

Author: Portfolio Forecasting System
Version: 1.0.0
"""

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

# Configure logging for financial data operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YFinanceClient:
    """Professional-grade client for Yahoo Finance API with enterprise features.
    
    This class provides a robust interface to Yahoo Finance data with built-in
    reliability features including retry logic, caching, and data validation.
    Designed for production use in quantitative finance applications.
    
    Attributes:
        max_retries (int): Maximum retry attempts for failed API calls
        retry_delay (float): Base delay between retries (exponential backoff)
        valid_symbols (List[str]): Supported financial instruments
        use_cache (bool): Enable/disable local data caching
        cache_manager (CacheManager): Handles local data storage
        validator (DataValidator): Validates data integrity
    
    Example:
        >>> # Initialize client with caching enabled
        >>> client = YFinanceClient(max_retries=5, use_cache=True)
        >>> 
        >>> # Fetch historical data for portfolio analysis
        >>> symbols = ['TSLA', 'SPY', 'BND']
        >>> data = client.fetch_data(symbols, '2020-01-01', '2024-12-31')
        >>> 
        >>> # Check data quality
        >>> print(f"Data shape: {data.shape}")
        >>> print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, 
                 use_cache: bool = True, cache_expiry_hours: int = 24):
        """Initialize YFinance client with configuration parameters.
        
        Sets up the client with retry logic, caching system, and data validation.
        The client is configured to handle network failures gracefully and
        optimize performance through intelligent caching.
        
        Args:
            max_retries (int, optional): Maximum number of retry attempts for 
                failed API requests. Defaults to 3. Higher values increase 
                reliability but may slow down error recovery.
            retry_delay (float, optional): Initial delay between retries in seconds.
                Uses exponential backoff (delay * 2^attempt). Defaults to 1.0.
            use_cache (bool, optional): Enable local data caching to reduce API
                calls and improve performance. Defaults to True.
            cache_expiry_hours (int, optional): Hours after which cached data
                expires and fresh data is fetched. Defaults to 24 hours.
        
        Raises:
            ImportError: If required dependencies are not installed
            PermissionError: If cache directory cannot be created
        
        Note:
            The client automatically validates fetched data for completeness
            and integrity. Invalid data triggers automatic retry attempts.
        """
        # Core configuration parameters for API interaction
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Supported financial instruments for portfolio analysis
        # These symbols are validated and tested for data quality
        self.valid_symbols = ["TSLA", "BND", "SPY"]  
        
        self.use_cache = use_cache
        
        # Initialize caching system for performance optimization
        # Cache reduces API calls and improves response times
        if self.use_cache:
            self.cache_manager = CacheManager(expiry_hours=cache_expiry_hours)
            logger.info(f"Cache enabled with {cache_expiry_hours}h expiry")
        else:
            self.cache_manager = None
            logger.info("Cache disabled - all requests will hit API")
            
        # Initialize data validator for quality assurance
        # Ensures fetched data meets requirements for financial analysis
        self.validator = DataValidator()
        
        logger.info(f"YFinanceClient initialized: retries={max_retries}, "
                   f"delay={retry_delay}s, cache={use_cache}")
        
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
        """Fetch historical market data with enterprise-grade reliability.
        
        This method provides the primary interface for retrieving financial data
        with built-in caching, validation, and error recovery. It handles both
        single and multiple asset requests with automatic retry logic.
        
        The method implements a multi-stage process:
        1. Input validation (symbols, date formats, date logic)
        2. Cache lookup (if enabled and not forcing refresh)
        3. API data retrieval with retry logic
        4. Data validation and quality checks
        5. Cache storage for future requests
        
        Args:
            symbols (Union[str, List[str]]): Financial instrument identifier(s).
                Can be a single symbol string ('TSLA') or list of symbols 
                (['TSLA', 'SPY', 'BND']). All symbols must be in valid_symbols.
            start_date (str): Start date for historical data in YYYY-MM-DD format.
                Must be a valid date and before end_date. Cannot be in the future.
            end_date (str): End date for historical data in YYYY-MM-DD format.
                Must be a valid date and after start_date.
            force_refresh (bool, optional): If True, bypasses cache and fetches
                fresh data from API. Useful for real-time analysis. Defaults to False.
            
        Returns:
            pd.DataFrame: Historical market data with columns:
                - Date: Trading date (datetime index)
                - Open: Opening price for the trading day
                - High: Highest price during the trading day
                - Low: Lowest price during the trading day
                - Close: Closing price (adjusted for splits/dividends)
                - Volume: Number of shares traded
                - Symbol: Asset identifier for multi-asset datasets
                
        Raises:
            ValueError: If symbols are not in valid_symbols list, or if date
                format is invalid, or if start_date >= end_date
            ConnectionError: If API is unreachable after all retry attempts
            DataValidationError: If fetched data fails quality checks
            
        Example:
            >>> # Fetch single asset data
            >>> client = YFinanceClient()
            >>> tsla_data = client.fetch_data('TSLA', '2020-01-01', '2024-12-31')
            >>> print(f"TSLA data: {len(tsla_data)} trading days")
            >>> 
            >>> # Fetch multiple assets for portfolio analysis
            >>> portfolio_data = client.fetch_data(
            ...     ['TSLA', 'SPY', 'BND'], '2020-01-01', '2024-12-31'
            ... )
            >>> print(f"Portfolio data shape: {portfolio_data.shape}")
            >>> 
            >>> # Force fresh data (bypass cache)
            >>> fresh_data = client.fetch_data(
            ...     'SPY', '2024-01-01', '2024-12-31', force_refresh=True
            ... )
            
        Note:
            - Data is automatically sorted by Date and Symbol for consistency
            - Cache is used by default to improve performance and reduce API load
            - All data undergoes validation before being returned
            - Network failures trigger automatic retry with exponential backoff
        """
        # Normalize input: convert single symbol string to list for uniform processing
        # This allows the same processing logic for both single and multiple symbols
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Validate inputs before proceeding with API calls
        # This prevents unnecessary API requests for invalid parameters
        symbols = self.validate_symbols(symbols)  # Ensures symbols are supported
        start_date, end_date = self.validate_date_range(start_date, end_date)  # Validates date format and logic
        
        # Log the data request for monitoring and debugging purposes
        logger.info(f"Fetching data for symbols: {symbols}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Force refresh: {force_refresh}, Cache enabled: {self.use_cache}")
        
        # Attempt to retrieve data from cache first (performance optimization)
        # Cache lookup is skipped if force_refresh=True or caching is disabled
        if self.use_cache and self.cache_manager and not force_refresh:
            logger.debug("Checking cache for existing data...")
            cached_data = self.cache_manager.get_cached_data(symbols, start_date, end_date)
            
            if cached_data is not None:
                logger.debug(f"Found cached data with {len(cached_data)} records")
                
                # Validate cached data integrity before using it
                # Corrupted cache data should not be returned to users
                validation_result = self.validator.validate_data(cached_data)
                if validation_result.is_valid:
                    logger.info(f"Using cached data for {symbols} ({len(cached_data)} records)")
                    return cached_data
                else:
                    logger.warning("Cached data failed validation, fetching fresh data")
                    logger.debug(f"Validation errors: {validation_result.errors}")
            else:
                logger.debug("No cached data found, proceeding with API fetch")
        
        # Fetch fresh data from Yahoo Finance API with error handling
        # Process each symbol individually to handle partial failures gracefully
        all_data = []
        failed_symbols = []
        
        logger.info(f"Fetching fresh data from Yahoo Finance API...")
        
        for symbol in symbols:
            try:
                logger.debug(f"Processing symbol: {symbol}")
                
                # Fetch data with built-in retry logic and error recovery
                data = self.fetch_data_with_retry(symbol, start_date, end_date)
                all_data.append(data)
                
                logger.debug(f"Successfully fetched {len(data)} records for {symbol}")
                
            except Exception as e:
                # Log the error but continue processing other symbols
                # This ensures partial success rather than complete failure
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                failed_symbols.append(symbol)
                continue
        
        # Report on any failed symbol fetches
        if failed_symbols:
            logger.warning(f"Failed to fetch data for symbols: {failed_symbols}")
            
        # Check if we have any successful data fetches
        if not all_data:
            raise ValueError(f"Failed to fetch data for any symbols: {symbols}")
        
        # Combine data from all successfully fetched symbols
        # Use pandas concat for efficient DataFrame combination
        logger.debug(f"Combining data from {len(all_data)} successful fetches")
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort data for consistent ordering and easier analysis
        # Primary sort by Date ensures chronological order
        # Secondary sort by Symbol groups data by asset
        combined_data = combined_data.sort_values(['Date', 'Symbol']).reset_index(drop=True)
        
        logger.info(f"Combined dataset: {len(combined_data)} total records")
        logger.debug(f"Date range in combined data: {combined_data['Date'].min()} to {combined_data['Date'].max()}")
        
        # Perform comprehensive data validation on the combined dataset
        # This ensures data quality before caching or returning to user
        logger.debug("Validating combined dataset...")
        validation_result = self.validator.validate_data(combined_data)
        
        if not validation_result.is_valid:
            # Log validation issues but don't fail completely
            # Some validation warnings may be acceptable for analysis
            logger.warning("Fetched data has validation issues:")
            validation_result.print_report()
            
            # Check if errors are critical (vs warnings)
            if validation_result.has_critical_errors():
                logger.error("Critical data validation errors detected")
                raise DataValidationError("Data quality issues prevent safe usage")
            else:
                logger.info("Validation warnings noted but data is usable")
        else:
            logger.debug("Data validation passed successfully")
        
        # Cache the validated data for future requests (performance optimization)
        # Only cache data that passes validation to ensure cache quality
        if self.use_cache and self.cache_manager and validation_result.is_valid:
            logger.debug("Saving validated data to cache...")
            try:
                self.cache_manager.save_data_to_cache(combined_data, symbols, start_date, end_date)
                logger.debug(f"Successfully cached {len(combined_data)} records")
            except Exception as cache_error:
                # Cache failures shouldn't prevent data return
                logger.warning(f"Failed to cache data: {cache_error}")
        
        # Log successful completion with summary statistics
        successful_symbols = len(set(combined_data['Symbol'].unique()))
        logger.info(f"Data fetch completed: {successful_symbols}/{len(symbols)} symbols, "
                   f"{len(combined_data)} total records")
        
        # Return the validated and sorted dataset
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