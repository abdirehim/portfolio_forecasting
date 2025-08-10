"""Tests for YFinance client."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.append('../../src')

from src.data.yfinance_client import YFinanceClient
from src.data.data_validator import ValidationResult


class TestYFinanceClient:
    """Test cases for YFinanceClient."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        self.client = YFinanceClient(use_cache=True, cache_expiry_hours=1)
        # Override cache directory for testing
        if self.client.cache_manager:
            self.client.cache_manager.cache_dir = Path(self.temp_dir)
            self.client.cache_manager.cache_dir.mkdir(exist_ok=True)
            (self.client.cache_manager.cache_dir / "data").mkdir(exist_ok=True)
            (self.client.cache_manager.cache_dir / "metadata").mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_symbols_valid(self):
        """Test symbol validation with valid symbols."""
        valid_symbols = ["TSLA", "BND", "SPY"]
        result = self.client.validate_symbols(valid_symbols)
        assert result == valid_symbols
    
    def test_validate_symbols_invalid(self):
        """Test symbol validation with invalid symbols."""
        invalid_symbols = ["INVALID", "FAKE"]
        
        with pytest.raises(ValueError) as exc_info:
            self.client.validate_symbols(invalid_symbols)
        
        assert "Invalid symbols" in str(exc_info.value)
    
    def test_validate_date_range_valid(self):
        """Test date range validation with valid dates."""
        start_date = "2020-01-01"
        end_date = "2020-12-31"
        
        result_start, result_end = self.client.validate_date_range(start_date, end_date)
        assert result_start == start_date
        assert result_end == end_date
    
    def test_validate_date_range_invalid_format(self):
        """Test date range validation with invalid format."""
        start_date = "2020/01/01"  # Wrong format
        end_date = "2020-12-31"
        
        with pytest.raises(ValueError) as exc_info:
            self.client.validate_date_range(start_date, end_date)
        
        assert "Invalid date format" in str(exc_info.value)
    
    def test_validate_date_range_wrong_order(self):
        """Test date range validation with wrong order."""
        start_date = "2020-12-31"
        end_date = "2020-01-01"
        
        with pytest.raises(ValueError) as exc_info:
            self.client.validate_date_range(start_date, end_date)
        
        assert "Start date must be before end date" in str(exc_info.value)
    
    def test_get_info(self):
        """Test getting symbol information."""
        # Mock the yfinance ticker
        with patch('src.data.yfinance_client.yf.Ticker') as mock_ticker:
            mock_info = {
                'longName': 'Tesla Inc',
                'sector': 'Consumer Discretionary',
                'industry': 'Auto Manufacturers',
                'marketCap': 800000000000,
                'currency': 'USD',
                'exchange': 'NASDAQ'
            }
            mock_ticker.return_value.info = mock_info
            
            result = self.client.get_info("TSLA")
            
            assert result['symbol'] == 'TSLA'
            assert result['longName'] == 'Tesla Inc'
            assert result['sector'] == 'Consumer Discretionary'
    
    def test_check_data_availability(self):
        """Test checking data availability."""
        symbols = ["TSLA"]
        start_date = "2020-01-01"
        end_date = "2020-01-31"
        
        # Mock the yfinance ticker
        with patch('src.data.yfinance_client.yf.Ticker') as mock_ticker:
            # Create mock data
            mock_data = pd.DataFrame({
                'Open': [100, 101, 102],
                'High': [105, 106, 107],
                'Low': [95, 96, 97],
                'Close': [103, 104, 105],
                'Volume': [1000, 1100, 1200]
            }, index=pd.date_range('2020-01-01', periods=3))
            
            mock_ticker.return_value.history.return_value = mock_data
            
            result = self.client.check_data_availability(symbols, start_date, end_date)
            
            assert result['TSLA'] is True
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        # Test that cache info works
        cache_info = self.client.get_cache_info()
        assert 'cache_dir' in cache_info
        assert cache_info['total_entries'] == 0
        
        # Test clearing cache
        cleared = self.client.clear_cache()
        assert cleared >= 0  # Should return number of cleared entries
    
    def test_client_without_cache(self):
        """Test client functionality without caching."""
        client_no_cache = YFinanceClient(use_cache=False)
        
        cache_info = client_no_cache.get_cache_info()
        assert cache_info['message'] == 'Caching is disabled'
        
        cleared = client_no_cache.clear_cache()
        assert cleared == 0


if __name__ == "__main__":
    pytest.main([__file__])