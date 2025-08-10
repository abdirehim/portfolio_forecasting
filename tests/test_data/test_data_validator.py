"""Tests for data validator."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.append('../../src')

from src.data.data_validator import DataValidator, ValidationResult


class TestDataValidator:
    """Test cases for DataValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
        
        # Create sample valid data
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        self.valid_data = pd.DataFrame({
            'Date': dates,
            'Open': [100 + i for i in range(10)],
            'High': [105 + i for i in range(10)],
            'Low': [95 + i for i in range(10)],
            'Close': [103 + i for i in range(10)],
            'Adj Close': [103 + i for i in range(10)],
            'Volume': [1000 + i*100 for i in range(10)],
            'Symbol': ['TSLA'] * 10
        })
    
    def test_validate_valid_data(self):
        """Test validation with valid data."""
        result = self.validator.validate_data(self.valid_data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.summary['total_rows'] == 10
        assert result.summary['unique_symbols'] == 1
        assert 'TSLA' in result.summary['symbols']
    
    def test_validate_empty_data(self):
        """Test validation with empty DataFrame."""
        empty_data = pd.DataFrame()
        result = self.validator.validate_data(empty_data)
        
        assert result.is_valid is False
        assert "DataFrame is empty" in result.errors
    
    def test_validate_missing_columns(self):
        """Test validation with missing required columns."""
        incomplete_data = self.valid_data.drop(columns=['Volume', 'Symbol'])
        result = self.validator.validate_data(incomplete_data)
        
        assert result.is_valid is False
        assert any("Missing required columns" in error for error in result.errors)
    
    def test_validate_invalid_symbols(self):
        """Test validation with invalid symbols."""
        invalid_data = self.valid_data.copy()
        invalid_data['Symbol'] = 'INVALID'
        
        result = self.validator.validate_data(invalid_data)
        
        assert result.is_valid is False
        assert any("Invalid symbols" in error for error in result.errors)
    
    def test_validate_duplicate_dates(self):
        """Test validation with duplicate date-symbol combinations."""
        duplicate_data = pd.concat([self.valid_data, self.valid_data.iloc[:2]], ignore_index=True)
        result = self.validator.validate_data(duplicate_data)
        
        assert result.is_valid is False
        assert any("duplicate Date-Symbol combinations" in error for error in result.errors)
    
    def test_validate_missing_values(self):
        """Test validation with missing values."""
        data_with_missing = self.valid_data.copy()
        data_with_missing.loc[0, 'Close'] = np.nan
        data_with_missing.loc[1, 'Volume'] = np.nan
        
        result = self.validator.validate_data(data_with_missing)
        
        # Should still be valid but have warnings
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("missing values" in warning for warning in result.warnings)
    
    def test_validate_future_dates(self):
        """Test validation with future dates."""
        future_data = self.valid_data.copy()
        future_date = datetime.now() + timedelta(days=30)
        future_data.loc[0, 'Date'] = future_date
        
        result = self.validator.validate_data(future_data)
        
        # Should be valid but have warnings about future dates
        assert result.is_valid is True
        assert any("future dates" in warning for warning in result.warnings)
    
    def test_validate_price_consistency(self):
        """Test validation of price consistency."""
        inconsistent_data = self.valid_data.copy()
        # Make high price lower than low price (invalid)
        inconsistent_data.loc[0, 'High'] = 90
        inconsistent_data.loc[0, 'Low'] = 95
        
        result = self.validator.validate_data(inconsistent_data)
        
        # Should be valid but have warnings about price inconsistency
        assert result.is_valid is True
        assert len(result.warnings) > 0
    
    def test_validate_negative_volume(self):
        """Test validation with negative volume."""
        negative_volume_data = self.valid_data.copy()
        negative_volume_data.loc[0, 'Volume'] = -100
        
        result = self.validator.validate_data(negative_volume_data)
        
        # Should be valid but have warnings about negative volume
        assert result.is_valid is True
        assert any("negative volume" in warning for warning in result.warnings)
    
    def test_validate_symbols_list(self):
        """Test validation of symbols list."""
        # Valid symbols
        valid_result = self.validator.validate_symbols_list(['TSLA', 'BND', 'SPY'])
        assert valid_result.is_valid is True
        assert len(valid_result.errors) == 0
        
        # Invalid symbols
        invalid_result = self.validator.validate_symbols_list(['INVALID', 'FAKE'])
        assert invalid_result.is_valid is False
        assert any("Invalid symbols" in error for error in invalid_result.errors)
        
        # Empty list
        empty_result = self.validator.validate_symbols_list([])
        assert empty_result.is_valid is False
        assert any("Symbol list is empty" in error for error in empty_result.errors)
    
    def test_quick_validate(self):
        """Test quick validation method."""
        # Valid data
        assert self.validator.quick_validate(self.valid_data) is True
        
        # Invalid data (empty)
        empty_data = pd.DataFrame()
        assert self.validator.quick_validate(empty_data) is False
    
    def test_validation_result_properties(self):
        """Test ValidationResult properties."""
        result = ValidationResult(
            is_valid=False,
            errors=['Error 1', 'Error 2'],
            warnings=['Warning 1'],
            summary={'test': 'value'}
        )
        
        assert result.has_errors is True
        assert result.has_warnings is True
        assert len(result.errors) == 2
        assert len(result.warnings) == 1


if __name__ == "__main__":
    pytest.main([__file__])