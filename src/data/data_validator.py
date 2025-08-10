"""Data validation utilities for the portfolio forecasting system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation process."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]
    
    def __post_init__(self):
        """Add convenience properties."""
        self.has_errors = len(self.errors) > 0
        self.has_warnings = len(self.warnings) > 0
    
    def print_report(self) -> None:
        """Print validation report."""
        print(f"Validation Result: {'FAILED' if not self.is_valid else 'PASSED'}")
        print(f"Errors: {len(self.errors)}, Warnings: {len(self.warnings)}")
        
        if self.errors:
            print("\nERRORS:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.summary:
            print("\nSUMMARY:")
            for key, value in self.summary.items():
                print(f"  {key}: {value}")


class DataValidator:
    """Validates financial data integrity and completeness."""
    
    def __init__(self):
        """Initialize data validator."""
        self.required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Symbol']
        self.numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        self.valid_symbols = ['TSLA', 'BND', 'SPY']
    
    def validate_data(self, data: pd.DataFrame) -> ValidationResult:
        """Comprehensive data validation.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        summary = {}
        
        # Basic structure validation
        structure_errors = self._validate_structure(data)
        errors.extend(structure_errors)
        
        if not structure_errors:  # Only proceed if structure is valid
            # Data type validation
            type_errors = self._validate_data_types(data)
            errors.extend(type_errors)
            
            # Missing values validation
            missing_warnings = self._validate_missing_values(data)
            warnings.extend(missing_warnings)
            
            # Duplicate validation
            duplicate_errors = self._validate_duplicates(data)
            errors.extend(duplicate_errors)
            
            # Symbol validation
            symbol_errors = self._validate_symbols(data)
            errors.extend(symbol_errors)
            
            # Date validation
            date_errors, date_warnings = self._validate_dates(data)
            errors.extend(date_errors)
            warnings.extend(date_warnings)
            
            # Price validation
            price_warnings = self._validate_prices(data)
            warnings.extend(price_warnings)
            
            # Volume validation
            volume_warnings = self._validate_volume(data)
            warnings.extend(volume_warnings)
            
            # Generate summary
            summary = self._generate_summary(data)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            summary=summary
        )
    
    def _validate_structure(self, data: pd.DataFrame) -> List[str]:
        """Validate DataFrame structure."""
        errors = []
        
        if data.empty:
            errors.append("DataFrame is empty")
            return errors
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(data.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        return errors
    
    def _validate_data_types(self, data: pd.DataFrame) -> List[str]:
        """Validate data types."""
        errors = []
        
        # Check if Date column can be converted to datetime
        try:
            pd.to_datetime(data['Date'])
        except Exception as e:
            errors.append(f"Date column cannot be converted to datetime: {e}")
        
        # Check numeric columns
        for col in self.numeric_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    try:
                        pd.to_numeric(data[col], errors='coerce')
                    except Exception as e:
                        errors.append(f"Column {col} cannot be converted to numeric: {e}")
        
        return errors
    
    def _validate_missing_values(self, data: pd.DataFrame) -> List[str]:
        """Check for missing values."""
        warnings = []
        
        missing_counts = data.isnull().sum()
        total_rows = len(data)
        
        for col, missing_count in missing_counts.items():
            if missing_count > 0:
                percentage = (missing_count / total_rows) * 100
                warnings.append(f"Column {col} has {missing_count} missing values ({percentage:.2f}%)")
        
        return warnings
    
    def _validate_duplicates(self, data: pd.DataFrame) -> List[str]:
        """Check for duplicate records."""
        errors = []
        
        # Check for duplicate Date-Symbol combinations
        if 'Date' in data.columns and 'Symbol' in data.columns:
            duplicates = data.duplicated(subset=['Date', 'Symbol'], keep=False)
            if duplicates.any():
                duplicate_count = duplicates.sum()
                errors.append(f"Found {duplicate_count} duplicate Date-Symbol combinations")
        
        return errors
    
    def _validate_symbols(self, data: pd.DataFrame) -> List[str]:
        """Validate asset symbols."""
        errors = []
        
        if 'Symbol' in data.columns:
            unique_symbols = data['Symbol'].unique()
            invalid_symbols = set(unique_symbols) - set(self.valid_symbols)
            
            if invalid_symbols:
                errors.append(f"Invalid symbols found: {invalid_symbols}")
        
        return errors
    
    def _validate_dates(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate date ranges and consistency."""
        errors = []
        warnings = []
        
        if 'Date' not in data.columns:
            return errors, warnings
        
        try:
            dates = pd.to_datetime(data['Date'])
            
            # Check for future dates
            today = pd.Timestamp.now()
            future_dates = dates > today
            if future_dates.any():
                future_count = future_dates.sum()
                warnings.append(f"Found {future_count} future dates")
            
            # Check date range
            min_date = dates.min()
            max_date = dates.max()
            
            # Expected range based on project requirements
            expected_start = pd.Timestamp('2015-07-01')
            expected_end = pd.Timestamp('2025-07-31')
            
            if min_date > expected_start:
                warnings.append(f"Data starts later than expected: {min_date.date()} vs {expected_start.date()}")
            
            if max_date < expected_end and max_date < today:
                warnings.append(f"Data ends earlier than expected: {max_date.date()} vs {expected_end.date()}")
            
        except Exception as e:
            errors.append(f"Error validating dates: {e}")
        
        return errors, warnings
    
    def _validate_prices(self, data: pd.DataFrame) -> List[str]:
        """Validate price data consistency."""
        warnings = []
        
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        available_price_cols = [col for col in price_columns if col in data.columns]
        
        if len(available_price_cols) < 4:
            return warnings
        
        for idx, row in data.iterrows():
            try:
                open_price = row.get('Open', 0)
                high_price = row.get('High', 0)
                low_price = row.get('Low', 0)
                close_price = row.get('Close', 0)
                
                # Skip if any price is missing or zero
                if any(pd.isna([open_price, high_price, low_price, close_price])) or \
                   any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                    continue
                
                # High should be >= Open, Close, Low
                if high_price < max(open_price, close_price, low_price):
                    warnings.append(f"Row {idx}: High price ({high_price}) is less than other prices")
                
                # Low should be <= Open, Close, High
                if low_price > min(open_price, close_price, high_price):
                    warnings.append(f"Row {idx}: Low price ({low_price}) is greater than other prices")
                
                # Check for extreme price movements (>50% in one day)
                if abs(close_price - open_price) / open_price > 0.5:
                    warnings.append(f"Row {idx}: Extreme price movement detected ({((close_price - open_price) / open_price * 100):.1f}%)")
                
            except Exception:
                continue  # Skip problematic rows
        
        return warnings
    
    def _validate_volume(self, data: pd.DataFrame) -> List[str]:
        """Validate volume data."""
        warnings = []
        
        if 'Volume' not in data.columns:
            return warnings
        
        # Check for negative volumes
        negative_volumes = data['Volume'] < 0
        if negative_volumes.any():
            negative_count = negative_volumes.sum()
            warnings.append(f"Found {negative_count} negative volume values")
        
        # Check for zero volumes (might be normal for some assets)
        zero_volumes = data['Volume'] == 0
        if zero_volumes.any():
            zero_count = zero_volumes.sum()
            total_rows = len(data)
            percentage = (zero_count / total_rows) * 100
            if percentage > 5:  # More than 5% zero volumes might be concerning
                warnings.append(f"Found {zero_count} zero volume values ({percentage:.2f}%)")
        
        return warnings
    
    def _generate_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate data summary statistics."""
        summary = {}
        
        try:
            summary['total_rows'] = len(data)
            summary['total_columns'] = len(data.columns)
            
            if 'Symbol' in data.columns:
                summary['unique_symbols'] = data['Symbol'].nunique()
                summary['symbols'] = sorted(data['Symbol'].unique().tolist())
            
            if 'Date' in data.columns:
                dates = pd.to_datetime(data['Date'])
                summary['date_range'] = {
                    'start': dates.min().strftime('%Y-%m-%d'),
                    'end': dates.max().strftime('%Y-%m-%d'),
                    'total_days': (dates.max() - dates.min()).days
                }
            
            # Missing value summary
            missing_summary = {}
            for col in data.columns:
                missing_count = data[col].isnull().sum()
                if missing_count > 0:
                    missing_summary[col] = missing_count
            
            if missing_summary:
                summary['missing_values'] = missing_summary
            
        except Exception as e:
            summary['error'] = f"Error generating summary: {e}"
        
        return summary
    
    def validate_symbols_list(self, symbols: List[str]) -> ValidationResult:
        """Validate a list of symbols.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        if not symbols:
            errors.append("Symbol list is empty")
        else:
            invalid_symbols = set(symbols) - set(self.valid_symbols)
            if invalid_symbols:
                errors.append(f"Invalid symbols: {invalid_symbols}")
        
        summary = {
            'total_symbols': len(symbols),
            'valid_symbols': len(set(symbols) & set(self.valid_symbols)),
            'symbols': symbols
        }
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            summary=summary
        )
    
    def quick_validate(self, data: pd.DataFrame) -> bool:
        """Quick validation check - returns True if data passes basic validation.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        result = self.validate_data(data)
        return result.is_valid