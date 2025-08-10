"""Common utility functions for the portfolio forecasting system."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import json
from datetime import datetime, timedelta


def ensure_directory(path: str) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_pickle(obj: Any, filepath: str) -> None:
    """Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """Calculate returns from price series.
    
    Args:
        prices: Price series
        method: Return calculation method ('simple' or 'log')
        
    Returns:
        Returns series
    """
    if method == 'simple':
        return prices.pct_change().dropna()
    elif method == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Method must be 'simple' or 'log'")


def calculate_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """Calculate rolling volatility from returns.
    
    Args:
        returns: Returns series
        window: Rolling window size
        
    Returns:
        Volatility series
    """
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
    volatility = returns.std() * np.sqrt(252)  # Annualized
    
    if volatility == 0:
        return 0.0
    
    return excess_returns / volatility


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns.
    
    Args:
        returns: Returns series
        
    Returns:
        Maximum drawdown as a positive number
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return abs(drawdown.min())


def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk (VaR).
    
    Args:
        returns: Returns series
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        
    Returns:
        VaR value
    """
    return np.percentile(returns, confidence_level * 100)


def validate_date_format(date_str: str) -> bool:
    """Validate date string format (YYYY-MM-DD).
    
    Args:
        date_str: Date string to validate
        
    Returns:
        True if valid format, False otherwise
    """
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def get_trading_days(start_date: str, end_date: str) -> int:
    """Calculate number of trading days between dates.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        Number of trading days
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create business day range
    business_days = pd.bdate_range(start=start, end=end)
    return len(business_days)


def split_data_chronologically(
    data: pd.DataFrame,
    train_end_date: str,
    test_start_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train and test sets.
    
    Args:
        data: DataFrame with datetime index
        train_end_date: End date for training data
        test_start_date: Start date for test data
        
    Returns:
        Tuple of (train_data, test_data)
    """
    train_data = data[data.index <= train_end_date]
    test_data = data[data.index >= test_start_date]
    
    return train_data, test_data


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string.
    
    Args:
        value: Value to format (e.g., 0.1234)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string (e.g., "12.34%")
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, currency: str = "$") -> str:
    """Format value as currency string.
    
    Args:
        value: Value to format
        currency: Currency symbol
        
    Returns:
        Formatted currency string
    """
    return f"{currency}{value:,.2f}"


def get_file_age_hours(filepath: str) -> float:
    """Get age of file in hours.
    
    Args:
        filepath: Path to file
        
    Returns:
        Age in hours, or float('inf') if file doesn't exist
    """
    path = Path(filepath)
    if not path.exists():
        return float('inf')
    
    modified_time = datetime.fromtimestamp(path.stat().st_mtime)
    age = datetime.now() - modified_time
    return age.total_seconds() / 3600