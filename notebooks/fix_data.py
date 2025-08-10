# Quick fix for the notebook data issue
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../src')

from data.yfinance_client import YFinanceClient
from analysis.eda_engine import EDAEngine

# Initialize client
client = YFinanceClient(use_cache=True, cache_expiry_hours=24)

# Fetch data
SYMBOLS = ['TSLA', 'BND', 'SPY']
START_DATE = '2015-07-01'
END_DATE = '2025-07-31'

print("📊 Fetching and fixing data...")

try:
    raw_data = client.fetch_data(
        symbols=SYMBOLS,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # Add Adj Close column if missing
    if 'Adj Close' not in raw_data.columns and 'Close' in raw_data.columns:
        raw_data['Adj Close'] = raw_data['Close']
        print("✅ Added 'Adj Close' column using 'Close' values")
    
    print(f"✅ Data ready: {raw_data.shape}")
    print(f"Columns: {list(raw_data.columns)}")
    
    # Test EDA engine
    eda_engine = EDAEngine(figsize=(12, 8))
    
    # Simple price plot
    plt.figure(figsize=(12, 6))
    for symbol in SYMBOLS:
        symbol_data = raw_data[raw_data['Symbol'] == symbol].sort_values('Date')
        plt.plot(symbol_data['Date'], symbol_data['Close'], label=symbol, linewidth=2)
    
    plt.title('Closing Prices - TSLA, BND, SPY')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print("✅ Basic plot created successfully!")
    
    # Summary stats
    print("\n📊 Data Summary:")
    for symbol in SYMBOLS:
        symbol_data = raw_data[raw_data['Symbol'] == symbol]
        returns = symbol_data['Close'].pct_change().dropna()
        
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        
        print(f"{symbol}: Return {annual_return:.2%}, Volatility {annual_vol:.2%}")
    
except Exception as e:
    print(f"❌ Error: {e}")