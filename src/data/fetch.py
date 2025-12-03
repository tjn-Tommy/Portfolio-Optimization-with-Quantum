import yfinance as yf
import numpy as np

# 1. list of tickers
tickers = ['NVDA', 'JPM', 'JNJ', 'PG', 'XOM']

# 2. download historical data (usually past 1-3 years of closing prices)
data = yf.download(tickers, start="2022-01-01", end="2025-01-01")

# 3. store data to csv
data.to_csv('historical_data.csv')