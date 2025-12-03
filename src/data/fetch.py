import yfinance as yf
import numpy as np

# 1. list of tickers
hybird_tickers = ['NVDA', 'JPM', 'JNJ', 'PG', 'XOM', 'BA', 'WMT']
tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']
# 2. download historical data (usually past 1-3 years of closing prices)
data = yf.download(hybird_tickers, start="2022-01-01", end="2025-01-01")
tech_data = yf.download(tech_tickers, start="2022-01-01", end="2025-01-01")

# 3. store data to csv
data.to_csv('hybrid_historical_data.csv')
tech_data.to_csv('tech_historical_data.csv')