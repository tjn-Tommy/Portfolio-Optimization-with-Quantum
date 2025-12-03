import pandas as pd
import glob
import os, sys

sys.path.append("..")
data_path = "data/*.csv"  

def parse_stock_csv(filename):
    df_raw = pd.read_csv(filename, header=None)

    # Row 0 = Price types (Close / High / Low...)
    types = df_raw.iloc[0, 1:].values
    
    # Row 1 = Ticker names
    tickers = df_raw.iloc[1, 1:].values
    
    # Row 2 = "Date" label row (ignored)
    # Row 3+ = data
    df_data = df_raw.iloc[3:].reset_index(drop=True)

    df_data.columns = ["Date"] + list(df_raw.columns[1:])
    df_data["Date"] = pd.to_datetime(df_data["Date"])

    close_cols = []
    close_tickers = []
    for col, ttype, ticker in zip(df_raw.columns[1:], types, tickers):
        if ttype == "Close":
            close_cols.append(col)
            close_tickers.append(ticker)

    df_close = df_data[["Date"] + close_cols].copy()

    df_close.columns = ["Date"] + close_tickers

    for t in close_tickers:
        df_close[t] = pd.to_numeric(df_close[t], errors="coerce")

    return df_close.set_index("Date")

all_close = None

for file in glob.glob(data_path):
    print(f"Processing {file} ...")
    df = parse_stock_csv(file)

    if all_close is None:
        all_close = df
    else:
        all_close = all_close.join(df, how="outer")

all_close = all_close.dropna(axis=1, how="all")
all_close = all_close.sort_index()

returns = all_close.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

all_close.to_csv("data/all_close_prices.csv")
returns.to_csv("data/returns.csv")
mean_returns.to_csv("data/mean.csv")
cov_matrix.to_csv("data/cov.csv")

print("\nDone!")
print("Generated files:")
print("- all_close_prices.csv")
print("- returns.csv")
print("- mean.csv")
print("- cov.csv")