import pandas as pd
import glob
import os, sys

sys.path.append("..")
data_path = "data/*.csv"  

# 定义脚本生成的输出文件列表，这些文件不应被再次处理为原始输入
generated_files = [
    os.path.join("data", "all_close_prices.csv"),
    os.path.join("data", "all_open_prices.csv"),
    os.path.join("data", "returns.csv"),
    os.path.join("data", "mean.csv"),
    os.path.join("data", "cov.csv")
]

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
    # 建议：如果日期格式固定，可以指定format以避免UserWarning
    # 例如：df_data["Date"] = pd.to_datetime(df_data["Date"], format="%Y-%m-%d")
    df_data["Date"] = pd.to_datetime(df_data["Date"])

    close_cols = []
    close_tickers = []
    open_cols = []  # 新增：用于存储"Open"价格的列
    open_tickers = [] # 新增：用于存储"Open"价格的股票代码

    for col, ttype, ticker in zip(df_raw.columns[1:], types, tickers):
        if ttype == "Close":
            close_cols.append(col)
            close_tickers.append(ticker)
        elif ttype == "Open": # 新增：检查是否是"Open"价格类型
            open_cols.append(col)
            open_tickers.append(ticker)

    df_close = df_data[["Date"] + close_cols].copy()
    df_close.columns = ["Date"] + close_tickers
    for t in close_tickers:
        df_close[t] = pd.to_numeric(df_close[t], errors="coerce")

    df_open = df_data[["Date"] + open_cols].copy() # 新增：创建"Open"价格的DataFrame
    df_open.columns = ["Date"] + open_tickers
    for t in open_tickers:
        df_open[t] = pd.to_numeric(df_open[t], errors="coerce")

    return df_close.set_index("Date"), df_open.set_index("Date") # 返回"Close"和"Open"两个DataFrame

all_close = None
all_open = None # 新增：用于累积所有"Open"价格的DataFrame

for file in glob.glob(data_path):
    # 检查当前文件是否是生成的输出文件，如果是则跳过
    if file in generated_files:
        print(f"Skipping generated file: {file}")
        continue

    print(f"Processing {file} ...")
    df_close, df_open = parse_stock_csv(file) # 解包返回的两个DataFrame

    if all_close is None:
        all_close = df_close
    else:
        all_close = all_close.join(df_close, how="outer")

    if all_open is None: # 新增：处理all_open的累积
        all_open = df_open
    else:
        all_open = all_open.join(df_open, how="outer")

all_close = all_close.dropna(axis=1, how="all")
all_close = all_close.sort_index()

all_open = all_open.dropna(axis=1, how="all") # 新增：对all_open进行数据清理
all_open = all_open.sort_index()

returns = all_close.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

all_close.to_csv("data/all_close_prices.csv")
all_open.to_csv("data/all_open_prices.csv") # 新增：保存all_open_prices.csv
returns.to_csv("data/returns.csv")
mean_returns.to_csv("data/mean.csv")
cov_matrix.to_csv("data/cov.csv")

print("\nDone!")
print("Generated files:")
print("- all_close_prices.csv")
print("- all_open_prices.csv") # 新增：打印生成的文件名
print("- returns.csv")
print("- mean.csv")
print("- cov.csv")