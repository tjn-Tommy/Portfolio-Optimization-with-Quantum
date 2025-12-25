from benchmark.benchmark import Benchmark, BenchmarkConfig
from data.dataset import StockDataset
from optimizer import *
import yaml
import numpy as np
import pandas as pd

def load_config(config_path: str) -> BenchmarkConfig:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return BenchmarkConfig(**config_dict)


if __name__ == "__main__":
    stock_list = ["AAPL","AMZN","GOOGL","META","MSFT"] #,"TSLA"]
    benchmark_config = BenchmarkConfig(
        start_date="2024-12-01",
        start_budget=1000,
        max_iter=50,
        asset_count=5,
        stock_list=stock_list,
        history_window=30,
        alpha=0.1,
        data_dir="./data",
    )
    benchmark = Benchmark(benchmark_config)
    # optimizer = QuantumAnnealingOptimizer(
    #     risk_aversion=0.5,
    #     lam=0.1,
    #     alpha=5.0,
    #     bits_per_asset=3,
    #     bits_slack=5,
    #     time=8,
    #     steps=80,
    #     traverse=1.0,
    # )
    optimizer = ScipOptimizer(
        risk_aversion=0.3,
        lam=0.1,
        upper_bounds=[8]*len(stock_list),
    )
    benchmark.run(optimizer.optimizer)