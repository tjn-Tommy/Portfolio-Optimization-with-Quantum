from typing import Any, Dict, Optional, Union, List, Callable
from dataclasses import dataclass
from data.dataset import StockDataset
import numpy as np
import itertools
from qiskit import QuantumCircuit
import pandas as pd
import matplotlib.pyplot as plt
import os

@dataclass
class BenchmarkConfig:
    start_date : str
    start_budget : int
    max_iter : int
    asset_count : int
    stock_list : Union[None, List[str]]
    history_window : int
    alpha : Union[float, None] # penalty coefficient
    data_dir : Optional[str] = None

class Benchmark():
    def __init__(
            self, 
            benchmark_config: BenchmarkConfig
            ):
        self.benchmark_config = benchmark_config
        self.budget = benchmark_config.start_budget
        self.date = self.benchmark_config.start_date
        self.dataset = StockDataset(data_dir=self.benchmark_config.data_dir if self.benchmark_config.data_dir else "./data", 
                                    stock_list=self.benchmark_config.stock_list)
        self.dataset.set_date(self.benchmark_config.start_date)
        
        if self.benchmark_config.stock_list is not None:
            self.benchmark_config.asset_count = len(self.benchmark_config.stock_list)
            print(f"Resetting asset count to {self.benchmark_config.asset_count}: asset count should be equal to stock list length")
    

    def run(
            self,
            optimizer: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
            **kwargs,
            ):
        budget_history = []
        date_history = []
        budget_history.append(self.budget)
        date_history.append(pd.to_datetime(self.date))

        self.date = self.dataset.next_date()
        iterations = 0

        while self.dataset.has_next() and iterations < self.benchmark_config.max_iter:
            mu = np.array(self.dataset.get_mu(self.benchmark_config.history_window))
            sigma = np.array(self.dataset.get_cov(self.benchmark_config.history_window))
            open_prices = np.array(self.dataset.get_open_price())

            if mu is None or sigma is None or open_prices is None or len(mu) == 0:
                break
            
            best_x = optimizer(mu, open_prices, sigma, self.budget, **kwargs)
            print(f"On date {self.date.strftime('%Y-%m-%d')} best x is {best_x}") 

            if best_x is None:
                print(f"Optimization failed for date {self.date.strftime('%Y-%m-%d')}. Stopping benchmark.")
                break

            close_prices = self.dataset.get_close_price()
            if close_prices is None:
                print(f"Could not get close prices for date {self.date.strftime('%Y-%m-%d')}. Stopping benchmark.")
                break

            try:
                self.budget = self.budget + best_x @ (close_prices - open_prices) # 使用 .values 确保是 numpy 数组
            except ValueError as e:
                print(f"Error calculating new budget on date {self.date.strftime('%Y-%m-%d')}: {e}")
                print(f"best_x shape: {best_x.shape}, close_prices shape: {close_prices.shape}")
                break

            budget_history.append(self.budget)
            date_history.append(pd.to_datetime(self.date))

            self.date = self.dataset.next_date()
            iterations += 1
        
        if date_history:
            plt.figure(figsize=(12, 6))
            plt.plot(date_history, budget_history, marker='o', linestyle='-')
            plt.title('Budget Evolution Over Time')
            plt.xlabel('Date')
            plt.ylabel('Budget')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            os.makedirs("result", exist_ok=True)
            # Add real time date to filename
            plt.savefig("result/result"+pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")+".png", dpi=600)
        else:
            print("No budget history to plot.")

