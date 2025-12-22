from typing import Any, Dict, Union, List, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from src.benchmark.dataset import StockDataset
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

    risk_aversion : float
    alpha : Union[float, None] # penalty coefficient

    spin_count : Union[int, None]
    bits_per_asset : Union[int, None]
    slack_variable_bits : Union[int, None]
    
    upper_bound_for_all_stocks : Union[List[int],None]

class Benchmark(ABC):
    def __init__(self, optimizer: Any, optimizer_args: Union[Dict[str, Any],None], benchmark_config: BenchmarkConfig):
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        if self.optimizer_args is None:
            self.optimizer_args = {}
        self.benchmark_config = benchmark_config
        self.budget = benchmark_config.start_budget
        self.date = self.benchmark_config.start_date
        self.dataset = StockDataset(data_dir="data", 
                                    stock_list=benchmark_config.stock_list)
        self.dataset.set_date(self.benchmark_config.start_date)
        
        if self.benchmark_config.stock_list is not None:
            self.benchmark_config.asset_count = len(benchmark_config.stock_list)
            print(f"Resetting asset count to {self.benchmark_config.asset_count}: asset count should be equal to stock list length")
    
    @abstractmethod
    def _optimize(self, mu, open_prices, sigma):
        pass

    @abstractmethod
    def run(self):
        pass

    def run(self):
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
            
            best_x = self._optimize(mu, open_prices, sigma)
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
            plt.savefig("result/result.png", dpi=600)
            plt.show()
        else:
            print("No budget history to plot.")

class QuantumBenchmark(Benchmark):
    def __init__(self, optimizer: Callable[..., Union[np.array, List, None]], optimizer_args: Dict, benchmark_config: BenchmarkConfig):
        super().__init__(optimizer, optimizer_args, benchmark_config)
        assert benchmark_config.alpha is not None, "Alpha (penalty coefficient) should not be none"
        assert benchmark_config.spin_count is not None, "Spin count should not be none"
        assert benchmark_config.bits_per_asset is not None, "Bits per asset should not be none"

    def _convert_data_to_optimizer_input(self, mu, prices, sigma):
        n = self.benchmark_config.asset_count
        N = self.benchmark_config.spin_count
        K = self.benchmark_config.bits_per_asset
        lam = self.benchmark_config.risk_aversion
        alpha = self.benchmark_config.alpha
        B = self.budget
        Ks = self.benchmark_config.slack_variable_bits

        def H_factor(s):
            # s is a list of spins (+1/-1), we need to map it to binary variables
            v = [(s[i]+1)//2 for i in range(len(s))]

            H = 0.0
            for i in range(n):
                for j in range(n):
                    for p1 in range(K):
                        for p2 in range(K):
                            idx_i = i*K + p1
                            idx_j = j*K + p2
                            coeff = (lam * sigma[i,j] + alpha * prices[i] * prices[j]) * (2**p1) * (2**p2)
                            H += coeff * v[idx_i] * v[idx_j]
            
            for i in range(n):
                for p in range(K):
                    idx = i*K + p
                    coeff = - (mu[i] + 2 * alpha * B * prices[i]) * (2**p)
                    H += coeff * v[idx]

            for i in range(n):
                for p1 in range(K):
                    for p2 in range(Ks):
                        idx1 = i*K + p1
                        idx2 = n*K + p2
                        coeff = 2 * alpha * prices[i] * (2**p1) * (2**p2)
                        H += coeff * v[idx1] * v[idx2]

            for p1 in range(Ks):
                for p2 in range(Ks):
                    idx1 = n*K + p1
                    idx2 = n*K + p2
                    coeff =alpha * (2**p1) * (2**p2)
                    H += coeff * v[idx1] * v[idx2]

            for p in range(Ks):
                idx = n*K + p
                coeff = - alpha * (2 * B) * (2**p)
                H += coeff * v[idx]

            H += alpha * B * B
            return H
    
        configs = np.array(list(itertools.product([1,-1], repeat=N)))
        H_values = np.array([H_factor(s) for s in configs])

        num_terms = 1 + N + N*(N-1)//2 
        X = np.ones((2**N, num_terms))
        X[:,1:1+N] = configs

        idx = 1 + N
        for i in range(N):
            for j in range(i+1, N):
                X[:, idx] = configs[:,i]*configs[:,j]
                idx +=1

        coeffs, *_ = np.linalg.lstsq(X, H_values, rcond=None)

        C = coeffs[0]
        h = coeffs[1:1+N]
        J = np.zeros((N,N))
        idx = 1 + N
        for i in range(N):
            for j in range(i+1, N):
                J[i,j] = coeffs[idx]
                idx +=1

        return h, J, C

    def get_current_hamiltonian(self):
        mu = np.array(self.dataset.get_mu(self.benchmark_config.history_window))
        sigma = np.array(self.dataset.get_cov(self.benchmark_config.history_window))
        open_prices = np.array(self.dataset.get_open_price())

        return self._convert_data_to_optimizer_input(mu, open_prices, sigma)

    def optimize(self):
        h, J, C = self.get_current_hamiltonian()
        return self.optimizer(h, J, C, self.budget, self.benchmark_config, **self.optimizer_args)
    
    def _optimize(self, mu, open_prices, sigma):
        h, J, C = self._convert_data_to_optimizer_input(mu, open_prices, sigma)
        return self.optimizer(h, J, C, self.budget, self.benchmark_config, **self.optimizer_args)
    
    def run(self):
        return super().run()


class ClassicalBenchmark(Benchmark):
    def __init__(self, optimizer: Callable[..., Union[np.array, List, None]], optimizer_args: Dict, benchmark_config: BenchmarkConfig):
        super().__init__(optimizer, optimizer_args, benchmark_config)
    
    def _optimize(self, mu, open_prices, sigma):
        return self.optimizer(mu, open_prices, sigma, self.budget, self.benchmark_config, **self.optimizer_args)
    
    def run(self):
        return super().run()

if __name__ == "__main__":
    pass
