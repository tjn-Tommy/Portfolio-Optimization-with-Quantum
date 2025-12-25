from typing import Optional, Union, List, Callable, Sequence, Tuple, Dict, Any
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.dataset import StockDataset
from optimizer.base import BaseOptimizer

@dataclass
class BenchmarkConfig:
    start_date : str
    start_budget : int
    max_iter : int
    asset_count : int
    stock_list : Union[None, List[str]]
    history_window : int
    alpha : Optional[float] = None # penalty coefficient
    data_dir : Optional[str] = None
    result_dir : Optional[str] = None

    @classmethod
    def init(cls, config: Dict[str, Any]) -> "BenchmarkConfig":
        data_cfg = config.get("data", {})
        problem_cfg = config.get("problem", {})

        stock_list = data_cfg.get("stock_list")
        asset_count = data_cfg.get("asset_count")
        if asset_count is None:
            asset_count = len(stock_list) if stock_list else 0

        return cls(
            start_date=data_cfg["start_date"],
            start_budget=problem_cfg["start_budget"],
            max_iter=data_cfg["max_iter"],
            asset_count=asset_count,
            stock_list=stock_list,
            history_window=data_cfg["history_window"],
            data_dir=data_cfg.get("data_dir"),
            result_dir=data_cfg.get("result_dir"),
        )

class Benchmark():
    def __init__(
            self, 
            benchmark_config: BenchmarkConfig
            ):
        self.benchmark_config = benchmark_config
        self.dataset = StockDataset(data_dir=self.benchmark_config.data_dir if self.benchmark_config.data_dir else "./data", 
                                    stock_list=self.benchmark_config.stock_list)
        
        if self.benchmark_config.stock_list is not None:
            self.benchmark_config.asset_count = len(self.benchmark_config.stock_list)
            print(f"Resetting asset count to {self.benchmark_config.asset_count}: asset count should be equal to stock list length")
    
    def _resolve_optimizer(self, optimizer) -> Tuple[str, Callable]:
        if isinstance(optimizer, BaseOptimizer):
            return optimizer.__class__.__name__, optimizer
        if callable(optimizer):
            name = getattr(optimizer, "__name__", optimizer.__class__.__name__)
            return name, optimizer
        raise TypeError("Optimizer must be a BaseOptimizer or a callable.")

    def _normalize_optimizers(
        self,
        optimizer_input,
    ) -> List[Tuple[str, Callable]]:
        if isinstance(optimizer_input, (list, tuple)):
            items = list(optimizer_input)
        else:
            items = [optimizer_input]

        resolved = []
        for item in items:
            resolved.append(self._resolve_optimizer(item))
        return resolved

    def _sanitize_name(self, name: str) -> str:
        if not name:
            return "optimizer"
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)

    def _plot_history(
        self,
        name: str,
        date_history: Sequence[pd.Timestamp],
        budget_history: Sequence[float],
        timestamp: str,
    ) -> None:
        if not date_history:
            print("No budget history to plot.")
            return

        result_dir = self.benchmark_config.result_dir or "result"
        os.makedirs(result_dir, exist_ok=True)
        safe_name = self._sanitize_name(name)
        filename = f"{safe_name}_{timestamp}.png"

        plt.figure(figsize=(12, 6))
        title = "Budget Evolution Over Time"
        if name:
            title = f"{title} ({name})"
        plt.title(title)
        plt.plot(date_history, budget_history, marker="o", linestyle="-")
        plt.xlabel("Date")
        plt.ylabel("Budget")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, filename), dpi=600)
        plt.close()

    def _plot_compare(
        self,
        results: Sequence[dict],
        timestamp: str,
    ) -> None:
        result_dir = self.benchmark_config.result_dir or "result"
        os.makedirs(result_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))
        has_series = False
        for result in results:
            name = result.get("name") or "optimizer"
            dates = result.get("date_history", [])
            budgets = result.get("budget_history", [])
            if not dates:
                continue
            has_series = True
            plt.plot(dates, budgets, marker="o", linestyle="-", label=name)

        if not has_series:
            print("No budget history to plot.")
            return

        plt.title("Budget Evolution Comparison")
        plt.xlabel("Date")
        plt.ylabel("Budget")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        filename = f"compare_{timestamp}.png"
        plt.savefig(os.path.join(result_dir, filename), dpi=600)
        plt.close()

    def _run_single(
        self,
        optimizer: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
        name: str,
        **kwargs,
    ) -> dict:
        budget = self.benchmark_config.start_budget
        self.dataset.set_date(self.benchmark_config.start_date)
        current_date = pd.to_datetime(self.benchmark_config.start_date)

        budget_history = [budget]
        date_history = [current_date]

        current_date = self.dataset.next_date()
        iterations = 0
        prefix = f"[{name}] " if name else ""

        while current_date is not None and self.dataset.has_next() and iterations < self.benchmark_config.max_iter:
            mu = np.array(self.dataset.get_mu(self.benchmark_config.history_window))
            sigma = np.array(self.dataset.get_cov(self.benchmark_config.history_window))
            open_prices = np.array(self.dataset.get_open_price())

            if mu is None or sigma is None or open_prices is None or len(mu) == 0:
                break

            best_x = optimizer(mu, open_prices, sigma, budget, **kwargs)
            print(f"{prefix}On date {current_date.strftime('%Y-%m-%d')} best x is {best_x}")

            if best_x is None:
                print(f"{prefix}Optimization failed for date {current_date.strftime('%Y-%m-%d')}. Stopping benchmark.")
                break

            close_prices = self.dataset.get_close_price()
            if close_prices is None:
                print(f"{prefix}Could not get close prices for date {current_date.strftime('%Y-%m-%d')}. Stopping benchmark.")
                break

            try:
                budget = budget + best_x @ (close_prices - open_prices)
            except ValueError as e:
                print(f"{prefix}Error calculating new budget on date {current_date.strftime('%Y-%m-%d')}: {e}")
                print(f"{prefix}best_x shape: {best_x.shape}, close_prices shape: {close_prices.shape}")
                break

            budget_history.append(budget)
            date_history.append(pd.to_datetime(current_date))

            current_date = self.dataset.next_date()
            iterations += 1

        return {
            "name": name,
            "date_history": date_history,
            "budget_history": budget_history,
        }

    def run(
        self,
        optimizer,
        **kwargs,
    ):
        optimizers = self._normalize_optimizers(optimizer)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results = []

        for name, opt_fn in optimizers:
            result = self._run_single(opt_fn, name, **kwargs)
            self._plot_history(name, result["date_history"], result["budget_history"], timestamp)
            results.append(result)

        if len(results) > 1:
            self._plot_compare(results, timestamp)

        return results
