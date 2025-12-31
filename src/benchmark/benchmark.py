from typing import Optional, Union, List, Callable, Sequence, Tuple, Dict, Any
from dataclasses import dataclass
import os
import json
from pathlib import Path

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
    lam : float = 0.0 # risk aversion coefficient
    beta : float = 0.0 # transaction cost coefficient
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
            lam=problem_cfg.get("lam", 0.0),
            beta=problem_cfg.get("beta", 0.0),
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
        result: dict,
        timestamp: str,
    ) -> None:
        name = result.get("name", "optimizer")
        date_history = result.get("date_history", [])
        budget_history = result.get("budget_history", [])
        objective_history = result.get("objective", [])
        transaction_cost_history = result.get("transaction_cost_history", [])

        if not date_history:
            print("No history to plot.")
            return

        result_dir = self.benchmark_config.result_dir or "result"
        os.makedirs(result_dir, exist_ok=True)
        safe_name = self._sanitize_name(name)

        # Plot Budget
        filename_budget = f"{safe_name}_budget_{timestamp}.png"
        plt.figure(figsize=(12, 6))
        title = f"Budget Evolution Over Time ({name})"
        plt.title(title)
        plt.plot(date_history, budget_history, marker="o", linestyle="-")
        plt.xlabel("Date")
        plt.ylabel("Budget")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, filename_budget), dpi=600)
        plt.close()

        # Plot Objective
        if objective_history:
            # Ensure lengths match
            length = min(len(date_history), len(objective_history))
            filename_obj = f"{safe_name}_objective_{timestamp}.png"
            plt.figure(figsize=(12, 6))
            title = f"Objective Evolution Over Time ({name})"
            plt.title(title)
            plt.plot(date_history[:length], objective_history[:length], marker="o", linestyle="-", color="orange")
            plt.xlabel("Date")
            plt.ylabel("Objective")
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, filename_obj), dpi=600)
            plt.close()

        # Plot Transaction Cost
        if transaction_cost_history:
            length = min(len(date_history), len(transaction_cost_history))
            filename_tc = f"{safe_name}_transaction_cost_{timestamp}.png"
            plt.figure(figsize=(12, 6))
            title = f"Transaction Cost Evolution Over Time ({name})"
            plt.title(title)
            plt.plot(date_history[:length], transaction_cost_history[:length], marker="o", linestyle="-", color="green")
            plt.xlabel("Date")
            plt.ylabel("Transaction Cost")
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, filename_tc), dpi=600)
            plt.close()

    def _plot_compare(
        self,
        results: Sequence[dict],
        timestamp: str,
    ) -> None:
        result_dir = self.benchmark_config.result_dir or "result"
        os.makedirs(result_dir, exist_ok=True)

        def plot_metric(metric_key, title, ylabel, filename_suffix):
            plt.figure(figsize=(12, 6))
            has_series = False
            for result in results:
                name = result.get("name") or "optimizer"
                dates = result.get("date_history", [])
                values = result.get(metric_key, [])
                if not dates or not values:
                    continue
                length = min(len(dates), len(values))
                has_series = True
                plt.plot(dates[:length], values[:length], marker="o", linestyle="-", label=name)

            if not has_series:
                # print(f"No {metric_key} history to plot.")
                plt.close()
                return

            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            filename = f"compare_{filename_suffix}_{timestamp}.png"
            plt.savefig(os.path.join(result_dir, filename), dpi=600)
            plt.close()

        plot_metric("budget_history", "Budget Evolution Comparison", "Budget", "budget")
        plot_metric("objective", "Objective Evolution Comparison", "Objective", "objective")
        plot_metric("transaction_cost_history", "Transaction Cost Comparison", "Transaction Cost", "transaction_cost")

    def _run_single(
        self,
        optimizer: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
        name: str,
        **kwargs,
    ) -> dict:
        budget = self.benchmark_config.start_budget
        lam = self.benchmark_config.lam
        beta = self.benchmark_config.beta
        self.dataset.set_date(self.benchmark_config.start_date)
        current_date = pd.to_datetime(self.benchmark_config.start_date)

        budget_history = [budget]
        date_history = [current_date]
        objective = [0.0]
        transaction_cost_history = [0.0]
        best_xs = []
        latest_best_x = None

        current_date = self.dataset.next_date()
        iterations = 0
        prefix = f"[{name}] " if name else ""

        while current_date is not None and self.dataset.has_next() and iterations < self.benchmark_config.max_iter:
            mu = np.array(self.dataset.get_mu(self.benchmark_config.history_window))
            sigma = np.array(self.dataset.get_cov(self.benchmark_config.history_window))
            open_prices = np.array(self.dataset.get_open_price())

            if mu is None or sigma is None or open_prices is None or len(mu) == 0:
                break

            best_x = optimizer(mu, open_prices, sigma, budget, latest_best_x, **kwargs)
            # Calculate the objective:
            assets_change = best_x - (latest_best_x if latest_best_x is not None else np.zeros_like(best_x))
            transaction_cost = beta * np.sum(open_prices * np.abs(assets_change))
            objective.append(float(mu @ best_x - lam * best_x @ sigma @ best_x - beta * np.sum(np.abs(assets_change))))
            transaction_cost_history.append(transaction_cost)
            best_xs.append(best_x)
            latest_best_x = best_x

            print(f"{prefix}On date {current_date.strftime('%Y-%m-%d')} best x is {best_x}, objective value: {objective[-1]:.8f}, transaction cost: {transaction_cost:.8f}, budget: {budget:.8f}")

            if best_x is None:
                print(f"{prefix}Optimization failed for date {current_date.strftime('%Y-%m-%d')}. Stopping benchmark.")
                break

            close_prices = self.dataset.get_close_price()
            if close_prices is None:
                print(f"{prefix}Could not get close prices for date {current_date.strftime('%Y-%m-%d')}. Stopping benchmark.")
                break

            try:
                budget = budget + best_x @ (close_prices - open_prices) - transaction_cost
            except ValueError as e:
                print(f"{prefix}Error calculating new budget on date {current_date.strftime('%Y-%m-%d')}: {e}")
                print(f"{prefix}best_x shape: {best_x.shape}, close_prices shape: {close_prices.shape}")
                break

            # final transaction cost update
            transaction_cost = beta * np.sum(open_prices * best_x)
            budget -= transaction_cost

            budget_history.append(budget)
            date_history.append(pd.to_datetime(current_date))

            current_date = self.dataset.next_date()
            iterations += 1

        return {
            "name": name,
            "date_history": date_history,
            "budget_history": budget_history,
            "objective": objective,
            "transaction_cost_history": transaction_cost_history,
            "best_xs": best_xs,
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
            self._plot_history(result, timestamp)
            results.append(result)

        if len(results) > 1:
            self._plot_compare(results, timestamp)

        result_dir = self.benchmark_config.result_dir or "result"
        filename = f"benchmark_results_{timestamp}.json"
        save_path = Path(result_dir) / filename
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Benchmark results saved to {save_path}")

        return results
