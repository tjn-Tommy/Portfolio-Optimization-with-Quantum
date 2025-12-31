import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Callable
from benchmark.benchmark import Benchmark, BenchmarkConfig
from config_loader import build_benchmark_config, build_optimizers, load_config

class SynthesizeBenchmark(Benchmark):
    def __init__(self,
                 benchmark_config: BenchmarkConfig,
                 mu: np.ndarray,
                 sigma: np.ndarray,
                 prices: np.ndarray
                 ):
        super().__init__(benchmark_config)
        self.mu = mu
        self.sigma = sigma
        self.prices = prices

    def run_single(
        self,
        optimizer: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
        name: str,
        **kwargs,
    ) -> dict:
        budget = self.benchmark_config.start_budget
        lam = self.benchmark_config.lam
        beta = self.benchmark_config.beta
        self.dataset.set_date(self.benchmark_config.start_date)

        best_x = optimizer(self.mu, self.prices, self.sigma, budget, None, **kwargs)
        # Calculate the objective:
        assets_change = best_x
        buy_price = np.sum(self.prices * np.abs(assets_change))
        # objective = float(self.mu @ best_x - 0.5 * best_x @ self.sigma @ best_x - beta * np.sum(np.abs(assets_change)))
        objective = float(self.mu @ best_x - lam * best_x @ self.sigma @ best_x)

        prefix = f"[{name}]" if name else ""
        print(f"{prefix} best x: {best_x}, objective value: {objective:.8f}, buy price: {buy_price:.3f}, budget: {budget:.3f}")
        return {
            "name": name,
            "objective": objective,
            "buy_price": buy_price,
            "best_x": best_x,
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
            result = self.run_single(opt_fn, name, **kwargs)
            results.append(result)

        result_dir = "result_synthesize"
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        filename = f"benchmark_results_{timestamp}.json"
        save_path = Path(result_dir) / filename
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Benchmark results saved to {save_path}")

        return results
    
def main():
    config_path = str(Path(__file__).resolve().parent / "config.yaml")

    config = load_config(config_path)
    benchmark_config = build_benchmark_config(config)
    optimizers = build_optimizers(config)

    mu = np.array([0.12, 0.10, 0.15, 0.09, 0.11])
    prices = np.array([10, 12, 8, 15, 7])
    sigma = np.array([
        [0.04, 0.01, 0.00, 0.00, 0.01],
        [0.01, 0.05, 0.01, 0.00, 0.00],
        [0.00, 0.01, 0.06, 0.02, 0.00],
        [0.00, 0.00, 0.02, 0.07, 0.01],
        [0.01, 0.00, 0.00, 0.01, 0.03]
    ])

    benchmark = SynthesizeBenchmark(benchmark_config, mu, sigma, prices)
    benchmark.run(optimizers)

if __name__ == "__main__":
    main()