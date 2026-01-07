from __future__ import annotations

from pathlib import Path
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config_loader import load_config, build_benchmark_config
from benchmark.benchmark import Benchmark
from optimizer.quantum_annealing import QuantumAnnealingOptimizer


STEPS = 50


def make_named_callable(opt, name: str):
    def _fn(mu, prices, sigma, budget, x0=None, **kwargs):
        return opt(mu, prices, sigma, budget, x0=x0, **kwargs)

    _fn.__name__ = name
    return _fn


def main() -> None:
    base_cfg = load_config(str(Path(__file__).resolve().parent / "config.yaml"))
    cfg = copy.deepcopy(base_cfg)

    # Match the QA depth experiment's data window & portfolio universe
    cfg.setdefault("data", {})
    cfg["data"]["start_date"] = "2024-12-01"
    cfg["data"]["max_iter"] = 30
    cfg["data"]["stock_list"] = ["MSFT", "TSLA", "META"]#, "BA", "JNJ"]

    # Match the QA depth experiment's problem settings
    cfg.setdefault("problem", {})
    cfg["problem"]["start_budget"] = 500
    cfg["problem"]["lam"] = 0.3

    # Solver: quantum annealing only
    cfg["solvers"] = ["quantum_annealing"]

    # Base QA config overrides
    qa_cfg = copy.deepcopy(cfg.get("quantum_annealing", {}))
    qa_cfg["bits_per_asset"] = 2
    qa_cfg["bits_slack"] = 8
    qa_cfg["time"] = 10
    qa_cfg["steps"] = int(STEPS)
    qa_cfg["transact_opt"] = "ignore"
    cfg["quantum_annealing"] = qa_cfg

    benchmark_config = build_benchmark_config(cfg)
    bench = Benchmark(benchmark_config)

    lam = float(cfg.get("problem", {}).get("lam", 0.0))
    beta = float(cfg.get("problem", {}).get("beta", 0.0))

    # Noise config: use config if present; otherwise fall back to the (commented) defaults from config.yaml
    noise_cfg = qa_cfg.get("noise")
    if noise_cfg is None:
        noise_cfg = {
            "type": "depolarizing",
            "p1": 0.001,
            "p2": 0.01,
            "readout": 0.02,
        }
        print("[qa_noise_experiment] NOTE: No quantum_annealing.noise found in config; using fallback depolarizing noise.")

    # Build two optimizers: no-noise vs with-noise
    qa_cfg_clean = copy.deepcopy(qa_cfg)
    qa_cfg_clean.pop("noise", None)

    qa_cfg_noisy = copy.deepcopy(qa_cfg)
    qa_cfg_noisy["noise"] = noise_cfg

    opt_clean = QuantumAnnealingOptimizer.init(qa_cfg_clean, lam=lam, beta=beta)
    opt_noisy = QuantumAnnealingOptimizer.init(qa_cfg_noisy, lam=lam, beta=beta)

    optimizers = [
        make_named_callable(opt_clean, f"QA_steps_{STEPS}_no_noise"),
        make_named_callable(opt_noisy, f"QA_steps_{STEPS}_with_noise"),
    ]

    results = bench.run(optimizers)

    # Custom plot: objective vs realized cumulative return (from budget_history)
    result_dir = Path(benchmark_config.result_dir or "result")
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for r in results:
        name = r.get("name", "optimizer")
        dates = pd.to_datetime(r["date_history"])

        objective = np.asarray(r["objective"], dtype=float)
        budgets = np.asarray(r["budget_history"], dtype=float)

        cum_return = budgets / float(benchmark_config.start_budget) - 1.0

        m = min(len(dates), len(objective), len(cum_return))
        axes[0].plot(dates[:m], objective[:m], marker="o", linewidth=1.5, label=name)
        axes[1].plot(dates[:m], cum_return[:m], marker="o", linewidth=1.5, label=name)

    axes[0].set_title(f"Objective（题目目标函数）随时间变化 - Quantum Annealing steps={STEPS} 噪声对比")
    axes[0].set_ylabel("Objective")
    axes[0].grid(True)

    axes[1].set_title("实际收益（累计收益，用 budget 计算）随时间变化")
    axes[1].set_ylabel("Cumulative Return")
    axes[1].set_xlabel("Date")
    axes[1].grid(True)

    axes[0].legend()
    axes[1].legend()
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    out_path = result_dir / f"qa_noise_objective_vs_return_{timestamp}.png"
    fig.savefig(out_path, dpi=600)
    plt.close(fig)

    print("\n=== Summary (end of window) ===")
    for r in results:
        name = r["name"]
        obj = np.asarray(r["objective"], dtype=float)
        budgets = np.asarray(r["budget_history"], dtype=float)
        final_ret = budgets[-1] / float(benchmark_config.start_budget) - 1.0
        print(f"{name:>20s} | final_return={final_ret:+.4%} | final_objective={obj[-1]:+.6f}")

    print(f"\nSaved figure: {out_path}")


if __name__ == "__main__":
    main()
