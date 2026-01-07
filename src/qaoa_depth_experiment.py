from __future__ import annotations

from pathlib import Path
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from config_loader import load_config, build_benchmark_config
from benchmark.benchmark import Benchmark
from optimizer.qaoa import QAOAOptimizer


PS = [1, 2, 5, 10]


def make_named_callable(opt, name: str):
    def _fn(mu, prices, sigma, budget, x0=None, **kwargs):
        return opt(mu, prices, sigma, budget, x0=x0, **kwargs)

    _fn.__name__ = name
    return _fn


def main() -> None:
    # 1) Setup Configuration
    base_cfg = load_config(str(Path(__file__).resolve().parent / "config.yaml"))
    cfg = copy.deepcopy(base_cfg)

    # Use JAX QAOA only
    cfg["solvers"] = ["qaoa"]

    # Enforce trials and specific settings
    qaoa_cfg = copy.deepcopy(cfg.get("qaoa", {}))
    qaoa_cfg["n_trials"] = 10
    cfg["qaoa"] = qaoa_cfg

    benchmark_config = build_benchmark_config(cfg)
    bench = Benchmark(benchmark_config)

    lam = float(cfg.get("problem", {}).get("lam", 0.0))
    beta = float(cfg.get("problem", {}).get("beta", 0.0))

    optimizers = []
    for p in PS:
        solver_cfg = copy.deepcopy(qaoa_cfg)
        solver_cfg["p"] = int(p)

        opt = QAOAOptimizer.init(solver_cfg, lam=lam, beta=beta)
        optimizers.append(make_named_callable(opt, f"QAOA_p_{p}"))

    # 2) Run Benchmark
    results = bench.run(optimizers)

    # =========================================================
    # 3) Plotting: Optimized Visualization (Consistent Style)
    # =========================================================

    # 设置风格：清晰的网格，适合学术展示
    sns.set_theme(style="whitegrid", context="talk", palette="viridis")
    plt.rcParams['font.family'] = 'sans-serif'

    result_dir = Path(benchmark_config.result_dir or "result")
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # 获取颜色盘 (Viridis 非常适合展示参数 p 的递增关系)
    colors = sns.color_palette("viridis", n_colors=len(results))

    # 定义绘图辅助函数
    def plot_metric_line(ax, dates, values, label, color):
        ax.plot(dates, values, color=color, label=label,
                marker='o', markersize=6, linewidth=2.5, alpha=0.85)

    # --- 图 1: Objective vs Time ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    for idx, r in enumerate(results):
        name = r.get("name", "optimizer")
        dates = pd.to_datetime(r["date_history"])
        objective = np.asarray(r["objective"], dtype=float)

        m = min(len(dates), len(objective))
        plot_metric_line(ax1, dates[:m], objective[:m], name, colors[idx])

    ax1.set_title("QAOA Objective Evolution (Layer Depth Analysis)", fontsize=18, pad=15, fontweight='bold')
    ax1.set_ylabel("Hamiltonian Energy / Objective", fontsize=14)
    ax1.set_xlabel("Date", fontsize=14)
    ax1.legend(loc='best', frameon=True, framealpha=0.9, shadow=True, fontsize=12)

    # 日期格式化
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    fig1.autofmt_xdate(rotation=0)

    out_path_obj = result_dir / f"qaoa_depth_objective_{timestamp}.png"
    fig1.tight_layout()
    fig1.savefig(out_path_obj, dpi=300)
    plt.close(fig1)

    # --- 图 2: Cumulative Return vs Time ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    for idx, r in enumerate(results):
        name = r.get("name", "optimizer")
        dates = pd.to_datetime(r["date_history"])
        budgets = np.asarray(r["budget_history"], dtype=float)
        cum_return = budgets / float(benchmark_config.start_budget) - 1.0

        m = min(len(dates), len(cum_return))
        plot_metric_line(ax2, dates[:m], cum_return[:m], name, colors[idx])

    ax2.set_title("Cumulative Return Trajectory", fontsize=18, pad=15, fontweight='bold')
    ax2.set_ylabel("Return (%)", fontsize=14)
    ax2.set_xlabel("Date", fontsize=14)

    # Y轴百分比格式化
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.1%}'.format(x) for x in vals])

    ax2.legend(loc='best', frameon=True, framealpha=0.9, shadow=True, fontsize=12)

    # 日期格式化
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    fig2.autofmt_xdate(rotation=0)

    out_path_ret = result_dir / f"qaoa_depth_return_{timestamp}.png"
    fig2.tight_layout()
    fig2.savefig(out_path_ret, dpi=300)
    plt.close(fig2)

    # --- Summary ---
    print("\n=== Summary (end of window) ===")
    for r in results:
        name = r["name"]
        obj = np.asarray(r["objective"], dtype=float)
        budgets = np.asarray(r["budget_history"], dtype=float)
        final_ret = budgets[-1] / float(benchmark_config.start_budget) - 1.0
        print(f"{name:>14s} | final_return={final_ret:+.4%} | final_objective={obj[-1]:+.6f}")

    print(f"\nSaved figures:\n1. {out_path_obj}\n2. {out_path_ret}")


if __name__ == "__main__":
    main()