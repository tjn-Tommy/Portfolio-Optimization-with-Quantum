from pathlib import Path
import sys
from typing import Optional
import argparse
import json
from benchmark.benchmark import Benchmark
from config_loader import build_benchmark_config, build_optimizers, load_config


def main(config_path: Optional[str] = None) -> None:
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent / "config.yaml")

    config = load_config(config_path)
    benchmark_config = build_benchmark_config(config)
    optimizers = build_optimizers(config)

    benchmark = Benchmark(benchmark_config)
    results = benchmark.run(optimizers)
    results_dir = benchmark_config.result_dir or "./results"
    save_path = Path(results_dir) / "benchmark_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"Benchmark results saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to config.yaml",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
