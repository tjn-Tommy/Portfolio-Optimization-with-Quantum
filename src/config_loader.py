from typing import Any, Dict, List, Type
from pathlib import Path

import yaml

from benchmark.benchmark import BenchmarkConfig
from optimizer.base import BaseOptimizer
from optimizer.qaoa import QAOAOptimizer
from optimizer.quantum_annealing import QuantumAnnealingOptimizer
from optimizer.scip_solver import ScipOptimizer
from optimizer.tensor_network import TensorNetworkOptimizer


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def build_benchmark_config(config: Dict[str, Any]) -> BenchmarkConfig:
    return BenchmarkConfig.init(config)


def build_optimizers(config: Dict[str, Any]) -> List[BaseOptimizer]:
    problem_cfg = config.get("problem", {})
    lam = float(problem_cfg.get("lam", 0.3))
    beta = float(problem_cfg.get("beta", 0.0))

    solver_names = config.get("solvers") or config.get("solver")
    if solver_names is None:
        solver_names = [name for name in _SOLVER_CLASSES.keys() if name in config]
    elif isinstance(solver_names, str):
        solver_names = [solver_names]

    if not solver_names:
        raise ValueError("No solver names configured. Add 'solvers' or solver sections to the config.")

    optimizers: List[BaseOptimizer] = []
    for solver_name in solver_names:
        solver_cls = _SOLVER_CLASSES.get(solver_name)
        if solver_cls is None:
            raise ValueError(f"Unsupported solver in config: {solver_name}")
        solver_cfg = config.get(solver_name, {})
        if not hasattr(solver_cls, "init"):
            raise ValueError(f"{solver_name} does not provide an init method.")
        optimizers.append(solver_cls.init(solver_cfg, lam, beta))

    return optimizers


_SOLVER_CLASSES: Dict[str, Type[BaseOptimizer]] = {
    "quantum_annealing": QuantumAnnealingOptimizer,
    "qaoa": QAOAOptimizer,
    "tensor_network": TensorNetworkOptimizer,
    "scip": ScipOptimizer,
}
