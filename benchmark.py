from src.benchmark.benchmark import *
from src.benchmark.solvers import *

# =============================================== Optimizer Function ===============================================
# optimizer is a Callable with inputs:
# 1) Classical Solvers
# mu, open_prices, sigma, budget, benchmark_config, **optimizer_args
#
# 2) Quantum Solvers
# h, J, C, budget, benchmark_config, **optimizer_args
#
# Here the class benchmark_config contains
#
# class BenchmarkConfig:
#     start_date : str
#     start_budget : int
#     max_iter : int
#     asset_count : int
#     stock_list : Union[None, List[str]]
#     history_window : int
#     risk_aversion : float
#     alpha : Union[float, None] # penalty coefficient
#     spin_count : Union[int, None]
#     bits_per_asset : Union[int, None]
#     slack_variable_bits : Union[int, None]
#     upper_bound_for_all_stocks : Union[List[int],None]
#
# optimizer should return either
# 1) a list of int
# 2) None
# =================================================================================================================s

config = BenchmarkConfig(
    start_date='2024-06-01',
    start_budget=10000,
    max_iter=1000,
    asset_count=13,
    stock_list=None,
    history_window=100,
    risk_aversion=0.3,
    alpha=None,
    spin_count=None,
    bits_per_asset=None,
    slack_variable_bits=None,
    upper_bound_for_all_stocks=[8]*13
)

optimizer = scip_solve

benchmark = ClassicalBenchmark(
    optimizer=optimizer,
    optimizer_args=None,
    benchmark_config=config
)

benchmark.run()