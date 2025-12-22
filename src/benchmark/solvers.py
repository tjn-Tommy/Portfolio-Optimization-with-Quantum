from pyscipopt import Model, quicksum
import numpy as np
from src.benchmark.benchmark import *

def scip_solve(mu, prices, sigma, budget, benchmark_config : BenchmarkConfig):
    model = Model("mean_variance_mip")
    model.hideOutput() 
    upper_bounds = benchmark_config.upper_bound_for_all_stocks
    n = benchmark_config.asset_count
    B = budget
    lamb = benchmark_config.risk_aversion

    x = [model.addVar(vtype="I", lb=0, ub=upper_bounds[i], name=f"x{i}") for i in range(n)]
    model.addCons(quicksum(prices[i] * x[i] for i in range(n)) <= B)

    objvar = model.addVar(vtype="C", name="objvar")

    linear_term = quicksum(mu[i] * x[i] for i in range(n))
    quadratic_term = quicksum(sigma[i, j] * x[i] * x[j] for i in range(n) for j in range(n))

    model.addCons(objvar - linear_term + lamb * quadratic_term <= 0)

    model.setObjective(objvar, sense="maximize")

    model.optimize()
    sol = model.getBestSol()

    if sol:
        x_opt = np.array([sol[x[i]] for i in range(n)], dtype=int)
        return x_opt
    else:
        return None