from typing import Callable, Optional, Sequence, Union, Any, Dict

import numpy as np
from pyscipopt import Model, quicksum

from optimizer.base import BaseOptimizer


class ScipOptimizer(BaseOptimizer):
    def __init__(
        self,
        risk_aversion: float,
        lam: float,
        upper_bounds: Union[list, np.ndarray, int],
        suppress_output: bool = True,
    ):
        super().__init__(risk_aversion, lam)
        self.upper_bounds = upper_bounds
        self.suppress_output = suppress_output

    @classmethod
    def init(cls, cfg: Dict[str, Any], risk_aversion: float, lam: float) -> "ScipOptimizer":
        upper_bounds = cfg.get("upper_bounds")
        if upper_bounds is None:
            raise ValueError("scip.upper_bounds is required to build ScipOptimizer.")
        return cls(
            risk_aversion=risk_aversion,
            lam=lam,
            upper_bounds=upper_bounds,
            suppress_output=cfg.get("suppress_output", True),
        )

    @property
    def optimizer(self) -> Callable:
        return self.optimize

    def optimize(
        self,
        mu: np.ndarray,
        prices: np.ndarray,
        sigma: np.ndarray,
        budget: float,
    ) -> Optional[np.ndarray]:
        n = len(mu)
        model = Model("mean_variance_mip")
        if self.suppress_output:
            model.hideOutput()
        if isinstance(self.upper_bounds, int):
            self.upper_bounds = [self.upper_bounds] * n
        x = [
            model.addVar(vtype="I", lb=0, ub=int(self.upper_bounds[i]), name=f"x{i}")
            for i in range(n)
        ]
        model.addCons(quicksum(prices[i] * x[i] for i in range(n)) <= budget)

        objvar = model.addVar(vtype="C", name="objvar")
        linear_term = quicksum(mu[i] * x[i] for i in range(n))
        quadratic_term = quicksum(
            sigma[i, j] * x[i] * x[j] for i in range(n) for j in range(n)
        )
        model.addCons(objvar - linear_term + self.lam * quadratic_term <= 0)
        model.setObjective(objvar, sense="maximize")

        model.optimize()
        sol = model.getBestSol()

        if sol is None:
            return None

        return np.array([sol[x[i]] for i in range(n)], dtype=int)
