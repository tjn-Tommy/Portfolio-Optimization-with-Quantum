from typing import Callable, Optional, Sequence, Union, Any, Dict

import numpy as np
from pyscipopt import Model, quicksum

from optimizer.base import BaseOptimizer


class ScipOptimizer(BaseOptimizer):
    def __init__(
        self,
        lam: float,
        beta: Optional[float],
        upper_bounds: Union[list, np.ndarray, int],
        suppress_output: bool = True,
        transact_opt: str = "ignore"
    ):
        super().__init__(lam, beta)
        self.upper_bounds = upper_bounds
        self.suppress_output = suppress_output
        self.transact_opt = transact_opt

    @classmethod
    def init(cls, cfg: Dict[str, Any], lam: float, beta: Optional[float]) -> "ScipOptimizer":
        upper_bounds = cfg.get("upper_bounds")
        if upper_bounds is None:
            raise ValueError("scip.upper_bounds is required to build ScipOptimizer.")
        return cls(
            lam=lam,
            beta=beta,
            transact_opt=cfg.get("transact_opt", "ignore"),
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
        x0: Optional[np.ndarray] = None,
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

        # transaction cost
        transaction_flag = False
        if self.beta > 0.0:
            # print(f"SCIP: beta={self.beta}")
            transaction_flag = True
            if x0 is None:
                x0 = np.zeros(n, dtype=int)
            z = []
            for i in range(n):
                zi = model.addVar(vtype="C", lb=0, name=f"abs_dev{i}")
                model.addCons(zi >= x[i] - x0[i])
                model.addCons(zi >= -(x[i] - x0[i]))
                z.append(zi)
            transaction_term = quicksum(self.beta * z[i] for i in range(n))

        objvar = model.addVar(vtype="C", name="objvar")
        linear_term = quicksum(mu[i] * x[i] for i in range(n))
        quadratic_term = quicksum(
            sigma[i, j] * x[i] * x[j] for i in range(n) for j in range(n)
        )
        if transaction_flag:
            model.addCons(objvar - linear_term + self.lam * quadratic_term + transaction_term <= 0)
        else:
            model.addCons(objvar - linear_term + self.lam * quadratic_term <= 0)
        model.setObjective(objvar, sense="maximize")

        model.optimize()
        sol = model.getBestSol()

        if sol is None:
            return None

        return np.array([sol[x[i]] for i in range(n)], dtype=int)
