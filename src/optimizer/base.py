from typing import Any, Dict, Union, List, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class BaseOptimizer(ABC):
    def __init__(self, 
                risk_aversion: float,
                lam: float,
                beta: Optional[float],
                ) -> None:
        
        self.risk_aversion = risk_aversion
        self.lam = lam
        self.beta = beta if beta is not None else 0.0

    @classmethod
    def init(cls, cfg: Dict[str, Any], risk_aversion: float, lam: float, beta: Optional[float]) -> "BaseOptimizer":
        raise NotImplementedError("init method must be implemented in subclass.")

    @abstractmethod
    def optimize(self,
                 mu: np.ndarray,
                 prices: np.ndarray,
                 sigma: np.ndarray,
                 budget: float,
                 **args,
                 ) -> np.ndarray:
        pass

    @property
    def optimizer(self) -> Callable:
        return self.optimize

    def __call__(
        self,
        mu: np.ndarray,
        prices: np.ndarray,
        sigma: np.ndarray,
        budget: float,
        x0: Optional[np.ndarray] = None,
        **args,
    ) -> np.ndarray:
        return self.optimize(mu, prices, sigma, budget, x0, **args)
