from typing import Any, Dict, Union, List, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class BaseOptimizer(ABC):
    def __init__(self, 
                lam: float):
        
        self.lam = lam

    @classmethod
    def init(cls, cfg: Dict[str, Any], lam: float) -> "BaseOptimizer":
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
        **args,
    ) -> np.ndarray:
        return self.optimize(mu, prices, sigma, budget, **args)
