from typing import Any, Dict, Union, List, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class BaseOptimizer(ABC):
    def __init__(self, 
                risk_aversion: float,
                lam: float):
        
        self.risk_aversion = risk_aversion
        self.lam = lam

    @abstractmethod
    def optimize(self,
                 mu: np.ndarray,
                 prices: np.ndarray,
                 sigma: np.ndarray,
                 budget: float,
                 **args,
                 ) -> np.ndarray:
        pass