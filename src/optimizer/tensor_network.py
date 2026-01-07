from typing import Callable, Optional, Any, Dict

import numpy as np

from optimizer.base import BaseOptimizer
from optimizer.utils.qubo_utils import qubo_factor as qubo_factor_optimized
from optimizer.utils.qubo_utils import get_ising_coeffs as get_ising_coeffs_optimized
from optimizer.utils.qubo_utils import compute_num_spins as compute_num_spins_optimized
from optimizer.utils.qubo_utils import spins_to_asset_counts
from tensor_network.ED import ExactDiagonalization
from tensor_network.VariationalMPS import VariationalMPS
from tensor_network.tenpy_dmrg import tenpy_dmrg


class TensorNetworkOptimizer(BaseOptimizer):
    def __init__(
        self,
        lam: float,
        alpha: float,
        beta: Optional[float],
        bits_per_asset: int,
        bits_slack: int,
        transact_opt: str = "ignore",
        method: str = "dmrg",
        mps_rank: int = 10,
        mps_bond_dim: int = 6,
        mps_opt: int = 1,
        mps_seed: int = 1,
    ):
        super().__init__(lam, beta)
        self.alpha = alpha
        self.bits_per_asset = bits_per_asset
        self.bits_slack = bits_slack
        self.transact_opt = transact_opt
        self.method = method
        self.mps_rank = mps_rank
        self.mps_bond_dim = mps_bond_dim
        self.mps_opt = mps_opt
        self.mps_seed = mps_seed
        self.num_spins = 0

    @classmethod
    def init(cls, cfg: Dict[str, Any], lam: float, beta: Optional[float]) -> "TensorNetworkOptimizer":
        return cls(
            lam=lam,
            alpha=cfg["alpha"],
            beta = beta,
            bits_per_asset=cfg["bits_per_asset"],
            bits_slack=cfg["bits_slack"],
            transact_opt=cfg.get("transact_opt", "ignore"),
            method=cfg.get("method", "dmrg"),
            mps_rank=cfg.get("mps_rank", 10),
            mps_bond_dim=cfg.get("mps_bond_dim", 6),
            mps_opt=cfg.get("mps_opt", 1),
        )

    def qubo_factor(
        self,
        n: int,
        mu: np.ndarray,
        sigma: np.ndarray,
        prices: np.ndarray,
        n_spins: int,
        budget: float,
        x0: Optional[np.ndarray] = None,
    ):
        return qubo_factor_optimized(
            n=n,
            mu=mu,
            sigma=sigma,
            prices=prices,
            n_spins=n_spins,
            budget=budget,
            bits_per_asset=self.bits_per_asset,
            bits_slack=self.bits_slack,
            lam=self.lam,
            alpha=self.alpha,
            beta=self.beta,
            transact_opt=self.transact_opt,
            x0=x0,
        )

    def get_ising_coeffs(self, Q: np.ndarray, L: np.ndarray, constant: float):
        return get_ising_coeffs_optimized(Q, L, constant)
    
    def compute_num_spins(self,
                          n_assets: int,
                          x0: np.ndarray = None
    ):
        return compute_num_spins_optimized(
            n_assets=n_assets,
            bits_per_asset=self.bits_per_asset,
            bits_slack=self.bits_slack,
            transact_opt=self.transact_opt,
            x0=x0
        )
    
    def _spins_to_asset_counts(self,
               spins: np.ndarray,
               n_assets: int,
               x0: np.ndarray = None
    ):
        return spins_to_asset_counts(
            spins=spins,
            n_assets=n_assets,
            bits_per_asset=self.bits_per_asset,
            bits_plus=self.bits_plus,
            bits_minus=self.bits_minus,
            transact_opt=self.transact_opt,
            x0=x0
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
        method: Optional[str] = None,
        mps_rank: Optional[int] = None,
        mps_bond_dim: Optional[int] = None,
        mps_opt: Optional[int] = None,
        mps_seed: Optional[int] = None,
        **kwargs,
    ) -> Optional[np.ndarray]:
        n = len(mu)
        self.num_spins, self.bits_plus, self.bits_minus = self.compute_num_spins(n, x0)
        Q, L, constant = self.qubo_factor(
            n=n,
            mu=mu,
            sigma=sigma,
            prices=prices,
            n_spins=self.num_spins,
            budget=budget,
            x0=x0,
        )
        h, J, _ = self.get_ising_coeffs(Q, L, constant)

        chosen_method = (method or self.method).lower()
        if chosen_method == "variational_mps":
            _, spins = VariationalMPS(
                J,
                h,
                R=mps_rank if mps_rank is not None else self.mps_rank,
                Dp=2,
                Ds=mps_bond_dim if mps_bond_dim is not None else self.mps_bond_dim,
                opt=mps_opt if mps_opt is not None else self.mps_opt,
                seed=mps_seed if mps_seed is not None else self.mps_seed,
            )
        elif chosen_method == "dmrg":
            spins = tenpy_dmrg(J, h)
        elif chosen_method == "exact":
            _, spins = ExactDiagonalization(J, h)
        else:
            raise ValueError(f"Unsupported tensor network method: {chosen_method}")

        spins = np.array(spins, dtype=int)
        return self._spins_to_asset_counts(spins, n, x0)
