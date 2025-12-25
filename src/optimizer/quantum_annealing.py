from typing import Callable
import numpy as np
import pandas as pd
from pyscipopt import Model, quicksum
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from optimizer.base import BaseOptimizer
from optimizer.utils.qubo_utils import qubo_factor as qubo_factor_optimized
from optimizer.utils.qubo_utils import get_ising_coeffs as get_ising_coeffs_optimized

class QuantumAnnealingOptimizer(BaseOptimizer):
    def __init__(
        self,
        risk_aversion: float,
        lam: float,
        alpha: float,
        bits_per_asset: int,
        bits_slack: int,
        time: float = 10,
        steps: int = 100,
        traverse: float = 1.0,
    ):
        super().__init__(risk_aversion, lam)
        self.alpha = alpha
        self.bits_per_asset = bits_per_asset
        self.bits_slack = bits_slack
        self.time = time
        self.steps = steps
        self.traverse = traverse

    def qubo_factor(self, 
                    n: int, 
                    mu: np.ndarray, 
                    sigma: np.ndarray, 
                    prices: np.ndarray, 
                    n_spins: int, 
                    budget: float
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
        )
    
    def get_ising_coeffs(
            self,
            Q: np.ndarray, 
            L: np.ndarray, 
            constant: float
            ):
            return get_ising_coeffs_optimized(Q, L, constant)

    @property
    def optimizer(self) -> Callable:
        return self.optimize

    def optimize(self,
                mu: np.ndarray,
                prices: np.ndarray,
                sigma: np.ndarray,
                budget: float,
                ) -> np.ndarray:
        n = len(mu)
        self.num_spins = n * self.bits_per_asset + self.bits_slack
        Q, L, constant = self.qubo_factor(n=n, mu=mu, sigma=sigma, prices=prices, n_spins=self.num_spins, budget=budget)
        h, J, C = self.get_ising_coeffs(Q, L, constant)

        max_strength = np.max([np.max(np.abs(J)), np.max(np.abs(h))])
        
        # 防止除以 0
        if max_strength == 0:
            scale_factor = 1.0
        else:
            # 将最大系数归一化为 1.0 (或者类似于 B 的量级)
            scale_factor = 1.0 / max_strength

        # 应用归一化
        J_scaled = J * scale_factor
        h_scaled = h * scale_factor
        C_scaled = C * scale_factor

        def U_H(J, h, t):
            qc = QuantumCircuit(self.num_spins)
            for i in range(self.num_spins):
                if h[i] != 0:
                    qc.rz(2 * h[i] * t, i)
            for i in range(self.num_spins):
                for j in range(i + 1, self.num_spins):
                    if J[i, j] != 0:
                        qc.cx(i, j)
                        qc.rz(2 * J[i, j] * t, j)
                        qc.cx(i, j)
            return qc

        def U_x(B, t):
            qc = QuantumCircuit(self.num_spins)
            for i in range(self.num_spins):
                qc.rx(- 2 * B * t, i)
            return qc

        def trotter_annealing(T=10.0, M=100, B=1.0):
            """Simulate quantum annealing using first-order Trotter decomposition."""
            dt = T / M
            qc = QuantumCircuit(self.num_spins)
            qc.h(range(self.num_spins))  # Initialize in |+> state
            for i in range(M):
                s = i / M
                qc.append(U_x(B * (1 - s), dt), range(self.num_spins))
                qc.append(U_H(J_scaled, h_scaled, dt * s), range(self.num_spins))
            return qc


        def compute_energy(bitstring, J, h, C):
            """Compute Ising energy given spin configuration (+1/-1)."""
            S = np.array([1 if b == '0' else -1 for b in bitstring[::-1]])
            return S @ J @ S + np.dot(h, S) + C

        qc = trotter_annealing(T=self.time, M=self.steps, B=self.traverse) 
        qc.measure_all()
        sim = AerSimulator()
        result = sim.run(transpile(qc, sim), shots=1000).result()
        counts = result.get_counts()

        # Compute energies for each measurement
        energies = []
        min_energy = np.inf
        ground_state = np.zeros(self.num_spins)
        for bitstring, count in counts.items():
            E = compute_energy(bitstring, J_scaled, h_scaled, C_scaled )
            energies += [E] * count
            if E < min_energy:
                min_energy = E
                ground_state = np.array([1 if b == '0' else -1 for b in bitstring[::-1]])
        
        asset_counts = []
        for i in range(n):
            count = 0
            for p in range(self.bits_per_asset):
                idx = i*self.bits_per_asset + p
                if ground_state[idx] == -1:
                    count += 2**p
            asset_counts.append(count)
        
        return np.array(asset_counts)
