from typing import Callable, Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from contextlib import redirect_stdout
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit, transpile
from optimizer.base import BaseOptimizer
from optimizer.utils.qubo_utils import qubo_factor as qubo_factor_optimized
from optimizer.utils.qubo_utils import get_ising_coeffs as get_ising_coeffs_optimized
from optimizer.utils.qubo_utils import normalize_ising_coeffs
from optimizer.utils.noise_utils import build_aer_simulator

class QuantumAnnealingOptimizer(BaseOptimizer):
    def __init__(
        self,
        lam: float,
        alpha: float,
        beta: Optional[float],
        bits_per_asset: int,
        bits_slack: int,
        time: float = 10,
        steps: int = 100,
        traverse: float = 1.0,
        transact_opt: str = "ignore",
        noise_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(lam, beta)
        self.alpha = alpha
        self.bits_per_asset = bits_per_asset
        self.bits_slack = bits_slack
        self.time = time
        self.steps = steps
        self.traverse = traverse
        self.transact_opt = transact_opt
        self.noise_config = noise_config
        self.backend = build_aer_simulator(noise_config)
        # try:
        #     # 1. 强制使用 GPU
        #     self.backend.set_options(device='GPU')
            
        #     # 2. 关键：设置精度为单精度
        #     # "single": complex64 (对应 float32)
        #     # "double": complex128 (对应 float64) - 默认值
        #     self.backend.set_options(precision='single',
        #                              batched_shots_gpu=True,
        #                              max_shot_size=1000,
        #                             #  cuStateVec_enable=True,
        #                              batched_shots_gpu_max_qubits=22,
        #                              ) 
            
        #     print("✅ GPU Acceleration enabled with Single Precision.")
        # except Exception as e:
        #     print(f"⚠️ GPU setup failed, falling back to CPU: {e}")
        #     self.backend.set_options(device='CPU')

        # === 缓存变量 ===
        self._cached_circuit: Optional[QuantumCircuit] = None
        self._cached_num_spins: int = -1
        self._params_h: Optional[ParameterVector] = None
        self._params_J: Optional[ParameterVector] = None
        # 映射表：将 (i, j) 映射到 params_J 中的索引
        self._J_param_map: Dict[Tuple[int, int], int] = {}

    @classmethod
    def init(cls, cfg: Dict[str, Any], lam: float, beta: Optional[float]) -> "QuantumAnnealingOptimizer":
        return cls(
            lam=lam,
            alpha=cfg["alpha"],
            beta=beta,
            transact_opt=cfg.get("transact_opt", "ignore"),
            bits_per_asset=cfg["bits_per_asset"],
            bits_slack=cfg["bits_slack"],
            time=cfg.get("time", 10),
            steps=cfg.get("steps", 100),
            traverse=cfg.get("traverse", 1.0),
            noise_config=cfg.get("noise"),
        )

    def qubo_factor(self, 
                    n: int, 
                    mu: np.ndarray, 
                    sigma: np.ndarray, 
                    prices: np.ndarray, 
                    n_spins: int, 
                    budget: float,
                    x0: Optional[np.ndarray] = None
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
    def _build_and_transpile_circuit(self, num_spins: int) -> QuantumCircuit:
        print(f"Building and transpiling 2nd-order Trotter circuit (Full J) for {num_spins} spins...")
        
        # 1. 定义参数容器
        self._params_h = ParameterVector("h", num_spins)
        interactions = []
        for i in range(num_spins):
            for j in range(num_spins):
                interactions.append((i, j))
        
        self._params_J = ParameterVector("J", len(interactions))
        self._J_param_map = {pair: idx for idx, pair in enumerate(interactions)}

        # 2. 构建电路
        qc = QuantumCircuit(num_spins)
        qc.h(range(num_spins))  # Initialize in |+>
        
        dt = self.time / self.steps
        
        # 预先取出来，避免循环里频繁访问属性
        J_params_list = list(self._params_J)
        h_params_list = list(self._params_h)
        
        # Trotter Loop
        for step in range(self.steps):
            s = step / self.steps
            
            # =========================================================
            # 2阶 Trotter (Suzuki-Trotter): 
            # Sequence: [Half X] -> [Full Z] -> [Half X]
            # =========================================================

            # --- [Step A] 前半步横向场 (Half X) ---
            # Angle = -2 * B * (1-s) * (dt / 2)
            rx_angle_half = -2 * self.traverse * (1 - s) * (dt / 2)
            
            qc.rx(rx_angle_half, range(num_spins))
            
            # --- [Step B] 完整一步问题哈密顿量 (Full Z) ---
            # 系数保持完整的 dt
            # Angle = 2 * Coeff * s * dt
            z_coeff = 2 * s * dt
            
            # 1. Linear terms (Rz)
            for i in range(num_spins):
                qc.rz(h_params_list[i] * z_coeff, i)
        
            # 2. Quadratic terms (Rzz -> CNOT-Rz-CNOT) - 全矩阵遍历
            # 这里我们直接遍历 idx，它对应 interactions 列表中的 (i, j)
            for idx, (i, j) in enumerate(interactions):
                if i == j:
                    continue                
                # 应用 Rzz
                angle = J_params_list[idx] * z_coeff
                qc.cx(i, j)
                qc.rz(angle, j)
                qc.cx(i, j)

            # --- [Step C] 后半步横向场 (Half X) ---
            qc.rx(rx_angle_half, range(num_spins))

        qc.measure_all()
        
        # 3. 编译电路
        # 使用 level 1 或 2 均可，level 2 会稍微压缩一下相邻的 RX 门
        transpiled_qc = transpile(qc, self.backend, optimization_level=2)
        return transpiled_qc

    def optimize(self, mu, prices, sigma, budget, x0) -> np.ndarray:
        n = len(mu)
        num_spins = n * self.bits_per_asset + self.bits_slack
        
        # 1. 计算 Ising 系数
        Q, L, constant = self.qubo_factor(n=n, mu=mu, sigma=sigma, prices=prices, n_spins=num_spins, budget=budget, x0=x0)
        h, J, C = self.get_ising_coeffs(Q, L, constant)
        h_scaled, J_scaled, C_scaled = normalize_ising_coeffs(h, J, C)

        # 2. 检查缓存
        if self._cached_circuit is None or num_spins != self._cached_num_spins:
            self._cached_circuit = self._build_and_transpile_circuit(num_spins)
            self._cached_num_spins = num_spins

        # 3. 准备绑定参数
        full_dict = dict(zip(self._params_h, h_scaled))
        full_dict.update(zip(self._params_J, J_scaled.flatten()))
        valid_params_set = self._cached_circuit.parameters
        clean_dict = {k: v for k, v in full_dict.items() if k in valid_params_set}
        
        bound_qc = self._cached_circuit.assign_parameters(clean_dict)

        # 5. 运行
        result = self.backend.run(bound_qc, shots=1000).result()
        counts = result.get_counts()

        # 6. 向量化计算能量
        bitstrings = list(counts.keys())
        
        n_samples = len(bitstrings)
        spins = np.zeros((n_samples, num_spins))
        
        for k, bs in enumerate(bitstrings):
            bs_rev = bs[::-1]
            for idx, char in enumerate(bs_rev):
                spins[k, idx] = 1.0 if char == '0' else -1.0
        
        # 能量计算：需要适配全矩阵 J
        # E = sum( (s @ J) * s ) + s @ h
        # 这里的 J_scaled 是 N x N 矩阵，公式直接成立
        quad = np.sum((spins @ J_scaled) * spins, axis=1)
        linear = spins @ h_scaled
        
        energies = quad + linear + C_scaled
        
        # 找到最小能量
        min_idx = np.argmin(energies)
        best_spins = spins[min_idx]
        
        # 7. 解码
        asset_counts = []
        for i in range(n):
            count = 0
            for p in range(self.bits_per_asset):
                idx = i * self.bits_per_asset + p
                if best_spins[idx] == -1:
                    count += 2**p
            asset_counts.append(count)
        
        return np.array(asset_counts)
    # def optimize(self,
    #             mu: np.ndarray,
    #             prices: np.ndarray,
    #             sigma: np.ndarray,
    #             budget: float,
    #             ) -> np.ndarray:
    #     n = len(mu)
    #     self.num_spins = n * self.bits_per_asset + self.bits_slack
    #     Q, L, constant = self.qubo_factor(n=n, mu=mu, sigma=sigma, prices=prices, n_spins=self.num_spins, budget=budget)
    #     h, J, C = self.get_ising_coeffs(Q, L, constant)
    #     h_scaled, J_scaled, C_scaled = normalize_ising_coeffs(h, J, C)

    #     def U_H(J, h, t):
    #         qc = QuantumCircuit(self.num_spins)
    #         for i in range(self.num_spins):
    #             if h[i] != 0:
    #                 qc.rz(2 * h[i] * t, i)
    #         for i in range(self.num_spins):
    #             for j in range(i + 1, self.num_spins):
    #                 if J[i, j] != 0:
    #                     qc.cx(i, j)
    #                     qc.rz(2 * J[i, j] * t, j)
    #                     qc.cx(i, j)
    #         return qc

    #     def U_x(B, t):
    #         qc = QuantumCircuit(self.num_spins)
    #         for i in range(self.num_spins):
    #             qc.rx(- 2 * B * t, i)
    #         return qc

    #     def trotter_annealing(T=10.0, M=100, B=1.0):
    #         """Simulate quantum annealing using first-order Trotter decomposition."""
    #         dt = T / M
    #         qc = QuantumCircuit(self.num_spins)
    #         qc.h(range(self.num_spins))  # Initialize in |+> state
    #         for i in range(M):
    #             s = i / M
    #             qc.append(U_x(B * (1 - s), dt), range(self.num_spins))
    #             qc.append(U_H(J_scaled, h_scaled, dt * s), range(self.num_spins))
    #         return qc


    #     def compute_energy(bitstring, J, h, C):
    #         """Compute Ising energy given spin configuration (+1/-1)."""
    #         S = np.array([1 if b == '0' else -1 for b in bitstring[::-1]])
    #         return S @ J @ S + np.dot(h, S) + C

    #     qc = trotter_annealing(T=self.time, M=self.steps, B=self.traverse) 
    #     qc.measure_all()
    #     result = self.backend.run(transpile(qc, self.backend), shots=1000).result()
    #     counts = result.get_counts()

    #     # Compute energies for each measurement
    #     energies = []
    #     min_energy = np.inf
    #     ground_state = np.zeros(self.num_spins)
    #     for bitstring, count in counts.items():
    #         E = compute_energy(bitstring, J_scaled, h_scaled, C_scaled )
    #         energies += [E] * count
    #         if E < min_energy:
    #             min_energy = E
    #             ground_state = np.array([1 if b == '0' else -1 for b in bitstring[::-1]])
        
    #     asset_counts = []
    #     for i in range(n):
    #         count = 0
    #         for p in range(self.bits_per_asset):
    #             idx = i*self.bits_per_asset + p
    #             if ground_state[idx] == -1:
    #                 count += 2**p
    #         asset_counts.append(count)
        
    #     return np.array(asset_counts)
