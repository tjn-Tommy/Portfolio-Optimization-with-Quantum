from typing import Callable, Optional, Sequence, Any, Dict

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from optimizer.base import BaseOptimizer
from optimizer.utils.qubo_utils import qubo_factor as qubo_factor_optimized
from optimizer.utils.qubo_utils import get_ising_coeffs as get_ising_coeffs_optimized
from optimizer.utils.noise_utils import build_aer_simulator


class QAOAOptimizer(BaseOptimizer):
    def __init__(
        self,
        lam: float,
        alpha: float,
        beta: Optional[float],
        bits_per_asset: int,
        bits_slack: int,
        transact_opt: str = "ignore",
        p: int = 1,
        shots: int = 1000,
        n_trials: int = 1,
        maxiter: int = 100,
        grad_delta: float = 0.01,
        init_spread: float = 0.0,
        seed: Optional[int] = None,
        use_gradient: bool = True,
        noise_config: Optional[Dict[str, Any]] = None,
        use_gpu: bool = False,
    ):
        super().__init__(lam, beta)
        self.alpha = alpha
        self.bits_per_asset = bits_per_asset
        self.bits_slack = bits_slack
        self.transact_opt = transact_opt
        self.p = p
        self.shots = shots
        self.n_trials = n_trials
        self.maxiter = maxiter
        self.grad_delta = grad_delta
        self.init_spread = init_spread
        self.seed = seed
        self.use_gradient = use_gradient
        self.noise_config = noise_config
        self.backend = build_aer_simulator(noise_config)
        if use_gpu:
            # 1. 强制使用 GPU
            self.backend.set_options(device='GPU')
            
            # 2. 关键：设置精度为单精度
            # "single": complex64 (对应 float32)
            # "double": complex128 (对应 float64) - 默认值
            self.backend.set_options(precision='single') 
            
            print("✅ GPU Acceleration enabled with Single Precision.")
        self.num_spins = 0

    @classmethod
    def init(cls, cfg: Dict[str, Any], lam: float, beta: Optional[float]) -> "QAOAOptimizer":
        return cls(
            lam=lam,
            alpha=cfg["alpha"],
            beta=beta,
            transact_opt=cfg.get("transact_opt", "ignore"),
            bits_per_asset=cfg["bits_per_asset"],
            bits_slack=cfg["bits_slack"],
            p=cfg.get("p", 1),
            shots=cfg.get("shots", 1000),
            n_trials=cfg.get("n_trials", 1),
            maxiter=cfg.get("maxiter", 100),
            grad_delta=cfg.get("grad_delta", 0.01),
            init_spread=cfg.get("init_spread", 0.0),
            seed=cfg.get("seed"),
            use_gradient=cfg.get("use_gradient", True),
            noise_config=cfg.get("noise"),
            use_gpu=cfg.get("use_gpu", False),
        )

    def qubo_factor(
        self,
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

    def get_ising_coeffs(self, Q: np.ndarray, L: np.ndarray, constant: float):
        return get_ising_coeffs_optimized(Q, L, constant)

    @property
    def optimizer(self) -> Callable:
        return self.optimize
    
    def _build_circuit(self, p: int, h: np.ndarray, J: np.ndarray) -> QuantumCircuit:
        betas = ParameterVector("betas", p)
        gammas = ParameterVector("gammas", p)
        qc = QuantumCircuit(self.num_spins)
        qc.h(range(self.num_spins))

        for layer in range(p):
            for i in range(self.num_spins):
                if h[i] != 0:
                    qc.rz(gammas[layer] * 2 * h[i], i)
            for i in range(self.num_spins):
                for j in range(i + 1, self.num_spins):
                    if J[i, j] != 0:
                        qc.cx(i, j)
                        qc.rz(gammas[layer] * 2 * J[i, j], j)
                        qc.cx(i, j)
            for i in range(self.num_spins):
                qc.rx(betas[layer] * 2, i)

        qc.measure_all()
        return qc

    def _build_bind_dict(
        self,
        circ: QuantumCircuit,
        p: int,
        betas: np.ndarray,
        gammas: np.ndarray,
    ):
        param_map = {param.name: param for param in circ.parameters}
        bind_dict = {}
        for i in range(p):
            bind_dict[param_map[f"betas[{i}]"]] = [float(betas[i])]
            bind_dict[param_map[f"gammas[{i}]"]] = [float(gammas[i])]
        return bind_dict

    def _run_counts(
        self,
        circ: QuantumCircuit,
        bind_dict,
        shots: int,
    ):
        job = self.backend.run(circ, shots=shots, parameter_binds=[bind_dict])
        counts = job.result().get_counts()
        if isinstance(counts, list):
            return counts[0]
        return counts

    def _bitstring_to_spins(self, bitstring: str) -> np.ndarray:
        bits = bitstring.replace(" ", "")
        spins = np.empty(self.num_spins, dtype=int)
        for i, char in enumerate(reversed(bits)):
            spins[i] = 1 if char == "0" else -1
        return spins

    # def _compute_expectation(
    #     self,
    #     counts,
    #     h: np.ndarray,
    #     J: np.ndarray,
    # ) -> float:
    #     if not counts:
    #         return float("inf")
    #     bitstrings = list(counts.keys())
    #     counts_arr = np.array(list(counts.values()), dtype=float)
    #     total_shots = counts_arr.sum()
    #     if total_shots <= 0:
    #         return float("inf")

    #     spins = np.zeros((len(bitstrings), self.num_spins))
    #     for k, bitstring in enumerate(bitstrings):
    #         spins[k] = self._bitstring_to_spins(bitstring)

    #     term1 = spins @ h
    #     term2 = np.sum((spins @ J) * spins, axis=1)
    #     energies = term1 + term2
    #     return float(np.sum(energies * counts_arr) / total_shots)

    # def _objective(
    #     self,
    #     x0: np.ndarray,
    #     circ: QuantumCircuit,
    #     p: int,
    #     h: np.ndarray,
    #     J: np.ndarray,
    #     shots: int,
    # ) -> float:
    #     betas = x0[:p]
    #     gammas = x0[p:]
    #     bind_dict = self._build_bind_dict(circ, p, betas, gammas)
    #     counts = self._run_counts(circ, bind_dict, shots)
    #     return self._compute_expectation(counts, h, J)

    # def _gradient(
    #     self,
    #     x0: np.ndarray,
    #     circ: QuantumCircuit,
    #     p: int,
    #     h: np.ndarray,
    #     J: np.ndarray,
    #     shots: int,
    #     delta: float,
    # ) -> np.ndarray:
    #     num_params = len(x0)
    #     param_sets = []
    #     for i in range(num_params):
    #         x_plus = x0.copy()
    #         x_plus[i] += delta
    #         param_sets.append(x_plus)

    #         x_minus = x0.copy()
    #         x_minus[i] -= delta
    #         param_sets.append(x_minus)

    #     param_map = {param.name: param for param in circ.parameters}
    #     binds = []
    #     for params in param_sets:
    #         betas = params[:p]
    #         gammas = params[p:]
    #         bind_dict = {}
    #         for i in range(p):
    #             bind_dict[param_map[f"betas[{i}]"]] = [float(betas[i])]
    #             bind_dict[param_map[f"gammas[{i}]"]] = [float(gammas[i])]
    #         binds.append(bind_dict)
    #     circuits_to_run = [circ] * len(binds)
    #     job = self.backend.run(circuits_to_run, shots=shots, parameter_binds=binds)
    #     counts_list = job.result().get_counts()
    #     if not isinstance(counts_list, list):
    #         counts_list = [counts_list]

    #     gradients = np.zeros(num_params)
    #     for i in range(num_params):
    #         counts_plus = counts_list[2 * i]
    #         counts_minus = counts_list[2 * i + 1]

    #         e_plus = self._compute_expectation(counts_plus, h, J)
    #         e_minus = self._compute_expectation(counts_minus, h, J)
    #         gradients[i] = (e_plus - e_minus) / (2 * delta)

    #     return gradients

    def _initial_params(
        self,
        p: int,
        initial_betas: Optional[Sequence[float]],
        initial_gammas: Optional[Sequence[float]],
    ) -> np.ndarray:
        if initial_betas is None:
            betas = 0.05 * np.linspace(1, 0, p)
        else:
            betas = np.asarray(initial_betas, dtype=float)
            if betas.size != p:
                raise ValueError("initial_betas must have length p")

        if initial_gammas is None:
            gammas = 0.05 * np.linspace(0, 1, p)
        else:
            gammas = np.asarray(initial_gammas, dtype=float)
            if gammas.size != p:
                raise ValueError("initial_gammas must have length p")

        return np.concatenate([betas, gammas])

    def _spins_to_asset_counts(self, spins: np.ndarray, n: int) -> np.ndarray:
        asset_counts = []
        for i in range(n):
            count = 0
            for p in range(self.bits_per_asset):
                idx = i * self.bits_per_asset + p
                if spins[idx] == -1:
                    count += 2**p
            asset_counts.append(count)
        return np.array(asset_counts, dtype=int)

    # def optimize(
    #     self,
    #     mu: np.ndarray,
    #     prices: np.ndarray,
    #     sigma: np.ndarray,
    #     budget: float,
    #     p: Optional[int] = None,
    #     shots: Optional[int] = None,
    #     n_trials: Optional[int] = None,
    #     maxiter: Optional[int] = None,
    #     initial_betas: Optional[Sequence[float]] = None,
    #     initial_gammas: Optional[Sequence[float]] = None,
    #     init_spread: Optional[float] = None,
    #     seed: Optional[int] = None,
    #     use_gradient: Optional[bool] = None,
    # ) -> Optional[np.ndarray]:
    #     n = len(mu)
    #     self.num_spins = n * self.bits_per_asset + self.bits_slack

    #     Q, L, constant = self.qubo_factor(
    #         n=n,
    #         mu=mu,
    #         sigma=sigma,
    #         prices=prices,
    #         n_spins=self.num_spins,
    #         budget=budget,
    #     )
    #     h, J, C = self.get_ising_coeffs(Q, L, constant)

    #     chosen_p = p if p is not None else self.p
    #     chosen_shots = shots if shots is not None else self.shots
    #     chosen_trials = n_trials if n_trials is not None else self.n_trials
    #     chosen_maxiter = maxiter if maxiter is not None else self.maxiter
    #     chosen_spread = init_spread if init_spread is not None else self.init_spread
    #     chosen_seed = seed if seed is not None else self.seed
    #     chosen_grad = use_gradient if use_gradient is not None else self.use_gradient

    #     circuit = self._build_circuit(chosen_p, h, J)
    #     circuit = transpile(circuit, self.backend)

    #     base_params = self._initial_params(
    #         chosen_p,
    #         initial_betas,
    #         initial_gammas,
    #     )

    #     rng = np.random.default_rng(chosen_seed)
    #     best_solution = None
    #     best_value = float("inf")

    #     def jac_fn(x, *args):
    #         return self._gradient(x, *args, delta=self.grad_delta)

    #     for trial in range(chosen_trials):
    #         x0 = base_params.copy()
    #         if trial > 0 and chosen_spread > 0:
    #             x0 = x0 + rng.normal(scale=chosen_spread, size=2 * chosen_p)

    #         jac = None
    #         if chosen_grad:
    #             jac = jac_fn
    #         sol = minimize(
    #             self._objective,
    #             x0=x0,
    #             args=(circuit, chosen_p, h, J, chosen_shots),
    #             method="BFGS",
    #             jac=jac,
    #             options={"maxiter": chosen_maxiter},
    #         )
    #         if np.isfinite(sol.fun) and sol.fun < best_value:
    #             best_value = sol.fun
    #             best_solution = sol

    #     if best_solution is None:
    #         return None

    #     best_params = best_solution.x
    #     betas = best_params[:chosen_p]
    #     gammas = best_params[chosen_p:]
    #     bind_dict = self._build_bind_dict(circuit, chosen_p, betas, gammas)
    #     counts = self._run_counts(circuit, bind_dict, chosen_shots)
    #     if not counts:
    #         return None

    #     min_energy = float("inf")
    #     best_spins = None
    #     for bitstring in counts:
    #         spins = self._bitstring_to_spins(bitstring)
    #         energy = spins @ J @ spins + h @ spins + C
    #         if energy < min_energy:
    #             min_energy = energy
    #             best_spins = spins

    #     if best_spins is None:
    #         return None

    #     return self._spins_to_asset_counts(best_spins, n)


    def _get_hamiltonian(self, h: np.ndarray, J: np.ndarray) -> SparsePauliOp:
        num_qubits = len(h)
        pauli_list = []
        
        # 线性项 h * Z_i
        for i, coeff in enumerate(h):
            if abs(coeff) > 1e-8:
                pauli_str = ["I"] * num_qubits
                pauli_str[num_qubits - 1 - i] = "Z" # Qiskit 是 Little Endian，索引要反转
                pauli_list.append(("".join(pauli_str), coeff))
        
        # 二次项 J * Z_i * Z_j
        rows, cols = np.nonzero(J)
        for i, j in zip(rows, cols):
            if i < j: # 只取上三角，避免重复
                coeff = J[i, j]
                if abs(coeff) > 1e-8:
                    pauli_str = ["I"] * num_qubits
                    pauli_str[num_qubits - 1 - i] = "Z"
                    pauli_str[num_qubits - 1 - j] = "Z"
                    pauli_list.append(("".join(pauli_str), coeff))
        
        if not pauli_list:
            return SparsePauliOp(["I" * num_qubits], [0.0])
            
        return SparsePauliOp.from_list(pauli_list)

    # --- 优化点 1: 向量化计算期望值 (速度极大提升) ---
    def _compute_expectation(
        self,
        counts: Dict[str, int],
        h: np.ndarray,
        J: np.ndarray,
    ) -> float:
        if not counts:
            return float("inf")
            
        # 1. 提取所有 bitstrings 和对应的频率
        bitstrings = list(counts.keys())
        freqs = np.array(list(counts.values()), dtype=float)
        total_shots = np.sum(freqs)
        
        if total_shots <= 0:
            return float("inf")
            
        # 2. 向量化转换：Bitstring (str) -> Spins (numpy array)
        # 这是一个技巧：将字符串列表转为字符矩阵，再转为 int
        # 注意：Qiskit 的 bitstring 是从右到左对应 qubit 0 到 N
        n_spins = len(h)
        
        # 创建字符矩阵 (M samples x N spins)
        # 例如 ['10', '01'] -> [['1','0'], ['0','1']]
        char_matrix = np.array([list(s) for s in bitstrings])
        
        # 将 '0'->1, '1'->-1。Qiskit输出中 '0'是+1态, '1'是-1态
        # 注意：需要反转列顺序以匹配你的 J 矩阵索引（通常 Qiskit输出是 qubit N...0）
        # 你的原代码用了 reversed(bits)，这里我们通过 flip 模拟
        spins_matrix = np.ones(char_matrix.shape, dtype=float)
        spins_matrix[char_matrix == '1'] = -1.0
        
        # 如果你的 qubit 0 对应 bitstring 的最右边（标准 Qiskit），则需翻转矩阵列
        spins_matrix = np.flip(spins_matrix, axis=1)

        # 3. 向量化计算能量
        # E = h . s + s . J . s^T
        # term1: (M, N) @ (N,) -> (M,)
        term1 = spins_matrix @ h
        
        # term2: 逐元素计算二次项
        # (spins @ J) 得到 (M, N)，然后与 spins (M, N) 逐元素相乘并求和
        term2 = np.sum((spins_matrix @ J) * spins_matrix, axis=1)
        
        energies = term1 + term2
        
        # 4. 加权平均
        avg_energy = np.sum(energies * freqs) / total_shots
        return float(avg_energy)

    # --- 优化点 2: 目标函数修改 (去掉梯度计算，改用无梯度优化器) ---
    def _objective(
        self,
        x0: np.ndarray,
        circ: QuantumCircuit,
        p: int,
        h: np.ndarray,
        J: np.ndarray,
        shots: int,
    ) -> float:
        betas = x0[:p]
        gammas = x0[p:]
        bind_dict = self._build_bind_dict(circ, p, betas, gammas)
        counts = self._run_counts(circ, bind_dict, shots)
        return self._compute_expectation(counts, h, J)
    
    def optimize(
        self,
        mu: np.ndarray,
        prices: np.ndarray,
        sigma: np.ndarray,
        budget: float,
        x0: Optional[np.ndarray] = None,
        p: Optional[int] = None,
        shots: Optional[int] = None,
        n_trials: Optional[int] = None,
        maxiter: Optional[int] = None,
        initial_betas: Optional[Sequence[float]] = None,
        initial_gammas: Optional[Sequence[float]] = None,
        init_spread: Optional[float] = None,
        seed: Optional[int] = None,
        use_gradient: Optional[bool] = False, # 建议默认为 False
    ) -> Optional[np.ndarray]:
        # ... (参数初始化代码保持不变) ...
        n = len(mu)
        self.num_spins = n * self.bits_per_asset + self.bits_slack
        
        # ... (QUBO 生成代码保持不变) ...
        Q, L, constant = self.qubo_factor(n, mu, sigma, prices, self.num_spins, budget, x0)
        h, J, C = self.get_ising_coeffs(Q, L, constant)

        # 参数设置
        chosen_p = p if p is not None else self.p
        chosen_shots = shots if shots is not None else self.shots
        chosen_trials = n_trials if n_trials is not None else self.n_trials
        chosen_maxiter = maxiter if maxiter is not None else self.maxiter
        chosen_spread = init_spread if init_spread is not None else self.init_spread
        chosen_seed = seed if seed is not None else self.seed
        
        # 构建电路
        circuit = self._build_circuit(chosen_p, h, J)
        circuit = transpile(circuit, self.backend)

        base_params = self._initial_params(chosen_p, initial_betas, initial_gammas)
        rng = np.random.default_rng(chosen_seed)
        best_solution = None
        best_value = float("inf")

        

        for trial in range(chosen_trials):
            x0 = base_params.copy()
            if trial > 0 and chosen_spread > 0:
                x0 = x0 + rng.normal(scale=chosen_spread, size=2 * chosen_p)

            # --- 关键修改: 推荐使用 COBYLA ---
            # COBYLA 不需要梯度函数，且对噪声更有鲁棒性
            method = "COBYLA"
            jac = None
            
            # 如果你确实非常想用梯度 (BFGS)，请保留原有的 jac_fn 逻辑，
            # 但要注意 BFGS 在含噪量子模拟中性能通常很差。
            if use_gradient: 
                method = "BFGS"
                jac = lambda x, *args: self._gradient(x, *args, delta=self.grad_delta)

            sol = minimize(
                self._objective,
                x0=x0,
                args=(circuit, chosen_p, h, J, chosen_shots),
                method=method, # 使用 COBYLA 或 BFGS
                jac=jac,
                options={"maxiter": chosen_maxiter},
                tol=1e-4 # 对于 COBYLA 有用
            )

            if sol.fun < best_value:
                best_value = sol.fun
                best_solution = sol

        if best_solution is None:
            return None

        # --- 结果解析 ---
        best_params = best_solution.x
        betas = best_params[:chosen_p]
        gammas = best_params[chosen_p:]
        bind_dict = self._build_bind_dict(circuit, chosen_p, betas, gammas)
        counts = self._run_counts(circuit, bind_dict, chosen_shots)
        
        if not counts:
            return None

        # 同样使用向量化方法寻找最优解
        bitstrings = list(counts.keys())
        char_matrix = np.array([list(s) for s in bitstrings])
        spins_matrix = np.ones(char_matrix.shape, dtype=float)
        spins_matrix[char_matrix == '1'] = -1.0
        spins_matrix = np.flip(spins_matrix, axis=1) # 记得翻转
        
        # 批量计算能量
        term1 = spins_matrix @ h
        term2 = np.sum((spins_matrix @ J) * spins_matrix, axis=1)
        energies = term1 + term2 + C # 加上常数项
        
        min_idx = np.argmin(energies)
        best_spins = spins_matrix[min_idx].astype(int)

        return self._spins_to_asset_counts(best_spins, n)