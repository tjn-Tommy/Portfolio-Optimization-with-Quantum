from typing import Callable, Optional, Sequence, Any, Dict

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize, OptimizeResult
from qiskit.quantum_info import SparsePauliOp
from optimizer.base import BaseOptimizer
from optimizer.utils.qubo_utils import compute_num_spins as compute_num_spins_optimized
from optimizer.utils.qubo_utils import spins_to_asset_counts
from optimizer.utils.qubo_utils import qubo_factor as qubo_factor_optimized
from optimizer.utils.qubo_utils import get_ising_coeffs as get_ising_coeffs_optimized
from optimizer.utils.noise_utils import build_aer_simulator
from qiskit_aer.primitives import EstimatorV2


GRADIENT_BASED_METHODS = {
    "BFGS",
    "L-BFGS-B",
    "CG",
    "NEWTON-CG",
    "DOGLEG",
    "TRUST-NCG",
    "TRUST-KRYLOV",
    "TRUST-EXACT",
    "TRUST-CONSTR",
    "TNC",
    "SLSQP",
}
VALID_GRAD_METHODS = {"shot_based", "estimator"}


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
        optimization_algorithm: str = "COBYLA",
        grad_method: str = "param_shift",
        spsa_options: Optional[Dict[str, float]] = None,
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
        self.optimization_algorithm = optimization_algorithm
        self.grad_method = grad_method
        self.spsa_options = spsa_options or {}
        self.noise_config = noise_config
        self.backend = build_aer_simulator(noise_config)
        if use_gpu:
            # 1. 强制使用 GPU
            self.backend.set_options(device='GPU')
            
            # 2. 关键：设置精度为单精度
            # "single": complex64 (对应 float32)
            # "double": complex128 (对应 float64) - 默认值
            self.backend.set_options(precision='single', cuStateVec_enable=True) 

            
            print("✅ GPU Acceleration enabled with Single Precision.")
        self.num_spins = 0
        self.estimator = EstimatorV2(
            options={
            "run_options":{"shots": None, "seed": 42},
            "backend_options":{
                "method": "statevector",      
                "device": "GPU",              
                "precision": "single",        
                "cuStateVec_enable": True 
            },}
            )

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
            optimization_algorithm=cfg.get(
                "optimization_algorithm",
                cfg.get("optimzation_algorithm", "COBYLA"),
            ),
            grad_method=cfg.get("grad_method", "param_shift"),
            spsa_options=cfg.get("spsa"),
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

    def _build_circuit(self, p: int, h: np.ndarray, J: np.ndarray, measure: bool=True) -> QuantumCircuit:
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
                        # qc.cx(i, j)
                        # qc.rz(gammas[layer] * 2 * J[i, j], j)
                        # qc.cx(i, j)
                        qc.rzz(gammas[layer] * 2 * J[i, j], i, j)
            for i in range(self.num_spins):
                qc.rx(betas[layer] * 2, i)
        if measure:
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

    def _initial_params(
        self,
        p: int,
        initial_betas: Optional[Sequence[float]],
        initial_gammas: Optional[Sequence[float]],
    ) -> np.ndarray:
        if initial_betas is None:
            betas = 1 * np.linspace(1, 0, p)
        else:
            betas = np.asarray(initial_betas, dtype=float)
            if betas.size != p:
                raise ValueError("initial_betas must have length p")

        if initial_gammas is None:
            gammas = 1 * np.linspace(0, 1, p)
        else:
            gammas = np.asarray(initial_gammas, dtype=float)
            if gammas.size != p:
                raise ValueError("initial_gammas must have length p")

        return np.concatenate([betas, gammas])

    def _get_hamiltonian(self, h: np.ndarray, J: np.ndarray) -> SparsePauliOp:
        num_qubits = len(h)
        pauli_list = []
    
        for i, coeff in enumerate(h):
            if abs(coeff) > 1e-8:
                pauli_str = ["I"] * num_qubits
                pauli_str[num_qubits - 1 - i] = "Z" # Qiskit 是 Little Endian，索引要反转
                pauli_list.append(("".join(pauli_str), coeff))
        
        rows, cols = np.nonzero(J)
        for i, j in zip(rows, cols):
            if i < j: 
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
        term1 = spins_matrix @ h
        term2 = np.sum((spins_matrix @ J) * spins_matrix, axis=1)
        energies = term1 + term2
        
        # 4. 加权平均
        avg_energy = np.sum(energies * freqs) / total_shots
        return float(avg_energy)


    def _gradient_estimator(  # 修正拼写: gradiant -> gradient, estimater -> estimator
        self,
        x_init: np.ndarray,
        circ: QuantumCircuit,
        p: int,
        h: np.ndarray,
        J: np.ndarray,
        shots: int,
        method: str,
    ) -> np.ndarray:
        num_params = len(x_init)
        
        # --- 1. 构建参数矩阵 (Batching) ---
        # 我们不再创建 list of dicts，而是创建一个大的 numpy array
        # 形状: (2 * num_params, num_params)
        batch_params = np.empty((2 * num_params, num_params))
        
        for i in range(num_params):
            # x + delta
            batch_params[2 * i] = x_init.copy()
            batch_params[2 * i, i] += self.grad_delta
            
            # x - delta
            batch_params[2 * i + 1] = x_init.copy()
            batch_params[2 * i + 1, i] -= self.grad_delta

        # --- 3. 构建单一 PUB (Broadcasting) ---
        hamiltonian = self._get_hamiltonian(h, J)
        pub = (circ, hamiltonian, batch_params)
        
        # --- 4. 一次性执行 ---
        job = self.estimator.run([pub]) 
        result = job.result()
        
        # --- 5. 获取结果 ---
        evs = result[0].data.evs
        
        # --- 6. 计算梯度 ---
        gradients = (evs[0::2] - evs[1::2]) / (2.0 * self.grad_delta)
        
        return gradients

    # --- 优化点 2: 目标函数 ---
    def _objective(
        self,
        x_init: np.ndarray,
        circ: QuantumCircuit,
        p: int,
        h: np.ndarray,
        J: np.ndarray,
        shots: int,
    ) -> float:
        betas = x_init[:p]
        gammas = x_init[p:]
        bind_dict = self._build_bind_dict(circ, p, betas, gammas)
        counts = self._run_counts(circ, bind_dict, shots)
        return self._compute_expectation(counts, h, J)

    def _evaluate_expectations(
        self,
        param_sets: Sequence[np.ndarray],
        circ: QuantumCircuit,
        p: int,
        h: np.ndarray,
        J: np.ndarray,
        shots: int,
    ) -> Sequence[float]:
        if not param_sets:
            return []

        binds = []
        for params in param_sets:
            betas = params[:p]
            gammas = params[p:]
            binds.append(self._build_bind_dict(circ, p, betas, gammas))

        circuits_to_run = [circ] * len(binds)
        job = self.backend.run(circuits_to_run, shots=shots, parameter_binds=binds)
        counts_list = job.result().get_counts()
        if not isinstance(counts_list, list):
            counts_list = [counts_list]

        return [self._compute_expectation(counts, h, J) for counts in counts_list]

    def _gradient(
        self,
        x_init: np.ndarray,
        circ: QuantumCircuit,
        p: int,
        h: np.ndarray,
        J: np.ndarray,
        shots: int,
        method: str,
    ) -> np.ndarray:
        method_key = (method or "").lower()
        step = float(self.grad_delta)
        if step <= 0:
            raise ValueError("grad_delta must be positive for finite_diff.")
        scale = 1.0 / (2.0 * step)

        param_sets = []
        for i in range(len(x_init)):
            x_plus = x_init.copy()
            x_plus[i] += step
            x_minus = x_init.copy()
            x_minus[i] -= step
            param_sets.append(x_plus)
            param_sets.append(x_minus)

        energies = self._evaluate_expectations(param_sets, circ, p, h, J, shots)
        gradients = np.zeros(len(x_init))
        for i in range(len(x_init)):
            gradients[i] = scale * (energies[2 * i] - energies[2 * i + 1])
        return gradients

    @staticmethod
    def _algorithm_uses_gradient(method: str) -> bool:
        return method.strip().upper() in GRADIENT_BASED_METHODS

    def _optimize_spsa(
        self,
        objective_fn: Callable[[np.ndarray], float],
        x_init: np.ndarray,
        maxiter: int,
        rng: Optional[np.random.Generator],
    ) -> OptimizeResult:
        if rng is None:
            rng = np.random.default_rng()

        options = self.spsa_options
        a = float(options.get("a", 0.2))
        c = float(options.get("c", 0.1))
        alpha = float(options.get("alpha", 0.602))
        gamma = float(options.get("gamma", 0.101))
        A = float(options.get("A", max(1, maxiter // 10)))

        x = x_init.copy()
        best_x = x.copy()
        best_val = objective_fn(x)
        n_params = len(x)

        for k in range(maxiter):
            ak = a / ((k + 1 + A) ** alpha)
            ck = c / ((k + 1) ** gamma)
            delta = rng.choice([-1.0, 1.0], size=n_params)
            x_plus = x + ck * delta
            x_minus = x - ck * delta
            f_plus = objective_fn(x_plus)
            f_minus = objective_fn(x_minus)
            g_hat = (f_plus - f_minus) / (2.0 * ck) * delta
            x = x - ak * g_hat

            f_val = objective_fn(x)
            if f_val < best_val:
                best_val = f_val
                best_x = x.copy()

        return OptimizeResult(x=best_x, fun=best_val, nit=maxiter)
    
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
        optimization_algorithm: Optional[str] = None,
        grad_method: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        n = len(mu)
        self.num_spins, self.bits_plus, self.bits_minus = self.compute_num_spins(n, x0)
        
        Q, L, constant = self.qubo_factor(n, mu, sigma, prices, self.num_spins, budget, x0)
        h, J, C = self.get_ising_coeffs(Q, L, constant)

        # 参数设置
        chosen_p = p if p is not None else self.p
        chosen_shots = shots if shots is not None else self.shots
        chosen_trials = n_trials if n_trials is not None else self.n_trials
        chosen_maxiter = maxiter if maxiter is not None else self.maxiter
        chosen_spread = init_spread if init_spread is not None else self.init_spread
        chosen_seed = seed if seed is not None else self.seed
        chosen_algorithm = (
            optimization_algorithm
            if optimization_algorithm is not None
            else self.optimization_algorithm
        )
        if not chosen_algorithm:
            chosen_algorithm = "COBYLA"
        chosen_grad_method = grad_method if grad_method is not None else self.grad_method
        method_key = chosen_algorithm.strip().upper()
        use_spsa = method_key == "SPSA"
        requires_gradient = self._algorithm_uses_gradient(method_key)
        grad_method_key = (chosen_grad_method or "").lower()
        if requires_gradient and grad_method_key not in VALID_GRAD_METHODS:
            raise ValueError(
                f"Unsupported grad_method: {chosen_grad_method}. "
                f"Choose from {sorted(VALID_GRAD_METHODS)}."
            )
        
        # 构建电路
        circuit = self._build_circuit(chosen_p, h, J)
        circuit = transpile(circuit, self.backend)

        base_params = self._initial_params(chosen_p, initial_betas, initial_gammas)
        rng = np.random.default_rng(chosen_seed)
        best_solution = None
        best_value = float("inf")
        objective_fn = lambda params: self._objective(
            params, circuit, chosen_p, h, J, chosen_shots
        )

        for trial in range(chosen_trials):
            x_init = base_params.copy()
            if trial > 0 and chosen_spread > 0:
                x_init = x_init + rng.normal(scale=chosen_spread, size=2 * chosen_p)

            if use_spsa:
                sol = self._optimize_spsa(objective_fn, x_init, chosen_maxiter, rng)
            else:
                jac = None
                if requires_gradient:
                    if grad_method_key == "shot_based" 
                        jac = lambda x, *args: self._gradient(
                            x, *args, method="finite_diff"
                        )
                    elif grad_method_key == "estimator":
                        jac = lambda x, *args: self._gradient_estimator(
                            x, *args, method=grad_method_key
                        )
                minimize_kwargs = {
                    "x0": x_init,
                    "args": (circuit, chosen_p, h, J, chosen_shots),
                    "method": chosen_algorithm,
                    "options": {"maxiter": chosen_maxiter},
                    "tol": 1e-4,
                }
                if jac is not None:
                    minimize_kwargs["jac"] = jac

                sol = minimize(self._objective, **minimize_kwargs)

            if np.isfinite(sol.fun) and sol.fun < best_value:
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

        return self._spins_to_asset_counts(best_spins, n, x0)
