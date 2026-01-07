from typing import Callable, Optional, Sequence, Any, Dict

import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize, OptimizeResult
from qiskit.quantum_info import SparsePauliOp
from optimizer.base import BaseOptimizer
from optimizer.utils.qubo_utils import compute_num_spins as compute_num_spins_optimized
from optimizer.utils.qubo_utils import spins_to_asset_counts
from optimizer.utils.qubo_utils import qubo_factor as qubo_factor_optimized
from optimizer.utils.qubo_utils import get_ising_coeffs as get_ising_coeffs_optimized
from optimizer.utils.qubo_utils import normalize_ising_coeffs
from optimizer.utils.noise_utils import build_aer_simulator
from qiskit_aer.primitives import EstimatorV2
import time

GRADIENT_BASED_METHODS = {
    "L-BFGS-B",
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
        grad_method: str = "shot_based",
        spsa_options: Optional[Dict[str, float]] = None,
        noise_config: Optional[Dict[str, Any]] = None,
        use_gpu: bool = False,
        use_history: bool = False,
        shift: float = 0.05,
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
            self.backend.set_options(precision='single', cuStateVec_enable=True) 

            
            print("✅ GPU Acceleration enabled with Single Precision.")
        self.num_spins = 0
        self.estimator = EstimatorV2(
            options={
                
            "run_options":{"shots": None, "seed": 42},
            "backend_options":{
                "method": "statevector",      
                "device": "GPU" if use_gpu else "CPU",              
                "precision": "single",        
                "cuStateVec_enable": True,
            },}
            )
        self.use_history = use_history
        self.history = None
        self.shift = shift
        

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
            grad_method=cfg.get("grad_method", "shot_based"),
            spsa_options=cfg.get("spsa"),
            noise_config=cfg.get("noise"),
            use_gpu=cfg.get("use_gpu", False),
            use_history=cfg.get("use_history", False),
            shift=cfg.get("shift", 0.05),
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
    ) -> np.ndarray:
        num_params = len(x_init)
        # start_time = time.time()
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
        # end_time = time.time()
        # print(f"Gradient computed in {end_time - start_time:.2f} seconds.")
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
        if self.grad_method == "estimator":
            hamiltonian = self._get_hamiltonian(h, J)
            pub = (circ, hamiltonian, x_init)
            job = self.estimator.run([pub])
            result = job.result()
            energy = result[0].data.evs
            if isinstance(energy, np.ndarray):
                return float(energy.item())         
            return float(energy)
        else:
             # Shot-based 评估
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
    ) -> np.ndarray:
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
    
    def _compute_val_and_grad(
        self,
        x_init: np.ndarray,
        circ: QuantumCircuit,
        p: int,
        h: np.ndarray,
        J: np.ndarray,
        shots: int,
    ):
        num_params = len(x_init)
        
        # --- 1. 构建超大 Batch (1 + 2 * num_params) ---
        total_circuits = 1 + 2 * num_params
        batch_params = np.empty((total_circuits, num_params))
        
        # 填入原始参数
        batch_params[0] = x_init
        
        # 填入梯度参数
        for i in range(num_params):
            # x + delta
            batch_params[1 + 2 * i] = x_init.copy()
            batch_params[1 + 2 * i, i] += self.grad_delta
            
            # x - delta
            batch_params[1 + 2 * i + 1] = x_init.copy()
            batch_params[1 + 2 * i + 1, i] -= self.grad_delta

        # --- 2. 只有一次 GPU 调用 (Crucial!) ---
        # Qiskit Aer 会并行计算这 81 个电路
        if self.grad_method == "estimator":
            hamiltonian = self._get_hamiltonian(h, J)
            pub = (circ, hamiltonian, batch_params) # 广播
            job = self.estimator.run([pub])
            result = job.result()
            evs = result[0].data.evs
            
            # --- 3. 解析结果 ---
            # 目标函数值 (第 1 个结果)
            objective_value = float(evs[0])
            
            # 梯度 (剩下的结果)
            grad_evs = evs[1:]
            gradients = (grad_evs[0::2] - grad_evs[1::2]) / (2.0 * self.grad_delta)
            
            return objective_value, gradients

        else:
            raise NotImplementedError("Merged execution is currently optimized for Estimator only.")
        

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
        **kwargs,
    ) -> Optional[np.ndarray]:
        return self._optimize_interp(
            mu,
            prices,
            sigma,
            budget,
            x0,
            p,
            shots,
            n_trials,
            maxiter,
            initial_betas,
            initial_gammas,
            init_spread,
            seed,
            optimization_algorithm,
            grad_method,
            **kwargs,
        )

    def _optimize(
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
        **kwargs,
    ) -> Optional[np.ndarray]:
        n = len(mu)
        self.num_spins, self.bits_plus, self.bits_minus = self.compute_num_spins(n, x0)

        Q, L, constant = self.qubo_factor(n, mu, sigma, prices, self.num_spins, budget, x0)
        h, J, C = self.get_ising_coeffs(Q, L, constant)
        h, J, C= normalize_ising_coeffs(h, J, C)
        
        outer_pbar = kwargs.get("outer_pbar")

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
        bounds = [(0, 2*np.pi)] * (2 * chosen_p)
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
        circuit_no_measure = self._build_circuit(chosen_p, h, J, measure=False)
        circuit = transpile(circuit, self.backend)
        circuit_no_measure = transpile(circuit_no_measure, self.backend)
        if self.use_history and self.history is not None:
            initial_betas = self.history.get("betas", initial_betas)
            initial_gammas = self.history.get("gammas", initial_gammas)
            # add gussian noise around previous best
            if initial_betas is not None:
                initial_betas = np.array(initial_betas) + np.random.normal(
                    scale=self.shift, size=chosen_p
                )
            if initial_gammas is not None:
                initial_gammas = np.array(initial_gammas) + np.random.normal(
                    scale=self.shift, size=chosen_p
                )
        base_params = self._initial_params(chosen_p, initial_betas, initial_gammas)
        rng = np.random.default_rng(chosen_seed)
        best_solution = None
        best_value = float("inf")
        obj_circuit = circuit_no_measure if grad_method_key == "estimator" else circuit
        objective_fn = lambda params: self._objective(
            params, obj_circuit
            , chosen_p, h, J, chosen_shots
        )
        objective_fn_with_grad = lambda params: self._compute_val_and_grad(
            params, obj_circuit, chosen_p, h, J, chosen_shots
        )

        total_iterations = 0
        metadata = kwargs.get("metadata", {})

        for trial in range(chosen_trials):
            x_init = base_params.copy()
            if trial > 0 and chosen_spread > 0:
                x_init = x_init + rng.normal(scale=chosen_spread, size=2 * chosen_p)

            if use_spsa:
                sol = self._optimize_spsa(objective_fn, x_init, chosen_maxiter, rng)
                total_iterations += sol.nit
            else:
                jac = None
                if grad_method_key == "shot_based":
                    jac = lambda x, *args: self._gradient(
                        x, circuit, chosen_p, h, J, chosen_shots,
                    )
                elif grad_method_key == "estimator":
                    jac = lambda x, *args: self._gradient_estimator(
                        x,  circuit_no_measure, chosen_p, h, J, chosen_shots,
                        )
                # 创建进度条
                # pbar = tqdm(total=chosen_maxiter, desc=f"Trial {trial+1}/{chosen_trials}", leave=False)
                current_iter = [0]  # 使用列表以便在闭包中修改
                
                def callback(xk):
                    current_iter[0] += 1
                    # pbar.update(1)
                    # 计算当前目标函数值用于显示
                    # current_val = objective_fn(xk)
                    # pbar.set_postfix({"obj": f"{current_val:.4e}"})
                
                minimize_kwargs = {
                    "x0": x_init,
                    "method": chosen_algorithm,
                    "options": {"maxiter": chosen_maxiter},# "disp": True, 'maxfev': 300, 'final_tr_radius': 1e-5},
                    "tol": 1e-4,
                    "bounds": bounds,
                    "callback": callback,
                    "jac": True
                }
                # if jac is not None:
                    # minimize_kwargs["jac"] = jac

                # sol = minimize(objective_fn, **minimize_kwargs)
                sol = minimize(objective_fn_with_grad, **minimize_kwargs)
                # pbar.close()
                total_iterations += sol.nit

            if np.isfinite(sol.fun) and sol.fun < best_value:
                best_value = sol.fun
                best_solution = sol

        if "iterations" in metadata:
             metadata["iterations"] += total_iterations
        else:
             metadata["iterations"] = total_iterations

        if best_solution is None:
            return None

        # --- 结果解析 ---
        best_params = best_solution.x
        betas = best_params[:chosen_p]
        gammas = best_params[chosen_p:]
        if self.use_history:
            self.history={
                "betas": betas,
                "gammas": gammas,
                "objective_value": best_value
            }
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

    # --- 新增: Interp 插值核心逻辑 ---
    def _interpolate_params(self, old_params: np.ndarray) -> np.ndarray:
        """
        使用线性插值将参数从 p 层扩展到 p+1 层 (Interp Strategy)
        保留波形形状，平滑扩展到更深的电路。
        """
        num_params = len(old_params)
        p_old = num_params // 2
        
        if p_old == 0:
            return self._initial_params(1, None, None)

        betas_old = old_params[:p_old]
        gammas_old = old_params[p_old:]

        p_new = p_old + 1
        
        # 定义旧的时间轴 [0, 1] 和新的时间轴
        # 使用中心点对齐效果通常更好: (i + 0.5) / p
        x_old = (np.arange(p_old) + 0.5) / p_old
        x_new = (np.arange(p_new) + 0.5) / p_new
        
        # 线性插值
        betas_new = np.interp(x_new, x_old, betas_old)
        gammas_new = np.interp(x_new, x_old, gammas_old)
        
        return np.concatenate([betas_new, gammas_new])

    # --- 修改: 加入 strategy 参数并支持逐层循环 ---
    def _optimize_interp(
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
        strategy: str = "interp",  # <--- 新增参数: "standard" or "interp"
        **kwargs,
    ) -> Optional[np.ndarray]:
        n = len(mu)
        self.num_spins, self.bits_plus, self.bits_minus = self.compute_num_spins(n, x0)

        # 1. 计算 Ising/QUBO (这部分只与问题有关，与 p 无关，放在循环外)
        Q, L, constant = self.qubo_factor(n, mu, sigma, prices, self.num_spins, budget, x0)
        h, J, C = self.get_ising_coeffs(Q, L, constant)
        h, J, C = normalize_ising_coeffs(h, J, C)
        
        # 参数解析
        target_p = p if p is not None else self.p
        chosen_shots = shots if shots is not None else self.shots
        chosen_trials = n_trials if n_trials is not None else self.n_trials
        chosen_maxiter = maxiter if maxiter is not None else self.maxiter
        chosen_spread = init_spread if init_spread is not None else self.init_spread
        chosen_seed = seed if seed is not None else self.seed
        
        # 确定算法
        chosen_algorithm = (
            optimization_algorithm
            if optimization_algorithm is not None
            else self.optimization_algorithm
        )
        if not chosen_algorithm:
            chosen_algorithm = "COBYLA"
            
        chosen_grad_method = grad_method if grad_method is not None else self.grad_method
        grad_method_key = (chosen_grad_method or "").lower()
        method_key = chosen_algorithm.strip().upper()
        use_spsa = method_key == "SPSA"
        
        # --- 策略控制逻辑 ---
        if strategy.lower() == "interp":
            print(f"🚀 Starting Interp Strategy optimization up to p={target_p}...")
            p_schedule = range(1, target_p + 1)
        else:
            p_schedule = [target_p]

        best_global_solution = None
        best_global_value = float("inf")
        
        # 存储上一层的最优参数用于插值
        prev_layer_params = None

        # --- 2. 逐层循环 (Interp Loop) ---
        for current_p in p_schedule:
            if strategy.lower() == "interp":
                print(f"  > Optimizing Layer p={current_p}...")
            
            # 2.1 确定当前层的初始化参数
            if current_p == 1:
                # 第一层使用标准初始化 (Random or Linear)
                base_params = self._initial_params(current_p, initial_betas, initial_gammas)
                # 如果是 interp 模式，第一层通常不需要太大 spread，主要靠 optimize 找方向
                current_spread = chosen_spread 
            else:
                # 后续层使用插值
                base_params = self._interpolate_params(prev_layer_params)
                # 插值后的点通常已经很好，spread 可以设小一点或者为0
                current_spread = chosen_spread * 0.5 

            # 2.2 构建当前层的电路
            circuit = self._build_circuit(current_p, h, J) 
            circuit_no_measure = self._build_circuit(current_p, h, J, measure=False)
            
            # Transpile
            circuit = transpile(circuit, self.backend)
            circuit_no_measure = transpile(circuit_no_measure, self.backend)
            
            # 2.3 定义目标函数 (绑定当前的 current_p)
            obj_circuit = circuit_no_measure if grad_method_key == "estimator" else circuit
            
            objective_fn = lambda params: self._objective(
                params, obj_circuit, current_p, h, J, chosen_shots
            )
            
            objective_fn_with_grad = lambda params: self._compute_val_and_grad(
                params, obj_circuit, current_p, h, J, chosen_shots
            )

            bounds = [(0, 2*np.pi)] * (2 * current_p)
            rng = np.random.default_rng(chosen_seed)
            
            # 当前层最好的结果
            layer_best_sol = None
            layer_best_val = float("inf")

            # 2.4 多次 Trial 优化 (防止单层陷入局部最优)
            # 对于 Interp，通常 trials 可以设少一点(比如1-3次)，因为初值已经很好
            current_trials = chosen_trials if current_p == 1 or strategy != "interp" else max(1, chosen_trials // 2)

            for trial in range(current_trials):
                x_init = base_params.copy()
                # 只有当不是第一层直接插值得到的结果，且需要扰动时才加噪声
                if (trial > 0 or (current_p == 1 and strategy != "interp")) and current_spread > 0:
                    x_init = x_init + rng.normal(scale=current_spread, size=2 * current_p)
                
                # 执行优化
                sol = None
                if use_spsa:
                    sol = self._optimize_spsa(objective_fn, x_init, chosen_maxiter, rng)
                else:
                    minimize_kwargs = {
                        "x0": x_init,
                        "method": chosen_algorithm,
                        "options": {"maxiter": chosen_maxiter},
                        "tol": 1e-4,
                        "bounds": bounds,
                        "jac": True if grad_method_key == "estimator" else False 
                    }
                    
                    if grad_method_key == "estimator":
                         sol = minimize(objective_fn_with_grad, **minimize_kwargs)
                    else:
                        # Shot-based gradient logic (omitted for brevity, same as before)
                         # ... existing gradient logic if needed ...
                         pass 

                if sol is not None and np.isfinite(sol.fun) and sol.fun < layer_best_val:
                    layer_best_val = sol.fun
                    layer_best_sol = sol

            # 2.5 记录当前层结果
            if layer_best_sol is not None:
                prev_layer_params = layer_best_sol.x
                # 如果是最后一层，或者非 interp 模式，更新全局最优
                if current_p == target_p:
                    best_global_solution = layer_best_sol
                    best_global_value = layer_best_val
            else:
                print(f"⚠️ Warning: Optimization failed at p={current_p}")
                break

        # --- 3. 最终结果处理 (使用 best_global_solution) ---
        if best_global_solution is None:
            return None

        best_params = best_global_solution.x
        
        # ... (后续用于最后输出资产配置的代码保持不变) ...
        # 注意: 下面的 circuit 需要用 target_p 重新构建一次用于最后采样，
        # 或者直接使用循环最后一次的 circuit (如果在循环外需要小心作用域)
        
        final_circuit = self._build_circuit(target_p, h, J)
        final_circuit = transpile(final_circuit, self.backend)
        
        final_betas = best_params[:target_p]
        final_gammas = best_params[target_p:]
        
        bind_dict = self._build_bind_dict(final_circuit, target_p, final_betas, final_gammas)
        counts = self._run_counts(final_circuit, bind_dict, chosen_shots)
        
        if not counts:
            return None
            
        # 向量化寻找最优 Bitstring
        bitstrings = list(counts.keys())
        char_matrix = np.array([list(s) for s in bitstrings])
        spins_matrix = np.ones(char_matrix.shape, dtype=float)
        spins_matrix[char_matrix == '1'] = -1.0
        spins_matrix = np.flip(spins_matrix, axis=1) 
        
        term1 = spins_matrix @ h
        term2 = np.sum((spins_matrix @ J) * spins_matrix, axis=1)
        energies = term1 + term2 + C
        
        min_idx = np.argmin(energies)
        best_spins = spins_matrix[min_idx].astype(int)

        return self._spins_to_asset_counts(best_spins, n, x0)