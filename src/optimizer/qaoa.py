from typing import Callable, Optional, Sequence, Any, Dict

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize

from optimizer.base import BaseOptimizer
from optimizer.utils.qubo_utils import qubo_factor as qubo_factor_optimized
from optimizer.utils.qubo_utils import get_ising_coeffs as get_ising_coeffs_optimized
from optimizer.utils.noise_utils import build_aer_simulator


class QAOAOptimizer(BaseOptimizer):
    def __init__(
        self,
        risk_aversion: float,
        lam: float,
        alpha: float,
        bits_per_asset: int,
        bits_slack: int,
        p: int = 1,
        shots: int = 1000,
        n_trials: int = 1,
        maxiter: int = 100,
        grad_delta: float = 0.01,
        init_spread: float = 0.0,
        seed: Optional[int] = None,
        use_gradient: bool = True,
        noise_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(risk_aversion, lam)
        self.alpha = alpha
        self.bits_per_asset = bits_per_asset
        self.bits_slack = bits_slack
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
        self.num_spins = 0

    @classmethod
    def init(cls, cfg: Dict[str, Any], risk_aversion: float, lam: float) -> "QAOAOptimizer":
        return cls(
            risk_aversion=risk_aversion,
            lam=lam,
            alpha=cfg["alpha"],
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
        )

    def qubo_factor(
        self,
        n: int,
        mu: np.ndarray,
        sigma: np.ndarray,
        prices: np.ndarray,
        n_spins: int,
        budget: float,
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

    def get_ising_coeffs(self, Q: np.ndarray, L: np.ndarray, constant: float):
        return get_ising_coeffs_optimized(Q, L, constant)

    @property
    def optimizer(self) -> Callable:
        return self.optimize
    
    def __call__(self, mu: np.ndarray[tuple[Any, ...], np.dtype[Any]], prices: np.ndarray[tuple[Any, ...], np.dtype[Any]], sigma: np.ndarray[tuple[Any, ...], np.dtype[Any]], budget: float, **args) -> Optional[np.ndarray]:
        return self.optimize(mu, prices, sigma, budget, **args)

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

    def _compute_expectation(
        self,
        counts,
        h: np.ndarray,
        J: np.ndarray,
    ) -> float:
        if not counts:
            return float("inf")
        bitstrings = list(counts.keys())
        counts_arr = np.array(list(counts.values()), dtype=float)
        total_shots = counts_arr.sum()
        if total_shots <= 0:
            return float("inf")

        spins = np.zeros((len(bitstrings), self.num_spins))
        for k, bitstring in enumerate(bitstrings):
            spins[k] = self._bitstring_to_spins(bitstring)

        term1 = spins @ h
        term2 = np.sum((spins @ J) * spins, axis=1)
        energies = term1 + term2
        return float(np.sum(energies * counts_arr) / total_shots)

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

    def _gradient(
        self,
        x0: np.ndarray,
        circ: QuantumCircuit,
        p: int,
        h: np.ndarray,
        J: np.ndarray,
        shots: int,
        delta: float,
    ) -> np.ndarray:
        num_params = len(x0)
        param_sets = []
        for i in range(num_params):
            x_plus = x0.copy()
            x_plus[i] += delta
            param_sets.append(x_plus)

            x_minus = x0.copy()
            x_minus[i] -= delta
            param_sets.append(x_minus)

        param_map = {param.name: param for param in circ.parameters}
        binds = []
        for params in param_sets:
            betas = params[:p]
            gammas = params[p:]
            bind_dict = {}
            for i in range(p):
                bind_dict[param_map[f"betas[{i}]"]] = [float(betas[i])]
                bind_dict[param_map[f"gammas[{i}]"]] = [float(gammas[i])]
            binds.append(bind_dict)
        circuits_to_run = [circ] * len(binds)
        job = self.backend.run(circuits_to_run, shots=shots, parameter_binds=binds)
        counts_list = job.result().get_counts()
        if not isinstance(counts_list, list):
            counts_list = [counts_list]

        gradients = np.zeros(num_params)
        for i in range(num_params):
            counts_plus = counts_list[2 * i]
            counts_minus = counts_list[2 * i + 1]

            e_plus = self._compute_expectation(counts_plus, h, J)
            e_minus = self._compute_expectation(counts_minus, h, J)
            gradients[i] = (e_plus - e_minus) / (2 * delta)

        return gradients

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

    def optimize(
        self,
        mu: np.ndarray,
        prices: np.ndarray,
        sigma: np.ndarray,
        budget: float,
        p: Optional[int] = None,
        shots: Optional[int] = None,
        n_trials: Optional[int] = None,
        maxiter: Optional[int] = None,
        initial_betas: Optional[Sequence[float]] = None,
        initial_gammas: Optional[Sequence[float]] = None,
        init_spread: Optional[float] = None,
        seed: Optional[int] = None,
        use_gradient: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        n = len(mu)
        self.num_spins = n * self.bits_per_asset + self.bits_slack

        Q, L, constant = self.qubo_factor(
            n=n,
            mu=mu,
            sigma=sigma,
            prices=prices,
            n_spins=self.num_spins,
            budget=budget,
        )
        h, J, C = self.get_ising_coeffs(Q, L, constant)

        chosen_p = p if p is not None else self.p
        chosen_shots = shots if shots is not None else self.shots
        chosen_trials = n_trials if n_trials is not None else self.n_trials
        chosen_maxiter = maxiter if maxiter is not None else self.maxiter
        chosen_spread = init_spread if init_spread is not None else self.init_spread
        chosen_seed = seed if seed is not None else self.seed
        chosen_grad = use_gradient if use_gradient is not None else self.use_gradient

        circuit = self._build_circuit(chosen_p, h, J)
        circuit = transpile(circuit, self.backend)

        base_params = self._initial_params(
            chosen_p,
            initial_betas,
            initial_gammas,
        )

        rng = np.random.default_rng(chosen_seed)
        best_solution = None
        best_value = float("inf")

        def jac_fn(x, *args):
            return self._gradient(x, *args, delta=self.grad_delta)

        for trial in range(chosen_trials):
            x0 = base_params.copy()
            if trial > 0 and chosen_spread > 0:
                x0 = x0 + rng.normal(scale=chosen_spread, size=2 * chosen_p)

            jac = None
            if chosen_grad:
                jac = jac_fn
            sol = minimize(
                self._objective,
                x0=x0,
                args=(circuit, chosen_p, h, J, chosen_shots),
                method="BFGS",
                jac=jac,
                options={"maxiter": chosen_maxiter},
            )
            if np.isfinite(sol.fun) and sol.fun < best_value:
                best_value = sol.fun
                best_solution = sol

        if best_solution is None:
            return None

        best_params = best_solution.x
        betas = best_params[:chosen_p]
        gammas = best_params[chosen_p:]
        bind_dict = self._build_bind_dict(circuit, chosen_p, betas, gammas)
        counts = self._run_counts(circuit, bind_dict, chosen_shots)
        if not counts:
            return None

        min_energy = float("inf")
        best_spins = None
        for bitstring in counts:
            spins = self._bitstring_to_spins(bitstring)
            energy = spins @ J @ spins + h @ spins + C
            if energy < min_energy:
                min_energy = energy
                best_spins = spins

        if best_spins is None:
            return None

        return self._spins_to_asset_counts(best_spins, n)
