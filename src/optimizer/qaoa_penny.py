from typing import Optional, Any, Dict, Tuple
import numpy as np
import pennylane as qml
import jax
import jax.numpy as jnp
import jax.lax
from catalyst import qjit, grad
import optax

from optimizer.base import BaseOptimizer

from optimizer.utils.qubo_utils import compute_num_spins as compute_num_spins_optimized
from optimizer.utils.qubo_utils import spins_to_asset_counts
from optimizer.utils.qubo_utils import qubo_factor as qubo_factor_optimized
from optimizer.utils.qubo_utils import get_ising_coeffs as get_ising_coeffs_optimized
from optimizer.utils.qubo_utils import normalize_ising_coeffs


class QAOAOptimizerJAX(BaseOptimizer):
    """
    Key idea:
      - Keep circuit structure constant for fixed (num_spins, p)
      - Pass (h_vec, J_upper, C) as DATA into the QNode and qjit function
      - QNode returns ONE scalar expval using qml.dot(coeffs, ops) to avoid Catalyst adjoint crash
      - Cache compiled kernels by (num_spins, p, n_trials)

    Optimizer is FIXED in-code:
      - Optax Adam (scale_by_adam) + global-norm clip
      - Learning-rate schedule: warmup -> cosine decay
      - Parameter wrapping to keep angles in valid ranges after each update
    """

    # ----------------------------
    # Fixed optimizer hyperparams (NOT configurable via cfg/init)
    # ----------------------------
    _ADAM_B1 = 0.9
    _ADAM_B2 = 0.999
    _ADAM_EPS = 1e-8
    _CLIP_NORM = 1.0

    # LR schedule (works well for analytic expval QAOA; stable defaults)
    _LR_PEAK = 0.05     # peak LR after warmup
    _LR_END = 0.005     # final LR at end of decay (10x decay)
    _WARMUP_MAX = 10    # warmup steps cap; also uses ~num_steps//5

    _DTYPE = jnp.float64

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
        init_spread: float = 0.1,
        seed: Optional[int] = None,
        use_history: bool = False,
        shift: float = 0.05,
        learning_rate: float = 0.05,  # kept for backward-compat; ignored intentionally
        **kwargs,
    ):
        super().__init__(lam, beta)
        self.alpha = alpha
        self.bits_per_asset = bits_per_asset
        self.bits_slack = bits_slack
        self.transact_opt = transact_opt
        self.p = int(p)
        self.shots = int(shots)
        self.n_trials = int(n_trials)
        self.maxiter = int(maxiter)

        # kept but not used (compat)
        self.init_spread = float(init_spread)
        self.seed = seed
        self.use_history = use_history
        self.history = None
        self.shift = float(shift)

        # backend
        self.device_name = "lightning.gpu"
        print(f"✅ QAOAOptimizerJAX initialized. Target Backend: {self.device_name} (Catalyst Enabled)")

        # cache: key -> (pairs, qaoa_cost_qnode, run_batched_optimization_qjit, sample_qnode)
        self._kernel_cache: Dict[Tuple[int, int, int], Any] = {}

    @classmethod
    def init(cls, cfg: Dict[str, Any], lam: float, beta: Optional[float]) -> "QAOAOptimizerJAX":
        # NOTE: optimizer choice + hyperparams are fixed in code, so we do NOT read LR-related options from cfg.
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
            seed=cfg.get("seed"),
            use_history=cfg.get("use_history", False),
        )

    # ----------------- classical preprocessing (NumPy) -----------------

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

    def compute_num_spins(self, n_assets: int, x0: np.ndarray = None):
        return compute_num_spins_optimized(
            n_assets=n_assets,
            bits_per_asset=self.bits_per_asset,
            bits_slack=self.bits_slack,
            transact_opt=self.transact_opt,
            x0=x0,
        )

    def _spins_to_asset_counts(self, spins: np.ndarray, n_assets: int, x0: np.ndarray = None):
        return spins_to_asset_counts(
            spins=spins,
            n_assets=n_assets,
            bits_per_asset=self.bits_per_asset,
            bits_plus=self.bits_plus,
            bits_minus=self.bits_minus,
            transact_opt=self.transact_opt,
            x0=x0,
        )

    # ----------------- cached quantum + qjit kernels -----------------

    def _get_kernels(self, num_spins: int, p: int, n_trials: int):
        key = (num_spins, p, n_trials)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        # fixed pair ordering for upper triangle (i<j)
        pairs = [(i, j) for i in range(num_spins) for j in range(i + 1, num_spins)]
        num_pairs = len(pairs)

        # fixed operator list for measurement: [Z_i] + [Z_i Z_j] in same order as coeffs concat
        ops_z = [qml.PauliZ(i) for i in range(num_spins)]
        ops_zz = [qml.PauliZ(i) @ qml.PauliZ(j) for (i, j) in pairs]
        ops_cost = ops_z + ops_zz  # length = num_spins + num_pairs

        dev = qml.device(self.device_name, wires=num_spins)

        def apply_cost_layer(gamma, h_vec, J_upper):
            for w in range(num_spins):
                qml.RZ(2.0 * gamma * h_vec[w], wires=w)
            for k in range(num_pairs):
                i, j = pairs[k]
                qml.MultiRZ(2.0 * gamma * J_upper[k], wires=[i, j])

        def apply_mixer_layer(beta):
            for w in range(num_spins):
                qml.RX(2.0 * beta, wires=w)

        @qml.qnode(dev, interface="jax", diff_method="adjoint")
        def qaoa_cost(params, h_vec, J_upper):
            # params shape: (2, p)
            betas = params[0]
            gammas = params[1]

            for w in range(num_spins):
                qml.Hadamard(wires=w)

            for layer in range(p):
                apply_cost_layer(gammas[layer], h_vec, J_upper)
                apply_mixer_layer(betas[layer])

            # return ONE scalar
            coeffs = jnp.concatenate([h_vec, J_upper])  # (num_spins + num_pairs,)
            H = qml.dot(coeffs, ops_cost)
            return qml.expval(H)

        def wrap_angles(batch_params):
            # batch_params: (n_trials, 2, p)  OR (2, p)
            # canonical ranges: betas in [0, pi), gammas in [0, 2pi)
            if batch_params.ndim == 3:
                betas = jnp.mod(batch_params[:, 0, :], jnp.pi)
                gammas = jnp.mod(batch_params[:, 1, :], 2.0 * jnp.pi)
                return jnp.stack([betas, gammas], axis=1)
            else:
                betas = jnp.mod(batch_params[0, :], jnp.pi)
                gammas = jnp.mod(batch_params[1, :], 2.0 * jnp.pi)
                return jnp.stack([betas, gammas], axis=0)

        def lr_schedule(step_i, num_steps):
            # warmup -> cosine decay
            # warmup = min(WARMUP_MAX, max(1, num_steps//5))
            ns = jnp.asarray(num_steps, dtype=self._DTYPE)
            t = jnp.asarray(step_i, dtype=self._DTYPE)

            warmup = jnp.minimum(
                jnp.asarray(self._WARMUP_MAX, dtype=self._DTYPE),
                jnp.maximum(jnp.asarray(1.0, dtype=self._DTYPE), jnp.floor(ns / 5.0)),
            )
            decay_steps = jnp.maximum(jnp.asarray(1.0, dtype=self._DTYPE), ns - warmup)

            peak = jnp.asarray(self._LR_PEAK, dtype=self._DTYPE)
            end = jnp.asarray(self._LR_END, dtype=self._DTYPE)

            # Use (t+1) in warmup so step 0 isn't exactly zero
            lr_warm = peak * (t + 1.0) / warmup

            progress = (t - warmup) / decay_steps
            progress = jnp.clip(progress, 0.0, 1.0)
            lr_cos = end + 0.5 * (peak - end) * (1.0 + jnp.cos(jnp.pi * progress))

            return jnp.where(t < warmup, lr_warm, lr_cos)

        # qjit optimizer: no vmap; fori_loop over trials
        @qjit(static_argnums=(2,))  # n_trials is static => compile once per n_trials
        def run_batched_optimization(batch_init_params, num_steps, n_trials, h_vec, J_upper, const_C):
            # batch_init_params: (n_trials, 2, p)
            # NOTE: optimizer + schedule fixed inside this function.

            def cost_one(params):
                return qaoa_cost(params, h_vec, J_upper) + const_C

            # Use MEAN over trials so gradient scale doesn't grow with n_trials.
            def mean_cost(batch_params):
                def body_fn(i, acc):
                    return acc + cost_one(batch_params[i])
                total = jax.lax.fori_loop(0, n_trials, body_fn, jnp.asarray(0.0, dtype=self._DTYPE))
                return total / jnp.asarray(n_trials, dtype=self._DTYPE)

            batch_grad_fn = grad(mean_cost, method="auto")

            # Optax: clip + Adam preconditioner (no LR baked in; we apply LR ourselves per-step)
            tx = optax.chain(
                optax.clip_by_global_norm(self._CLIP_NORM),
                optax.scale_by_adam(b1=self._ADAM_B1, b2=self._ADAM_B2, eps=self._ADAM_EPS),
            )
            params = wrap_angles(batch_init_params)
            opt_state = tx.init(params)

            def update_step(i, carry):
                params, opt_state = carry
                grads = batch_grad_fn(params)

                updates, opt_state = tx.update(grads, opt_state, params)

                lr = lr_schedule(i, num_steps)
                params = params - lr * updates  # Adam direction returned by scale_by_adam
                params = wrap_angles(params)
                return (params, opt_state)

            final_params, _ = jax.lax.fori_loop(0, num_steps, update_step, (params, opt_state))

            def scan_body(carry, i):
                return carry, cost_one(final_params[i])

            _, final_costs = jax.lax.scan(scan_body, None, jnp.arange(n_trials))
            return final_params, final_costs

        # sampling QNode (shots set on QNode, not device)
        dev_samp = qml.device(self.device_name, wires=num_spins)

        @qml.qnode(dev_samp, interface="jax")
        def sample_circuit(params, h_vec, J_upper):
            betas = params[0]
            gammas = params[1]

            for w in range(num_spins):
                qml.Hadamard(wires=w)

            for layer in range(p):
                apply_cost_layer(gammas[layer], h_vec, J_upper)
                apply_mixer_layer(betas[layer])

            return qml.sample(wires=range(num_spins))

        sample_circuit = qml.set_shots(sample_circuit, shots=self.shots)

        self._kernel_cache[key] = (pairs, qaoa_cost, run_batched_optimization, sample_circuit)
        return self._kernel_cache[key]

    # ----------------- main optimize -----------------

    def optimize(
        self,
        mu: np.ndarray,
        prices: np.ndarray,
        sigma: np.ndarray,
        budget: float,
        x0: Optional[np.ndarray] = None,
        p: Optional[int] = None,
        **kwargs,
    ) -> Optional[np.ndarray]:

        n = len(mu)
        self.num_spins, self.bits_plus, self.bits_minus = self.compute_num_spins(n, x0)

        Q, L, constant = self.qubo_factor(n, mu, sigma, prices, self.num_spins, budget, x0)
        h, J, C = self.get_ising_coeffs(Q, L, constant)
        h, J, C = normalize_ising_coeffs(h, J, C)

        chosen_p = int(p if p is not None else self.p)
        n_trials = int(self.n_trials)

        # build / reuse compiled kernels (compile only when shapes change)
        pairs, _, run_batched_optimization, sample_circuit = self._get_kernels(self.num_spins, chosen_p, n_trials)

        # pack Ising data into vectors aligned with ops/pairs order
        h_vec = np.asarray(h, dtype=np.float64)  # (num_spins,)
        J_upper = np.asarray([J[i, j] for (i, j) in pairs], dtype=np.float64)  # (num_pairs,)
        const_C = float(C)

        # init params batch (CPU)
        rng = np.random.default_rng(self.seed)
        batch_params_list = []
        for _ in range(n_trials):
            betas_init = rng.uniform(0.0, np.pi, chosen_p)
            gammas_init = rng.uniform(0.0, 2.0 * np.pi, chosen_p)

            if self.use_history and self.history:
                betas_init = self.history["betas"] + rng.normal(0.0, self.shift, chosen_p)
                gammas_init = self.history["gammas"] + rng.normal(0.0, self.shift, chosen_p)

            batch_params_list.append(np.stack([betas_init, gammas_init]))

        if not batch_params_list:
            return None

        batch_init_params = jnp.asarray(np.stack(batch_params_list), dtype=self._DTYPE)

        # move coefficients to JAX
        h_jax = jnp.asarray(h_vec, dtype=self._DTYPE)
        Jupper_jax = jnp.asarray(J_upper, dtype=self._DTYPE)
        C_jax = jnp.asarray(const_C, dtype=self._DTYPE)

        # optimize (qjit)
        all_final_params, all_costs = run_batched_optimization(
            batch_init_params,
            int(self.maxiter),
            n_trials,
            h_jax,
            Jupper_jax,
            C_jax,
        )

        # pick best trial (CPU)
        all_costs_np = np.array(all_costs)
        best_idx = int(np.argmin(all_costs_np))
        best_value = float(all_costs_np[best_idx])
        best_params = np.array(all_final_params[best_idx])

        # sampling
        best_params_jax = jnp.asarray(best_params, dtype=self._DTYPE)
        final_samples = np.array(sample_circuit(best_params_jax, h_jax, Jupper_jax))  # (shots, num_spins)

        spins_samples = 1 - 2 * final_samples  # {0,1} -> {+1,-1}

        # energy evaluate (NumPy) using same convention as your existing code
        spins_samples = spins_samples.astype(np.float64)
        term1 = spins_samples @ h_vec
        term2 = np.sum((spins_samples @ J) * spins_samples, axis=1)
        energies = term1 + term2 + const_C

        best_sample_idx = int(np.argmin(energies))
        best_spin_config = spins_samples[best_sample_idx].astype(int)

        if self.use_history:
            self.history = {
                "betas": best_params[0],
                "gammas": best_params[1],
                "objective_value": best_value,
            }

        return self._spins_to_asset_counts(best_spin_config, n, x0)
