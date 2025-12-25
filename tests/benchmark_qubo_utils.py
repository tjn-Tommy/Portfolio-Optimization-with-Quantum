import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from optimizer.utils.qubo_utils import qubo_factor as optimized_qubo_factor
from optimizer.utils.qubo_utils import get_ising_coeffs as optimized_get_ising_coeffs


def original_qubo_factor(
    n: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    prices: np.ndarray,
    n_spins: int,
    budget: float,
    bits_per_asset: int,
    bits_slack: int,
    lam: float,
    alpha: float,
):
    Q = np.zeros((n_spins, n_spins))
    L = np.zeros(n_spins)
    constant = 0.0
    for i in range(n):
        for j in range(n):
            for p1 in range(bits_per_asset):
                for p2 in range(bits_per_asset):
                    idx_i = i * bits_per_asset + p1
                    idx_j = j * bits_per_asset + p2
                    coeff = (
                        lam * sigma[i, j] + alpha * prices[i] * prices[j]
                    ) * (2**p1) * (2**p2)
                    Q[idx_i, idx_j] += coeff

    for i in range(n):
        for p in range(bits_per_asset):
            idx = i * bits_per_asset + p
            coeff = -(
                mu[i] + 2 * alpha * budget * prices[i]
            ) * (2**p)
            L[idx] += coeff

    for i in range(n):
        for p1 in range(bits_per_asset):
            for p2 in range(bits_slack):
                idx1 = i * bits_per_asset + p1
                idx2 = n * bits_per_asset + p2
                coeff = alpha * prices[i] * (2**p1) * (2**p2)
                Q[idx1, idx2] += coeff
                Q[idx2, idx1] += coeff

    for p1 in range(bits_slack):
        for p2 in range(bits_slack):
            idx1 = n * bits_per_asset + p1
            idx2 = n * bits_per_asset + p2
            coeff = alpha * (2**p1) * (2**p2)
            Q[idx1, idx2] += coeff

    for p in range(bits_slack):
        idx = n * bits_per_asset + p
        coeff = -alpha * (2 * budget) * (2**p)
        L[idx] += coeff

    constant += alpha * budget * budget

    return Q, L, constant


def original_get_ising_coeffs(Q: np.ndarray, L: np.ndarray, constant: float):
    num_vars = Q.shape[0]
    J = np.zeros((num_vars, num_vars))
    h = np.zeros(num_vars)
    C = constant

    for i in range(num_vars):
        L[i] += Q[i, i]
        Q[i, i] = 0.0

    for i in range(num_vars):
        h[i] -= L[i] / 2.0
        C += L[i] / 2.0

    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            val = Q[i, j] + Q[j, i]

            term = val / 4.0
            C += term
            h[i] -= term
            h[j] -= term
            J[i, j] += term

    return h, J, C


def time_fn(fn, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def format_ms(seconds: float) -> str:
    return f"{seconds * 1e3:.3f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick benchmark for QUBO/Ising helpers."
    )
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--bits-per-asset", type=int, default=3)
    parser.add_argument("--bits-slack", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_spins = args.n * args.bits_per_asset + args.bits_slack

    mu = rng.normal(size=args.n)
    sigma = rng.normal(size=(args.n, args.n))
    prices = rng.uniform(1.0, 10.0, size=args.n)
    lam = 0.7
    alpha = 0.3
    budget = 100.0

    def run_original_qubo():
        return original_qubo_factor(
            n=args.n,
            mu=mu,
            sigma=sigma,
            prices=prices,
            n_spins=n_spins,
            budget=budget,
            bits_per_asset=args.bits_per_asset,
            bits_slack=args.bits_slack,
            lam=lam,
            alpha=alpha,
        )

    def run_optimized_qubo():
        return optimized_qubo_factor(
            n=args.n,
            mu=mu,
            sigma=sigma,
            prices=prices,
            n_spins=n_spins,
            budget=budget,
            bits_per_asset=args.bits_per_asset,
            bits_slack=args.bits_slack,
            lam=lam,
            alpha=alpha,
        )

    t_qubo_old = time_fn(run_original_qubo, args.repeats, args.warmup)
    t_qubo_new = time_fn(run_optimized_qubo, args.repeats, args.warmup)

    rng_ising = np.random.default_rng(args.seed + 1)
    Q_base = rng_ising.normal(size=(n_spins, n_spins))
    L_base = rng_ising.normal(size=n_spins)
    constant = rng_ising.normal()

    def run_original_ising():
        Q = Q_base.copy()
        L = L_base.copy()
        return original_get_ising_coeffs(Q, L, constant)

    def run_optimized_ising():
        Q = Q_base.copy()
        L = L_base.copy()
        return optimized_get_ising_coeffs(Q, L, constant)

    t_ising_old = time_fn(run_original_ising, args.repeats, args.warmup)
    t_ising_new = time_fn(run_optimized_ising, args.repeats, args.warmup)

    print("QUBO factor benchmark")
    print(f"  original : {format_ms(t_qubo_old)}")
    print(f"  optimized: {format_ms(t_qubo_new)}")
    print(f"  speedup  : {t_qubo_old / t_qubo_new:.2f}x")
    print()
    print("Ising coeffs benchmark")
    print(f"  original : {format_ms(t_ising_old)}")
    print(f"  optimized: {format_ms(t_ising_new)}")
    print(f"  speedup  : {t_ising_old / t_ising_new:.2f}x")


if __name__ == "__main__":
    main()
