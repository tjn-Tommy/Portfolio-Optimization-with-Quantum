import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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


def test_qubo_factor_matches_original():
    rng = np.random.default_rng(0)
    n = 4
    bits_per_asset = 3
    bits_slack = 2
    n_spins = n * bits_per_asset + bits_slack

    mu = rng.normal(size=n)
    sigma = rng.normal(size=(n, n))
    prices = rng.uniform(1.0, 10.0, size=n)
    lam = 0.7
    alpha = 0.3
    budget = 100.0

    Q_ref, L_ref, c_ref = original_qubo_factor(
        n=n,
        mu=mu,
        sigma=sigma,
        prices=prices,
        n_spins=n_spins,
        budget=budget,
        bits_per_asset=bits_per_asset,
        bits_slack=bits_slack,
        lam=lam,
        alpha=alpha,
    )
    Q_opt, L_opt, c_opt = optimized_qubo_factor(
        n=n,
        mu=mu,
        sigma=sigma,
        prices=prices,
        n_spins=n_spins,
        budget=budget,
        bits_per_asset=bits_per_asset,
        bits_slack=bits_slack,
        lam=lam,
        alpha=alpha,
    )

    np.testing.assert_allclose(Q_opt, Q_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(L_opt, L_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(c_opt, c_ref, rtol=0, atol=1e-12)


def test_get_ising_coeffs_matches_original():
    rng = np.random.default_rng(1)
    num_vars = 6
    Q = rng.normal(size=(num_vars, num_vars))
    L = rng.normal(size=num_vars)
    constant = rng.normal()

    Q_ref = Q.copy()
    L_ref = L.copy()
    h_ref, J_ref, C_ref = original_get_ising_coeffs(Q_ref, L_ref, constant)

    Q_opt = Q.copy()
    L_opt = L.copy()
    h_opt, J_opt, C_opt = optimized_get_ising_coeffs(Q_opt, L_opt, constant)

    np.testing.assert_allclose(h_opt, h_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(J_opt, J_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(C_opt, C_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Q_opt, Q_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(L_opt, L_ref, rtol=0, atol=1e-12)
