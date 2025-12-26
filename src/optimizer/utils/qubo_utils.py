from typing import Tuple

import numpy as np


def qubo_factor(
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
) -> Tuple[np.ndarray, np.ndarray, float]:
    n_asset_spins = n * bits_per_asset
    Q = np.zeros((n_spins, n_spins))
    L = np.zeros(n_spins)
    constant = alpha * budget * budget

    bit_weights = 2 ** np.arange(bits_per_asset)
    slack_weights = 2 ** np.arange(bits_slack)

    asset_coeff = lam * sigma + alpha * np.outer(prices, prices)
    bit_outer = np.outer(bit_weights, bit_weights)
    Q[:n_asset_spins, :n_asset_spins] = np.kron(asset_coeff, bit_outer)

    linear_asset = -(mu + 2 * alpha * budget * prices)
    L[:n_asset_spins] = (linear_asset[:, None] * bit_weights).reshape(-1)

    if bits_slack > 0:
        asset_bit_prices = np.repeat(prices, bits_per_asset) * np.tile(bit_weights, n)
        cross_block = alpha * np.outer(asset_bit_prices, slack_weights)
        slack_slice = slice(n_asset_spins, n_asset_spins + bits_slack)

        Q[:n_asset_spins, slack_slice] = cross_block
        Q[slack_slice, :n_asset_spins] = cross_block.T
        Q[slack_slice, slack_slice] = alpha * np.outer(slack_weights, slack_weights)

        L[slack_slice] = -(2 * alpha * budget) * slack_weights

    return Q, L, constant


def get_ising_coeffs(
    Q: np.ndarray,
    L: np.ndarray,
    constant: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert QUBO to Ising model coefficients."""
    num_vars = Q.shape[0]
    J = np.zeros((num_vars, num_vars))

    diag = np.diag(Q)
    L += diag
    np.fill_diagonal(Q, 0.0)

    h = -0.5 * L
    C = constant + 0.5 * L.sum()

    if num_vars > 1:
        sym = Q + Q.T
        upper = np.triu(sym, 1)
        J = 0.25 * upper
        C += 0.25 * upper.sum()
        h -= 0.25 * sym.sum(axis=1)

    return h, J, C

def normalize_ising_coeffs(
    h: np.ndarray,
    J: np.ndarray,
    C: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalize Ising coefficients to prevent large values."""
    max_strength = np.max([np.max(np.abs(J)), np.max(np.abs(h))])
    
    # Prevent division by zero
    if max_strength == 0:
        scale_factor = 1.0
    else:
        # Normalize the maximum coefficient to 1.0 (or a similar scale)
        scale_factor = 1.0 / max_strength

    # Apply normalization
    J_scaled = J * scale_factor
    h_scaled = h * scale_factor
    C_scaled = C * scale_factor

    return h_scaled, J_scaled, C_scaled