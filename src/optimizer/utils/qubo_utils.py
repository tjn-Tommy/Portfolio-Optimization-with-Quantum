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
    beta: float = 0.0,
    transact_opt: str = "ignore",
    x0: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Compute QUBO coefficients for portfolio optimization with slack variables.
    H = lam * x^T * sigma * x - mu^T * x + alpha (x^T * p - B)^2,
    p: price, B: budget.
    If bits_slack > 0, the last term is replaced with (x^T * p + s - B)^2, where
    x, s are represented in binary with bits_per_asset and bits_slack bits respectively.
    If beta > 0, we may add transaction fee term: H += beta * sum_i |x_i - x_i^0|,
    where x^0 is the initial asset allocation;
    transact_opt = "ignore": ignore transaction cost;
    transact_opt = "quadratic": use quadratic penalty to approximate |x_i - x_i^0|
    '''
    n_asset_spins = n * bits_per_asset
    Q = np.zeros((n_spins, n_spins))
    L = np.zeros(n_spins)
    constant = alpha * budget * budget

    # Handle transaction cost
    if beta == 0.0:
        transact_opt = "ignore" # ignore transaction cost
    if x0 is None:
        x0 = np.zeros(n, dtype=int)
    if transact_opt == "quadratic":
        max_x = 2 ** bits_per_asset - 1
        sigma_transact = np.zeros(n)
        mu_transact = np.zeros(n)
        const_transact = 0.0
        for i in range(n):
            # use quadratic function Ax^2 + Bx + C to approximate |x| in [-a,b]
            a = x0[i]
            b = max_x - x0[i]
            # result from least square fitting
            A = (30 * a**2 * b**2) / ((a + b)**5)
            B = (-(a**5-b**5) - 5*a*b*(a**3-b**3) + 26*a**2*b**2*(a-b)) / ((a + b)**5)
            C = (3*a**2*b**2 * (3*a**2 - 4*a*b + 3*b**2)) / ((a + b)**5)
            sigma_transact[i] = A
            mu_transact[i] = B - 2 * A * x0[i]
            const_transact += A * x0[i]**2 - B * x0[i] + C

        # add to Q, L, constant
        sigma += beta * np.diag(sigma_transact)
        mu -= beta * mu_transact
        constant += beta * const_transact

    # convert to binary representation
    bit_weights = 2 ** np.arange(bits_per_asset)
    slack_weights = 2 ** np.arange(bits_slack)

    asset_coeff = lam * sigma + alpha * np.outer(prices, prices)
    bit_outer = np.outer(bit_weights, bit_weights)
    Q[:n_asset_spins, :n_asset_spins] = np.kron(asset_coeff, bit_outer)

    linear_asset = -(mu + 2 * alpha * budget * prices)
    L[:n_asset_spins] = (linear_asset[:, None] * bit_weights).reshape(-1)

    # handle slack variables
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