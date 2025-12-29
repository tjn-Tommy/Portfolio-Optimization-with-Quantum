from typing import Tuple

import numpy as np
# from qubo_utils import qubo_factor as original_qubo_factor

def compute_num_spins(
    n_assets: int,
    bits_per_asset: int,
    bits_slack: int,
    transact_opt: str = "ignore",
    x0: np.ndarray = None
) -> int:
    '''
    compute the total number of spins needed for QUBO formulation
    '''
    if transact_opt == "ignore" or transact_opt == "quadratic":
        n_spins = n_assets * bits_per_asset + bits_slack
        bits_plus = None
        bits_minus = None
    elif transact_opt == "exact":
        if x0 is None:
            x0 = np.zeros(n_assets, dtype=int)
        max_x = 2 ** bits_per_asset - 1
        x_left = max_x - x0
        bits_plus = np.floor(np.log2(x_left + 1)).astype(int)
        bits_minus = np.floor(np.log2(x0 + 1)).astype(int)
        n_spins = np.sum(bits_plus + bits_minus) + bits_slack
    else:
        raise ValueError(f"Unsupported transact_opt: {transact_opt}")
    return n_spins, bits_plus, bits_minus

def compute_encode_matrix(
    n_spins: int,
    n_assets: int,
    bits_per_asset: int,
    bits_slack: int,
    transact_opt: str = "ignore",
    x0: np.ndarray = None
) -> int:
    '''
    convert from integer representation to binary representation
    '''
    if transact_opt == "ignore" or transact_opt == "quadratic":
        num_rows = n_assets + (1 if bits_slack > 0 else 0)
        encode_matrix = np.zeros((num_rows, n_spins))
        for i in range(n_assets):
            for b in range(bits_per_asset):
                encode_matrix[i, i * bits_per_asset + b] = 2 ** b
        if bits_slack > 0:
            for b in range(bits_slack):
                encode_matrix[-1, n_assets * bits_per_asset + b] = 2 ** b
    elif transact_opt == "exact":
        num_rows = 2 * n_assets + (1 if bits_slack > 0 else 0)
        if x0 is None:
            x0 = np.zeros(n_assets, dtype=int)
        max_x = 2 ** bits_per_asset - 1
        x_left = max_x - x0
        bits_plus = np.floor(np.log2(x_left + 1)).astype(int)
        bits_minus = np.floor(np.log2(x0 + 1)).astype(int)

        idx_row = 0
        idx_col = 0
        encode_matrix = np.zeros((n_assets * 2 + (1 if bits_slack > 0 else 0), n_spins))
        # x_plus
        for i in range(n_assets):
            for b in range(bits_plus[i]):
                encode_matrix[idx_row, idx_col] = 2 ** b
                idx_col += 1
            idx_row += 1
        # x_minus
        for i in range(n_assets):
            for b in range(bits_minus[i]):
                encode_matrix[idx_row, idx_col] = 2 ** b
                idx_col += 1
            idx_row += 1
        # slack variables
        if bits_slack > 0:
            for b in range(bits_slack):
                encode_matrix[idx_row, idx_col] = 2 ** b
                idx_col += 1
            idx_row += 1
        assert idx_col == n_spins
        assert idx_row == num_rows
    else:
        raise ValueError(f"Unsupported transact_opt: {transact_opt}")
    return encode_matrix

def spins_to_asset_counts(
    spins: np.ndarray,
    n_assets: int,
    bits_per_asset: int,
    transact_opt: str = "ignore",
    bits_plus: np.ndarray = None,
    bits_minus: np.ndarray = None,
    x0: np.ndarray = None
) -> np.ndarray:
    '''
    decode from binary representation to integer representation
    '''
    if transact_opt == "ignore" or transact_opt == "quadratic":
        x = np.zeros(n_assets, dtype=int)
        for i in range(n_assets):
            for b in range(bits_per_asset):
                if spins[i * bits_per_asset + b] == -1:
                    x[i] += 2**b
    elif transact_opt == "exact":
        if x0 is None:
            x0 = np.zeros(n_assets, dtype=int)
        x = x0.copy()
        idx = 0
        # x_plus
        for i in range(n_assets):
            for b in range(bits_plus[i]):
                if spins[idx] == -1:
                    x[i] += 2**b
                idx += 1
        # x_minus
        for i in range(n_assets):
            for b in range(bits_minus[i]):
                if spins[idx] == -1:
                    x[i] -= 2**b
                idx += 1
    else:
        raise ValueError(f"Unsupported transact_opt: {transact_opt}")
    return x
        

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
    Q0 = lam * sigma + alpha * np.outer(prices, prices)
    L0 = -mu - 2 * alpha * budget * prices
    constant = alpha * budget * budget

    # Handle transaction cost
    if beta == 0.0:
        transact_opt = "ignore" # ignore transaction cost
    if x0 is None:
        x0 = np.zeros(n, dtype=int)
    if transact_opt == "ignore" or transact_opt == "quadratic":
        # special treatment for quadratic approximation
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
            Q0 += lam * beta * np.diag(sigma_transact)
            L0 += beta * mu_transact
            constant += beta * const_transact

        if bits_slack == 0:
            Q = Q0
            L = L0
        else:
            Q = np.zeros((n+1, n+1))
            L = np.zeros(n+1)
            Q[:n, :n] = Q0
            Q[:n, n] = alpha * prices
            Q[n, :n] = alpha * prices
            Q[n, n] = alpha
            L[:n] = L0
            L[n] = -2 * alpha * budget
    elif transact_opt == "exact":
        if bits_slack == 0:
            Q = np.zeros((2*n, 2*n))
            L = np.zeros(2*n)
        else:
            Q = np.zeros((2*n+1, 2*n+1))
            L = np.zeros(2*n+1)
        Q[:n, :n] = Q0
        Q[n:2*n, n:2*n] = Q0
        Q[:n, n:2*n] = -Q0
        Q[n:2*n, :n] = -Q0
        L[:n] = L0 + 2 * Q0.T @ x0 + beta * np.ones(n)
        L[n:2*n] = -L0 - 2 * Q0.T @ x0 + beta * np.ones(n)
        constant = lam * x0.T @ sigma @ x0 - mu.T @ x0 + alpha * (x0.T @ prices - budget) ** 2
        if bits_slack > 0:
            Q[:n, 2*n] = alpha * prices
            Q[n:2*n, 2*n] = -alpha * prices
            Q[2*n, :n] = alpha * prices
            Q[2*n, n:2*n] = -alpha * prices
            Q[2*n, 2*n] = alpha
            L[2*n] = 2 * alpha * (x0.T @ prices - budget)
    else:
        raise ValueError(f"Unsupported transact_opt: {transact_opt}")
    
    C = compute_encode_matrix(n_spins, n, bits_per_asset, bits_slack, transact_opt, x0)
    Q = C.T @ Q @ C
    L = C.T @ L

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

# if __name__ == "__main__":
#     # Example usage
#     n = 5
#     mu = np.array([0.12, 0.10, 0.15, 0.09, 0.11]) 
#     sigma = np.array([
#         [0.04, 0.01, 0.00, 0.00, 0.01],
#         [0.01, 0.05, 0.01, 0.00, 0.00],
#         [0.00, 0.01, 0.06, 0.02, 0.00],
#         [0.00, 0.00, 0.02, 0.07, 0.01],
#         [0.01, 0.00, 0.00, 0.01, 0.03]
#     ])
#     prices = np.array([10, 12, 8, 15, 7])
#     x0 = np.array([1, 0, 2, 0, 3])
#     n_spins, _, _ = compute_num_spins(n, bits_per_asset=2, bits_slack=5, transact_opt="exact", x0=x0)
#     print("Number of spins:", n_spins)
#     Q1, L1, constant1 = qubo_factor(
#         n=n,
#         mu=mu,
#         sigma=sigma,
#         prices=prices,
#         n_spins=n_spins,
#         budget=150,
#         bits_per_asset=2,
#         bits_slack=5,
#         lam=0.1,
#         alpha=5.0,
#         beta=0.05,
#         transact_opt="exact",
#         x0=x0
#     )
#     print("QUBO Q:\n", Q1)
#     print("QUBO L:\n", L1)
#     print("QUBO constant:\n", constant1)