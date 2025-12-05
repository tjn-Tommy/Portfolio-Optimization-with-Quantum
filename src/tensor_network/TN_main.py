import numpy as np
import itertools
from VariationalMPS import VariationalMPS
from ED import ExactDiagonalization
from tenpy_dmrg import tenpy_dmrg

def RandomIsingCoeffs(N):
    np.random.seed(1)
    J = np.random.randn(N, N)    # full random couplings
    J = (J + J.T) / 2            # symmetrize
    h = np.random.randn(N)       # random fields

    # Only use i<j terms
    for i in range(N):
        for j in range(N):
            if i >= j:
                J[i, j] = 0.0
    return h, J, 0.0

def IsingCoeffs(n, n_per, sigma, mu, prices, B, lam, alpha):
    N = n * n_per
    price_rate = prices / B
    C = np.zeros((n,N))     # conversion matrix
    for i in range(n):
        for p in range(n_per):
            C[i, i*n_per + p] = 2**p

    # J_x = lambda * C^T * sigma * C + alpha C^T * p' * p'^T * C
    J_x = lam * C.T @ sigma @ C + alpha * np.outer(C.T @ price_rate, C.T @ price_rate)
    # h_x = - (C^T * mu + 2 alpha B C^T * p')
    h_x = - (C.T @ mu + 2 * alpha * (C.T @ price_rate))

    # x -> (1+s)/2
    J = 0.25 * J_x
    h = 0.5 * h_x + 0.25 * np.sum(J_x, axis=1) + 0.25 * np.sum(J_x, axis=0)
    C = 0.25 * np.sum(J_x) + 0.5 * np.sum(h_x) + alpha

    # symmetrize J
    # add diagonal terms of J to C
    C += np.sum(np.diag(J))
    J = J + J.T
    # make J upper triangular
    for i in range(N):
        for j in range(N):
            if i >= j:
                J[i,j] = 0.0
    return h, J, C

def CalculateEnergy(state, h, J, C=0):
    energy = 0.0
    L = len(state)
    for i in range(L):
        energy += h[i] * state[i]
        for j in range(L):
            if J[i, j] != 0:
                energy += J[i, j] * state[i] * state[j]
    energy += C
    return energy

def main():
    n = 5                # number of assets
    n_per = 2          # number of bits per asset

    mu = np.array([0.12, 0.10, 0.15, 0.09, 0.11])      # expected return per share
    prices = np.array([10, 12, 8, 15, 7])             # cost per share
    B = 50                                            # budget
    lam = 0.3                                        # risk aversion λ
    alpha = 5.0                                        # penalty coefficient

    # Example positive semidefinite covariance matrix
    Sigma = np.array([
        [0.04, 0.01, 0.00, 0.00, 0.01],
        [0.01, 0.05, 0.01, 0.00, 0.00],
        [0.00, 0.01, 0.06, 0.02, 0.00],
        [0.00, 0.00, 0.02, 0.07, 0.01],
        [0.01, 0.00, 0.00, 0.01, 0.03]
    ])

    # h, J, C = RandomIsingCoeffs(n * n_per)
    h, J, C = IsingCoeffs(n, n_per, Sigma, mu, prices, B, lam, alpha)
    print(h.shape, J.shape, C)
    N = n * n_per
    R = min(15, N)        # low-rank approximation rank

    print("=== Variational MPS ===")
    _, state = VariationalMPS(J, h, R=R, Dp=2, Ds=10, opt=1)
    energy = CalculateEnergy(state, h, J, C)
    print("Calculated energy from Variational MPS state:", energy)

    print("\n=== Exact Diagonalization ===")
    _, state = ExactDiagonalization(J, h)
    energy = CalculateEnergy(state, h, J, C)
    print("Calculated energy from Exact Diagonalization state:", energy)

    print("\n=== Tensor Network MPO ===")
    tenpy_dmrg(J, h)

if __name__ == "__main__":
    main()