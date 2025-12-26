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

def IsingCoeffsWithSlack(n, n_per, Ks, sigma, mu, prices, B, lam, alpha):
    N = n * n_per
    price_rate = prices / B
    C = np.zeros((n,N))     # conversion matrix
    for i in range(n):
        for p in range(n_per):
            C[i, i*n_per + p] = 2**p
    D = np.zeros((1,Ks))   # slack variable conversion
    for k in range(Ks):
        D[0,k] = 2**k

    # J_x = lambda * C^T * sigma * C + alpha C^T * p' * p'^T * C
    J_x = lam * C.T @ sigma @ C + alpha * np.outer(C.T @ price_rate, C.T @ price_rate)
    # h_x = - (C^T * mu + 2 alpha B C^T * p')
    h_x = - (C.T @ mu + 2 * alpha * (C.T @ price_rate))

    # J_s = alpha * D.T * D
    J_s = alpha * D.T @ D / (B**2)
    # h_s = -2 * alpha * B * D.T
    h_s = (-2 * alpha * D.T / B).flatten()
    # J_xs = alpha * (C.T * p') * (D)
    J_xs = alpha * np.outer(C.T @ price_rate, D[0,:]) / B

    # combine
    J_full = np.zeros((N+Ks, N+Ks))
    J_full[0:N, 0:N] = J_x
    J_full[N:N+Ks, N:N+Ks] = J_s
    J_full[0:N, N:N+Ks] = J_xs
    J_full[N:N+Ks, 0:N] = J_xs.T
    h_full = np.zeros(N+Ks)
    h_full[0:N] = h_x
    h_full[N:N+Ks] = h_s

    # x -> (1+s)/2
    J = 0.25 * J_full
    h = 0.5 * h_full + 0.25 * np.sum(J_full, axis=1) + 0.25 * np.sum(J_full, axis=0)
    C = 0.25 * np.sum(J_full) + 0.5 * np.sum(h_full) + alpha

    # add diagonal terms of J to C
    C += np.sum(np.diag(J))
    J = J + J.T
    # make J upper triangular
    for i in range(N+Ks):
        for j in range(N+Ks):
            if i >= j:
                J[i,j] = 0.0
    return h, J, C

def CalculateEnergy(state, h, J, Const=0):
    energy = state @ J @ state + h @ state + Const
    return energy

def main():
    n = 5                # number of assets
    n_per = 2          # number of bits per asset
    Ks = 5               # number of slack variables

    mu = np.array([0.12, 0.10, 0.15, 0.09, 0.11])      # expected return per share
    prices = np.array([10, 12, 8, 15, 7])             # cost per share
    B = 50                                            # budget
    lam = 0.3                                        # risk aversion λ
    alpha = 10.0                                        # penalty coefficient

    # Example positive semidefinite covariance matrix
    Sigma = np.array([
        [0.04, 0.01, 0.00, 0.00, 0.01],
        [0.01, 0.05, 0.01, 0.00, 0.00],
        [0.00, 0.01, 0.06, 0.02, 0.00],
        [0.00, 0.00, 0.02, 0.07, 0.01],
        [0.01, 0.00, 0.00, 0.01, 0.03]
    ])

    # h, J, C = RandomIsingCoeffs(n * n_per)
    # h, J, C = IsingCoeffs(n, n_per, Sigma, mu, prices, B, lam, alpha)
    h, J, C = IsingCoeffsWithSlack(n, n_per, Ks, Sigma, mu, prices, B, lam, alpha)
    # print(h)
    # print(J)
    # print(C)
    N = n * n_per
    R = min(15, N)        # low-rank approximation rank

    print("=== Variational MPS ===")
    _, state = VariationalMPS(J, h, R=R, Dp=2, Ds=20, opt=1, seed=1)
    energy = CalculateEnergy(state, h, J, C)
    print("Calculated energy from Variational MPS state:", energy)

    # print("\n=== Exact Diagonalization ===")
    # _, state = ExactDiagonalization(J, h)
    # energy = CalculateEnergy(state, h, J, C)
    # print("Calculated energy from Exact Diagonalization state:", energy)

    print("\n=== Tensor Network MPO ===")
    state2 = tenpy_dmrg(J, h)
    energy2 = CalculateEnergy(state2, h, J, C)
    print("Calculated energy from Tensor Network MPO state:", energy2)

    # print("\n=== Tensor Network VUMPS ===")
    # energy3, state3 = tenpy_vumps(J, h)
    # # print("Calculated energy from Tensor Network VUMPS state:", energy3)

    # state_gt = [-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,-1]  # hypothetical ground truth state
    # energy_gt = CalculateEnergy(state_gt, h, J, C)
    # print("\nGround truth state energy:", energy_gt)
    # x = np.array([2,0,2,0,2])

    print("\n======")
    x = [0]*n
    for i in range(n):
        for p in range(n_per):
            if state2[i*n_per + p] == 1:
                x[i] += 2**p
    x = np.array(x)
    print("Selected portfolio:", x, "investment:", x @ prices)
    # slack = 0
    # for k in range(Ks):
    #     if state[N + k] == 1:
    #         slack += 2**k
    # print("Selected portfolio:", x, "slack:", slack)
    E = lam * x @ Sigma @ x - mu @ x
    print("Ground truth portfolio energy:", E)

    x_gt = np.array([2,0,2,0,2])
    E_gt = lam * x_gt @ Sigma @ x_gt - mu @ x_gt
    print("Ground truth portfolio energy:", E_gt)

if __name__ == "__main__":
    main()