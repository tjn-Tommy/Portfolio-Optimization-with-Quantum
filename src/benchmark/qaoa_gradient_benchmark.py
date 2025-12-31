import time
import numpy as np
import sys
import os
from tqdm import tqdm
# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimizer.qaoa import QAOAOptimizer
from qiskit import transpile

def benchmark_qaoa_gradients():
    print("Initializing QAOA Optimizer for Benchmark...")
    print("(Note: Benchmarking src/optimizer/qaoa.py as quantum_annealing.py does not implement gradients)")
    
    # Config similar to config.yaml but simplified for benchmark
    lam = 0.5
    beta = 0.0
    alpha = 100.0
    bits_per_asset = 3
    bits_slack = 8
    p = 5 
    shots = 1000
    
    # Create Optimizer
    optimizer = QAOAOptimizer(
        lam=lam,
        alpha=alpha,
        beta=beta,
        bits_per_asset=bits_per_asset,
        bits_slack=bits_slack,
        p=p,
        shots=shots,
        grad_delta=0.01,
        use_gpu=True  # Attempt to use GPU
    )

    # Dummy Problem
    n_assets = 5 # Small problem
    print(f"Problem size: {n_assets} assets")
    
    mu = np.random.rand(n_assets)
    sigma = np.random.rand(n_assets, n_assets)
    sigma = (sigma + sigma.T) / 2 # Symmetric
    prices = np.random.rand(n_assets) * 100
    budget = 1000
    x0 = None

    # 1. Prepare Ising Hamiltonian (h, J)
    print("Preparing Hamiltonian...")
    n = len(mu)
    optimizer.num_spins, _, _ = optimizer.compute_num_spins(n, x0)
    Q, L, constant = optimizer.qubo_factor(n, mu, sigma, prices, optimizer.num_spins, budget, x0)
    h, J, C = optimizer.get_ising_coeffs(Q, L, constant)
    
    print(f"Number of Spins: {optimizer.num_spins}")

    # 2. Build Circuit
    print(f"Building and Transpiling Circuit (p={p})...")
    # Note: QAOAOptimizer uses measure=True by default for the circuit passed to optimization
    circuit = optimizer._build_circuit(p, h, J, measure=True) 
    circuit_no_measure = optimizer._build_circuit(p, h, J, measure=False)
    circuit = transpile(circuit, optimizer.backend)
    circuit_no_measure = transpile(circuit_no_measure, optimizer.backend)
    
    # Initial Params
    x_init = np.random.rand(2 * p)

    # --- Benchmark Estimator Gradient ---
    print("\n--- Benchmarking Estimator Gradient ---")
    # Warmup
    try:
        _ = optimizer._gradient_estimator(x_init, circuit_no_measure, p, h, J, shots, method="finite_diff")
    except Exception as e:
        print(f"Estimator warmup failed: {e}")
    print("Warmed up estimator.")
    start_time = time.time()
    n_loops = 2
    for _ in tqdm(range(n_loops), desc="Estimating Gradient"):
        grad_est = optimizer._gradient_estimator(x_init, circuit_no_measure, p, h, J, shots, method="finite_diff")
    end_time = time.time()
    est_duration = (end_time - start_time) / n_loops
    print(f"Estimator Gradient Average Time: {est_duration:.6f} s")
    print(f"Gradient Sample (first 5): {grad_est}")

    # --- Benchmark Finite Difference Gradient ---
    print("\n--- Benchmarking Finite Difference Gradient (Trivial) ---")
    start_time = time.time()
    for _ in tqdm(range(n_loops), desc="Finite Difference Gradient"):
        grad_fd = optimizer._gradient(x_init, circuit, p, h, J, shots, method="finite_diff")
    end_time = time.time()
    fd_duration = (end_time - start_time) / n_loops
    print(f"Finite Difference Gradient Average Time: {fd_duration:.6f} s")
    print(f"Gradient Sample (first 5): {grad_fd}")

    # --- Comparison ---
    print("\n--- Summary ---")
    if est_duration > 0:
        print(f"Speedup (Finite Diff / Estimator): {fd_duration / est_duration:.2f}x")
    else:
        print("Estimator time was 0, infinite speedup?")

if __name__ == "__main__":
    benchmark_qaoa_gradients()
