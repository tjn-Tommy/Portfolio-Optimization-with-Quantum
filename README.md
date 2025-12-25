# 顾全好帅！
# README for Portfolio Optimization

## Code Structure

The project is organized into the following key modules:

- **`src/`**: The core source code directory.
  - **`main.py`**: The entry point for the application. It loads the configuration, initializes the benchmark, and runs the selected optimizers.
  - **`config.yaml`**: Configuration file for defining experiment parameters, datasets, and solver settings.
  - **`benchmark/`**: Contains the benchmarking framework.
    - `benchmark.py`: Defines the `Benchmark` class which manages the simulation loop, budget tracking, and result plotting.
  - **`data/`**: Handles data ingestion and processing.
    - `dataset.py`: Implements `StockDataset` for managing historical stock data.
  - **`optimizer/`**: Implementations of various portfolio optimization algorithms.
    - `base.py`: Abstract base class for all optimizers.
    - `quantum_annealing.py`: Quantum Annealing solver using Qiskit.
    - `qaoa.py`: QAOA solver.
    - `tensor_network.py`: Tensor Network-based solver.
    - `scip_solver.py`: Classical solver using SCIP.
  - **`tensor_network/`**: Specialized tensor network libraries and utilities (e.g., variational MPS implementations).

# TODO:
[] Noise
[] real data
