import numpy as np
import itertools
from pyscipopt import Model, quicksum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt
import itertools
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator

import os
import pandas as pd

import os
import pandas as pd

class StockDataset:
    def __init__(self, data_dir="data", stock_list=None):
        """
        Initialize StockDataset.
        
        Args:
            data_dir (str): Directory path containing the data files
            stock_list (list, optional): List of stock symbols to include. 
                                         If None, uses all available stocks.
        """
        self.data_dir = data_dir
        self.stock_list = stock_list
        self.all_close_prices = None
        self.all_open_prices = None 
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.simulation_date = None

        self._load_data()
        self._filter_stocks()

    def _load_data(self):
        print(f"Loading data from directory: {self.data_dir}...")

        # all_close_prices.csv
        all_close_path = os.path.join(self.data_dir, "all_close_prices.csv")
        try:
            self.all_close_prices = pd.read_csv(all_close_path, index_col="Date", parse_dates=True)
            print(f"Loaded {all_close_path}")
        except FileNotFoundError:
            print(f"Warning: {all_close_path} not found. Please ensure the file exists.")
        except Exception as e:
            print(f"Error loading {all_close_path}: {e}")
        
        # all_open_prices.csv
        all_open_path = os.path.join(self.data_dir, "all_open_prices.csv")
        try:
            self.all_open_prices = pd.read_csv(all_open_path, index_col="Date", parse_dates=True)
            print(f"Loaded {all_open_path}")
        except FileNotFoundError:
            print(f"Warning: {all_open_path} not found. Please ensure the file exists.")
        except Exception as e:
            print(f"Error loading {all_open_path}: {e}")

        # returns.csv
        returns_path = os.path.join(self.data_dir, "returns.csv")
        try:
            self.returns = pd.read_csv(returns_path, index_col="Date", parse_dates=True)
            print(f"Loaded {returns_path}")
        except FileNotFoundError:
            print(f"Warning: {returns_path} not found. Please ensure the file exists.")
        except Exception as e:
            print(f"Error loading {returns_path}: {e}")

    def _filter_stocks(self):
        """Filter data to include only specified stocks if stock_list is provided."""
        if self.stock_list is None:
            print("No stock filter applied. Using all available stocks.")
            return
        
        if not isinstance(self.stock_list, (list, tuple)):
            print(f"Warning: stock_list should be a list or tuple. Got {type(self.stock_list)}. Using all stocks.")
            return
        
        if len(self.stock_list) == 0:
            print("Warning: stock_list is empty. No stocks will be loaded.")
            return
        
        print(f"Filtering data to include only specified stocks: {self.stock_list}")
        
        # Filter close prices
        if self.all_close_prices is not None:
            existing_close = [col for col in self.stock_list if col in self.all_close_prices.columns]
            if existing_close:
                self.all_close_prices = self.all_close_prices[existing_close]
                print(f"  - Filtered all_close_prices to {len(existing_close)} stocks")
            else:
                print("  - Warning: No specified stocks found in all_close_prices")
        
        # Filter open prices
        if self.all_open_prices is not None:
            existing_open = [col for col in self.stock_list if col in self.all_open_prices.columns]
            if existing_open:
                self.all_open_prices = self.all_open_prices[existing_open]
                print(f"  - Filtered all_open_prices to {len(existing_open)} stocks")
            else:
                print("  - Warning: No specified stocks found in all_open_prices")
        
        # Filter returns
        if self.returns is not None:
            existing_returns = [col for col in self.stock_list if col in self.returns.columns]
            if existing_returns:
                self.returns = self.returns[existing_returns]
                print(f"  - Filtered returns to {len(existing_returns)} stocks")
            else:
                print("  - Warning: No specified stocks found in returns")

    def get_all_close_prices(self):
        return self.all_close_prices

    def get_all_open_prices(self):
        return self.all_open_prices

    def get_returns(self):
        return self.returns

    def set_date(self, date):
        """Sets the simulation date."""
        try:
            self.simulation_date = pd.to_datetime(date)
            print(f"Simulation date set to: {self.simulation_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"Error setting simulation date: {e}. Please provide a valid date format.")
            self.simulation_date = None

    def get_cov(self, window=None):
        """
        Returns the covariance matrix of returns before the simulation date.
        
        Args:
            window (int, optional): Number of trading days to look back from simulation date.
                                   If None, uses all available historical data.
        """
        if self.simulation_date is None:
            print("Error: Simulation date not set. Please call set_date() first.")
            return None
        if self.returns is None:
            print("Warning: Returns data not loaded.")
            return None
        
        historical_returns = self.returns[self.returns.index < self.simulation_date]
        
        if window is not None:
            historical_returns = historical_returns.tail(window)
            print(f"Calculating covariance matrix using last {window} days of data before {self.simulation_date.strftime('%Y-%m-%d')}")
        else:
            print(f"Calculating covariance matrix using all available data before {self.simulation_date.strftime('%Y-%m-%d')}")
        
        if historical_returns.empty:
            print(f"Warning: No returns data available before {self.simulation_date.strftime('%Y-%m-%d')}.")
            return None
        
        if len(historical_returns) < 2:
            print(f"Warning: Insufficient data points ({len(historical_returns)}) to calculate covariance matrix.")
            return None
        
        self.cov_matrix = historical_returns.cov()
        return self.cov_matrix

    def get_mu(self, window=None):
        """
        Returns the mean returns before the simulation date.
        
        Args:
            window (int, optional): Number of trading days to look back from simulation date.
                                   If None, uses all available historical data.
        """
        if self.simulation_date is None:
            print("Error: Simulation date not set. Please call set_date() first.")
            return None
        if self.returns is None:
            print("Warning: Returns data not loaded.")
            return None
        
        historical_returns = self.returns[self.returns.index < self.simulation_date]
        
        if window is not None:
            historical_returns = historical_returns.tail(window)
            print(f"Calculating mean returns using last {window} days of data before {self.simulation_date.strftime('%Y-%m-%d')}")
        else:
            print(f"Calculating mean returns using all available data before {self.simulation_date.strftime('%Y-%m-%d')}")
        
        if historical_returns.empty:
            print(f"Warning: No returns data available before {self.simulation_date.strftime('%Y-%m-%d')}.")
            return None
        
        if len(historical_returns) < 1:
            print(f"Warning: Insufficient data points ({len(historical_returns)}) to calculate mean returns.")
            return None
        
        self.mean_returns = historical_returns.mean()
        return self.mean_returns

    def get_open_price(self):
        """Returns the first open price data point after the simulation date."""
        if self.simulation_date is None:
            print("Error: Simulation date not set. Please call set_date() first.")
            return None
        if self.all_open_prices is None:
            print("Warning: All open prices data not loaded.")
            return None
        
        future_open_prices = self.all_open_prices[self.all_open_prices.index > self.simulation_date]
        
        if future_open_prices.empty:
            print(f"Warning: No open price data available after {self.simulation_date.strftime('%Y-%m-d')}.")
            return None
        
        return future_open_prices.iloc[0]

    def get_close_price(self):
        """Returns the first close price data point after the simulation date."""
        if self.simulation_date is None:
            print("Error: Simulation date not set. Please call set_date() first.")
            return None
        if self.all_close_prices is None:
            print("Warning: All close prices data not loaded.")
            return None
        
        future_close_prices = self.all_close_prices[self.all_close_prices.index > self.simulation_date]
        
        if future_close_prices.empty:
            print(f"Warning: No close price data available after {self.simulation_date.strftime('%Y-%m-%d')}.")
            return None
        
        return future_close_prices.iloc[0]

    def next_date(self):
        """
        Advances the simulation date to the next available date in the dataset
        and returns that date.
        """
        if self.simulation_date is None:
            print("Error: Simulation date not set. Please call set_date() first.")
            return None
        
        # Use all_close_prices index as the primary source for available dates
        if self.all_close_prices is None:
            print("Warning: All close prices data not loaded. Cannot determine next date.")
            return None
        
        all_dates = self.all_close_prices.index.sort_values()
        
        # Find dates strictly after the current simulation date
        future_dates = all_dates[all_dates > self.simulation_date]
        
        if future_dates.empty:
            print(f"Warning: No more dates available after {self.simulation_date.strftime('%Y-%m-%d')}.")
            return None
        
        # The next date is the first one in the sorted future dates
        next_available_date = future_dates.min()
        self.simulation_date = next_available_date
        print(f"Simulation date advanced to: {self.simulation_date.strftime('%Y-%m-%d')}")
        return self.simulation_date
    
    def has_next(self):
        if self.simulation_date is None:
            print("Error: Simulation date not set. Please call set_date() first.")
            return False
        
        if self.all_close_prices is None:
            print("Warning: All close prices data not loaded. Cannot determine next date.")
            return False
        
        all_dates = self.all_close_prices.index.sort_values()
        future_dates = all_dates[all_dates > self.simulation_date]
        
        if future_dates.empty:
            return False
        
        return True

    
class BruteForceSolver():
    def __init__(self, stock_upper_bounds, budget, lam):
        self.upper_bounds = stock_upper_bounds
        self.B = budget
        self.lam = lam
    
    def set_budget(self, budget):
        self.B = budget

    def optimize(self, n, mu, sigma, prices):
        best_x = None
        best_value = -1e18
        records = []

        for x in tqdm(itertools.product(*(range(ub + 1) for ub in self.upper_bounds))):
            x = np.array(x)
            cost = prices @ x
            
            if cost <= self.B:
                value = mu @ x - self.lam * (x.T @ sigma @ x)
                records.append([*x, cost, value])
                
                if value > best_value:
                    best_value = value
                    best_x = x.copy()

        return best_x

class SCIPSolver():
    def __init__(self, stock_upper_bounds, budget, lam):
        self.upper_bounds = stock_upper_bounds
        self.B = budget
        self.lam = lam
    
    def set_budget(self, budget):
        self.B = budget

    def optimize(self, n, mu, sigma, prices):
        model = Model("mean_variance_mip")
        x = [model.addVar(vtype="I", lb=0, ub=self.upper_bounds[i], name=f"x{i}") for i in range(n)]

        # ----------- 2. budget constraint -----------------
        model.addCons(quicksum(prices[i] * x[i] for i in range(n)) <= self.B)

        # ----------- 3. 引入一个 objvar 作为线性目标 -----------------
        objvar = model.addVar(vtype="C", name="objvar")

        # ----------- 4. 构建线性项 μᵀx -----------
        linear_term = quicksum(mu[i] * x[i] for i in range(n))

        # ----------- 5. 构建二次项 xᵀΣx（SCIP 作为非线性表达式支持） -----------
        quadratic_term = quicksum(sigma[i, j] * x[i] * x[j] for i in range(n) for j in range(n))

        # ----------- 6. 添加非线性约束： objvar >= μᵀx − λ·xᵀΣx -----------
        # （即 objvar - linear_term + lam * quadratic_term >= 0）
        model.addCons(objvar - linear_term + self.lam * quadratic_term <= 0)

        # ----------- 7. 目标：最大化 objvar（线性的，SCIP 允许） -----------
        model.setObjective(objvar, sense="maximize")

        # ----------- 8. 求解 -----------------
        # 使用 redirect_stdout 抑制 SCIP 的输出
        with redirect_stdout(open(os.devnull, 'w')):
            model.optimize()

        # ----------- 9. 输出结果 -----------------
        # 移除了原始的 print 语句，因为用户要求不输出
        sol = model.getBestSol()

        if sol:
            x_opt = np.array([sol[x[i]] for i in range(n)], dtype=int)
            return x_opt
        else:
            return None

class QuantumAnnealingSolver():
    def __init__(self, stock_upper_bounds, budget, lam, alpha, num_spins, bits_per_asset, bits_slack):
        self.upper_bounds = stock_upper_bounds
        self.budget = budget
        self.lam = lam
        self.num_spins = num_spins
        self.bits_per_asset = bits_per_asset
        self.bits_slack = bits_slack
        self.alpha = alpha
    
    def set_budget(self, budget):
        self.budget = budget

    
    def optimize(self, n, mu, sigma, prices):
        def H_factor(s):
            
            # s is a list of spins (+1/-1), we need to map it to binary variables
            v = [(s[i]+1)//2 for i in range(len(s))]

            H = 0.0
            for i in range(n):
                for j in range(n):
                    for p1 in range(self.bits_per_asset):
                        for p2 in range(self.bits_per_asset):
                            idx_i = i*self.bits_per_asset + p1
                            idx_j = j*self.bits_per_asset + p2
                            coeff = (self.lam * sigma[i,j] + self.alpha * prices[i] * prices[j]) * (2**p1) * (2**p2)
                            H += coeff * v[idx_i] * v[idx_j]
            
            for i in range(n):
                for p in range(self.bits_per_asset):
                    idx = i*self.bits_per_asset + p
                    coeff = - (mu[i] + 2 * self.alpha * self.budget * prices[i]) * (2**p)
                    H += coeff * v[idx]

            for i in range(n):
                for p1 in range(self.bits_per_asset):
                    for p2 in range(self.bits_slack):
                        idx1 = i*self.bits_per_asset + p1
                        idx2 = n*self.bits_per_asset + p2
                        coeff = 2 * self.alpha * prices[i] * (2**p1) * (2**p2)
                        H += coeff * v[idx1] * v[idx2]

            for p1 in range(self.bits_slack):
                for p2 in range(self.bits_slack):
                    idx1 = n*self.bits_per_asset + p1
                    idx2 = n*self.bits_per_asset + p2
                    coeff = self.alpha * (2**p1) * (2**p2)
                    H += coeff * v[idx1] * v[idx2]

            for p in range(self.bits_slack):
                idx = n*self.bits_per_asset + p
                coeff = - self.alpha * (2 * self.budget) * (2**p)
                H += coeff * v[idx]

            H += self.alpha * self.budget * self.budget
            return H

        def get_ising_coeffs(H_func):
            configs = np.array(list(itertools.product([1,-1], repeat=self.num_spins)))
            H_values = np.array([H_func(s) for s in configs])

            num_terms = 1 + self.num_spins + self.num_spins*(self.num_spins-1)//2  # 常数 + h_i + J_ij
            X = np.ones((2**self.num_spins, num_terms))
            X[:,1:1+self.num_spins] = configs

            # 填充上三角 s_i s_j
            idx = 1 + self.num_spins
            for i in range(self.num_spins):
                for j in range(i+1, self.num_spins):
                    X[:, idx] = configs[:,i]*configs[:,j]
                    idx +=1

            coeffs, *_ = np.linalg.lstsq(X, H_values, rcond=None)

            C = coeffs[0]
            h = coeffs[1:1+self.num_spins]  # 注意标准形式 H = h_i s_i + sum J_ij s_i s_j + C  ##### IMPORTANT: all things in positive manner #####
            J = np.zeros((self.num_spins,self.num_spins))
            idx = 1 + self.num_spins
            for i in range(self.num_spins):
                for j in range(i+1, self.num_spins):
                    J[i,j] = coeffs[idx]
                    idx +=1

            return h, J, C

        h, J, C = get_ising_coeffs(H_factor)

        # max_energy = -np.inf
        # min_energy = np.inf
        # min_energy_state = None
        # for state in range(2**self.num_spins):
        #     z = np.array([1 if (state >> i) & 1 == 0 else -1 for i in range(self.num_spins)])
        #     energy = z @ J @ z + h @ z + C
        #     max_energy = max(max_energy, energy) 
        #     if energy < min_energy:
        #         min_energy = energy
        #         min_energy_state = z

        def U_H(J, h, t):
            qc = QuantumCircuit(self.num_spins)
            for i in range(self.num_spins):
                if h[i] != 0:
                    qc.rz(-2 * h[i] * t, i)
            for i in range(self.num_spins):
                for j in range(i + 1, self.num_spins):
                    if J[i, j] != 0:
                        qc.cx(i, j)
                        qc.rz(-2 * J[i, j] * t, j)
                        qc.cx(i, j)
            return qc

        def U_x(B, t):
            qc = QuantumCircuit(self.num_spins)
            for i in range(self.num_spins):
                qc.rx(2 * B * t, i)
            return qc

        def trotter_annealing(T=10, M=100, B=1.0):
            """Simulate quantum annealing using first-order Trotter decomposition."""
            dt = T / M
            qc = QuantumCircuit(self.num_spins)
            qc.h(range(self.num_spins))  # Initialize in |+> state
            for i in range(M):
                s = i / M
                qc.append(U_x(B * (1 - s), dt), range(self.num_spins))
                qc.append(U_H(J, h, dt * s), range(self.num_spins))
            return qc


        def compute_energy(bitstring, J, h, C):
            """Compute Ising energy given spin configuration (+1/-1)."""
            #print(bitstring)
            S = np.array([1 if b == '0' else -1 for b in bitstring[::-1]])
            return S @ J @ S + np.dot(h, S) + C

        qc = trotter_annealing(T=10, M=1000, B=1) # TODO: how to choose parameters
        qc.measure_all()
        sim = AerSimulator()
        result = sim.run(transpile(qc, sim), shots=10000).result()
        counts = result.get_counts()

        # Compute energies for each measurement
        energies = []
        min_energy = np.inf
        ground_state = None
        for bitstring, count in counts.items():
            E = compute_energy(bitstring, J, h, C)
            energies += [E] * count
            if E < min_energy:
                min_energy = E
                ground_state = np.array([1 if b == '0' else -1 for b in bitstring[::-1]])
        
        asset_counts = []
        for i in range(n):
            count = 0
            for p in range(self.bits_per_asset):
                idx = i*self.bits_per_asset + p
                if ground_state[idx] == 1:
                    count += 2**p
            asset_counts.append(count)
        
        return np.array(asset_counts)



class Benchmark():
    def __init__(self, optimizer, date, budget, max_iter, stock_list, history_window=None):
        self.optimizer = optimizer
        self.optimizer.set_budget(budget)
        self.dataset = StockDataset(stock_list=stock_list)
        self.dataset.set_date(date)
        self.date = date
        self.budget = budget
        self.max_iter = max_iter
        self.window = history_window
    
    def optimize(self, n, mu, sigma, open_prices):
        return self.optimizer.optimize(n, mu, sigma, open_prices)

    def run_benchmark(self):
        budget_history = []
        date_history = []
        budget_history.append(self.budget)
        date_history.append(pd.to_datetime(self.date))

        self.date = self.dataset.next_date()
        iterations = 0
        while self.dataset.has_next() and iterations < self.max_iter:
            mu = np.array(self.dataset.get_mu(self.window))
            sigma = np.array(self.dataset.get_cov(self.window))
            open_prices = np.array(self.dataset.get_open_price())
            
            if mu is None or sigma is None or open_prices is None or len(mu) == 0:
                break

            best_x = self.optimize(len(mu), mu, sigma, open_prices)
            print(f"On date {self.date.strftime('%Y-%m-%d')} best x is {best_x}") # 格式化日期输出

            if best_x is None:
                print(f"Optimization failed for date {self.date.strftime('%Y-%m-%d')}. Stopping benchmark.")
                break

            close_prices = self.dataset.get_close_price()
            if close_prices is None:
                print(f"Could not get close prices for date {self.date.strftime('%Y-%m-%d')}. Stopping benchmark.")
                break

            try:
                self.budget = self.budget + best_x @ (close_prices - open_prices) # 使用 .values 确保是 numpy 数组
            except ValueError as e:
                print(f"Error calculating new budget on date {self.date.strftime('%Y-%m-%d')}: {e}")
                print(f"best_x shape: {best_x.shape}, close_prices shape: {close_prices.shape}")
                break

            budget_history.append(self.budget)
            date_history.append(pd.to_datetime(self.date))

            self.date = self.dataset.next_date()
            iterations += 1
        
        if date_history:
            plt.figure(figsize=(12, 6))
            plt.plot(date_history, budget_history, marker='o', linestyle='-')
            plt.title('Budget Evolution Over Time')
            plt.xlabel('Date')
            plt.ylabel('Budget')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 确保 'result' 目录存在
            os.makedirs("result", exist_ok=True)
            plt.savefig("result/result.png", dpi=600)
            plt.show()
        else:
            print("No budget history to plot.")


if __name__ == "__main__":
    stock_list = None
    budget = 10000
    upper_bound_per_stock = 100
    n = 13
    if stock_list is not None:
        n = len(stock_list)
    solver = SCIPSolver([upper_bound_per_stock]*n ,budget, 0.3)
    # solver = QuantumAnnealingSolver([upper_bound_per_stock]*n, 1000, 0.3, 5.0, 15, 2, 5)
    benchmark = Benchmark(solver, "2023-01-01", budget, 1000, stock_list=stock_list, history_window=100)

    benchmark.run_benchmark()