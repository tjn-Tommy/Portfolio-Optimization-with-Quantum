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


class BaseSolver():
    def __init__(self, stock_upper_bounds, budget, lam):
        self.upper_bounds = stock_upper_bounds
        self.B = budget
        self.lam = lam
    
    def set_budget(self, budget):
        self.B = budget

    def optimize(self, n, mu, sigma, prices):
        raise NotImplementedError("This method should be overridden by subclasses.")  

class BruteForceSolver(BaseSolver):
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

class SCIPSolver(BaseSolver):
    def optimize(self, n, mu, sigma, prices):
        model = Model("mean_variance_mip")
        model.hideOutput() 
        x = [model.addVar(vtype="I", lb=0, ub=self.upper_bounds[i], name=f"x{i}") for i in range(n)]
        model.addCons(quicksum(prices[i] * x[i] for i in range(n)) <= self.B)

        # ----------- 3. 引入一个 objvar 作为线性目标 -----------------
        objvar = model.addVar(vtype="C", name="objvar")

        # ----------- 4. 构建参数 -----------
        linear_term = quicksum(mu[i] * x[i] for i in range(n))
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
        sol = model.getBestSol()

        if sol:
            x_opt = np.array([sol[x[i]] for i in range(n)], dtype=int)
            return x_opt
        else:
            return None

class QuantumAnnealingSolver(BaseSolver):
    def __init__(
        self,
        stock_upper_bounds,
        budget,
        lam,
        alpha,
        bits_per_asset,
        bits_slack,
    ):
        super().__init__(stock_upper_bounds, budget, lam)
        self.bits_per_asset = bits_per_asset
        self.bits_slack = bits_slack
        self.alpha = alpha
        self.num_spins = 0
    
    def optimize(self, n, mu, sigma, prices):
        def qubo_factor(num_spins):
            Q = np.zeros((num_spins, num_spins))
            L = np.zeros(num_spins)
            constant = 0.0
            # Quadratic Part
            for i in range(n):
                for j in range(n):
                    for p1 in range(self.bits_per_asset):
                        for p2 in range(self.bits_per_asset):
                            idx_i = i*self.bits_per_asset + p1
                            idx_j = j*self.bits_per_asset + p2
                            coeff = (self.lam * sigma[i,j] + self.alpha * prices[i] * prices[j]) * (2**p1) * (2**p2)
                            Q[idx_i, idx_j] += coeff
            
            # Linear Part
            for i in range(n):
                for p in range(self.bits_per_asset):
                    idx = i*self.bits_per_asset + p
                    coeff = - (mu[i] + 2 * self.alpha * self.B * prices[i]) * (2**p)
                    L[idx] += coeff

            for i in range(n):
                for p1 in range(self.bits_per_asset):
                    for p2 in range(self.bits_slack):
                        idx1 = i*self.bits_per_asset + p1
                        idx2 = n*self.bits_per_asset + p2
                        coeff = self.alpha * prices[i] * (2**p1) * (2**p2)
                        Q[idx1, idx2] += coeff
                        Q[idx2, idx1] += coeff 
            
            # Constants Part
            for p1 in range(self.bits_slack):
                for p2 in range(self.bits_slack):
                    idx1 = n*self.bits_per_asset + p1
                    idx2 = n*self.bits_per_asset + p2
                    coeff = self.alpha * (2**p1) * (2**p2)
                    Q[idx1, idx2] += coeff

            for p in range(self.bits_slack):
                idx = n*self.bits_per_asset + p
                coeff = - self.alpha * (2 * self.B) * (2**p)
                L[idx] += coeff

            constant += self.alpha * self.B * self.B

            return Q, L, constant

        def get_ising_coeffs(Q, L, constant):
            """Convert QUBO to Ising model coefficients."""
            num_vars = Q.shape[0]
            J = np.zeros((num_vars, num_vars))
            h = np.zeros(num_vars)
            C = constant

            # 对角线元素 Q_ii 转化
            for i in range(num_vars):
                L[i] += Q[i, i]
                Q[i, i] = 0.0

            # 线性项 L_i * q_i 转化
            # L_i * (1 - z_i)/2 = L_i/2 - (L_i/2) * z_i
            for i in range(num_vars):
                h[i] -= L[i] / 2.0
                C += L[i] / 2.0

            # 二次项 Q_ij * q_i * q_j 转化 (i != j)
            for i in range(num_vars):
                for j in range(i + 1, num_vars): # 只遍历上三角
                    val = Q[i, j] + Q[j, i] # 汇总对称位置的系数
                    
                    term = val / 4.0
                    C += term       # 常数部分
                    h[i] -= term          # z_i 部分
                    h[j] -= term          # z_j 部分
                    J[i, j] += term       # z_i*z_j 部分 (Ising 耦合)
            
            return h, J, C
        
        self.num_spins = n * self.bits_per_asset + self.bits_slack
        Q, L, constant = qubo_factor(self.num_spins)
        h, J, C = get_ising_coeffs(Q, L, constant)

        def U_H(J, h, t):
            qc = QuantumCircuit(self.num_spins)
            for i in range(self.num_spins):
                if h[i] != 0:
                    qc.rz(2 * h[i] * t, i)
            for i in range(self.num_spins):
                for j in range(i + 1, self.num_spins):
                    if J[i, j] != 0:
                        qc.cx(i, j)
                        qc.rz(2 * J[i, j] * t, j)
                        qc.cx(i, j)
            return qc

        def U_x(B, t):
            qc = QuantumCircuit(self.num_spins)
            for i in range(self.num_spins):
                qc.rx(- 2 * B * t, i)
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

        qc = trotter_annealing(T=10, M=100, B=1) 
        qc.measure_all()
        sim = AerSimulator()
        result = sim.run(transpile(qc, sim), shots=1000).result()
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
                if ground_state[idx] == -1:
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
    # stock_list = ['BA','JNJ','JPM','PG','WMT','XOM','AAPL','AMZN','GOOGL','META','MSFT','TSLA']
    stock_list = ['AAPL','MSFT','GOOGL']
    budget = 1000
    upper_bound_per_stock = 8
    n = 13
    if stock_list is not None:
        n = len(stock_list)
    # solver = SCIPSolver([upper_bound_per_stock]*n ,budget, 0.3)
    solver = QuantumAnnealingSolver([upper_bound_per_stock]*n, 1000, 0.3, 5.0, 3, 5)
    benchmark = Benchmark(solver, "2024-12-01", budget, 1000, stock_list=stock_list, history_window=100)

    benchmark.run_benchmark()