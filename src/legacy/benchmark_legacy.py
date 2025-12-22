import numpy as np
import pandas as pd
from pyscipopt import Model, quicksum
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import itertools
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from src.benchmark.dataset import StockDataset


class BaseSolver():
    def __init__(self, stock_upper_bounds, budget, lam):
        self.upper_bounds = stock_upper_bounds
        self.B = budget
        self.lam = lam
    
    def set_budget(self, budget):
        self.B = budget

    def optimize(self, n, mu, sigma, prices):
        raise NotImplementedError("This method should be overridden by subclasses.")  

class AutoSolver(BaseSolver):
    def __init__(self, optimizer, date, budget, max_iter, stock_list, history_window=None):
        self.optimizer = optimizer
        self.optimizer.set_budget(budget)
        self.dataset = StockDataset(stock_list=stock_list)
        self.dataset.set_date(date)
        self.date = date
        self.budget = budget
        self.max_iter = max_iter
        self.window = history_window
    
    def set_optimizer(fn: Any):
        pass

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