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