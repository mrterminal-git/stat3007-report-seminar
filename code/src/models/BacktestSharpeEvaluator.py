import numpy as np

class BacktestSharpeEvaluator:
    def __init__(self):
        self.portfolio_returns = []

    def add_return(self, r: float):
        """Add a single next-day portfolio return."""
        self.portfolio_returns.append(r)

    def add_returns(self, returns: list):
        """Add a list of next-day portfolio returns."""
        self.portfolio_returns.extend(returns)

    def reset(self):
        """Reset the stored returns."""
        self.portfolio_returns = []

    def calculate_sharpe(self, returns=None, risk_free_rate=0.0):
        """
        Calculate Sharpe Ratio from stored or passed-in returns.
        Sharpe Ratio = (mean - risk-free) / std deviation
        """
        r = self.portfolio_returns if returns is None else returns
        r = np.array(r)
        if len(r) == 0 or np.std(r) == 0:
            return np.nan
        excess_returns = r - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)

    def normalize_weights_l1(self, raw_weights, phi=None):
        """
        Normalize raw weights using Ordoñez's method:
        w_normalized = (w_raw^T * phi) / ||w_raw^T * phi||_1

        Parameters:
            raw_weights: numpy array of shape (n_assets,)
            phi: optional transformation matrix (e.g., identity or mapping from factor to asset space)

        Returns:
            L1-normalized weights: numpy array of shape (n_assets,)
        """
        if phi is None:
            phi = np.eye(len(raw_weights))  # default to identity if no mapping provided
        raw = raw_weights.T @ phi
        norm = np.sum(np.abs(raw))
        if norm == 0:
            return np.zeros_like(raw)
        return raw / norm

    def compute_portfolio_return(self, raw_weights, next_day_returns, phi=None):
        """
        Normalize weights, compute and store the next-day portfolio return.

        Parameters:
            raw_weights: numpy array of shape (n_assets,)
            next_day_returns: numpy array of shape (n_assets,)
            phi: optional transformation matrix

        Returns:
            Computed return (float)
        """
        w = self.normalize_weights_l1(raw_weights, phi)
        r = float(np.dot(w, next_day_returns))
        self.add_return(r)
        return r
