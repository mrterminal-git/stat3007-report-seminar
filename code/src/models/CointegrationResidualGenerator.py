import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Optional

class CointegrationResidualGenerator:
    """
    Generates cointegration residuals for a set of assets based on their price data.
    Residuals represent deviations from a linear combination of cumulative excess returns,
    useful for modeling stationary relationships between assets in portfolio optimization.

    Attributes:
        price_data (pd.DataFrame): Input price data with dates as index and assets (e.g., countries) as columns.
        risk_free_rate_daily (float): Daily risk-free rate, derived from annual rate divided by 252 trading days.
        returns (pd.DataFrame): Daily excess log returns for each asset.
        cumulative_returns (pd.DataFrame): Cumulative sum of excess returns over time.
        asset_residuals (pd.DataFrame): Residuals from cointegration regression for each asset.
        betas (dict): Dictionary storing regression coefficients (betas) for each asset.
    """
    def __init__(self, price_data: pd.DataFrame, risk_free_rate_annual: float = 0.01):
        """
        Initialize the CointegrationResidualGenerator with price data and risk-free rate.

        Args:
            price_data (pd.DataFrame): DataFrame with shape [num_days, num_assets], where:
                - Index: Datetime index representing trading days.
                - Columns: Asset names (e.g., countries like 'Austria', 'Germany').
                - Values: Daily prices for each asset.
            risk_free_rate_annual (float): Annual risk-free rate (default: 0.01, i.e., 1%).

        Initializes:
            - Converts annual risk-free rate to daily rate (assuming 252 trading days per year).
            - Computes excess log returns and their cumulative sum.
            - Sets up empty DataFrame for residuals and dictionary for betas.
        """
        self.price_data = price_data  # Store input price data
        # Convert annual risk-free rate to daily rate
        self.risk_free_rate_daily = risk_free_rate_annual / 252
        # Compute daily excess log returns
        self.returns = self._compute_excess_returns(price_data)
        # Compute cumulative excess returns
        self.cumulative_returns = self._compute_cumulative_returns(self.returns)
        # Initialize empty DataFrame for residuals, with same index as cumulative returns
        self.asset_residuals = pd.DataFrame(index=self.cumulative_returns.index)
        # Initialize dictionary to store regression coefficients
        self.betas = {}

    def _compute_excess_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily excess log returns for each asset.

        Args:
            price_data (pd.DataFrame): Price data with shape [num_days, num_assets].

        Returns:
            pd.DataFrame: Excess log returns with shape [num_days - 1, num_assets], where:
                - Index: Datetime index (one day less due to shift).
                - Columns: Asset names.
                - Values: Log returns minus the daily risk-free rate.

        Notes:
            - Log returns are computed as log(price_t / price_{t-1}).
            - Excess returns subtract the daily risk-free rate.
            - First row is dropped due to NaN from shift operation.
        """
        # Compute log returns: log(price_t / price_{t-1})
        log_returns = np.log(price_data / price_data.shift(1)).dropna()
        # Subtract daily risk-free rate to get excess returns
        excess_returns = log_returns - self.risk_free_rate_daily
        return excess_returns

    def _compute_cumulative_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cumulative excess returns by summing daily excess returns over time.

        Args:
            returns (pd.DataFrame): Daily excess returns with shape [num_days, num_assets].

        Returns:
            pd.DataFrame: Cumulative returns with shape [num_days, num_assets], where:
                - Index: Same datetime index as input returns.
                - Columns: Asset names.
                - Values: Cumulative sum of excess returns for each asset.
        """
        # Compute cumulative sum of returns along the time axis
        cumulative_returns = returns.cumsum()
        return cumulative_returns

    def compute_all_asset_residuals(self) -> None:
        """
        Compute cointegration residuals for each asset by treating it as the dependent variable
        in a linear regression against all other assets' cumulative returns.

        For each asset:
            - Dependent variable: Cumulative returns of the target asset.
            - Independent variables: Cumulative returns of all other assets.
            - Residuals: Difference between actual and predicted cumulative returns.

        Stores:
            - Residuals in self.asset_residuals (DataFrame with shape [num_days, num_assets]).
            - Regression coefficients (betas) in self.betas (dict with asset names as keys).

        Notes:
            - Uses sklearn.LinearRegression to fit the model.
            - Residuals represent stationary deviations from the cointegrating relationship.
        """
        # Iterate over each asset to treat it as the dependent variable
        for target_asset in self.cumulative_returns.columns:
            # Extract dependent variable (y): Cumulative returns of the target asset
            y = self.cumulative_returns[target_asset].values.reshape(-1, 1)  # Shape: [num_days, 1]
            # Extract independent variables (X): Cumulative returns of all other assets
            X = self.cumulative_returns.drop(columns=[target_asset]).values  # Shape: [num_days, num_assets - 1]
            X_cols = self.cumulative_returns.drop(columns=[target_asset]).columns  # Column names of other assets

            # Fit linear regression model: y ~ X
            model = LinearRegression().fit(X, y)
            betas = model.coef_[0]  # Regression coefficients, shape: [num_assets - 1]
            intercept = model.intercept_[0]  # Intercept, scalar

            # Predict values: y_pred = X * betas + intercept
            y_pred = model.predict(X).flatten()  # Shape: [num_days]
            # Compute residuals: y - y_pred
            residuals = y.flatten() - y_pred  # Shape: [num_days]

            # Store residuals in DataFrame
            self.asset_residuals[target_asset] = residuals
            # Store betas and intercept for this asset
            beta_series = pd.Series(betas, index=X_cols)
            beta_series['Intercept'] = intercept
            self.betas[target_asset] = beta_series

    def get_asset_residuals(self) -> pd.DataFrame:
        """
        Retrieve the computed cointegration residuals for all assets.

        Returns:
            pd.DataFrame: Residuals with shape [num_days, num_assets], where:
                - Index: Datetime index matching cumulative returns.
                - Columns: Asset names.
                - Values: Cointegration residuals for each asset.

        Raises:
            ValueError: If residuals have not been computed yet.
        """
        if self.asset_residuals.empty:
            raise ValueError("Asset residuals not yet computed.")
        return self.asset_residuals

    def get_betas_for_asset(self, asset: str) -> pd.Series:
        """
        Retrieve the regression coefficients (betas) used to compute residuals for a specific asset.

        Args:
            asset (str): Name of the asset (e.g., 'Austria').

        Returns:
            pd.Series: Series with:
                - Index: Names of other assets plus 'Intercept'.
                - Values: Regression coefficients and intercept for the cointegration model.

        Raises:
            ValueError: If betas for the asset are not found (i.e., residuals not computed).
        """
        if asset not in self.betas:
            raise ValueError(f"Betas for asset '{asset}' not found. Compute residuals first.")
        return self.betas[asset]

    def prepare_cnn_input_from_residuals(self, window: int = 30) -> np.ndarray:
        """
        Prepare a 3D array of cumulative residuals for input to a Convolutional Neural Network (CNN).

        For each window of residuals:
            - Compute the cumulative sum within the window to capture trends.
            - Organize into a 3D array suitable for CNN input.

        Args:
            window (int): Number of days in each window (default: 30).

        Returns:
            np.ndarray: 3D array with shape [num_samples, window, num_assets], where:
                - num_samples: Number of windows = (num_days - window + 1).
                - window: Number of days in each window.
                - num_assets: Number of assets (e.g., countries).
                - Values: Cumulative residuals within each window.

        Raises:
            ValueError: If residuals have not been computed yet.

        Example:
            - If num_days = 100, window = 30, num_assets = 5:
            - Output shape: [71, 30, 5], where 71 = (100 - 30 + 1).
        """
        if self.asset_residuals.empty:
            raise ValueError("Asset residuals not yet computed.")

        cnn_input_list = []

        # Iterate over possible windows
        for start_idx in range(len(self.asset_residuals) - window + 1):
            # Extract window of residuals
            window_residuals = self.asset_residuals.iloc[start_idx:start_idx + window]  # Shape: [window, num_assets]
            # Compute cumulative sum within the window
            cumulative_window = window_residuals.cumsum()  # Shape: [window, num_assets]
            # Append to list
            cnn_input_list.append(cumulative_window.values)

        # Convert to 3D numpy array
        cnn_input_array = np.array(cnn_input_list)  # Shape: [num_samples, window, num_assets]
        return cnn_input_array