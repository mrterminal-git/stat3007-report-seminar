import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

class DataPreparation:
    """
    Prepares combined price residuals and weather data for input into a machine learning model.
    Handles alignment of dates, creation of rolling windows, and train/validation/test splits.
    """
    def __init__(self, price_residuals: pd.DataFrame, weather_data: pd.DataFrame, 
                 countries: List[str], weather_features: List[str]):
        """
        Initialize the DataPreparation class with price residuals and weather data.

        Args:
            price_residuals (pd.DataFrame): DataFrame with countries as columns, dates as index, and price residuals as values.
            weather_data (pd.DataFrame): DataFrame with MultiIndex columns (feature, country) and dates as index.
            countries (List[str]): List of country names to process (e.g., ['Austria', 'Germany']).
            weather_features (List[str]): List of weather features to include (e.g., ['temperature_2m_mean', 'wind_speed_mean']).
        """
        self.price_residuals = price_residuals  # Store price residuals
        self.weather_data = weather_data  # Store weather data
        self.countries = countries  # Store list of countries
        self.weather_features = weather_features  # Store list of weather features
        
    def align_data_dates(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align price residuals and weather data to have the same dates.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Aligned price residuals and weather data DataFrames with common dates.
        """
        # Find common dates between price residuals and weather data
        common_dates = self.price_residuals.index.intersection(self.weather_data.index)
        # Filter both datasets to include only common dates
        aligned_price = self.price_residuals.loc[common_dates]
        aligned_weather = self.weather_data.loc[common_dates]
        return aligned_price, aligned_weather
    
    def prepare_rolling_windows(self, window_size: int = 30, stride: int = 1) -> np.ndarray:
        """
        Create rolling windows of combined price residuals and weather data for model input.

        Args:
            window_size (int): Number of days in each window (default: 30).
            stride (int): Number of days to shift between windows (default: 1).

        Returns:
            np.ndarray: Array of shape [num_samples, num_features, window_size], where num_features is
                        (num_countries + num_weather_features * num_countries).
        """
        # Align price and weather data by date
        price_data, weather_data = self.align_data_dates()
        # Calculate number of samples based on window size and stride
        num_samples = (len(price_data) - window_size) // stride + 1
        combined_windows = []
        
        # Iterate over the data to create rolling windows
        for i in range(0, len(price_data) - window_size + 1, stride):
            # Extract price residuals for the current window and compute cumulative sum
            price_window = price_data.iloc[i:i+window_size].cumsum()
            # Extract weather data for the same window
            weather_window = weather_data.iloc[i:i+window_size]
            window_features = [price_window.values]  # Start with price residuals
            
            # Add weather features for each country
            for feature in self.weather_features:
                feature_data = np.zeros((window_size, len(self.countries)))
                for country_idx, country in enumerate(self.countries):
                    if (feature, country) in weather_window.columns:
                        feature_data[:, country_idx] = weather_window[(feature, country)].values
                window_features.append(feature_data)
            
            # Combine price and weather features, transpose to [num_features, window_size]
            combined_feature_array = np.concatenate(window_features, axis=1).T
            combined_windows.append(combined_feature_array)
        
        # Convert to numpy array with shape [num_samples, num_features, window_size]
        return np.array(combined_windows)
    
    def prepare_next_day_returns(self, returns: pd.DataFrame, 
                                window_size: int = 30, 
                                stride: int = 1) -> np.ndarray:
        """
        Prepare next-day returns aligned with the rolling windows.

        Args:
            returns (pd.DataFrame): Daily returns with countries as columns and dates as index.
            window_size (int): Number of days in each window (default: 30).
            stride (int): Number of days to shift between windows (default: 1).

        Returns:
            np.ndarray: Array of shape [num_samples, num_countries] containing next-day returns for each window.
        """
        # Align returns with price residuals by date
        aligned_returns = returns.loc[self.price_residuals.index]
        # Calculate expected number of samples
        num_samples = (len(self.price_residuals) - window_size) // stride + 1
        next_day_returns = []
        
        # Collect next-day returns for each valid window
        for i in range(0, len(self.price_residuals) - window_size + 1, stride):
            if i + window_size < len(aligned_returns):  # Ensure next-day return exists
                next_day_return = aligned_returns.iloc[i + window_size]
                next_day_returns.append(next_day_return.values)
        
        # Convert to numpy array
        result = np.array(next_day_returns)
        # Check for sample count mismatch
        if len(result) != num_samples:
            print(f"Warning: Number of returns ({len(result)}) does not match expected samples ({num_samples})")
            # Pad with zeros if fewer returns than expected
            if len(result) < num_samples:
                result = np.pad(result, ((0, num_samples - len(result)), (0, 0)), mode='constant')
            # Truncate if more returns than expected
            elif len(result) > num_samples:
                result = result[:num_samples]
        
        return result

    def create_train_val_test_split(self, 
                                   combined_data: np.ndarray, 
                                   next_day_returns: np.ndarray,
                                   train_size: float = 0.7, 
                                   val_size: float = 0.15) -> Tuple:
        """
        Split data into training, validation, and test sets.

        Args:
            combined_data (np.ndarray): Combined price and weather data with shape [num_samples, num_features, window_size].
            next_day_returns (np.ndarray): Next-day returns with shape [num_samples, num_countries].
            train_size (float): Proportion of data for training (default: 0.7).
            val_size (float): Proportion of data for validation (default: 0.15).

        Returns:
            Tuple: Contains three tuples of (data, returns) for training, validation, and test sets.
        """
        # Validate that input data and returns have the same number of samples
        if combined_data.shape[0] != next_day_returns.shape[0]:
            raise ValueError(f"Mismatch in sample counts: combined_data has {combined_data.shape[0]} samples, "
                            f"next_day_returns has {next_day_returns.shape[0]} samples")
        
        # Calculate split indices
        n_samples = combined_data.shape[0]
        train_idx = int(n_samples * train_size)
        val_idx = train_idx + int(n_samples * val_size)
        
        # Split data into training, validation, and test sets
        train_data = combined_data[:train_idx]
        train_returns = next_day_returns[:train_idx]
        val_data = combined_data[train_idx:val_idx]
        val_returns = next_day_returns[train_idx:val_idx]
        test_data = combined_data[val_idx:]
        test_returns = next_day_returns[val_idx:]
        
        return (train_data, train_returns), (val_data, val_returns), (test_data, test_returns)