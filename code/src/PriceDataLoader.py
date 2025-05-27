import pandas as pd
from io import StringIO
from typing import Optional, List
import torch

class PriceDataLoader:
    """
    Parses European wholesale electricity price data, allowing filtering
    by country and date range.
    """
    def __init__(self, file_path="../data/european_wholesale_electricity_price_data_daily.csv"):
        """
        Initializes the parser and loads the data.

        Args:
            file_path (str): The path to the CSV file.
        """
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self):
        """Loads and preprocesses the data from the CSV file."""
        try:
            df = pd.read_csv(self.file_path)
            # Convert 'Date' column to datetime objects
            df['Date'] = pd.to_datetime(df['Date'])
            # Drop ISO3 Code column
            df.drop(columns={'ISO3 Code'}, inplace=True)
            # Rename price column for easier access
            df.rename(columns={'Price (EUR/MWhe)': 'Price'}, inplace=True)
            print(f"Data loaded successfully from {self.file_path}")
            
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return None
        except KeyError as e:
            print(f"Error: Expected column '{e}' not found in the CSV.")
            return None
        except Exception as e:
            print(f"Error loading or processing file: {e}")
            return None

    def get_data_by_country_and_range(self, time_range:str, country=None):
        """
        Filters the data for a specific country and time range.

        Args:
            country (str): The name of the country to filter by (e.g., 'Germany').
            time_range (str): A string representing the date range in the format
                              'YYYY-MM-DD,YYYY-MM-DD'.

        Returns:
            pandas.DataFrame: A DataFrame containing the filtered data,
                              or None if an error occurs or no data is found.
        """
        if self.data is None:
            print("Error: Data not loaded.")
            return None

        try:
            start_date_str, end_date_str = time_range.split(',')
            start_date = pd.to_datetime(start_date_str.strip())
            end_date = pd.to_datetime(end_date_str.strip())
        except ValueError:
            print("Error: Invalid time_range format. Please use 'YYYY-MM-DD,YYYY-MM-DD'.")
            return None
        except Exception as e:
             print(f"Error parsing time range: {e}")
             return None

        outputData = self.data.copy()
        # If a country is specified, filter the data by country, if not use all data
        if country is not None:
            outputData = self.data[self.data['Country'].str.lower() == country.lower()]

        # Filter by date range (inclusive)
        filtered_data = outputData[
            (outputData['Date'] >= start_date) & (outputData['Date'] <= end_date)
        ]

        if filtered_data.empty:
            print(f"Warning: No data found for country '{country}' within the range {time_range}.")
            return pd.DataFrame() # Return empty DataFrame

        return filtered_data.copy() # Return a copy to avoid SettingWithCopyWarning

    def get_all_data(self):
        """
        Returns the entire dataset.

        Returns:
            pandas.DataFrame: The entire dataset.
        """
        if self.data is None:
            print("Error: Data not loaded.")
            return None
        return self.data.copy()

    def get_country_list(self):
        """
        Returns a list of unique countries in the dataset.

        Returns:
            list: A list of unique country names.
        """
        if self.data is None:
            print("Error: Data not loaded.")
            return None
        return self.data['Country'].unique().tolist()
    
    def get_price_matrix(
        self,
        time_range: str,
        countries: List[str],
        fill_method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Returns a price matrix where:
        - Rows = dates
        - Columns = countries
        - Values = daily electricity prices

        Parameters:
        - time_range (str): e.g. "2021-05-10,2021-05-16"
        - countries (List[str]): list of country names to include
        - fill_method (Optional[str]): 'ffill', 'bfill', or None

        Returns:
        - pd.DataFrame: index=date, columns=country names, values=prices
        """
        start_date, end_date = time_range.split(",")

        # Filter the master data once
        df = self.data.copy()
        df = df[df["Country"].isin(countries)]
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

        # Pivot: index=date, columns=country, values=price
        price_matrix = df.pivot(index="Date", columns="Country", values="Price").sort_index()

        # Handle missing data
        if fill_method == "ffill":
            price_matrix = price_matrix.ffill()
        elif fill_method == "bfill":
            price_matrix = price_matrix.bfill()
        else:
            price_matrix = price_matrix.dropna()

        return price_matrix
    
    def get_countries_with_complete_data(self, time_range: str) -> List[str]:
        """
        Returns a list of countries with complete (no NaN) price data for the specified time range.

        Args:
            time_range (str): Dateyou want to use format 'YYYY-MM-DD,YYYY-MM-DD'.

        Returns:
            List[str]: List of countries with complete data for the given time range.
        """
        if self.data is None:
            print("Error: Data not loaded.")
            return []

        try:
            start_date_str, end_date_str = time_range.split(',')
            start_date = pd.to_datetime(start_date_str.strip())
            end_date = pd.to_datetime(end_date_str.strip())
        except ValueError:
            print("Error: Invalid time_range format. Please use 'YYYY-MM-DD,YYYY-MM-DD'.")
            return []

        # Calculate expected number of days
        expected_days = (end_date - start_date).days + 1

        # Filter data for the time range
        df = self.data[
            (self.data['Date'] >= start_date) & (self.data['Date'] <= end_date)
        ]

        # Group by country and count non-NaN prices
        country_counts = df.groupby('Country')['Price'].count()
        # Select countries with complete data (count equals expected days)
        complete_countries = country_counts[country_counts == expected_days].index.tolist()
        return complete_countries

    def get_price_matrix_rolling_window(
        self,
        one_window_days: int,
        window_stride_days: int,
        time_range: str,
        countries: List[str],
        fill_method: Optional[str] = None
    ) -> torch.Tensor:
        """
        Returns a tensor of price matrices for rolling windows.
        Shape: [num_samples, window_size, num_countries]

        Parameters:
        - one_window_days (int): Number of days in one window
        - window_stride_days (int): Number of days to stride the window
        - time_range (str): e.g., "2021-01-01,2021-12-31"
        - countries (List[str]): List of country names to include
        - fill_method (Optional[str]): 'ffill', 'bfill', or None

        Returns:
        - torch.Tensor: Shape [num_samples, window_size, num_countries]
        """
        price_matrix = self.get_price_matrix(time_range, countries, fill_method)
        num_samples = (len(price_matrix) - one_window_days) // window_stride_days + 1
        price_tensor = torch.zeros((num_samples, one_window_days, len(countries)), dtype=torch.float32)

        for i in range(0, len(price_matrix) - one_window_days + 1, window_stride_days):
            sample_idx = i // window_stride_days
            if sample_idx >= num_samples:
                break
            window = price_matrix.iloc[i:i + one_window_days]
            price_tensor[sample_idx] = torch.tensor(window.values, dtype=torch.float32)

        return price_tensor

    def get_next_day_returns(
        self,
        rolling_windows: torch.Tensor,
        price_matrix: pd.DataFrame,
        one_window_days: int,
        window_stride_days: int
    ) -> torch.Tensor:
        """
        Returns a tensor of next-day returns for each rolling window.
        Shape: [num_samples, num_countries]

        Parameters:
        - rolling_windows (torch.Tensor): Tensor of price matrices, shape [num_samples, window_size, num_countries]
        - price_matrix (pd.DataFrame): DataFrame of daily prices (index=date, columns=country names)
        - one_window_days (int): Number of days in one window
        - window_stride_days (int): Number of days to stride the window

        Returns:
        - torch.Tensor: Shape [num_samples, num_countries]
        """
        returns = price_matrix.pct_change().dropna()
        next_day_returns = torch.zeros((rolling_windows.shape[0], rolling_windows.shape[2]), dtype=torch.float32)

        for i in range(rolling_windows.shape[0]):
            last_date = price_matrix.index[i * 1 + one_window_days - window_stride_days]  # Assuming stride=1
            if last_date in returns.index:
                next_day_idx = returns.index.get_loc(last_date) + 1
                if next_day_idx < len(returns):
                    next_day_returns[i] = torch.tensor(returns.iloc[next_day_idx].values, dtype=torch.float32)
                else:
                    next_day_returns[i] = torch.zeros(rolling_windows.shape[2], dtype=torch.float32)
            else:
                next_day_returns[i] = torch.zeros(rolling_windows.shape[2], dtype=torch.float32)

        return next_day_returns