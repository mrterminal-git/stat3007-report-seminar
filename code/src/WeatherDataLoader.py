from typing import Optional, List
import pandas as pd

class WeatherDataLoader:
    """
    Loads and processes weather data from a CSV file, providing methods to filter and transform the data
    into a format suitable for analysis.
    """
    def __init__(self, file_path: str = "../data/aggregated_weather.csv"):
        """
        Initialize the WeatherDataLoader with the path to the weather data CSV file.

        Args:
            file_path (str): Path to the CSV file containing weather data (default: '../data/aggregated_weather.csv').
        """
        self.file_path = file_path  # Store file path
        self.data = self._load_data()  # Load and preprocess data

    def _load_data(self) -> Optional[pd.DataFrame]:
        """
        Load and preprocess the weather data from the CSV file.

        Returns:
            pd.DataFrame or None: Loaded DataFrame with 'Date' as datetime, or None if loading fails.
        """
        try:
            # Load CSV file into a DataFrame
            df = pd.read_csv(self.file_path)
            # Convert 'valid_time' to datetime and rename to 'Date'
            df['valid_time'] = pd.to_datetime(df['valid_time'])
            df.rename(columns={'valid_time': 'Date'}, inplace=True)
            print(f"Weather data loaded successfully from {self.file_path}")
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

    def get_data_by_country_and_range(self, time_range: str, country: Optional[str] = None) -> pd.DataFrame:
        """
        Filter weather data by country and date range.

        Args:
            time_range (str): Date range in format 'YYYY-MM-DD,YYYY-MM-DD' (e.g., '2021-01-01,2021-12-31').
            country (str, optional): Country to filter by (e.g., 'Austria'). If None, include all countries.

        Returns:
            pd.DataFrame: Filtered DataFrame with weather data, or empty DataFrame if no data is found.
        """
        if self.data is None:
            print("Error: Weather data not loaded.")
            return pd.DataFrame()

        try:
            # Parse start and end dates from time_range
            start_date_str, end_date_str = time_range.split(',')
            start_date = pd.to_datetime(start_date_str.strip())
            end_date = pd.to_datetime(end_date_str.strip())
        except ValueError:
            print("Error: Invalid time_range format. Please use 'YYYY-MM-DD,YYYY-MM-DD'.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error parsing time range: {e}")
            return pd.DataFrame()

        # Copy data to avoid modifying the original
        output_data = self.data.copy()
        # Filter by country if specified
        if country is not None:
            output_data = output_data[output_data['country'].str.lower() == country.lower()]

        # Filter by date range (inclusive)
        filtered_data = output_data[
            (output_data['Date'] >= start_date) & (output_data['Date'] <= end_date)
        ]

        # Warn if no data is found
        if filtered_data.empty:
            print(f"Warning: No weather data found for country '{country}' within the range {time_range}.")
            return pd.DataFrame()

        return filtered_data.copy()

    def get_all_data(self) -> Optional[pd.DataFrame]:
        """
        Return the entire weather dataset.

        Returns:
            pd.DataFrame or None: Copy of the entire weather dataset, or None if not loaded.
        """
        if self.data is None:
            print("Error: Weather data not loaded.")
            return None
        return self.data.copy()

    def get_country_list(self) -> Optional[List[str]]:
        """
        Get a list of unique countries in the weather dataset.

        Returns:
            List[str] or None: List of unique country names, or None if data is not loaded.
        """
        if self.data is None:
            print("Error: Weather data not loaded.")
            return None
        return self.data['country'].unique().tolist()

    def get_weather_matrix(
        self,
        time_range: str,
        countries: List[str],
        fill_method: Optional[str] = None,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create a weather feature matrix with dates as rows and (feature, country) as columns.

        Args:
            time_range (str): Date range in format 'YYYY-MM-DD,YYYY-MM-DD'.
            countries (List[str]): List of countries to include.
            fill_method (str, optional): Method to handle missing values ('ffill', 'bfill', or None).
            features (List[str], optional): List of weather features to include. If None, include all numerical columns.

        Returns:
            pd.DataFrame: DataFrame with MultiIndex columns (feature, country) and dates as index.
        """
        if self.data is None:
            print("Error: Weather data not loaded.")
            return pd.DataFrame()

        # Parse start and end dates
        start_date, end_date = time_range.split(",")
        start_date = pd.to_datetime(start_date.strip())
        end_date = pd.to_datetime(end_date.strip())

        # Filter data by countries and date range
        df = self.data.copy()
        df = df[df['country'].isin(countries)]
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        # Select features (default to all numerical columns except 'Date' and 'country')
        if features is None:
            features = [col for col in df.columns if col not in ['Date', 'country'] and df[col].dtype in ['float64', 'int64']]

        # Create a multi-index DataFrame for each feature
        df_list = []
        for feature in features:
            temp_df = df.pivot(index='Date', columns='country', values=feature)
            temp_df.columns = pd.MultiIndex.from_product([[feature], temp_df.columns])
            df_list.append(temp_df)
        
        # Concatenate all feature DataFrames
        weather_matrix = pd.concat(df_list, axis=1).sort_index()

        # Handle missing data based on fill_method
        if fill_method == "ffill":
            weather_matrix = weather_matrix.ffill()
        elif fill_method == "bfill":
            weather_matrix = weather_matrix.bfill()
        else:
            weather_matrix = weather_matrix.dropna()

        return weather_matrix

    def get_weather_matrix_rolling_window(
        self,
        one_window_days: int,
        window_stride_days: int,
        time_range: str,
        countries: List[str],
        fill_method: Optional[str] = None,
        features: Optional[List[str]] = None
    ) -> List[pd.DataFrame]:
        """
        Generate rolling windows of weather feature matrices.

        Args:
            one_window_days (int): Number of days in each window.
            window_stride_days (int): Number of days to stride between windows.
            time_range (str): Date range in format 'YYYY-MM-DD,YYYY-MM-DD'.
            countries (List[str]): List of countries to include.
            fill_method (str, optional): Method to handle missing values ('ffill', 'bfill', or None).
            features (List[str], optional): List of weather features to include.

        Returns:
            List[pd.DataFrame]: List of weather matrices, each representing a rolling window.
        """
        # Get the full weather matrix
        weather_matrix = self.get_weather_matrix(
            time_range=time_range,
            countries=countries,
            fill_method=fill_method,
            features=features
        )

        # Generate rolling windows
        rolling_windows = []
        for start_idx in range(0, len(weather_matrix) - one_window_days + 1, window_stride_days):
            end_idx = start_idx + one_window_days
            if end_idx > len(weather_matrix):
                break
            rolling_window = weather_matrix.iloc[start_idx:end_idx]
            rolling_windows.append(rolling_window)

        return rolling_windows