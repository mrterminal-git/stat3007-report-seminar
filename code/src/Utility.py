from models.CointegrationResidualGenerator import CointegrationResidualGenerator
import pandas as pd


class Utility:
    @staticmethod
    def process_training_data(
        x_tr_data: list,
        returns: pd.DataFrame,
        cumulative_residual_window: int = 30
    ):
        """
        Processes training data to generate CNN inputs and next-day returns, preserving date indices.

        Returns:
        - x_tr_data_cumulative_residuals (list of pd.DataFrame): Each DataFrame is (countries, window), indexed by window dates.
        - y_tr_data_next_day_returns (pd.DataFrame): (samples, countries), indexed by next-day date.
        """

        all_cnn_inputs = []
        all_next_day_returns = []
        all_next_day_dates = []

        for current_price_matrix in x_tr_data:
            residual_generator = CointegrationResidualGenerator(current_price_matrix)
            # Skip if there is any thing wrong with the data
            try:
                residual_generator.compute_all_asset_residuals()
            except ValueError as e:
                print(f"Skipping due to error: {e}")
                continue
            asset_residuals = residual_generator.get_asset_residuals()

            if len(asset_residuals) < cumulative_residual_window:
                print("The cumulative residual window size exceeds the available data.")
                continue
        
            cnn_input = residual_generator.prepare_cnn_input_from_residuals(window=cumulative_residual_window)
            # cnn_input shape: [samples, window, features]
            # Transpose to [samples, features, window]
            cnn_input_array = cnn_input.transpose(0, 2, 1)

            for i in range(cnn_input_array.shape[0]):
                # Get window dates for this sample
                window_start = i
                window_end = i + cumulative_residual_window
                window_dates = asset_residuals.index[window_start:window_end]
                window_end_date = window_dates[-1]

                # Find the next date in returns after window_end_date
                try:
                    next_day_loc = returns.index.get_loc(window_end_date) + 1
                    if next_day_loc >= len(returns):
                        continue
                    next_day_date = returns.index[next_day_loc]
                    next_day_return = returns.iloc[next_day_loc].values
                except KeyError:
                    continue

                # Create DataFrame for this sample: (countries, window), columns=window_dates
                sample_df = pd.DataFrame(
                    cnn_input_array[i],
                    index=returns.columns,  # countries
                    columns=window_dates    # window dates
                )
                all_cnn_inputs.append(sample_df)
                all_next_day_returns.append(next_day_return)
                all_next_day_dates.append(next_day_date)

        # y_tr_data_next_day_returns: (samples, countries), indexed by next-day date
        y_tr_data_next_day_returns = pd.DataFrame(
            all_next_day_returns, index=pd.Index(all_next_day_dates, name="date"), columns=returns.columns
        )

        # x_tr_data_cumulative_residuals: list of DataFrames, each (countries, window_dates)
        return all_cnn_inputs, y_tr_data_next_day_returns