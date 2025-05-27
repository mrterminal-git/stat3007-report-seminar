class TimeSeriesDataSplitter:
    def __init__(self, data_x, data_y, train_size=0.7, validation_size=0.15, test_size=0.15):
        """
        Initializes the TimeSeriesDataSplitter with the given data, train, validation and test size.

        :param data_x (np.array): The time series feature data to be split.
        :param data_y (pd.DataFrame): The target data to be split.
        :param train_size (float): The proportion of the dataset to include in the training set.
        :param validation_size (float): The proportion of the dataset to include in the validation set.
        :param test_size (float): The proportion of the dataset to include in the testing set.
        :raises ValueError: If the sum of train_size, validation_size, and test_size is not equal to 1.
        """
        self.data_x = data_x
        self.data_y = data_y
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size

        if not (0 < train_size < 1 and 0 < validation_size < 1 and 0 < test_size < 1):
            raise ValueError("train_size, validation_size, and test_size must be between 0 and 1.")
        if train_size + validation_size + test_size != 1:
            raise ValueError("The sum of train_size, validation_size, and test_size must equal 1.")

    def split_data(self):
        """
        Splits the data into training, validation, and testing sets.

        :return: A tuple containing the training, validation, and testing sets.
        """
        n = len(self.data_x)
        train_end = int(n * self.train_size)
        val_end = train_end + int(n * self.validation_size)

        train_data_x = self.data_x[:train_end]
        train_data_y = self.data_y[:train_end]

        val_data_x = self.data_x[train_end:val_end]
        val_data_y = self.data_y[train_end:val_end]

        test_data_x = self.data_x[val_end:]
        test_data_y = self.data_y[val_end:]
        return (train_data_x, train_data_y), (val_data_x, val_data_y), (test_data_x, test_data_y)