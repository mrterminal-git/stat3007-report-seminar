import torch
from PortfolioOptimizer import PortfolioOptimizer
from typing import Optional, Tuple, List, Dict
import torch.optim as optim
from models.BacktestSharpeEvaluator import BacktestSharpeEvaluator

class Trainer:
    """
    Manages training, validation, and evaluation of the PortfolioOptimizer model, including
    hyperparameter tuning via grid search and early stopping.
    """
    def __init__(
        self,
        optimizer: PortfolioOptimizer,
        train_data: torch.Tensor,
        train_returns: torch.Tensor,
        val_data: torch.Tensor,
        val_returns: torch.Tensor,
        lr: float = 0.001,
        num_epochs: int = 100,
        batch_size: int = 32,
        patience: Optional[int] = None,
        device: torch.device = None
    ):
        """
        Initialize the Trainer with the model and data.

        Args:
            optimizer (PortfolioOptimizer): The portfolio optimization model to train.
            train_data (torch.Tensor): Training input data with shape [num_samples, num_features, window_size].
            train_returns (torch.Tensor): Training next-day returns with shape [num_samples, num_countries].
            val_data (torch.Tensor): Validation input data with shape [num_samples, num_features, window_size].
            val_returns (torch.Tensor): Validation next-day returns with shape [num_samples, num_countries].
            lr (float): Learning rate for the optimizer (default: 0.001).
            num_epochs (int): Number of training epochs (default: 100).
            batch_size (int): Batch size for training (default: 32).
            patience (int, optional): Number of epochs to wait for improvement before early stopping (default: None).
            device (torch.device, optional): Device for computation; defaults to optimizer's device if None.
        """
        self.optimizer = optimizer  # Store PortfolioOptimizer instance
        self.train_data = train_data  # Store training input data
        self.train_returns = train_returns  # Store training returns
        self.val_data = val_data  # Store validation input data
        self.val_returns = val_returns  # Store validation returns
        self.lr = lr  # Store learning rate
        self.num_epochs = num_epochs  # Store number of epochs
        self.batch_size = batch_size  # Store batch size
        self.patience = patience  # Store early stopping patience
        self.device = device or optimizer.device  # Use optimizer's device if none specified
        self.evaluator = BacktestSharpeEvaluator()  # Initialize Sharpe ratio evaluator
        # Initialize Adam optimizer with model parameters
        self.torch_optimizer = optim.Adam(self.optimizer.get_parameters(), lr=self.lr)

    def sharpe_ratio_loss(self, returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
        """
        Compute the negative Sharpe ratio as the loss function for optimization.

        Args:
            returns (torch.Tensor): Portfolio returns with shape [batch_size].
            risk_free_rate (float): Risk-free rate (default: 0.0).

        Returns:
            torch.Tensor: Negative Sharpe ratio (scalar) to be minimized.
        """
        # Calculate excess returns
        excess_returns = returns - risk_free_rate
        # Compute mean and standard deviation of excess returns
        mean_excess = torch.mean(excess_returns)
        std_excess = torch.std(excess_returns, unbiased=False) + 1e-5  # Add epsilon to avoid division by zero
        # Calculate Sharpe ratio
        sharpe_ratio = mean_excess / std_excess
        # Return negative Sharpe ratio as loss (to maximize Sharpe ratio)
        return -sharpe_ratio

    def train_epoch(self) -> None:
        """
        Train the model for one epoch on the training data.
        """
        # Set models to training mode to enable dropout and batch normalization
        self.optimizer.cnn_model.train()
        if self.optimizer.use_transformer:
            self.optimizer.transformer_model.train()
        self.optimizer.ffn_model.train()

        # Iterate over batches of training data
        for batch_idx in range(0, len(self.train_data), self.batch_size):
            # Extract batch
            batch_end = min(batch_idx + self.batch_size, len(self.train_data))
            batch_inputs = self.train_data[batch_idx:batch_end]
            batch_returns = self.train_returns[batch_idx:batch_end]

            # Zero out gradients
            self.torch_optimizer.zero_grad()
            # Forward pass to get portfolio weights
            weights = self.optimizer.forward(batch_inputs)
            # Compute portfolio returns as weighted sum of asset returns
            portfolio_returns = torch.sum(weights * batch_returns, dim=1)
            # Calculate loss (negative Sharpe ratio)
            loss = self.sharpe_ratio_loss(portfolio_returns)
            # Backpropagate gradients
            loss.backward()
            # Update model parameters
            self.torch_optimizer.step()

    def evaluate(self, data: torch.Tensor, returns: torch.Tensor) -> float:
        """
        Evaluate the model on a dataset and compute the Sharpe ratio.

        Args:
            data (torch.Tensor): Input data with shape [num_samples, num_features, window_size].
            returns (torch.Tensor): Returns data with shape [num_samples, num_countries].

        Returns:
            float: Sharpe ratio computed on the dataset.
        """
        # Reset the evaluator to clear previous returns
        self.evaluator.reset()
        # Set models to evaluation mode
        self.optimizer.cnn_model.eval()
        if self.optimizer.use_transformer:
            self.optimizer.transformer_model.eval()
        self.optimizer.ffn_model.eval()

        # Disable gradient computation for evaluation
        with torch.no_grad():
            # Process data in batches
            for i in range(0, len(data), self.batch_size):
                batch_end = min(i + self.batch_size, len(data))
                batch_inputs = data[i:batch_end]
                batch_returns = returns[i:batch_end]
                # Get portfolio weights using inference mode
                weights = self.optimizer.infer(batch_inputs)
                # Compute portfolio returns
                portfolio_returns = torch.sum(weights * batch_returns, dim=1)
                # Store returns in evaluator
                for ret in portfolio_returns.cpu().numpy():
                    self.evaluator.add_return(ret)
        # Calculate and return Sharpe ratio
        return self.evaluator.calculate_sharpe()

    def train(self, verbose: bool = False) -> Tuple[float, List[float]]:
        """
        Train the model with early stopping based on validation Sharpe ratio.

        Args:
            verbose (bool): If True, print training progress every 10 epochs (default: False).

        Returns:
            Tuple[float, List[float]]: Best validation Sharpe ratio and corresponding portfolio returns.
        """
        best_sharpe = float('-inf')  # Track best Sharpe ratio
        best_returns = []  # Store portfolio returns for best Sharpe
        patience_counter = 0  # Counter for early stopping

        # Train for specified number of epochs
        for epoch in range(self.num_epochs):
            # Train for one epoch
            self.train_epoch()
            # Evaluate on validation set
            val_sharpe = self.evaluate(self.val_data, self.val_returns)

            # Print progress if verbose
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Validation Sharpe: {val_sharpe:.4f}")

            # Update best Sharpe and check for early stopping
            if val_sharpe > best_sharpe:
                best_sharpe = val_sharpe
                best_returns = self.evaluator.portfolio_returns.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if self.patience and patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        return best_sharpe, best_returns

    def grid_search(
        self,
        param_grid: Dict[str, List],
        verbose: bool = False
    ) -> Tuple[Dict, float, List[float]]:
        """
        Perform grid search over hyperparameters to find the best model configuration.

        Args:
            param_grid (Dict[str, List]): Dictionary with keys 'num_filters', 'filter_size', 'hidden_dim'
                                         and lists of values to try.
            verbose (bool): If True, print progress for each hyperparameter combination (default: False).

        Returns:
            Tuple[Dict, float, List[float]]: Best hyperparameters, best Sharpe ratio, and corresponding portfolio returns.
        """
        from itertools import product

        best_score = float('-inf')  # Track best Sharpe ratio
        best_params = None  # Store best hyperparameters
        best_returns = None  # Store best portfolio returns

        # Iterate over all combinations of hyperparameters
        for params in product(
            param_grid.get('num_filters', [self.optimizer.num_filters]),
            param_grid.get('filter_size', [self.optimizer.filter_size]),
            param_grid.get('hidden_dim', [self.optimizer.hidden_dim])
        ):
            num_filters, filter_size, hidden_dim = params
            print(f"Testing: num_filters={num_filters}, filter_size={filter_size}, hidden_dim={hidden_dim}")

            # Reinitialize optimizer with new hyperparameters
            self.optimizer = PortfolioOptimizer(
                window_size=self.optimizer.window_size,
                num_countries=self.optimizer.num_countries,
                num_weather_features=self.optimizer.num_weather_features,
                num_filters=num_filters,
                filter_size=filter_size,
                hidden_dim=hidden_dim,
                num_heads=self.optimizer.num_heads,
                use_transformer=self.optimizer.use_transformer,
                device=self.device
            )
            # Reinitialize optimizer with new model parameters
            self.torch_optimizer = optim.Adam(self.optimizer.get_parameters(), lr=self.lr)

            # Train and evaluate model
            sharpe, returns = self.train(verbose=verbose)
            # Update best results if current Sharpe is better
            if sharpe > best_score:
                best_score = sharpe
                best_params = {'num_filters': num_filters, 'filter_size': filter_size, 'hidden_dim': hidden_dim}
                best_returns = returns

        return best_params, best_score, best_returns

    def test(self, test_data: torch.Tensor, test_returns: torch.Tensor) -> Tuple[float, List[float]]:
        """
        Evaluate the trained model on a test dataset.

        Args:
            test_data (torch.Tensor): Test input data with shape [num_samples, num_features, window_size].
            test_returns (torch.Tensor): Test returns with shape [num_samples, num_countries].

        Returns:
            Tuple[float, List[float]]: Sharpe ratio and portfolio returns on the test set.
        """
        # Evaluate on test data using the same method as validation
        sharpe = self.evaluate(test_data, test_returns)
        return sharpe, self.evaluator.portfolio_returns