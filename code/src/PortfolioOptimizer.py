import torch
from typing import List
from models.FFN import FFN
from models.CNN import CNN
from models.Transformer import Transformer

class PortfolioOptimizer:
    """
    Manages the initialization and inference of a neural network model (CNN + optional Transformer + FFN)
    for portfolio optimization, generating weights to maximize the Sharpe ratio.
    """
    def __init__(
        self,
        window_size: int,
        num_countries: int,
        num_weather_features: int,
        num_filters: int,
        filter_size: int,
        hidden_dim: int,
        num_heads: int,
        use_transformer: bool,
        device: torch.device
    ):
        """
        Initialize the PortfolioOptimizer with model architecture parameters.

        Args:
            window_size (int): Number of days in the input window.
            num_countries (int): Number of countries (assets) in the portfolio.
            num_weather_features (int): Number of weather features per country.
            num_filters (int): Number of filters in the CNN.
            filter_size (int): Size of the convolutional filters in the CNN.
            hidden_dim (int): Hidden dimension for CNN output and FFN layers.
            num_heads (int): Number of attention heads in the Transformer.
            use_transformer (bool): Whether to include a Transformer layer between CNN and FFN.
            device (torch.device): Device for computation (e.g., 'cuda', 'cpu', or 'mps').
        """
        self.device = device  # Store computation device
        self.window_size = window_size  # Store window size
        self.num_countries = num_countries  # Store number of countries
        self.num_weather_features = num_weather_features  # Store number of weather features
        self.num_filters = num_filters  # Store number of CNN filters
        self.filter_size = filter_size  # Store CNN filter size
        self.hidden_dim = hidden_dim  # Store hidden dimension
        self.num_heads = num_heads  # Store number of Transformer heads
        self.use_transformer = use_transformer  # Store Transformer usage flag

        # Initialize neural network models
        self.cnn_model = self._initialize_cnn()  # CNN for feature extraction
        self.transformer_model = self._initialize_transformer() if use_transformer else None  # Optional Transformer
        self.ffn_model = self._initialize_ffn()  # FFN for weight generation

    def _initialize_cnn(self) -> CNN:
        """
        Initialize the CNN model for feature extraction.

        Returns:
            CNN: Initialized CNN model moved to the specified device.
        """
        return CNN(
            input_length=self.window_size,
            num_features=self.num_countries,
            num_weather_features=self.num_weather_features,
            num_filters=self.num_filters,
            num_classes=self.hidden_dim,
            filter_size=self.filter_size,
            use_transformer=self.use_transformer
        ).to(self.device)

    def _initialize_transformer(self) -> Transformer:
        """
        Initialize the Transformer model for sequence processing.

        Returns:
            Transformer: Initialized Transformer model moved to the specified device.
        """
        return Transformer(
            input_dim=self.num_filters,
            num_heads=self.num_heads
        ).to(self.device)

    def _initialize_ffn(self) -> FFN:
        """
        Initialize the Feed-Forward Network (FFN) for generating portfolio weights.

        Returns:
            FFN: Initialized FFN model moved to the specified device.
        """
        # Set input dimension based on whether Transformer is used
        ffn_input = self.num_filters if self.use_transformer else self.hidden_dim
        return FFN(
            input_dim=ffn_input,
            hidden_dim=self.hidden_dim,
            output_dim=self.num_countries
        ).to(self.device)

    def soft_normalize(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Softly normalize portfolio weights to be approximately in [-1, 1] and sum to 1.

        Args:
            weights (torch.Tensor): Raw weights of shape [batch_size, num_assets].

        Returns:
            torch.Tensor: Soft-normalized weights of shape [batch_size, num_assets].
        """
        # Squash to (-1, 1)
        squashed = torch.tanh(weights)

        # Normalize so the sum is 1 (signed sum, not absolute)
        sum_weights = torch.sum(squashed, dim=1, keepdim=True) + 1e-8
        normalized = squashed / sum_weights

        return normalized

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the model for training, retaining gradients.

        Args:
            inputs (torch.Tensor): Input data with shape [batch_size, num_features, window_size].

        Returns:
            torch.Tensor: Normalized portfolio weights with shape [batch_size, num_countries].
        """
        # Pass input through CNN
        cnn_output = self.cnn_model(inputs)
        # Pass through Transformer if enabled
        if self.use_transformer:
            transformer_output, _ = self.transformer_model(cnn_output)
            weights = self.ffn_model(transformer_output)
        else:
            weights = self.ffn_model(cnn_output)
        # Normalize weights
        return self.soft_normalize(weights)

    def infer(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass for inference (no gradients, evaluation mode).

        Args:
            inputs (torch.Tensor): Input data with shape [batch_size, num_features, window_size].

        Returns:
            torch.Tensor: Normalized portfolio weights with shape [batch_size, num_countries].
        """
        # Set models to evaluation mode
        self.cnn_model.eval()
        if self.use_transformer:
            self.transformer_model.eval()
        self.ffn_model.eval()

        # Disable gradient computation for inference
        with torch.no_grad():
            cnn_output = self.cnn_model(inputs)
            if self.use_transformer:
                transformer_output, _ = self.transformer_model(cnn_output)
                weights = self.ffn_model(transformer_output)
            else:
                weights = self.ffn_model(cnn_output)
        return self.soft_normalize(weights)

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get all model parameters for optimization.

        Returns:
            List[torch.nn.Parameter]: List of model parameters from CNN, Transformer (if used), and FFN.
        """
        params = list(self.cnn_model.parameters()) + list(self.ffn_model.parameters())
        if self.use_transformer:
            params += list(self.transformer_model.parameters())
        return params