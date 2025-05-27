import torch
import torch.nn as nn # building blocks for neural networks
import torch.nn.functional as F # access to functions like ReLU, sigmoid, etc.

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=31):
        """
        Initialize the Feedforward Neural Network.
        
        Args:
            input_dim (int): Input dimension (matches CNN output)
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Number of output weights (one per country)
        """
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_filters]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
        """
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        w_eps = self.out(x)  # Shape: [batch_size, output_dim]
        return w_eps
    
# equation (11) from project framework
def soft_normalize(weights):
    """
    Normalize allocation weights using L1 norm (sum of absolute values).
    weights: Tensor of shape [batch_size, 1]
    Returns: Normalized weights of shape [batch_size, 1]
    """
    l1_norm = torch.sum(torch.abs(weights), dim=0, keepdim=True) + 1e-8 # avoid division by zero
    normalized_weights = weights / l1_norm
    return normalized_weights
