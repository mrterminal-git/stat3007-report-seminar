import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    def __init__(self, input_dim=8, num_heads=4):
        """
        Initialize a single-layer Transformer with multi-head attention.

        Args:
            input_dim (int): Input dimension (D, number of filters from CNN, e.g. 8)
            num_heads (int): Number of attention heads (H=4)
        """

        super(Transformer, self).__init__() 
        assert input_dim % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.input_dim = input_dim
        self.d_per_head = input_dim // num_heads # Dimension per head (D/H), multi-head attention
        # Why multi-head?
        # Intuition: Imagine 4 "experts" (heads) analyzing the sequence, each looking at a 2-dimensional slice of the 8-dimensional features 

        # Linear projections for Queries, Keys and Values (Eq 6.)
        self.W_q = nn.Linear(input_dim, input_dim) # WQ: D -> D
        self.W_k = nn.Linear(input_dim, input_dim) # WK: D -> D
        self.W_v = nn.Linear(input_dim, input_dim) # WV: D -> D

        # Output projection after concatenation (combines the results from all heads into a unified representation)
        self.W_o = nn.Linear(input_dim, input_dim)

        self.scale = math.sqrt(self.d_per_head) # Scaling factor for attention

    def forward(self, x):
        """
        Forward pass through the Transformer layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, input_dim] (h_L from Eq. 9, final timestep)
        """
        batch_size, seq_len, input_dim = x.size()

        # Linear projections (Eq. 6: Q, K, V), Intuitions: 
        Q = self.W_q(x) # [batch_size, seq_len, input_dim] QUERIES (Q): Represent what the model is "looking for" at each time step ("e.g. Which days predict tomorrow's returns?")
        K = self.W_k(x) # [batch_size, seq_len, input_dim] KEYS (K): Represent the "identity/context" of each time step in the sequence ("e.g. I'm day 25, here's my signature")
        V = self.W_v(x) # [batch_size, seq_len, input_dim] VALUES (V): Contain the actual information to be weighted and extracted ("e.g. Here's my feature vector for day 25")

        # Reshape for multi-head attention
        # Each head processes a smaller portion of the input_dim dimension
        # With num_heads=4 and input_dim=8, each head handles input_dim // num_head = 8 // 4 = 2 dimensions
        # This allows each of the 4 heads to focus on different aspects of the 8-dimensional feature vector

        # [batch_size, seq_len, num_heads, input_dim] -> [batch_size, num_heads, seq_len, d_per_head]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_per_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_per_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_per_head).transpose(1, 2)

        # Scaled dot-product attention (Eq. 7)

        # Each score represents how much each time step (day) should attend to every other time step in the sequence for each head
        # Intuition: The scores are like a "relevance map", showing how much each day (e.g. day 25) should influence every other day (e.g. 28) for each head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale # [batch_size, num_heads, seq_len, d_per_head] x [batch_size, num_heads, d_per_head, seq_len] -> [batch_size, num_heads, seq_len, seq_len]
        attn_weights = torch.softmax(scores, dim=-1) # [batch_size, num_heads, seq_len, seq_len], normalize each score/weight (sum to 1)
        # Intuition: If day 25's residuals strongly predict day 28's returns, attn_weights[:, :, 28, 25] will be high, meaning the model "pays attention" to day 25 when processing day 28

        # Multiply the weights with the values to weight the values by their relevance
        # Intuition: This produces a new representation for each time step, combining information from all days, with more weight given to "important" days (based on attention weights)
        attn_out = torch.matmul(attn_weights, V) # [batch_size, num_heads, seq_len, seq_len] x [batch_size, num_heads, seq_len, d_per_head] -> [batch_size, num_heads, seq_len, d_per_head]

        # Reshape and project (concatenate heads)
        #print("Attention out shape:", attn_out.shape)
        # Intuition: Combines the 4 heads' outputs into a single 8-dimensional vector per time step, integrating their different perspectives
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, input_dim) # [batch_size, num_heads, seq_len, d_per_head] -> [batch_size, seq_len, num_heads, d_per_head] -> [batch_size, seq_len, num_heads x d_per_head = input_dim]
        #print("Attention out shape:", attn_out.shape)

        # Intuition: "Refines" the attention output, combining information from all heads into one representation
        output = self.W_o(attn_out) # [batch_size, seq_len, input_dim]

        # Extract the final timestep (Eq. 9)
        # Intuition: The final timestep summarizes the entire sequence (28 days) into a single 8-dimensional vector, capturing the most relevant information for prediciting portfolio weights
        final_out = output[:, -1, :] # [batch_size, input_dim]
        return final_out, attn_weights