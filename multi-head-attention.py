import torch
import math
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        print("Head Dimension: ", self.head_dim)

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
    

    def forward(self, X):
        batch_size, seq_len, _ = X.shape

        # Create Q, K and V
        Q = self.Wq(X)
        print("Q Shape:", Q.shape)
        K = self.Wk(X)
        V = self.Wv(X)

        # Split into heads
        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)
        print("Q Shape Split:", Q.shape)


        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)
        print("K Shape Split:", Q.shape)


        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Attension scores
        scores = Q @ K.transpose(-2, -1)
        print("Scores shape: ",scores.shape)

        # Scale
        scores = scores /  math.sqrt(self.head_dim)

        # Casual Masking
        mask = torch.tril(
            torch.ones(seq_len, seq_len)
        )

        scores = scores.masked_fill(mask==0, float("-inf"))

        # Attention weights
        weights = torch.softmax(scores, dim=-1)

        output = weights @ V

        # Combine heads
        output = output.transpose(1, 2)

        output = output.contiguous().view(
            batch_size,
            seq_len,
            self.d_model
        )

        print("Output", output.shape)

        # Final projection
        output = self.out_proj(output)
        return output


attention = MultiHeadAttention(d_model=12, num_heads=3)

X = torch.randn(4, 10, 12)

output = attention(X)
