import torch
import math
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
    

    def forward(self, X):
        batch_size, seq_len, _ = X.shape

        # Create Q, K and V
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        # Split into heads
        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Attension scores
        scores = Q @ K.transpose(-2, -1)

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

        # Final projection
        output = self.out_proj(output)
        return output


class TransfermerBlock(nn.Module):
    
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attention = MultiHeadAttention(d_model, num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )


    def forward(self, x):
        x = x + self.attention(self.norm1(x))

        x = x + self.ffn(self.norm1(x))

        return x


# Transformer
class Transformer(nn.Module):

    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransfermerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x
    
model = Transformer(
    d_model=64,
    num_heads=8,
    num_layers=12
)

x = torch.randn(4, 50, 64)

output = model(x)

print("Input :", x.shape)
print("Output:", output.shape)