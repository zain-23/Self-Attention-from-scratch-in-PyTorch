import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.Wq = nn.Linear(4, 4)
        self.Wk = nn.Linear(4, 4)
        self.Wv = nn.Linear(4, 4)

    def forward(self, X):
        # Create Q, K and V
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        # Attention scores
        scores = Q @ K.T

        # Scale
        d_k = Q.shape[-1]
        scores = scores / math.sqrt(d_k)

        # Casual mask
        mask = torch.tril(
            torch.ones(X.shape[0], X.shape[0])
        )

        scores = scores.masked_fill(mask==0, float("-inf"))


        # Attention weights
        weights = torch.softmax(scores, dim=-1)

        weighted_output = weights @ V

        return weighted_output
    


attention = SelfAttention()

X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0]
])

output = attention(X)

print(output)
print(output.shape)