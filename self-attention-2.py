import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.Wq = nn.Linear(32, 64)
        self.Wk = nn.Linear(32, 64)
        self.Wv = nn.Linear(32, 64)
    

    def forward(self, X):

        # Create Q, K, V
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        print(Q.shape)
        print(K.shape)
        print(V.shape)

        # Attention scores
        print(K.transpose(-2, -1))
        scores = Q @ K.transpose(-2, -1)

        # Scale
        d_k = Q.shape[-1]
        scores = scores / math.sqrt(d_k)
        print(scores.shape)


        # Casual Mask
        seq_len = X.shape[1]

        mask = torch.tril(
            torch.ones(seq_len, seq_len)
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )


        # Attention weights
        weights = torch.softmax(
            scores,
            dim=-1
        )

        # Weighted values
        output = weights @ V

        return output


attention = SelfAttention()

X = torch.tensor([
    [
        [1., 0., 1., 0.],
        [0., 1., 0., 1.],
        [1., 1., 1., 1.]
    ],

    [
        [1., 1., 0., 0.],
        [0., 0., 1., 1.],
        [1., 0., 1., 1.]
    ]
])

output = attention(X)
