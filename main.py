import torch
import math

Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

V = torch.tensor([
    [1.0, 0.0],
    [0.0, 2.0],
    [3.0, 1.0]
])

# 1. Similarity
scores = Q @ K.T

# 2. Scale
d_k = K.shape[-1]
scores = scores / math.sqrt(d_k)

# 3. Casual Masking
mask = torch.tril(torch.ones(3, 3))
scores = scores.masked_fill(mask==0, float("-inf"))

# 4. Convert scores -> probabilities
weights = torch.softmax(scores, dim=-1)

# 5. Weighted values
output = weights @ V

print(K.shape)
print(torch.tril(torch.ones(3, 3)))