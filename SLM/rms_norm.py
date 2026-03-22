import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # A learnable scaling parameter, initialized to 1s
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate the root mean square and normalize the input
        # torch.rsqrt is the reciprocal square root (1 / sqrt(x))
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
        # Scale by the learnable weight
        return (norm * self.weight).to(x.dtype)
    
if __name__ == "__main__":
    # 1. Initialize with dim=5 because our vector has 5 elements
    rms = RMSNorm(dim=5)
    
    # 2. Convert the Python list to a PyTorch float tensor
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # 3. Run the forward pass
    output = rms(x)
    
    print("Input: ", x)
    print("Output:", output)