import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SLMConfig

class SwiGLUFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Ensure ALL THREE are present
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up   = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        # The SwiGLU mathematical formula: 
        # 1. Project input through the gate and apply SiLU activation
        # 2. Project input through the up matrix (no activation)
        # 3. Multiply them together element-wise
        # 4. Project back down to the model dimension
        return self.down(F.silu(self.gate(x)) * self.up(x))

if __name__ == "__main__":
    # 1. Setup standard configuration
    cfg = SLMConfig()
    
    # 2. Initialize the SwiGLU network
    print(f"Building SwiGLUFFN (d_model={cfg.d_model}, ffn_dim={cfg.ffn_dim})...")
    ffn = SwiGLUFFN(cfg)
    
    # 3. Create dummy input data [Batch, Seq_Len, d_model]
    BATCH_SIZE = 2
    SEQ_LEN = 32
    x = torch.randn(BATCH_SIZE, SEQ_LEN, cfg.d_model)
    
    # 4. Run the forward pass
    print("Running forward pass...")
    output = ffn(x)
    
    # 5. Verify shapes
    print("\n--- Shape Verification ---")
    print(f"Input shape (x):      {list(x.shape)}")
    print(f"Output shape (out):   {list(output.shape)}")
    
    if x.shape == output.shape:
        print("\nSUCCESS! The SwiGLU FFN processed the data perfectly.")