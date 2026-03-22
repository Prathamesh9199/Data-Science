import torch

def precompute_freqs_cis(dim, seq_len, theta=500_000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)          # [seq_len, dim/2]
    cos   = torch.cos(freqs)               # [seq_len, dim/2]
    sin   = torch.sin(freqs)               # [seq_len, dim/2]
    return cos, sin                        # plain floats — no complex!

def _rotate_half(x):
    """Negate the second half and swap: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_emb(xq, xk, cos, sin):
    cos = cos.to(dtype=xq.dtype)
    sin = sin.to(dtype=xq.dtype)
    seq_len = xq.shape[1]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    xq_out = xq * cos + _rotate_half(xq) * sin
    xk_out = xk * cos + _rotate_half(xk) * sin
    return xq_out, xk_out
    

if __name__ == "__main__":
    DIM = 4
    SEQ_LEN = 2
    THETA = 10.0
    
    freqs_cis = precompute_freqs_cis(dim=DIM, seq_len=SEQ_LEN, theta=THETA)
    
    base_vector = [1.0, 2.0, 3.0, 4.0]
    
    # 4D Tensor: [Batch=1, Seq_Len=2, Num_Heads=1, Head_Dim=4]
    xq = torch.tensor([[
        [base_vector], # Token 0
        [base_vector]  # Token 1
    ]])
    xk = xq.clone()
    
    xq_rotated, xk_rotated = apply_rotary_emb(xq, xk, freqs_cis)
    
    print("--- Original Vectors ---")
    print(f"Token 0: {xq[0, 0, 0].tolist()}")
    print(f"Token 1: {xq[0, 1, 0].tolist()}\n")
    
    print("--- Rotated Vectors (RoPE Applied) ---")
    print(f"Token 0: {[round(num, 4) for num in xq_rotated[0, 0, 0].tolist()]}")
    print(f"Token 1: {[round(num, 4) for num in xq_rotated[0, 1, 0].tolist()]}")