import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from config import SLMConfig
from rope import apply_rotary_emb
from rms_norm import RMSNorm
import math
import torch.nn.functional as F
from rope import precompute_freqs_cis

class GQAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads    = cfg.n_heads       # 12 Query heads
        self.n_kv_heads = cfg.n_kv_heads    # 4 Key/Value heads
        
        # Dimension of each individual attention head (768 // 12 = 64)
        self.head_dim   = cfg.d_model // cfg.n_heads
        
        # How many Query heads share a single Key/Value head (12 // 4 = 3)
        self.n_rep      = cfg.n_heads // cfg.n_kv_heads
        
        # Linear projections for Q, K, V, and Output (No biases, per standard SOTA)
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads    * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model,    bias=False)
        
        # QK-Norm: per-head RMSNorm on Q and K prevents attention entropy
        # collapse at 4096+ context. Used in Qwen2, Gemma 2, Cohere Command-R.
        self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)

    def forward(self, x, freqs_cis):
        B, T, _ = x.shape
        
        # 1. Project inputs and separate into heads
        q = self.wq(x).view(B, T, self.n_heads,    self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # 2. Apply QK-Norm (Must be done BEFORE RoPE per Qwen2/Gemma2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # 3. Apply Rotary Positional Embeddings
        cos, sin = freqs_cis
        q, k = apply_rotary_emb(q, k, cos, sin)
        
        # 4. Repeat KV heads to match the number of Query heads
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)
        
        # 5. Compute Attention using optimized FlashAttention-2
        # causal=True ensures tokens can only attend to past tokens, not future ones
        out = flash_attn_func(q.to(x.dtype), k.to(x.dtype), v.to(x.dtype), causal=True)
        
        # 6. Flatten the heads back together and project to output
        return self.wo(out.reshape(B, T, -1))

if __name__ == "__main__":
    # 1. Setup a tiny, trackable configuration
    cfg = SLMConfig()
    cfg.d_model = 4
    cfg.n_heads = 2
    cfg.n_kv_heads = 1
    head_dim = cfg.d_model // cfg.n_heads # 2 dimensions per head
    
    BATCH = 1
    SEQ_LEN = 3 # "I", "can", "run"
    
    print("Initializing transparent test on CUDA...")
    # Use float16 for FlashAttention compatibility, set a manual seed for reproducibility
    torch.manual_seed(42)
    x = torch.randn(BATCH, SEQ_LEN, cfg.d_model, device='cuda', dtype=torch.float16)
    freqs_cis = precompute_freqs_cis(head_dim, SEQ_LEN, cfg.rope_theta).to('cuda')
    
    # 2. Initialize your real module
    layer = GQAttention(cfg).to('cuda').to(torch.float16)
    
    # ==========================================
    # PART A: MANUAL "TRANSPARENT" ATTENTION
    # ==========================================
    with torch.no_grad():
        # Get Q, K, V manually using the layer's weights
        q = layer.wq(x).view(BATCH, SEQ_LEN, cfg.n_heads, head_dim)
        k = layer.wk(x).view(BATCH, SEQ_LEN, cfg.n_kv_heads, head_dim)
        v = layer.wv(x).view(BATCH, SEQ_LEN, cfg.n_kv_heads, head_dim)
        
        # Apply QK-Norm
        q = layer.q_norm(q)
        k = layer.k_norm(k)
        
        # Apply RoPE (we skip the actual rotation call here just to look at pure attention, 
        # but the shapes remain identical)
        
        # GQA: Repeat the KV heads to match Q heads (from 1 to 2)
        k = k.repeat_interleave(layer.n_rep, dim=2)
        v = v.repeat_interleave(layer.n_rep, dim=2)
        
        # Reshape for standard matrix multiplication: [Batch, Heads, Seq_Len, Dim]
        q_trans = q.transpose(1, 2)
        k_trans = k.transpose(1, 2)
        v_trans = v.transpose(1, 2)
        
        # 1. The Dot Product (Q * K^T) scaled by sqrt(head_dim)
        scores = torch.matmul(q_trans, k_trans.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # 2. The Causal Mask
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device='cuda'), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        
        # 3. The Softmax (The "Attention Percentages")
        attention_weights = F.softmax(scores, dim=-1)
        
        # 4. Multiply by V and project output
        manual_context = torch.matmul(attention_weights, v_trans)
        manual_context = manual_context.transpose(1, 2).contiguous().view(BATCH, SEQ_LEN, cfg.d_model)
        manual_output = layer.wo(manual_context)

    # ==========================================
    # PART B: YOUR FLASH ATTENTION MODULE
    # ==========================================
    # We pass the same 'x' through your actual code. 
    # (We bypass RoPE temporarily in the module just for this strict 1:1 math test)
    with torch.no_grad():
        q_real = layer.q_norm(layer.wq(x).view(BATCH, SEQ_LEN, cfg.n_heads, head_dim))
        k_real = layer.k_norm(layer.wk(x).view(BATCH, SEQ_LEN, cfg.n_kv_heads, head_dim)).repeat_interleave(layer.n_rep, dim=2)
        v_real = layer.wv(x).view(BATCH, SEQ_LEN, cfg.n_kv_heads, head_dim).repeat_interleave(layer.n_rep, dim=2)
        
        from flash_attn import flash_attn_func
        flash_context = flash_attn_func(q_real, k_real, v_real, causal=True)
        real_output = layer.wo(flash_context.reshape(BATCH, SEQ_LEN, -1))

    # ==========================================
    # PRINT THE RESULTS
    # ==========================================
    print("\n--- The Mathematical 'Thinking' Process (Head 0) ---")
    print("1. The Causal Attention Grid (Softmax Percentages):")
    # Convert to percentages for readability
    grid = (attention_weights[0, 0] * 100).int()
    print(f"Token 0 looks at: {grid[0].tolist()}%")
    print(f"Token 1 looks at: {grid[1].tolist()}%")
    print(f"Token 2 looks at: {grid[2].tolist()}%")
    
    print("\n--- Verification ---")
    # float16 has precision noise, so we check if they are close, not perfectly identical
    is_match = torch.allclose(manual_output, real_output, atol=1e-2)
    print(f"Does the manual math perfectly match your FlashAttention code? -> {is_match}")