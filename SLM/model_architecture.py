import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint

# Import our custom, verified modules
from rope import precompute_freqs_cis
from rms_norm import RMSNorm
from attention import GQAttention
from ffn import SwiGLUFFN
from config import SLMConfig

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn   = GQAttention(cfg)
        self.ffn    = SwiGLUFFN(cfg)
        
        # Pre-attention and Pre-FFN normalization
        self.norm1  = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm2  = RMSNorm(cfg.d_model, cfg.norm_eps)

    def _impl(self, x, freqs_cis):
        # The Residual Connection: x = x + Layer(Norm(x))
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.ffn(self.norm2(x))
        return x

    def forward(self, x, freqs_cis):
        # Gradient Checkpointing: Recomputes activations during the backward 
        # pass instead of storing them in VRAM. 
        return checkpoint(self._impl, x, freqs_cis, use_reentrant=False)


class CustomTransformer(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg     = cfg or SLMConfig()
        
        # 1. Token Embeddings
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        
        # 2. Stack of 24 Transformer Blocks
        self.blocks  = nn.ModuleList([TransformerBlock(self.cfg) for _ in range(self.cfg.n_layers)])
        
        # 3. Final Output Normalization
        self.norm    = RMSNorm(self.cfg.d_model, self.cfg.norm_eps)
        
        # 4. Language Model Head (Vocab Predictor)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        
        # Tied Embeddings: Share weights between input and output
        self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE frequencies once for the whole model
        cos, sin = precompute_freqs_cis(
            self.cfg.d_model // self.cfg.n_heads,
            self.cfg.max_seq_len,
            self.cfg.rope_theta
        )
        self.register_buffer("freqs_cos", cos)
        self.register_buffer("freqs_sin", sin)
                
        self._init_weights()

    def _init_weights(self):
        """Depth-scaled initialization to keep the residual stream stable."""
        std_proj = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                is_output_proj = name.endswith(('wo', 'down'))
                std = std_proj if is_output_proj else 0.02
                nn.init.normal_(m.weight, mean=0.0, std=std)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # 1. Look up token vectors
        x = self.tok_emb(x)
        
        # 2. Slice RoPE frequencies to match the current sequence length
        freqs = (self.freqs_cos[:x.size(1)], self.freqs_sin[:x.size(1)])
        
        # 3. Pass through all 24 blocks
        for block in self.blocks:
            x = block(x, freqs)
            
        # 4. Normalize and predict the next token
        return self.lm_head(self.norm(x))