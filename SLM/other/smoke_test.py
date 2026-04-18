import torch
import torch.nn.functional as F
from model_architecture import CustomTransformer
from config import SLMConfig
from pretrain import build_optimizers

def run_final_smoke_test():
    print("🚀 Starting 36-Layer Deep Reasoning Smoke Test...\n")
    
    # ── 1. Injecting the 36-Layer Config ──
    cfg = SLMConfig()
    cfg.d_model  = 1024
    cfg.n_layers = 36    
    cfg.n_heads  = 16    
    cfg.n_kv_heads = 4
    cfg.ffn_dim  = 4096
    cfg.max_seq_len = 4096
    cfg.micro_batch = 2  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Building Deep Model ({cfg.n_layers} Layers, {cfg.d_model} Dim)...")
    model = CustomTransformer(cfg).to(device).to(torch.bfloat16)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model built! Total Parameters: {total_params / 1e6:.2f} M")

    print(f"Allocating dummy tensors (Micro-batch: {cfg.micro_batch}, Seq: {cfg.max_seq_len})...")
    dummy_x = torch.randint(0, cfg.vocab_size, (cfg.micro_batch, cfg.max_seq_len), device=device)
    dummy_y = torch.randint(0, cfg.vocab_size, (cfg.micro_batch, cfg.max_seq_len), device=device)
        
    print("Initializing Muon & 8-bit AdamW Optimizers...")
    opt_muon, opt_adam = build_optimizers(model)
    
    try:
        print("\n🔥 Running Forward Pass...")
        logits = model(dummy_x)
        
        print("🔥 Calculating Loss...")
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), dummy_y.view(-1))
        
        print("🔥 Running Backward Pass (Gradient calculation)...")
        loss.backward()
        
        print("🔥 Running Optimizer Steps...")
        opt_muon.step()
        opt_adam.step()
        opt_muon.zero_grad()
        opt_adam.zero_grad()
        
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\n🎉 SUCCESS! Your 8GB GPU handled the 36-Layer architecture.")
        print(f"📊 Peak VRAM usage: {peak_mem:.2f} GB / 8.00 GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print("\n💀 OOM ERROR: 36 Layers with a micro_batch of 2 is too much.")
            print("  Fix: Change `cfg.micro_batch = 1` in your config and try again.")
        else:
            print(f"\n❌ FAILED with error: {e}")

if __name__ == "__main__":
    run_final_smoke_test()