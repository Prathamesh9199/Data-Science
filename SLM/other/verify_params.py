from model_architecture import CustomTransformer
from config import SLMConfig

def count_parameters(model):
    # Sum up all parameters that require a gradient
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 1. Load your standard configuration
    cfg = SLMConfig()
    
    # 2. Initialize the full 24-layer model
    print(f"Initializing CustomTransformer with {cfg.n_layers} layers...")
    model = CustomTransformer(cfg)
    
    # 3. Calculate total params
    total_params = count_parameters(model)
    
    print("\n--- Parameter Tally ---")
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"Target Parameters:         ~209,000,000")
    
    # 4. Breakdown by component type for transparency
    emb_params = model.tok_emb.weight.numel()
    # Note: lm_head is tied to tok_emb, so it's only counted once by the loop
    
    # Calculate params for a single block
    one_block = model.blocks[0]
    block_params = sum(p.numel() for p in one_block.parameters())
    all_blocks_params = block_params * cfg.n_layers
    
    print(f"\n--- Component Breakdown ---")
    print(f"Embeddings (Tied):         {emb_params:,}")
    print(f"Total for 24 Blocks:       {all_blocks_params:,}")
    print(f"Per-Block Average:         {block_params:,}")
    
    diff = abs(total_params - 209_000_000)
    if diff < 1_000_000:
        print("\nVERIFICATION SUCCESS: Model matches the v5.0 Build Plan target.")
    else:
        print(f"\nNOTE: Model is at {total_params/1e6:.1f}M params. This is expected if 'd_model' or 'layers' were adjusted.")