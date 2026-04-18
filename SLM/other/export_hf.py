import torch
import os
from transformers import MistralConfig, MistralForCausalLM, AutoTokenizer
from config import SLMConfig

def convert_to_hf():
    print("🚀 Loading Custom Dual-Brain SLM Checkpoint...")
    cfg = SLMConfig()
    
    # 1. Define the equivalent Hugging Face Mistral Configuration
    hf_config = MistralConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.ffn_dim,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        rms_norm_eps=cfg.norm_eps,
        max_position_embeddings=cfg.max_seq_len,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=True, # You tied lm_head to tok_emb!
        architectures=["MistralForCausalLM"]
    )
    
    # 2. Initialize an empty HF model
    print("🏗️  Initializing Hugging Face Mistral Architecture...")
    hf_model = MistralForCausalLM(hf_config)
    
    # 3. Load your trained weights
    checkpoint_path = os.path.join(cfg.ckpt_dir, "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Remove '_orig_mod.' prefix if you used torch.compile
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
    
    hf_state_dict = {}
    
    print("🔄 Mapping custom state dictionary to Hugging Face keys...")
    # 4. Map the Base Weights
    hf_state_dict["model.embed_tokens.weight"] = state_dict["tok_emb.weight"]
    hf_state_dict["model.norm.weight"] = state_dict["norm.weight"]
    hf_state_dict["lm_head.weight"] = state_dict["lm_head.weight"]
    
    # 5. Map the Transformer Blocks
    for i in range(cfg.n_layers):
        # Layer Norms
        hf_state_dict[f"model.layers.{i}.input_layernorm.weight"] = state_dict[f"blocks.{i}.norm1.weight"]
        hf_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = state_dict[f"blocks.{i}.norm2.weight"]
        
        # Attention Projections
        hf_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = state_dict[f"blocks.{i}.attn.wq.weight"]
        hf_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = state_dict[f"blocks.{i}.attn.wk.weight"]
        hf_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = state_dict[f"blocks.{i}.attn.wv.weight"]
        hf_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = state_dict[f"blocks.{i}.attn.wo.weight"]
        
        # FFN (SwiGLU)
        hf_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = state_dict[f"blocks.{i}.ffn.gate.weight"]
        hf_state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = state_dict[f"blocks.{i}.ffn.up.weight"]
        hf_state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = state_dict[f"blocks.{i}.ffn.down.weight"]

    # 6. Load the mapped weights into the HF model
    # Note: strict=False is required here because standard Mistral doesn't use QK-Norm.
    # The script will intentionally ignore your 'q_norm' and 'k_norm' weights.
    hf_model.load_state_dict(hf_state_dict, strict=False)
    hf_model.bfloat16() # Keep it in bfloat16 as trained
    
    # 7. Save the model and your Mistral tokenizer
    out_dir = "dual-brain-slm-hf"
    print(f"\n💾 Saving Hugging Face model and tokenizer to ./{out_dir}")
    hf_model.save_pretrained(out_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    tokenizer.save_pretrained(out_dir)
    print("✅ Done! Ready for llama.cpp compilation.")

if __name__ == "__main__":
    convert_to_hf()