import os
import json
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model_architecture import CustomTransformer
from config import SLMConfig

# ── Configuration & Tokenizer ──
cfg = SLMConfig()
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

# ── Load Model ──
def load_best_model():
    path = os.path.join(cfg.ckpt_dir, "best_model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
    
    print(f"Loading checkpoint: {path}")
    model = CustomTransformer(cfg)
    model = model.to(torch.bfloat16)
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ── Generation Function ──
@torch.no_grad()
def generate(
    model,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float  = 0.3,  # Low temp for deterministic reasoning
    top_k: int          = 40,
    top_p: float        = 0.90,
    repetition_penalty: float = 1.2,
    device: str         = "cpu",
):
    model = model.to(device)
    ids   = tokenizer.encode(prompt, add_special_tokens=False)
    x     = torch.tensor([ids], dtype=torch.long, device=device)

    start_time = time.perf_counter()
    tokens_generated = 0

    for _ in range(max_new_tokens):
        x_cond = x[:, -cfg.max_seq_len:]
        logits = model(x_cond)
        logits = logits[:, -1, :] 

        if repetition_penalty > 1.0:
            for tok in set(x[0].tolist()):
                if logits[0, tok] < 0:
                    logits[0, tok] *= repetition_penalty
                else:
                    logits[0, tok] /= repetition_penalty

        logits = logits / temperature

        if top_k > 0:
            top_k_vals = torch.topk(logits, top_k).values
            logits[logits < top_k_vals[:, -1:]] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_idx_remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_idx_remove] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

        probs     = F.softmax(logits, dim=-1)
        next_tok  = torch.multinomial(probs, num_samples=1)

        if next_tok.item() == tokenizer.eos_token_id:
            break

        x = torch.cat([x, next_tok], dim=1)
        tokens_generated += 1

    end_time = time.perf_counter()
    generation_time = end_time - start_time
    tok_per_sec = tokens_generated / generation_time if generation_time > 0 else 0

    output_ids = x[0, len(ids):].tolist()
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return generated_text, tokens_generated, tok_per_sec

# ── Reasoning Benchmark Suite ──
BENCHMARK_SUITE = {
    "Math & Arithmetic": [
        "If I have 5 apples and eat 2, how many apples do I have left?",
        "A train leaves New York at 3 PM traveling 60 mph. How far has it traveled by 5 PM?",
        "Solve this equation for x: 3x + 9 = 24. Step 1: Subtract 9 from both sides. Step 2:"
    ],
    "Logic & Deduction": [
        "All cats have tails. Fluffy is a cat. Therefore,",
        "If Alice is taller than Bob, and Bob is taller than Charlie, who is the shortest?",
        "There are three doors. Door 1 has a bear. Door 2 has a lion. Door 3 is safe. Which door should I open?"
    ],
    "Spatial & Physical": [
        "If you drop a fragile glass on a concrete floor, what is most likely to happen?",
        "Which is heavier: 1 pound of feathers or 1 pound of solid iron?",
        "If I leave an ice cube on the kitchen counter in the middle of summer, what will happen to it?"
    ],
    "Coding & Syntax": [
        "Write a Python function to calculate the factorial of a number.",
        "def is_even(num):\n    \"\"\"Returns True if num is even, else False\"\"\"",
        "List three popular programming languages:"
    ]
}

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing hardware: {device}")
    
    model = load_best_model()
    
    results = {
        "model_params": sum(p.numel() for p in model.parameters()),
        "device": device,
        "temperature": 0.3, # Hardcoded for this run
        "evaluations": {}
    }

    print("\n" + "="*60)
    print("🚀 Initiating Automated Reasoning Benchmark")
    print("="*60 + "\n")

    for category, prompts in BENCHMARK_SUITE.items():
        print(f"\n--- Testing Domain: {category} ---")
        results["evaluations"][category] = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n[Prompt {i+1}/{len(prompts)}]: {prompt}")
            
            output, tok_count, tps = generate(model, prompt, device=device)
            
            print(f"Output: {output}")
            print(f"Metrics: {tok_count} tokens @ {tps:.1f} tok/sec")
            
            results["evaluations"][category].append({
                "prompt": prompt,
                "response": output.strip(),
                "metrics": {
                    "tokens_generated": tok_count,
                    "tokens_per_second": round(tps, 2)
                }
            })

    # Save to disk
    output_file = "benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*60)
    print(f"✅ Benchmark complete! Results saved to {output_file}")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()