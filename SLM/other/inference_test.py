import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model_architecture import CustomTransformer
from config import SLMConfig

# ── Load config & tokenizer ──────────────────────────────────────────────────
cfg       = SLMConfig()
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

# ── Load Best Model ──────────────────────────────────────────────────────────
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

# ── Generation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(
    model,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float  = 0.4,  # Lowered from 0.7 to 0.4 for strict logic
    top_k: int          = 40,   # Tightened from 50
    top_p: float        = 0.90, # Tightened from 0.95
    repetition_penalty: float = 1.2, # NEW: Penalize repeated tokens
    device: str         = "cpu",
):
    model = model.to(device)
    ids   = tokenizer.encode(prompt, add_special_tokens=False)
    x     = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -cfg.max_seq_len:]
        logits = model(x_cond)
        logits = logits[:, -1, :] # [1, vocab]

        # ── Apply Repetition Penalty ──
        if repetition_penalty > 1.0:
            for tok in set(x[0].tolist()):
                if logits[0, tok] < 0:
                    logits[0, tok] *= repetition_penalty
                else:
                    logits[0, tok] /= repetition_penalty

        # Apply Temperature
        logits = logits / temperature

        # Top-k
        if top_k > 0:
            top_k_vals = torch.topk(logits, top_k).values
            logits[logits < top_k_vals[:, -1:]] = float("-inf")

        # Top-p (nucleus)
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

    output_ids = x[0, len(ids):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)


# ── Reasoning Benchmark Suite ────────────────────────────────────────────────
REASONING_PROMPTS = [
    # Basic Math
    "If I have 5 apples and eat 2, how many apples do I have left?",
    "Solve for x: 2x + 5 = 15. x = ",
    "What is the square root of 64?",
    "A train leaves New York at 3 PM traveling 60 mph. How far has it traveled by 5 PM?",
    
    # Logic & Deductive Reasoning
    "All cats have tails. Fluffy is a cat. Does Fluffy have a tail?",
    "If A is taller than B, and B is taller than C, who is the tallest?",
    "Which is heavier: 1 pound of feathers or 1 pound of bricks?",
    "If you drop a glass of water on the floor, what happens?",
    
    # Pattern Recognition
    "What is the next number in the sequence: 2, 4, 6, 8, ",
    "Complete the pattern: A, C, E, G, ",
    
    # Coding
    "Write a Python function to calculate the factorial of a number.",
    "def is_even(num):\n    \"\"\"Returns True if num is even, else False\"\"\"",
    "Write a for loop in Python that prints numbers from 1 to 10.",
    
    # Common Sense & Knowledge
    "If it is raining outside, what should I take with me to stay dry?",
    "The capital of France is",
    "List three primary colors.",
    "Can a normal human jump over a two-story house?",
    "If I put a glass of water in the freezer, what happens to the water?",
    "Explain the theory of relativity in one simple sentence.",
    "Translate 'Hello, how are you?' to Spanish."
]

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    model = load_best_model()
    
    results = []
    output_file = "reasoning_results_v3.json"

    print(f"Starting reasoning benchmark ({len(REASONING_PROMPTS)} prompts)...\n")
    print("-" * 60)

    for i, prompt in enumerate(REASONING_PROMPTS, 1):
        print(f"Prompt {i}/{len(REASONING_PROMPTS)}: {prompt}")
        
        output = generate(
            model, 
            prompt,
            max_new_tokens=100, 
            # We are relying on the defaults set in the function definition now
            device=device,
        )
        
        print(f"SLM Output:\n{output}")
        print("-" * 60)
        
        results.append({
            "prompt": prompt,
            "output": output
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Benchmark complete! Results saved to {output_file}")