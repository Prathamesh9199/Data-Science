import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model_architecture import CustomTransformer
from config import SLMConfig
import glob
import os

# ── Load config & tokenizer ──────────────────────────────────────────────────
cfg       = SLMConfig()
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

# ── Load latest checkpoint ───────────────────────────────────────────────────
def load_model(model_name: str):
    ckpts = sorted(glob.glob(os.path.join('base_models', model_name)))
    if not ckpts:
        raise FileNotFoundError("No checkpoint found in checkpoints/")
    path = ckpts[-1]
    print(f"Loading checkpoint: {path}")

    model = CustomTransformer(cfg)
    model = model.to(torch.bfloat16)  # <-- ADD THIS
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
    max_new_tokens: int = 200,
    temperature: float  = 0.8,
    top_k: int          = 50,
    top_p: float        = 0.95,
    device: str         = "cpu",
):
    model = model.to(device)
    ids   = tokenizer.encode(prompt, add_special_tokens=False)
    x     = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop context to max_seq_len
        x_cond = x[:, -cfg.max_seq_len:]

        logits = model(x_cond)              # [1, T, vocab]
        logits = logits[:, -1, :]           # last token only → [1, vocab]

        # Temperature
        logits = logits / temperature

        # Top-k
        if top_k > 0:
            top_k_vals = torch.topk(logits, top_k).values
            logits[logits < top_k_vals[:, -1:]] = float("-inf")

        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens once cumulative prob exceeds top_p
            sorted_idx_remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_idx_remove] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

        probs     = F.softmax(logits, dim=-1)
        next_tok  = torch.multinomial(probs, num_samples=1)

        # Stop at EOS
        if next_tok.item() == tokenizer.eos_token_id:
            break

        x = torch.cat([x, next_tok], dim=1)

    output_ids = x[0, len(ids):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    model = load_model("tiny_story.pt")

    while True:
        query = input("User: ")

        if 'quit' in query:
            break

        output = generate(
            model, query,
            max_new_tokens=150,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            device=device,
        )
        print(f"SLM: {output}")