import os
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
    max_new_tokens: int = 200,  # Increased default for chat
    temperature: float  = 0.4,  
    top_k: int          = 40,   
    top_p: float        = 0.90, 
    repetition_penalty: float = 1.2, 
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

    # Return only the newly generated tokens
    output_ids = x[0, len(ids):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)


# ── Interactive CLI Chatbot ──────────────────────────────────────────────────
def chat_interface():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing hardware: {device}\n")

    try:
        model = load_best_model()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    print("\n" + "="*60)
    print("🧠 Dual-Brain SLM Chat Interface Online")
    print("Type 'exit' or 'quit' to terminate the session.")
    print("Type 'clear' to reset the conversation memory.")
    print("="*60 + "\n")

    chat_history = ""

    while True:
        try:
            # 1. Get user input
            user_input = input("\nYou: ")
            
            # 2. Handle commands
            command = user_input.strip().lower()
            if command in ['exit', 'quit']:
                print("Shutting down sequence initiated. Goodbye!")
                break
            elif command == 'clear':
                chat_history = ""
                print("--- Conversation memory wiped ---")
                continue
            elif not command:
                continue

            # 3. Format prompt with context history
            prompt = f"{chat_history}User: {user_input}\nSLM: "

            print("SLM: ", end="", flush=True)

            # 4. Generate response
            output = generate(
                model, 
                prompt,
                max_new_tokens=250, 
                device=device,
            )
            
            # The generate function returns the decoded string (without the prompt)
            print(output)
            
            # 5. Update history for the next turn
            chat_history += f"User: {user_input}\nSLM: {output}\n"

            # 6. Prevent history from blowing past the 4096 context window
            # Roughly 3 chars per token, capping at ~8000 characters to be safe
            if len(chat_history) > 8000:
                # Keep the latter half of the conversation
                chat_history = chat_history[-4000:]

        except KeyboardInterrupt:
            print("\nShutting down sequence initiated. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred during generation: {e}")

if __name__ == "__main__":
    chat_interface()