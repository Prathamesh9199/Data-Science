# sft.py
# Supervised Fine-Tuning on reasoning data
# Reuses: model architecture, optimizers, LR schedule, thermal management,
#         checkpointing, early stopping, CSV logging — all from pretrain.py

import os
import gc
import csv
import json
import math
import glob
import time
import torch
import subprocess
import numpy as np
import bitsandbytes as bnb
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model_architecture import CustomTransformer
from config import SLMConfig
from dataclasses import dataclass

# ── Config ────────────────────────────────────────────────────────────────────

cfg = SLMConfig()

@dataclass
class SFTConfig:
    # Data
    train_file      : str   = "data/sft/master/sft_train.jsonl"
    val_file        : str   = "data/sft/master/sft_val.jsonl"

    # Model
    base_ckpt       : str   = "checkpoints/best_model.pt"   # pretrained weights
    sft_ckpt_dir    : str   = "checkpoints/sft"
    sft_log_file    : str   = "sft_training_log.csv"

    # Sequence
    max_seq_len     : int   = 1024   # SFT examples are shorter than pretraining
                                     # avg_sol=88w → ~130 tokens, problem ~50 tokens
                                     # 1024 covers 99%+ of our data

    # Training
    epochs          : int   = 1      # SFT is short — 3 epochs over 90K examples
    micro_batch     : int   = 2      # sequences per GPU step
    grad_accum      : int   = 16     # effective batch = 2 × 16 = 32 sequences
    grad_clip       : float = 1.0

    # Learning rate — much lower than pretraining to avoid catastrophic forgetting
    lr_muon         : float = 5e-4   # 100x lower than pretrain (was 0.02)
    lr_adam         : float = 5e-5   # 10x lower than pretrain (was 3e-4)
    lr_min_ratio    : float = 0.1
    warmup_steps    : int   = 50     # short warmup — model already trained
    weight_decay    : float = 0.01   # lighter than pretrain (was 0.1)

    # Checkpointing
    save_every      : int   = 200
    keep_ckpts      : int   = 3
    val_every       : int   = 100
    log_every       : int   = 10

    # Early stopping
    early_stopping_patience  : int   = 10
    early_stopping_tolerance : float = 0.02

    # Thermal
    gpu_temp_threshold : int = 78
    cooling_break_s    : int = 600
    break_every_steps  : int = 1000

sft_cfg = SFTConfig()


# ── GPU Thermal (reused from pretrain.py verbatim) ────────────────────────────

def get_gpu_temp():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            encoding="utf-8"
        ).strip()
        return int(out)
    except Exception:
        return 0

def gpu_cooling_break(temp):
    if temp >= sft_cfg.gpu_temp_threshold:
        print(f"\n⚠ GPU at {temp}°C — cooling break ({sft_cfg.cooling_break_s // 60} min)...")
        time.sleep(sft_cfg.cooling_break_s)
        print("  Resuming.")
        return True
    return False


# ── Learning Rate Schedule ─────────────────────────────────────────────────────
# SFT uses warmup → cosine only. No stable plateau needed —
# dataset is small enough that we want decay to start earlier.

def get_lr(step, total_steps, peak_lr):
    if step < sft_cfg.warmup_steps:
        return peak_lr * (step + 1) / sft_cfg.warmup_steps
    progress = (step - sft_cfg.warmup_steps) / max(1, total_steps - sft_cfg.warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (sft_cfg.lr_min_ratio + (1 - sft_cfg.lr_min_ratio) * cosine)

def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


# ── Muon Optimizer (reused from pretrain.py verbatim) ─────────────────────────

def zeropower_via_newtonschulz(G, steps=5):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * A @ X + c * A @ A @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr       = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g     = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g_nesterov = g + momentum * buf
                if g_nesterov.ndim == 2:
                    update = zeropower_via_newtonschulz(g_nesterov, steps=ns_steps)
                    update = update * max(1, g_nesterov.size(0) / g_nesterov.size(1)) ** 0.5
                else:
                    update = g_nesterov
                p.add_(update, alpha=-lr)

def build_optimizers(model):
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and any(k in name for k in ["wq", "wk", "wv", "wo",
                                                     "gate", "up", "down"]):
            muon_params.append(p)
        else:
            adam_params.append(p)
    opt_muon = Muon(muon_params, lr=sft_cfg.lr_muon, momentum=0.95)
    opt_adam = bnb.optim.AdamW8bit(
        adam_params, lr=sft_cfg.lr_adam,
        betas=(0.9, 0.95), weight_decay=sft_cfg.weight_decay, eps=1e-8
    )
    print(f"  Muon params : {sum(p.numel() for p in muon_params)/1e6:.1f}M")
    print(f"  AdamW params: {sum(p.numel() for p in adam_params)/1e6:.1f}M")
    return opt_muon, opt_adam


# ── SFT Dataset ───────────────────────────────────────────────────────────────
# This is the core difference from pretraining.
# Pretraining: random token windows, loss on every token.
# SFT: structured problem+solution pairs, loss ONLY on solution tokens.

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.examples    = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        print(f"  Loaded {len(self.examples):,} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Format: "Problem: {problem}\n\nSolution:\n{solution}"
        # The model sees the full text but only trains on the solution part.
        problem_text  = f"Problem: {ex['problem']}\n\nSolution:\n"
        solution_text = ex["solution"]
        full_text     = problem_text + solution_text

        # Tokenize full text
        full_ids = self.tokenizer.encode(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len - 1
        )
        full_ids = full_ids + [self.tokenizer.eos_token_id]

        # Tokenize just the problem prefix to find where solution starts
        prefix_ids = self.tokenizer.encode(
            problem_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len
        )
        prefix_len = len(prefix_ids)

        # Build input_ids and labels
        # labels = -100 for all problem tokens (masked from loss)
        # labels = token_id for all solution tokens (trained on)
        input_ids = full_ids
        labels = [-100] * prefix_len + full_ids[prefix_len:]

        # Truncate to max_seq_len
        input_ids = input_ids[:self.max_seq_len]
        labels    = labels[:self.max_seq_len]

        # Verify at least some solution tokens survived truncation
        # If the solution was entirely truncated, skip this example
        # (handled by collate_fn returning None check)
        n_solution_tokens = sum(1 for l in labels if l != -100)

        return {
            "input_ids"        : input_ids,
            "labels"           : labels,
            "n_solution_tokens": n_solution_tokens,
        }


def collate_fn(batch, pad_token_id):
    """
    Pad all sequences in the batch to the same length.
    Labels are padded with -100 so padding never contributes to loss.
    Filters out examples where no solution tokens survived truncation.
    """
    # Filter examples with no solution tokens
    batch = [b for b in batch if b["n_solution_tokens"] > 0]
    if not batch:
        return None

    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids_padded = []
    labels_padded    = []
    attention_masks  = []

    for b in batch:
        seq_len  = len(b["input_ids"])
        pad_len  = max_len - seq_len

        input_ids_padded.append(b["input_ids"] + [pad_token_id] * pad_len)
        labels_padded.append(b["labels"]    + [-100]          * pad_len)
        attention_masks.append([1] * seq_len + [0] * pad_len)

    return {
        "input_ids"      : torch.tensor(input_ids_padded, dtype=torch.long),
        "labels"         : torch.tensor(labels_padded,    dtype=torch.long),
        "attention_mask" : torch.tensor(attention_masks,  dtype=torch.long),
    }


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_base_model(device):
    """Load pretrained weights, strip torch.compile prefix."""
    print(f"  Loading base model from {sft_cfg.base_ckpt}...")
    model = CustomTransformer(cfg).to(device).to(torch.bfloat16)

    ckpt  = torch.load(sft_cfg.base_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
             for k, v in state.items()}
    model.load_state_dict(state)

    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total/1e6:.1f}M")
    return model


# ── Checkpointing (reused pattern from pretrain.py) ───────────────────────────

def save_checkpoint(step, epoch, model, opt_muon, opt_adam, loss_history):
    os.makedirs(sft_cfg.sft_ckpt_dir, exist_ok=True)
    path = os.path.join(sft_cfg.sft_ckpt_dir, f"sft_ckpt_{step:05d}.pt")
    torch.save({
        "step"        : step,
        "epoch"       : epoch,
        "model"       : model.state_dict(),
        "opt_muon"    : opt_muon.state_dict(),
        "opt_adam"    : opt_adam.state_dict(),
        "loss_history": loss_history,
    }, path)
    print(f"  ✓ SFT checkpoint saved → {path}")
    _prune_checkpoints()

def save_best_model(step, val_loss, model):
    os.makedirs(sft_cfg.sft_ckpt_dir, exist_ok=True)
    path = os.path.join(sft_cfg.sft_ckpt_dir, "sft_best_model.pt")
    torch.save({
        "step"    : step,
        "val_loss": val_loss,
        "model"   : model.state_dict(),
    }, path)
    print(f"  ★ SFT best model saved → {path}  (val_loss={val_loss:.4f} @ step {step})")

def _prune_checkpoints():
    ckpts = glob.glob(os.path.join(sft_cfg.sft_ckpt_dir, "sft_ckpt_*.pt"))
    ckpts.sort(key=os.path.getmtime)
    for old in ckpts[:-sft_cfg.keep_ckpts]:
        os.remove(old)
        print(f"  ✗ Deleted old SFT checkpoint: {old}")

def load_latest_sft_checkpoint(model, opt_muon, opt_adam):
    ckpts = glob.glob(os.path.join(sft_cfg.sft_ckpt_dir, "sft_ckpt_*.pt"))
    if not ckpts:
        print("  No SFT checkpoint found — starting from base model.")
        return 0, 0, []
    ckpts.sort(key=os.path.getmtime)
    path = ckpts[-1]
    print(f"  Resuming SFT from {path}")
    ckpt = torch.load(path, map_location="cuda", weights_only=False)
    state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
             for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    opt_muon.load_state_dict(ckpt["opt_muon"])
    opt_adam.load_state_dict(ckpt["opt_adam"])
    return ckpt["step"], ckpt.get("epoch", 0), ckpt.get("loss_history", [])

def load_best_sft_val_loss():
    path = os.path.join(sft_cfg.sft_ckpt_dir, "sft_best_model.pt")
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("val_loss", float("inf"))
    return float("inf")


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_val_loss(model, val_loader, device):
    """
    Compute mean loss over validation set.
    Loss is computed only on solution tokens — same masking as training.
    """
    model.eval()
    losses = []
    try:
        for batch in val_loader:
            if batch is None:
                continue
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                # Shift: predict token i+1 from position i
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, cfg.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100   # masked problem tokens ignored
                )
            losses.append(loss.item())

        return sum(losses) / len(losses) if losses else float("nan")
    finally:
        model.train()


# ── CSV Logging (reused pattern) ──────────────────────────────────────────────

def init_log(resume):
    if not resume or not os.path.exists(sft_cfg.sft_log_file):
        with open(sft_cfg.sft_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "train_loss", "val_loss", "ppl",
                             "lr_muon", "lr_adam", "elapsed_s",
                             "best_val_loss", "patience_counter"])
    return open(sft_cfg.sft_log_file, "a", newline="")

def write_log(log_f, step, epoch, train_loss, val_loss, ppl,
              lr_m, lr_a, elapsed, best_val_loss, patience_counter):
    writer   = csv.writer(log_f)
    val_str  = f"{val_loss:.4f}" if not math.isnan(val_loss) else "n/a"
    best_str = f"{best_val_loss:.4f}" if best_val_loss < float("inf") else "n/a"
    writer.writerow([step, epoch, f"{train_loss:.4f}", val_str, f"{ppl:.2f}",
                     f"{lr_m:.2e}", f"{lr_a:.2e}", f"{elapsed:.1f}",
                     best_str, patience_counter])
    log_f.flush()


# ── Early Stopping (reused verbatim from pretrain.py) ─────────────────────────

class EarlyStopping:
    def __init__(self, patience=10, tolerance=0.02):
        self.patience    = patience
        self.tolerance   = tolerance
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    def update(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            return True
        elif val_loss <= self.best_loss + self.tolerance:
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def status(self):
        return f"strikes {self.counter}/{self.patience} | best_val={self.best_loss:.4f}"


# ── Main SFT Training Loop ────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SLM Supervised Fine-Tuning")
    print(f"  Device : {device}")
    print(f"  dtype  : bfloat16")
    print(f"{'='*60}\n")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Datasets + DataLoaders ─────────────────────────────────────────────────
    print("  Building datasets...")
    train_dataset = SFTDataset(sft_cfg.train_file, tokenizer, sft_cfg.max_seq_len)
    val_dataset   = SFTDataset(sft_cfg.val_file,   tokenizer, sft_cfg.max_seq_len)

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_dataset,
        batch_size  = sft_cfg.micro_batch,
        shuffle     = True,
        collate_fn  = lambda b: collate_fn(b, pad_id),
        num_workers = 2,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = sft_cfg.micro_batch * 2,   # can use larger batch for val
        shuffle     = False,
        collate_fn  = lambda b: collate_fn(b, pad_id),
        num_workers = 2,
        pin_memory  = True,
    )

    steps_per_epoch  = math.ceil(len(train_dataset) / (sft_cfg.micro_batch * sft_cfg.grad_accum))
    total_steps      = steps_per_epoch * sft_cfg.epochs

    print(f"\n  Train examples   : {len(train_dataset):,}")
    print(f"  Val examples     : {len(val_dataset):,}")
    print(f"  Micro batch      : {sft_cfg.micro_batch}")
    print(f"  Grad accum       : {sft_cfg.grad_accum}")
    print(f"  Effective batch  : {sft_cfg.micro_batch * sft_cfg.grad_accum} sequences")
    print(f"  Steps/epoch      : {steps_per_epoch:,}")
    print(f"  Epochs           : {sft_cfg.epochs}")
    print(f"  Total steps      : {total_steps:,}")
    print(f"  Warmup steps     : {sft_cfg.warmup_steps}")
    print(f"  Max seq len      : {sft_cfg.max_seq_len}\n")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = load_base_model(device)
    print("  Compiling model (first run ~60s)...")
    model = torch.compile(model)
    print("  Compile done.\n")

    # ── Optimizers ────────────────────────────────────────────────────────────
    print("  Building optimizers...")
    opt_muon, opt_adam = build_optimizers(model)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step, start_epoch, loss_history = load_latest_sft_checkpoint(
        model, opt_muon, opt_adam
    )
    resume = start_step > 0

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stop = EarlyStopping(
        patience  = sft_cfg.early_stopping_patience,
        tolerance = sft_cfg.early_stopping_tolerance,
    )
    early_stop.best_loss = load_best_sft_val_loss()
    if early_stop.best_loss < float("inf"):
        print(f"  Restored best SFT val loss: {early_stop.best_loss:.4f}")

    log_f = init_log(resume)

    model.train()
    opt_muon.zero_grad()
    opt_adam.zero_grad()

    global_step = start_step
    t0          = time.time()
    start_time  = time.time()

    print(f"\n{'='*60}")
    print(f"  Starting SFT from step {start_step} / {total_steps}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, sft_cfg.epochs):
        print(f"\n  ── Epoch {epoch+1}/{sft_cfg.epochs} ──")

        accum_loss    = 0.0
        accum_steps   = 0

        for batch in train_loader:
            if batch is None:
                continue

            # ── LR update ─────────────────────────────────────────────────────
            lr_m = get_lr(global_step, total_steps, sft_cfg.lr_muon)
            lr_a = get_lr(global_step, total_steps, sft_cfg.lr_adam)
            set_lr(opt_muon, lr_m)
            set_lr(opt_adam, lr_a)

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            # ── Forward + loss ────────────────────────────────────────────────
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)

                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                # Cross entropy — ignore_index=-100 masks problem tokens
                # This is the core SFT difference: we only learn on solution tokens
                loss = F.cross_entropy(
                    shift_logits.view(-1, cfg.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                ) / sft_cfg.grad_accum

            loss.backward()
            accum_loss  += loss.item()
            accum_steps += 1

            # ── Optimizer step (every grad_accum batches) ──────────────────────
            if accum_steps % sft_cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), sft_cfg.grad_clip)
                opt_muon.step()
                opt_adam.step()
                opt_muon.zero_grad()
                opt_adam.zero_grad()

                loss_history.append(accum_loss)

                # ── Validation ────────────────────────────────────────────────
                val_loss = float("nan")
                if global_step % sft_cfg.val_every == 0 and global_step > 0:
                    val_loss = compute_val_loss(model, val_loader, device)
                    improved = early_stop.update(val_loss)

                    if improved:
                        save_best_model(global_step, val_loss, model)

                    print(
                        f"  step {global_step:5d}/{total_steps}"
                        f" | epoch {epoch+1}"
                        f" | train {accum_loss:.4f}"
                        f" | val {val_loss:.4f}"
                        f" | ppl {math.exp(min(accum_loss * sft_cfg.grad_accum, 20)):6.2f}"
                        f" | lr_m {lr_m:.1e}"
                        f" | {early_stop.status()}"
                        f" | {'★ BEST' if improved else ''}"
                    )

                    if early_stop.should_stop:
                        print(f"\n{'='*60}")
                        print(f"  ⏹ Early stopping at step {global_step}.")
                        print(f"  Best val_loss: {early_stop.best_loss:.4f}")
                        print(f"{'='*60}\n")
                        save_checkpoint(global_step, epoch, model,
                                        opt_muon, opt_adam, loss_history)
                        log_f.close()
                        return

                # ── Logging ───────────────────────────────────────────────────
                if global_step % sft_cfg.log_every == 0:
                    ppl     = math.exp(min(accum_loss * sft_cfg.grad_accum, 20))
                    elapsed = time.time() - t0
                    t0      = time.time()
                    if math.isnan(val_loss):
                        print(
                            f"  step {global_step:5d}/{total_steps}"
                            f" | epoch {epoch+1}"
                            f" | loss {accum_loss:.4f}"
                            f" | ppl {ppl:6.2f}"
                            f" | lr_m {lr_m:.1e}"
                        )
                    write_log(log_f, global_step, epoch+1, accum_loss, val_loss,
                              ppl, lr_m, lr_a, time.time() - start_time,
                              early_stop.best_loss, early_stop.counter)

                # ── Checkpoint + thermal ──────────────────────────────────────
                if global_step % sft_cfg.save_every == 0 and global_step > start_step:
                    save_checkpoint(global_step, epoch, model,
                                    opt_muon, opt_adam, loss_history)
                    torch.cuda.empty_cache()
                    gc.collect()
                    temp = get_gpu_temp()
                    print(f"  GPU temp: {temp}°C")
                    gpu_cooling_break(temp)

                # ── Mandatory cooling break ───────────────────────────────────
                if global_step % sft_cfg.break_every_steps == 0 and global_step > 0:
                    if global_step % sft_cfg.save_every != 0:
                        save_checkpoint(global_step, epoch, model,
                                        opt_muon, opt_adam, loss_history)
                    print(f"\n⏸ Mandatory break at step {global_step} "
                          f"— {sft_cfg.cooling_break_s//60} min cool-down")
                    torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(sft_cfg.cooling_break_s)
                    print("  Resuming.\n")

                accum_loss = 0.0
                global_step += 1

        # ── End of epoch: always validate and checkpoint ───────────────────────
        print(f"\n  ── End of Epoch {epoch+1} ──")
        val_loss = compute_val_loss(model, val_loader, device)
        improved = early_stop.update(val_loss)
        if improved:
            save_best_model(global_step, val_loss, model)
        save_checkpoint(global_step, epoch+1, model, opt_muon, opt_adam, loss_history)
        print(f"  Epoch {epoch+1} val_loss={val_loss:.4f}  ppl={math.exp(min(val_loss,20)):.2f}"
              f"  {'★ BEST' if improved else ''}")

        if early_stop.should_stop:
            print(f"\n  ⏹ Early stopping after epoch {epoch+1}.")
            break

    # ── Done ──────────────────────────────────────────────────────────────────
    log_f.close()
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  SFT complete!")
    print(f"  Total time    : {total_time/3600:.2f} hours")
    print(f"  Total steps   : {global_step:,}")
    print(f"  Best val loss : {early_stop.best_loss:.4f}")
    print(f"  Best model    : {sft_cfg.sft_ckpt_dir}/sft_best_model.pt")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()