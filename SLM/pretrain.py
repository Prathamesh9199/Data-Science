import os
import gc
import csv
import math
import glob
import time
import torch
import numpy as np
import bitsandbytes as bnb
import torch.nn.functional as F
from model_architecture import CustomTransformer
from config import SLMConfig
import subprocess

cfg = SLMConfig()


# == GPU thermal helpers =======================================================

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
    """Pause training if GPU is above threshold. Returns True if a break was taken."""
    if temp >= cfg.gpu_temp_threshold:
        print(f"\n⚠ GPU at {temp}°C — cooling break ({cfg.cooling_break_s // 60} min)...")
        time.sleep(cfg.cooling_break_s)
        print("  Resuming training.")
        return True
    return False


# == Learning rate schedule ====================================================

def get_lr(step, total_steps, peak_lr, warmup_steps=cfg.warmup_steps):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    stable_end = warmup_steps + int((total_steps - warmup_steps) * 0.70)
    if step < stable_end:
        return peak_lr
    progress = (step - stable_end) / max(1, total_steps - stable_end)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (cfg.lr_min_ratio + (1 - cfg.lr_min_ratio) * cosine)


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


# == Muon optimizer ============================================================

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
                g = p.grad
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
    opt_muon = Muon(muon_params, lr=cfg.lr_muon, momentum=0.95)
    opt_adam = bnb.optim.AdamW8bit(adam_params, lr=cfg.lr_adam, betas=(0.9, 0.95),
                                    weight_decay=cfg.weight_decay, eps=1e-8)
    print(f"  Muon params : {sum(p.numel() for p in muon_params)/1e6:.1f}M")
    print(f"  AdamW params: {sum(p.numel() for p in adam_params)/1e6:.1f}M")
    return opt_muon, opt_adam


# == Data loaders ==============================================================

def get_batch(data, micro_batch, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (micro_batch,))
    x  = torch.stack([
        torch.from_numpy(data[i : i + seq_len].astype(np.int64))
        for i in ix
    ]).to(device)
    y  = torch.stack([
        torch.from_numpy(data[i + 1 : i + seq_len + 1].astype(np.int64))
        for i in ix
    ]).to(device)
    return x, y


@torch.no_grad()
def compute_val_loss(model, val_data, device, n_seqs=50):
    model.eval()
    try:
        losses = []
        for i in range(n_seqs):
            start = i * cfg.max_seq_len
            if start + cfg.max_seq_len + 1 > len(val_data):
                break
            x = torch.from_numpy(
                val_data[start : start + cfg.max_seq_len].astype(np.int64)
            ).unsqueeze(0).to(device)
            y = torch.from_numpy(
                val_data[start + 1 : start + cfg.max_seq_len + 1].astype(np.int64)
            ).unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss   = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
            losses.append(loss.item())
        return sum(losses) / len(losses) if losses else float("nan")
    finally:
        model.train()


# == Checkpoint helpers ========================================================

def save_checkpoint(step, model, opt_muon, opt_adam, loss_history):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_dir, f"ckpt_{step:05d}.pt")
    torch.save({
        "step":         step,
        "model":        model.state_dict(),
        "opt_muon":     opt_muon.state_dict(),
        "opt_adam":     opt_adam.state_dict(),
        "loss_history": loss_history,
    }, path)
    print(f"  ✓ Checkpoint saved → {path}")
    _prune_checkpoints()


def save_best_model(step, val_loss, model):
    """Save a separate best model checkpoint — never pruned."""
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_dir, "best_model.pt")
    torch.save({
        "step":     step,
        "val_loss": val_loss,
        "model":    model.state_dict(),
    }, path)
    print(f"  ★ Best model saved → {path}  (val_loss={val_loss:.4f} @ step {step})")


def _prune_checkpoints():
    ckpts = glob.glob(os.path.join(cfg.ckpt_dir, "ckpt_*.pt"))
    # Sort by actual file creation/modification time
    ckpts.sort(key=os.path.getmtime) 
    for old in ckpts[:-cfg.keep_ckpts]:
        os.remove(old)
        print(f"  ✗ Deleted old checkpoint: {old}")


def load_latest_checkpoint(model, opt_muon, opt_adam):
    ckpts = glob.glob(os.path.join(cfg.ckpt_dir, "ckpt_*.pt"))
    if not ckpts:
        print("  No checkpoint found — starting from scratch.")
        return 0, []
    
    # Sort by actual file creation/modification time
    ckpts.sort(key=os.path.getmtime)
    path = ckpts[-1]
    
    print(f"  Resuming from {path}")
    ckpt = torch.load(path, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt_muon.load_state_dict(ckpt["opt_muon"])
    opt_adam.load_state_dict(ckpt["opt_adam"])
    loss_history = ckpt.get("loss_history", [])
    print(f"  Resumed at step {ckpt['step']}")
    return ckpt["step"], loss_history


def load_best_val_loss():
    """Read the best val loss from existing best_model.pt if it exists."""
    path = os.path.join(cfg.ckpt_dir, "best_model.pt")
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("val_loss", float("inf"))
    return float("inf")


# == CSV logger ================================================================

def init_log(resume):
    if not resume or not os.path.exists(cfg.log_file):
        with open(cfg.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "val_loss", "ppl",
                             "lr_muon", "lr_adam", "tokens_seen", "elapsed_s",
                             "best_val_loss", "patience_counter"])
    return open(cfg.log_file, "a", newline="")


def write_log(log_f, step, train_loss, val_loss, ppl, lr_m, lr_a,
              tokens_seen, elapsed, best_val_loss, patience_counter):
    writer = csv.writer(log_f)
    val_str  = f"{val_loss:.4f}"      if not math.isnan(val_loss)      else "n/a"
    best_str = f"{best_val_loss:.4f}" if best_val_loss < float("inf")  else "n/a"
    writer.writerow([step, f"{train_loss:.4f}", val_str, f"{ppl:.2f}",
                     f"{lr_m:.2e}", f"{lr_a:.2e}", tokens_seen,
                     f"{elapsed:.1f}", best_str, patience_counter])
    log_f.flush()


# == Early stopping ============================================================

class EarlyStopping:
    """
    Monitors val_loss. 
    Allows the model to flatline indefinitely.
    ONLY stops if the loss actively degrades (increases) for `patience` consecutive checks.
    """
    def __init__(self, patience=15, tolerance=0.05):
        self.patience    = patience
        self.tolerance   = tolerance # How much it can bounce up without counting as "diverging"
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    def update(self, val_loss):
        # 1. Did it hit a new personal best?
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            return True   # improved → save best model
            
        # 2. Is it flatlining or just bouncing slightly? (Safe Zone)
        elif val_loss <= self.best_loss + self.tolerance:
            self.counter = 0 # Reset counter, we are still stable
            return False
            
        # 3. Is it actively diverging/getting worse? (Danger Zone)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def status(self):
        return f"divergence strikes {self.counter}/{self.patience} | best_val={self.best_loss:.4f}"
    
# == Main training loop ========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SLM Pre-training")
    print(f"  Device : {device}")
    print(f"  dtype  : bfloat16")
    print(f"{'='*60}\n")

    tokens_per_step = cfg.micro_batch * cfg.grad_accum * cfg.max_seq_len
    total_steps     = cfg.target_tokens // tokens_per_step
    print(f"  Token budget     : {cfg.target_tokens/1e9:.1f}B")
    print(f"  Tokens/step      : {tokens_per_step:,}")
    print(f"  Total steps      : {total_steps:,}")
    print(f"  Warmup steps     : {cfg.warmup_steps}")
    print(f"  Save every       : {cfg.save_every} steps  (rolling {cfg.keep_ckpts} kept)")
    print(f"  Val every        : {cfg.val_every} steps")
    print(f"  Early stop       : patience={cfg.early_stopping_patience}, "
          f"min_delta={cfg.early_stopping_min_delta}\n")

    assert os.path.exists(cfg.train_data), \
        f"train.bin not found at '{cfg.train_data}'. Run data pipeline first."
    assert os.path.exists(cfg.validation_data), \
        f"val.bin not found at '{cfg.validation_data}'. Run data pipeline first."

    data     = np.memmap(cfg.train_data,      dtype=np.uint16, mode="r")
    val_data = np.memmap(cfg.validation_data, dtype=np.uint16, mode="r")
    print(f"  train.bin : {len(data)/1e9:.3f}B tokens")
    print(f"  val.bin   : {len(val_data)/1e6:.1f}M tokens\n")

    model = CustomTransformer(cfg).to(device).to(torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params  : {total_params/1e6:.1f}M\n")

    print("  Compiling model (first run ~60s)...")
    model = torch.compile(model)
    print("  Compile done.\n")

    print("  Building optimizers...")
    opt_muon, opt_adam = build_optimizers(model)

    start_step, loss_history = load_latest_checkpoint(model, opt_muon, opt_adam)
    resume      = start_step > 0
    tokens_seen = start_step * tokens_per_step

    log_f = init_log(resume)

    # == Early stopping setup ==================================================
    early_stop = EarlyStopping()
    # Restore best val loss from disk if resuming
    early_stop.best_loss = load_best_val_loss()
    if early_stop.best_loss < float("inf"):
        print(f"  Restored best val loss: {early_stop.best_loss:.4f}")

    model.train()
    opt_muon.zero_grad()
    opt_adam.zero_grad()

    accum_loss = 0.0
    t0         = time.time()
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  Starting from step {start_step} / {total_steps}")
    print(f"{'='*60}\n")

    for global_step in range(start_step, total_steps):

        lr_m = get_lr(global_step, total_steps, cfg.lr_muon)
        lr_a = get_lr(global_step, total_steps, cfg.lr_adam)
        set_lr(opt_muon, lr_m)
        set_lr(opt_adam, lr_a)

        for _ in range(cfg.grad_accum):
            x, y = get_batch(data, cfg.micro_batch, cfg.max_seq_len, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits  = model(x)
                ce_loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
                z_loss  = 1e-4 * logits.logsumexp(-1).pow(2).mean()
                loss    = (ce_loss + z_loss) / cfg.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt_muon.step()
        opt_adam.step()
        opt_muon.zero_grad()
        opt_adam.zero_grad()

        tokens_seen += tokens_per_step
        loss_history.append(accum_loss)

        # == Validation + early stopping check =================================
        val_loss = float("nan")
        if global_step % cfg.val_every == 0 and global_step > 0:
            val_loss  = compute_val_loss(model, val_data, device)
            improved  = early_stop.update(val_loss)

            if improved:
                # New best — save dedicated best_model.pt (never pruned)
                save_best_model(global_step, val_loss, model)

            status_str = (
                f"  step {global_step:5d}/{total_steps}"
                f" | train {accum_loss:.4f}"
                f" | val {val_loss:.4f}"
                f" | ppl {math.exp(min(accum_loss, 20)):7.2f}"
                f" | lr_m {lr_m:.1e}"
                f" | {tokens_seen/1e9:.3f}B tok"
                f" | {early_stop.status()}"
                f" | {'★ NEW BEST' if improved else ''}"
            )
            print(status_str)

            if early_stop.should_stop:
                print(f"\n{'='*60}")
                print(f"  ⏹ Early stopping triggered at step {global_step}.")
                print(f"  No val_loss improvement for {early_stop.patience} checks.")
                print(f"  Best val_loss: {early_stop.best_loss:.4f}")
                print(f"  Best model  : {cfg.ckpt_dir}/best_model.pt")
                print(f"{'='*60}\n")
                save_checkpoint(global_step, model, opt_muon, opt_adam, loss_history)
                break

        # == Logging ===========================================================
        if global_step % cfg.log_every == 0:
            ppl     = math.exp(min(accum_loss, 20))
            elapsed = time.time() - t0
            tok_s   = (cfg.log_every * tokens_per_step) / elapsed if global_step > 0 else 0
            t0      = time.time()

            if math.isnan(val_loss):  # no val this step — print train only
                print(
                    f"  step {global_step:5d}/{total_steps}"
                    f" | loss {accum_loss:.4f}"
                    f" | ppl {ppl:7.2f}"
                    f" | lr_m {lr_m:.1e}"
                    f" | {tokens_seen/1e9:.3f}B tok"
                    f" | {tok_s/1e3:.1f}k tok/s"
                )

            write_log(log_f, global_step, accum_loss, val_loss, ppl,
                      lr_m, lr_a, tokens_seen, time.time() - start_time,
                      early_stop.best_loss, early_stop.counter)

        # == Checkpoint + thermal check ========================================
        if global_step % cfg.save_every == 0 and global_step > start_step:
            save_checkpoint(global_step, model, opt_muon, opt_adam, loss_history)
            torch.cuda.empty_cache()
            gc.collect()
            temp = get_gpu_temp()
            print(f"  GPU temp: {temp}°C")
            gpu_cooling_break(temp)

        # == Mandatory cooling break ===========================================
        if global_step % cfg.break_every_steps == 0 and global_step > 0:
            if global_step % cfg.save_every != 0:
                save_checkpoint(global_step, model, opt_muon, opt_adam, loss_history)
            print(f"\n⏸ Mandatory break at step {global_step} — {cfg.cooling_break_s//60} min cool-down")
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(cfg.cooling_break_s)
            print("  Resuming.\n")

        accum_loss = 0.0

    else:
        # == Final checkpoint (only if training completed without early stop) ==
        save_checkpoint(total_steps, model, opt_muon, opt_adam, loss_history)

    log_f.close()

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Pre-training complete!")
    print(f"  Total time    : {total_time/3600:.2f} hours")
    print(f"  Tokens seen   : {tokens_seen/1e9:.3f}B")
    print(f"  Final loss    : {loss_history[-1]:.4f}")
    print(f"  Best val loss : {early_stop.best_loss:.4f}")
    print(f"  Best model    : {cfg.ckpt_dir}/best_model.pt")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()