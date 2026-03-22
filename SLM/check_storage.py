import shutil
import os

def check_storage():
    print("=" * 50)
    print("  Storage Check")
    print("=" * 50)

    # Overall disk usage
    total, used, free = shutil.disk_usage("/")
    print(f"\n  Disk (total) : {total / 1e9:.1f} GB")
    print(f"  Disk (used)  : {used  / 1e9:.1f} GB")
    print(f"  Disk (free)  : {free  / 1e9:.1f} GB")

    # What the training run needs
    print(f"\n  --- Training Run Requirements ---")
    train_bin   = 4_000_000_000 * 2 / 1e9   # 4B tokens × 2 bytes
    val_bin     = 20_000_000    * 2 / 1e9   # 20M tokens × 2 bytes
    checkpoints = 5 * 400 / 1e3             # 5 checkpoints × ~400MB each
    total_needed = train_bin + val_bin + checkpoints

    print(f"  train.bin    : {train_bin:.1f} GB  (4B tokens × 2 bytes)")
    print(f"  val.bin      :  {val_bin*1000:.0f} MB (20M tokens × 2 bytes)")
    print(f"  checkpoints  : {checkpoints:.1f} GB  (5 × ~400MB)")
    print(f"  ─────────────────────────────────")
    print(f"  Total needed : {total_needed:.1f} GB")

    # Verdict
    print(f"\n  --- Verdict ---")
    if free / 1e9 >= total_needed + 5:  # +5GB safety buffer
        print(f"  ✅ Enough space. ({free/1e9:.1f}GB free, {total_needed:.1f}GB needed)")
    else:
        shortfall = total_needed + 5 - free / 1e9
        print(f"  ❌ Not enough space. Need {shortfall:.1f} GB more.")

    # Current SLM folder usage
    slm_dir = os.path.dirname(os.path.abspath(__file__))
    slm_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(slm_dir)
        for f in files
    ) / 1e9
    print(f"\n  SLM folder   : {slm_size:.2f} GB  ({slm_dir})")

    # Existing .bin files
    print(f"\n  --- Existing data files ---")
    for fname in ["train.bin", "val.bin"]:
        fpath = os.path.join(slm_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1e9
            print(f"  {fname:12} : {size:.2f} GB")
        else:
            print(f"  {fname:12} : not found")

    # Checkpoints
    ckpt_dir = os.path.join(slm_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
        ckpts = sorted(os.listdir(ckpt_dir))
        ckpt_size = sum(
            os.path.getsize(os.path.join(ckpt_dir, f))
            for f in ckpts
        ) / 1e9
        print(f"\n  Checkpoints  : {len(ckpts)} files, {ckpt_size:.2f} GB total")
        for c in ckpts:
            size = os.path.getsize(os.path.join(ckpt_dir, c)) / 1e6
            print(f"    {c} : {size:.1f} MB")
    else:
        print(f"\n  Checkpoints  : none yet")

    print(f"\n{'='*50}\n")

if __name__ == "__main__":
    check_storage()