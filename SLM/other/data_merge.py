"""
Master Merge & Split Pipeline (CORRECTED)
------------------------------
Pools only train.bin and val.bin files together and enforces an exact global split:
  - Train: 4,000,000,000 tokens
  - Val  : Remaining tokens (~19.86 Million)
"""

import os
import glob
import numpy as np
from tqdm import tqdm

DATA_DIR = "data"
MASTER_TRAIN = os.path.join(DATA_DIR, "master_train.bin")
MASTER_VAL   = os.path.join(DATA_DIR, "master_val.bin")

TARGET_TRAIN = 4_000_000_000

def get_clean_files():
    # Find ONLY train.bin and val.bin, strictly ignoring raw.bin, master_, and tiny_story
    files = []
    for split in ["train.bin", "val.bin"]:
        files.extend(glob.glob(os.path.join(DATA_DIR, "**", split), recursive=True))
        
    clean_files = [f for f in files if "master_" not in f and "tiny_story" not in f]
    return sorted(clean_files)

def build_master_dataset():
    files = get_clean_files()
    if not files:
        print("⚠ No valid train.bin or val.bin files found!")
        return

    # 1. Calculate the exact global token count
    total_tokens = sum(os.path.getsize(f) // 2 for f in files)
    target_val = total_tokens - TARGET_TRAIN

    print(f"\n{'='*60}")
    print(f"  MASTER MERGE INITIALIZED (Strict Mode)")
    print(f"{'='*60}")
    print(f"  Total Found : {total_tokens:>15,} tokens")
    print(f"  -> Train    : {TARGET_TRAIN:>15,} tokens -> {MASTER_TRAIN}")
    print(f"  -> Val      : {target_val:>15,} tokens -> {MASTER_VAL}")
    print(f"{'='*60}\n")

    # 2. Allocate the master files on disk
    train_mmap = np.memmap(MASTER_TRAIN, dtype=np.uint16, mode="w+", shape=(TARGET_TRAIN,))
    val_mmap   = np.memmap(MASTER_VAL,   dtype=np.uint16, mode="w+", shape=(target_val,))

    # 3. Stream data across
    CHUNK_SIZE   = 10_000_000  # 20MB chunks
    train_offset = 0
    val_offset   = 0

    for f in files:
        ds_name = "/".join(f.split("/")[-2:])
        tokens_in_file = os.path.getsize(f) // 2
        source = np.memmap(f, dtype=np.uint16, mode="r")
        
        for i in tqdm(range(0, tokens_in_file, CHUNK_SIZE), desc=f"  Merging {ds_name:<25}"):
            end = min(i + CHUNK_SIZE, tokens_in_file)
            chunk = source[i:end]
            chunk_len = len(chunk)
            
            if train_offset < TARGET_TRAIN:
                space_left = TARGET_TRAIN - train_offset
                
                if chunk_len <= space_left:
                    train_mmap[train_offset : train_offset + chunk_len] = chunk
                    train_offset += chunk_len
                else:
                    train_mmap[train_offset : TARGET_TRAIN] = chunk[:space_left]
                    train_offset = TARGET_TRAIN
                    
                    val_chunk = chunk[space_left:]
                    val_mmap[val_offset : val_offset + len(val_chunk)] = val_chunk
                    val_offset += len(val_chunk)
            else:
                val_mmap[val_offset : val_offset + chunk_len] = chunk
                val_offset += chunk_len
                
        train_mmap.flush()
        val_mmap.flush()
        del source

    del train_mmap, val_mmap
    print(f"\n  ✅ Master Merge Complete.")
    print(f"  Train: {os.path.getsize(MASTER_TRAIN) / 1e9:.2f} GB")
    print(f"  Val  : {os.path.getsize(MASTER_VAL) / 1e6:.2f} MB")

def sanity_check():
    print(f"\n{'='*60}")
    print(f"  Final Sanity Check")
    print(f"{'='*60}")
    for path, label in [(MASTER_TRAIN, "master_train"), (MASTER_VAL, "master_val")]:
        data   = np.memmap(path, dtype=np.uint16, mode="r")
        max_id = int(data.max())
        status = "✓ valid" if max_id < 32_000 else "✗ INVALID"
        print(f"  [{label}] Tokens: {len(data):,} | Max ID: {max_id} ({status})")
        del data

if __name__ == "__main__":
    build_master_dataset()
    sanity_check()