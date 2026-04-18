"""
Cosmopedia v2 Data Pipeline
------------------------------
Source  : HuggingFaceTB/smollm-corpus (cosmopedia-v2)
Target  : 1,206,000,000 tokens total (1.2B train + 6M val)
Filter  : 40% middle school, 30% college, 30% other (interleaved)
Output  : data/cosmopedia/raw.bin   (1,206M tokens)
          data/cosmopedia/train.bin (1,200,000,000 tokens)
          data/cosmopedia/val.bin   (    6,000,000 tokens)
"""

import os
import gc
import random
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from config import SLMConfig

cfg = SLMConfig()

# == Paths ===================================================================
OUT_DIR   = "data/cosmopedia"
RAW_PATH  = os.path.join(OUT_DIR, "raw.bin")
TRAIN_PATH = os.path.join(OUT_DIR, "train.bin")
VAL_PATH   = os.path.join(OUT_DIR, "val.bin")
os.makedirs(OUT_DIR, exist_ok=True)

# == Targets =================================================================
TOTAL_TOKENS = 1_206_000_000
TRAIN_TOKENS = 1_200_000_000
VAL_TOKENS   =     6_000_000

# == Tokenizer ===============================================================
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
EOS       = tokenizer.eos_token_id
assert tokenizer.vocab_size == 32_000
print(f"Tokenizer ready. eos_id={EOS}")

# == Progress ================================================================
PROGRESS_FILE = os.path.join(OUT_DIR, "progress.json")

def save_progress(doc_offset, token_offset):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"doc_offset": doc_offset, "token_offset": token_offset}, f)

def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return None, None
    with open(PROGRESS_FILE) as f:
        p = json.load(f)
    if p.get("token_offset", 0) > 0:
        return p["doc_offset"], p["token_offset"]
    return None, None

# == Collect raw.bin =========================================================
def collect_raw():
    saved_doc, saved_tok = load_progress()

    if saved_doc is not None:
        print(f"\n  Resuming from doc {saved_doc:,}, token {saved_tok:,}")
        doc_offset   = saved_doc
        token_offset = saved_tok
        mmap_mode    = "r+"
    else:
        print(f"\n  Starting fresh.")
        doc_offset   = 0
        token_offset = 0
        mmap_mode    = "w+"
        save_progress(0, 0)

    print(f"\n{'='*60}")
    print(f"  [cosmopedia] Target : {TOTAL_TOKENS:,} tokens -> {RAW_PATH}")
    print(f"  [cosmopedia] Ratios : 40% Middle School / 30% College / 30% Other")
    print(f"{'='*60}")

    mmap     = np.memmap(RAW_PATH, dtype=np.uint16, mode=mmap_mode, shape=(TOTAL_TOKENS,))
    offset   = token_offset
    overflow = np.array([], dtype=np.uint16)
    batch_num = 0

    token_bar = tqdm(
        total=TOTAL_TOKENS,
        initial=offset,
        desc="  [cosmopedia] tokens",
        unit="tok",
        position=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    print(f"  Establishing optimized HF stream...")
    
    # We use HF Datasets streaming. This completely bypasses DuckDB's HTTP 429 issues.
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        name="cosmopedia-v2",
        split="train",
        streaming=True,
        token=cfg.hf_token
    )

    buffer_ms  = []
    buffer_col = []
    buffer_oth = []

    TARGET_MS  = 800
    TARGET_COL = 600
    TARGET_OTH = 600

    valid_docs_seen = 0

    for row in ds:
        if offset >= TOTAL_TOKENS:
            break

        text = row.get("text", "")
        
        # Apply the same length filter we had in DuckDB
        if not (100 <= len(text) <= 50000):
            continue
            
        # Fast-forward to where the script crashed last night
        if valid_docs_seen < doc_offset:
            valid_docs_seen += 1
            continue

        doc_offset += 1
        valid_docs_seen += 1

        audience = str(row.get("audience", "other")).lower()
        
        # Route texts into the proper ratio buckets
        if "middle_school" in audience:
            if len(buffer_ms) < TARGET_MS: buffer_ms.append(text)
        elif "college" in audience:
            if len(buffer_col) < TARGET_COL: buffer_col.append(text)
        else:
            if len(buffer_oth) < TARGET_OTH: buffer_oth.append(text)

        # Only tokenize and write when we have a perfect 40/30/30 blend
        if len(buffer_ms) == TARGET_MS and len(buffer_col) == TARGET_COL and len(buffer_oth) == TARGET_OTH:
            combined_texts = buffer_ms + buffer_col + buffer_oth
            random.shuffle(combined_texts)
            
            buffer_ms.clear()
            buffer_col.clear()
            buffer_oth.clear()
            
            batch_num += 1

            encoded = tokenizer(
                combined_texts,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            del combined_texts

            flat = []
            for seq in encoded:
                flat.extend(seq)
                flat.append(EOS)
            del encoded

            batch_tokens = np.array(flat, dtype=np.uint16)
            del flat

            combined   = np.concatenate([overflow, batch_tokens])
            del batch_tokens

            n_complete = len(combined) // cfg.max_seq_len
            write_len  = n_complete * cfg.max_seq_len

            written_this_batch = 0
            if write_len > 0:
                tokens_to_write    = min(write_len, TOTAL_TOKENS - offset)
                mmap[offset : offset + tokens_to_write] = combined[:tokens_to_write]
                mmap.flush()
                offset             += tokens_to_write
                written_this_batch  = tokens_to_write
                token_bar.update(tokens_to_write)

            overflow = combined[write_len:].copy()
            del combined

            save_progress(doc_offset, offset)

            tqdm.write(
                f"  batch {batch_num:>5} | "
                f"doc_offset: {doc_offset:>8,} | "
                f"written: {written_this_batch:>7,} tok | "
                f"total: {offset:>13,} / {TOTAL_TOKENS:,} tok"
            )

            gc.collect()

    token_bar.close()
    del mmap
    del overflow
    gc.collect()
    print(f"\n  [cosmopedia] raw collection complete. {offset:,} tokens -> {RAW_PATH}")
    return offset

# == Split raw.bin → train.bin + val.bin =====================================
def split_raw(real_tokens):
    print(f"\n{'='*60}")
    print(f"  Splitting tokens → {TRAIN_TOKENS:,} train / {VAL_TOKENS:,} val")
    print(f"{'='*60}")

    train_tokens = TRAIN_TOKENS
    val_tokens   = min(VAL_TOKENS, real_tokens - train_tokens)

    print(f"  Train : {train_tokens:,} tokens -> {TRAIN_PATH}")
    print(f"  Val   : {val_tokens:,}   tokens -> {VAL_PATH}")

    raw   = np.memmap(RAW_PATH,   dtype=np.uint16, mode="r")
    train = np.memmap(TRAIN_PATH, dtype=np.uint16, mode="w+", shape=(train_tokens,))
    val   = np.memmap(VAL_PATH,   dtype=np.uint16, mode="w+", shape=(val_tokens,))

    CHUNK = 10_000_000
    for i in tqdm(range(0, train_tokens, CHUNK), desc="  Writing train"):
        end = min(i + CHUNK, train_tokens)
        train[i:end] = raw[i:end]
    train.flush()

    for i in tqdm(range(0, val_tokens, CHUNK), desc="  Writing val"):
        end = min(i + CHUNK, val_tokens)
        val[i:end] = raw[train_tokens + i : train_tokens + end]
    val.flush()

    del raw, train, val
    gc.collect()

    print(f"\n  ✓ Split complete.")
    print(f"  train.bin : {train_tokens:,} tokens ({train_tokens*2/1e9:.3f} GB)")
    print(f"  val.bin   : {val_tokens:,} tokens ({val_tokens*2/1e6:.1f} MB)")

# == Sanity check ============================================================
def sanity_check():
    print(f"\n{'='*60}")
    print(f"  Sanity Check")
    print(f"{'='*60}")
    for path, label in [(TRAIN_PATH, "train"), (VAL_PATH, "val")]:
        data   = np.memmap(path, dtype=np.uint16, mode="r")
        max_id = int(data.max())
        status = "✓ valid" if max_id < 32_000 else "✗ INVALID"
        decoded = tokenizer.decode(data[:32].tolist())
        print(f"\n  [{label}]")
        print(f"    Tokens      : {len(data):,}")
        print(f"    Max token id: {max_id} ({status})")
        print(f"    Decoded     : \"{decoded[:100]}\"")
        del data

# == Run =====================================================================
if __name__ == "__main__":
    real_tokens = collect_raw()
    
    # SAFETY GUARD: Only split if we hit the target
    if real_tokens >= TOTAL_TOKENS:
        split_raw(real_tokens)
        sanity_check()
        print(f"\n  ✅ Cosmopedia v2 pipeline complete.")
        print(f"  Next: run PeS2o pipeline.")
    else:
        print(f"\n  ⏸ Collection paused or dataset exhausted early at {real_tokens:,} tokens.")