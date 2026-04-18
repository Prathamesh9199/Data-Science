"""
Cosmopedia v2 Top-Off Pipeline
------------------------------
Target  : 1,608,000,000 tokens total (1.6B train + 8M val)
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

OUT_DIR   = "data/cosmopedia"
RAW_PATH  = os.path.join(OUT_DIR, "raw.bin")
TRAIN_PATH = os.path.join(OUT_DIR, "train.bin")
VAL_PATH   = os.path.join(OUT_DIR, "val.bin")

TOTAL_TOKENS = 1_608_000_000
TRAIN_TOKENS = 1_600_000_000
VAL_TOKENS   =     8_000_000

tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
EOS       = tokenizer.eos_token_id

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

def collect_raw():
    saved_doc, saved_tok = load_progress()

    if saved_doc is not None:
        print(f"\n  Resuming from doc {saved_doc:,}, token {saved_tok:,}")
        doc_offset   = saved_doc
        token_offset = saved_tok
        mmap_mode    = "r+"
        
        # MAGIC TRICK: Instantly expand the file on disk to fit the new target
        print(f"  Expanding raw.bin to fit {TOTAL_TOKENS:,} tokens...")
        with open(RAW_PATH, "a") as f:
            f.truncate(TOTAL_TOKENS * 2) 
    else:
        print(f"\n  Starting fresh.")
        doc_offset   = 0
        token_offset = 0
        mmap_mode    = "w+"
        save_progress(0, 0)

    print(f"\n{'='*60}")
    print(f"  [cosmopedia] NEW Target : {TOTAL_TOKENS:,} tokens -> {RAW_PATH}")
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
        if not (100 <= len(text) <= 50000):
            continue
            
        if valid_docs_seen < doc_offset:
            valid_docs_seen += 1
            continue

        doc_offset += 1
        valid_docs_seen += 1

        audience = str(row.get("audience", "other")).lower()
        
        if "middle_school" in audience:
            if len(buffer_ms) < TARGET_MS: buffer_ms.append(text)
        elif "college" in audience:
            if len(buffer_col) < TARGET_COL: buffer_col.append(text)
        else:
            if len(buffer_oth) < TARGET_OTH: buffer_oth.append(text)

        if len(buffer_ms) == TARGET_MS and len(buffer_col) == TARGET_COL and len(buffer_oth) == TARGET_OTH:
            combined_texts = buffer_ms + buffer_col + buffer_oth
            random.shuffle(combined_texts)
            
            buffer_ms.clear()
            buffer_col.clear()
            buffer_oth.clear()
            
            batch_num += 1

            encoded = tokenizer(
                combined_texts, add_special_tokens=False, return_attention_mask=False
            )["input_ids"]

            flat = [tok for seq in encoded for tok in seq + [EOS]]
            batch_tokens = np.array(flat, dtype=np.uint16)
            combined   = np.concatenate([overflow, batch_tokens])

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
            save_progress(doc_offset, offset)
            
            tqdm.write(f"  batch {batch_num:>5} | total: {offset:>13,} / {TOTAL_TOKENS:,} tok")
            gc.collect()

    token_bar.close()
    del mmap, overflow
    gc.collect()
    return offset

def split_raw(real_tokens):
    print(f"\n  Splitting tokens → {TRAIN_TOKENS:,} train / {VAL_TOKENS:,} val")
    train_tokens = TRAIN_TOKENS
    val_tokens   = min(VAL_TOKENS, real_tokens - train_tokens)

    raw   = np.memmap(RAW_PATH,   dtype=np.uint16, mode="r")
    train = np.memmap(TRAIN_PATH, dtype=np.uint16, mode="w+", shape=(train_tokens,))
    val   = np.memmap(VAL_PATH,   dtype=np.uint16, mode="w+", shape=(val_tokens,))

    CHUNK = 10_000_000
    for i in tqdm(range(0, train_tokens, CHUNK), desc="  Writing train"):
        train[i:min(i + CHUNK, train_tokens)] = raw[i:min(i + CHUNK, train_tokens)]
    train.flush()

    for i in tqdm(range(0, val_tokens, CHUNK), desc="  Writing val"):
        val[i:min(i + CHUNK, val_tokens)] = raw[train_tokens + i : train_tokens + min(i + CHUNK, val_tokens)]
    val.flush()
    del raw, train, val
    gc.collect()

if __name__ == "__main__":
    real_tokens = collect_raw()
    if real_tokens >= TOTAL_TOKENS:
        split_raw(real_tokens)
        print(f"\n  ✅ Cosmopedia topped off to 1.608B!")