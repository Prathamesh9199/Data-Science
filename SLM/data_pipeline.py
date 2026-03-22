import os
import gc
import time
import json
import duckdb
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from config import SLMConfig

cfg = SLMConfig()

# == DuckDB setup ============================================================
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute(f"""
    CREATE OR REPLACE SECRET hf_token (
        TYPE huggingface,
        TOKEN '{cfg.hf_token}'
    );
""")

GLOB         = "hf://datasets/HuggingFaceTB/dclm-edu@~parquet/**/*.parquet"
PROGRESS_FILE = "pipeline_progress.json"

# == Tokenizer ===============================================================
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
EOS       = tokenizer.eos_token_id
assert tokenizer.vocab_size == 32_000
print(f"Tokenizer ready. eos_id={EOS}")

# == Token targets ===========================================================
TOTAL_TRAIN = cfg.target_tokens
TOTAL_VAL   = cfg.validation_tokens

# == Progress save/load ======================================================
def save_progress(label, doc_offset, token_offset):
    """Save current progress to disk after every batch."""
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
    progress[label] = {"doc_offset": doc_offset, "token_offset": token_offset}
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

def load_progress(label):
    """Load saved progress. Returns (doc_offset, token_offset) or (None, None)."""
    if not os.path.exists(PROGRESS_FILE):
        return None, None
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)
    if label in progress and progress[label]["token_offset"] > 0:
        return progress[label]["doc_offset"], progress[label]["token_offset"]
    return None, None

def clear_progress(label):
    """Remove progress entry once a phase is complete."""
    if not os.path.exists(PROGRESS_FILE):
        return
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)
    progress.pop(label, None)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

# == Retry wrapper ===========================================================
def fetch_batch(offset, batch_size=500, retries=5, wait=30):
    """Fetch one batch of docs with retry on 429 rate limit."""
    query = f"""
        SELECT text
        FROM '{GLOB}'
        WHERE edu_int_score = 3
          AND length(text) BETWEEN 300 AND 25000
        LIMIT {batch_size} OFFSET {offset}
    """
    for attempt in range(retries):
        try:
            return con.execute(query).fetchall()
        except Exception as e:
            if "429" in str(e):
                print(f"\n  ⚠ Rate limited. Waiting {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
                wait = min(wait * 2, 300)
            else:
                raise
    raise RuntimeError("Max retries exceeded.")

# == Core loop ===============================================================
def collect_tokens(doc_offset_start, token_target, out_path, label):
    # Check for existing progress
    saved_doc_offset, saved_token_offset = load_progress(label)

    if saved_doc_offset is not None:
        print(f"\n  ✓ Resuming [{label}] from doc {saved_doc_offset:,}, token {saved_token_offset:,}")
        doc_offset   = saved_doc_offset
        token_offset = saved_token_offset
        mmap_mode    = "r+"   # append to existing file
    else:
        print(f"\n  Starting [{label}] fresh from doc offset {doc_offset_start:,}")
        doc_offset   = doc_offset_start
        token_offset = 0
        mmap_mode    = "w+"   # create new file

    print(f"\n{'='*60}")
    print(f"  [{label}] Target : {token_target:,} tokens -> {out_path}")
    print(f"  [{label}] Doc offset start : {doc_offset:,}")
    print(f"  [{label}] Tokens written   : {token_offset:,}")
    print(f"{'='*60}")

    mmap      = np.memmap(out_path, dtype=np.uint16, mode=mmap_mode, shape=(token_target,))
    offset    = token_offset
    batch_num = 0
    overflow  = np.array([], dtype=np.uint16)

    token_bar = tqdm(
        total=token_target,
        initial=offset,
        desc=f"  [{label}] tokens",
        unit="tok",
        position=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    while offset < token_target:
        # ── STEP 1: fetch 500 docs ───────────────────────────────────────────
        rows = fetch_batch(doc_offset, batch_size=500)
        if not rows:
            print(f"\n  ⚠ Dataset exhausted at doc offset {doc_offset:,}.")
            break

        texts      = [row[0] for row in rows]
        batch_num += 1
        doc_offset += len(texts)
        del rows

        # ── STEP 2: tokenize ─────────────────────────────────────────────────
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        del texts

        # ── STEP 3: flatten with EOS ─────────────────────────────────────────
        flat = []
        for seq in encoded:
            flat.extend(seq)
            flat.append(EOS)
        del encoded

        batch_tokens = np.array(flat, dtype=np.uint16)
        del flat

        # ── STEP 4: combine with overflow, write complete seqs ───────────────
        combined   = np.concatenate([overflow, batch_tokens])
        del batch_tokens

        n_complete = len(combined) // cfg.max_seq_len
        write_len  = n_complete * cfg.max_seq_len

        written_this_batch = 0
        if write_len > 0:
            tokens_to_write    = min(write_len, token_target - offset)
            mmap[offset : offset + tokens_to_write] = combined[:tokens_to_write]
            mmap.flush()
            offset             += tokens_to_write
            written_this_batch  = tokens_to_write
            token_bar.update(tokens_to_write)

        overflow = combined[write_len:].copy()
        del combined

        # ── STEP 5: save progress after every batch ───────────────────────────
        save_progress(label, doc_offset, offset)

        # ── STEP 6: progress print ────────────────────────────────────────────
        tqdm.write(
            f"  batch {batch_num:>5} | "
            f"doc_offset: {doc_offset:>8,} | "
            f"written: {written_this_batch:>7,} tok | "
            f"total: {offset:>12,} / {token_target:,} tok"
        )

        # ── STEP 7: clear RAM ─────────────────────────────────────────────────
        gc.collect()

    token_bar.close()
    del mmap
    del overflow
    gc.collect()
    clear_progress(label)
    print(f"\n  [{label}] complete. {offset:,} tokens written to {out_path}")
    print(f"  [{label}] final doc offset: {doc_offset:,}")
    return doc_offset

# == Determine val end offset for train start ================================
# If val is already complete, read its saved final doc offset
def get_val_end_offset():
    """Return the doc offset where val ended, to avoid overlap with train."""
    # Check progress file for a completed val run
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
        if "val_end" in progress:
            return progress["val_end"]
    return None

def save_val_end(doc_offset):
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
    progress["val_end"] = doc_offset
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

# == Run =====================================================================
print("\n" + "=" * 60)
print("  Phase 1 — Validation set (dclm-edu, edu_int_score = 3)")
print("=" * 60)
os.makedirs(os.path.dirname(cfg.validation_data) or ".", exist_ok=True)

val_end = get_val_end_offset()
if val_end is not None:
    print(f"  ✓ Val already complete. End offset: {val_end:,}. Skipping.")
else:
    val_end = collect_tokens(
        doc_offset_start=0,
        token_target=TOTAL_VAL,
        out_path=cfg.validation_data,
        label="val"
    )
    save_val_end(val_end)

print("\n" + "=" * 60)
print("  Phase 2 — Training set (dclm-edu, edu_int_score = 3)")
print("=" * 60)
os.makedirs(os.path.dirname(cfg.train_data) or ".", exist_ok=True)
collect_tokens(
    doc_offset_start=val_end,
    token_target=TOTAL_TRAIN,
    out_path=cfg.train_data,
    label="train"
)

# == Sanity check ============================================================
print("\n" + "=" * 60)
print("  Sanity check")
print("=" * 60)
train_data = np.memmap(cfg.train_data, dtype=np.uint16, mode="r")
val_data   = np.memmap(cfg.validation_data, dtype=np.uint16, mode="r")
print(f"  train.bin : {len(train_data):,} tokens ({len(train_data)/1e9:.3f}B)")
print(f"  val.bin   : {len(val_data):,} tokens ({len(val_data)/1e6:.1f}M)")
print(f"  First 16 ids : {train_data[:16].tolist()}")
max_id = int(train_data.max())
status = "✓ valid" if max_id < 32_000 else "✗ INVALID — must be < 32000"
print(f"  Max token id  : {max_id}  ({status})")
decoded = tokenizer.decode(train_data[:32].tolist())
print(f"  Decoded       : \"{decoded}\"")