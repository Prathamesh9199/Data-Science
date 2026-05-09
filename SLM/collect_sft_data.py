# collect_sft_data.py

import os
import re
import sys
import json
import random
import hashlib
import logging
from datetime import datetime
from collections import defaultdict

from datasets import load_dataset
from huggingface_hub import login
from config import SLMConfig

cfg = SLMConfig()
if cfg.hf_token:
    login(token=cfg.hf_token)

# ── Configuration ──────────────────────────────────────────────────────────────

SEED       = 42
OUTPUT_DIR = "data/sft"
RAW_DIR    = os.path.join(OUTPUT_DIR, "raw")       # one file per dataset
MASTER_DIR = os.path.join(OUTPUT_DIR, "master")    # merged + deduped
LOG_FILE   = os.path.join(OUTPUT_DIR, "collection_log.txt")
VAL_RATIO  = 0.05

DATASET_TARGETS = {
    "gsm8k"         :  7_000,
    "metamath"      : 45_000,
    "orca_math"     : 20_000,
    "math_lighteval":  5_000,
    "logiqa"        :  5_500,
    "arc_challenge" :  1_100,
    "arc_easy"      :  2_200,
    "sciq"          : 10_000,
    "openbookqa"    :  4_500,
    "openhermes"    : 40_000,
}

OPENHERMES_KEEP_CATEGORIES = {
    "coding", "mathematics", "science", "data_analysis",
    "reasoning", "stem", "computer_science",
    "math", "logic", "algorithm",
}

random.seed(SEED)

# ── Logging ────────────────────────────────────────────────────────────────────

for d in [OUTPUT_DIR, RAW_DIR, MASTER_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger()


# ── I/O helpers ────────────────────────────────────────────────────────────────

def raw_path(name):
    return os.path.join(RAW_DIR, f"{name}.jsonl")

def save_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info(f"    → Saved {len(examples):,} examples to {path}")

def load_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

def already_collected(name):
    p = raw_path(name)
    if os.path.exists(p):
        count = sum(1 for _ in open(p, encoding="utf-8"))
        log.info(f"  [{name}] already exists — {count:,} examples. Skipping collection.")
        return True
    return False


# ── Format + Quality helpers ───────────────────────────────────────────────────

def build_example(source, category, problem, steps, answer, difficulty="medium"):
    step_lines = "\n\n".join(
        f"Step {i+1}: {s.strip()}"
        for i, s in enumerate(steps) if s.strip()
    )
    solution = f"{step_lines}\n\nTherefore, the answer is {answer}."
    uid = hashlib.md5((source + problem).encode("utf-8")).hexdigest()[:12]
    return {
        "id"        : f"{source}_{uid}",
        "source"    : source,
        "category"  : category,
        "difficulty": difficulty,
        "problem"   : problem.strip(),
        "solution"  : solution.strip(),
        "answer"    : str(answer).strip(),
    }

def is_quality(problem, solution, answer,
               min_prob_words=10, max_prob_words=200,
               min_sol_words=20,  max_sol_words=600):
    if not problem or not solution or not answer:
        return False
    pw = len(problem.split())
    sw = len(solution.split())
    if not (min_prob_words <= pw <= max_prob_words):
        return False
    if not (min_sol_words <= sw <= max_sol_words):
        return False
    sol_words = solution.lower().split()
    if len(sol_words) > 20:
        ngrams = [tuple(sol_words[i:i+5]) for i in range(len(sol_words) - 5)]
        if ngrams and len(set(ngrams)) / len(ngrams) < 0.5:
            return False
    if not re.search(r"\d", solution):
        return False
    if len(str(answer).strip()) == 0:
        return False
    return True

def log_dataset_stats(name, examples):
    if not examples:
        log.info(f"    [stats] No examples.")
        return
    sol_lens = [len(ex["solution"].split()) for ex in examples]
    by_diff  = defaultdict(int)
    for ex in examples:
        by_diff[ex["difficulty"]] += 1
    log.info(f"    [stats] count={len(examples):,}  "
             f"avg_sol={sum(sol_lens)//len(sol_lens)}w  "
             f"min={min(sol_lens)}w  max={max(sol_lens)}w  "
             f"diff={dict(by_diff)}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — COLLECT + WRITE EACH DATASET INDIVIDUALLY
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. GSM8K ──────────────────────────────────────────────────────────────────

def clean_gsm8k_solution(raw):
    lines   = raw.strip().split("\n")
    answer  = None
    content = []
    for line in lines:
        line = line.strip()
        if line.startswith("####"):
            answer = line.replace("####", "").strip()
        elif line:
            cleaned = re.sub(r"<<[^>]+>>", "", line).strip()
            if cleaned:
                content.append(cleaned)
    steps = []
    for line in content:
        for s in re.split(r"(?<=[.!?])\s+", line):
            s = s.strip()
            if s:
                steps.append(s)
    return steps, answer

def collect_gsm8k():
    name = "gsm8k"
    if already_collected(name): return
    log.info(f"\n{'='*55}\n  [1/10] Collecting GSM8K\n{'='*55}")
    target   = DATASET_TARGETS[name]
    ds       = load_dataset("openai/gsm8k", "main")
    examples = []
    for split in ["train", "test"]:
        if split not in ds: continue
        for row in ds[split]:
            steps, answer = clean_gsm8k_solution(row["answer"])
            if not steps or not answer: continue
            ex = build_example("gsm8k", "math_arithmetic",
                               row["question"], steps, answer, "medium")
            if is_quality(ex["problem"], ex["solution"], ex["answer"]):
                examples.append(ex)
    random.shuffle(examples)
    examples = examples[:target]
    log_dataset_stats(name, examples)
    save_jsonl(examples, raw_path(name))

# ── 2. MetaMathQA ─────────────────────────────────────────────────────────────

def clean_metamath_response(response):
    text = re.sub(r"^let'?s?\s+think\s+step\s+by\s+step\.?\s*", "",
                  response.strip(), flags=re.IGNORECASE)
    answer = None
    m = re.search(r"the answer is[:\s]+([^\n.]+)", text, re.IGNORECASE)
    if m:
        answer = m.group(1).strip().rstrip(".")
        text   = text[:m.start()].strip()
    if not answer:
        m = re.search(r"####\s*(.+)", text)
        if m:
            answer = m.group(1).strip()
            text   = text[:m.start()].strip()
    if not answer:
        return None, None
    text  = re.sub(r"^\d+\.\s*",      "", text, flags=re.MULTILINE)
    text  = re.sub(r"^Step\s+\d+:?\s*","", text, flags=re.MULTILINE|re.IGNORECASE)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    steps = [l for l in lines if len(l.split()) >= 3]
    return steps, answer

def collect_metamath():
    name = "metamath"
    if already_collected(name): return
    log.info(f"\n{'='*55}\n  [2/10] Collecting MetaMathQA\n{'='*55}")
    target     = DATASET_TARGETS[name]
    keep_types = {"GSM_Rephrased", "MATH_Rephrased"}
    ds         = load_dataset("meta-math/MetaMathQA", streaming=True)
    examples   = []
    for row in ds["train"]:
        if row.get("type","") not in keep_types: continue
        steps, answer = clean_metamath_response(row["response"])
        if not steps or not answer: continue
        diff = "medium" if "GSM" in row.get("type","") else "hard"
        ex   = build_example("metamath", "math_arithmetic",
                             row["query"], steps, answer, diff)
        if is_quality(ex["problem"], ex["solution"], ex["answer"]):
            examples.append(ex)
        if len(examples) >= target * 2:   # early exit — we have enough
            break
    random.shuffle(examples)
    examples = examples[:target]
    log_dataset_stats(name, examples)
    save_jsonl(examples, raw_path(name))

# ── 3. Orca-Math ──────────────────────────────────────────────────────────────

def clean_orca_math_answer(text):
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    answer = None
    for pattern in [r"####\s*([^\n]+)",
                    r"the answer is[:\s]+([^\n.]+)",
                    r"answer[:\s]+([^\n.]+)",
                    r"=\s*\*?\*?\$?([\d,\.]+)\*?\*?$"]:
        m = re.search(pattern, text, re.IGNORECASE|re.MULTILINE)
        if m:
            answer = m.group(1).strip().rstrip(".").replace(",","")
            break
    if not answer:
        lines = [l.strip() for l in text.split("\n") if re.search(r"\d", l)]
        if lines:
            nums = re.findall(r"[\d,\.]+", lines[-1])
            if nums:
                answer = nums[-1].replace(",","")
    if not answer:
        return None, None
    lines = re.split(r"\n+|\.\s+", text)
    steps = []
    for line in lines:
        line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
        if len(line.split()) >= 5 and re.search(r"\d", line):
            steps.append(line)
    if not steps:
        steps = [text[:300]]
    return steps[:8], answer

def collect_orca_math():
    name = "orca_math"
    if already_collected(name): return
    log.info(f"\n{'='*55}\n  [3/10] Collecting Orca-Math\n{'='*55}")
    target   = DATASET_TARGETS[name]
    ds       = load_dataset("microsoft/orca-math-word-problems-200k")
    examples = []
    for row in ds["train"]:
        steps, answer = clean_orca_math_answer(row["answer"])
        if not steps or not answer: continue
        ex = build_example("orca_math", "math_arithmetic",
                           row["question"], steps, answer, "medium")
        if is_quality(ex["problem"], ex["solution"], ex["answer"]):
            examples.append(ex)
    random.shuffle(examples)
    examples = examples[:target]
    log_dataset_stats(name, examples)
    save_jsonl(examples, raw_path(name))

# ── 4. MATH (lighteval) ───────────────────────────────────────────────────────

MATH_DIFF_MAP = {"Level 1":"easy","Level 2":"easy",
                 "Level 3":"medium","Level 4":"hard","Level 5":"hard"}

def clean_math_solution(solution):
    boxed = re.search(r"\\boxed\{([^}]+)\}", solution)
    if not boxed: return None, None
    answer = boxed.group(1).strip()
    text   = solution[:boxed.start()].strip()
    paras  = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if not paras:
        paras = [s.strip() for s in re.split(r"\.(?=\s+[A-Z])", text) if s.strip()]
    steps = [p for p in paras if len(p.split()) >= 5]
    return steps, answer

def collect_math_lighteval():
    name = "math_lighteval"
    if already_collected(name): return
    log.info(f"\n{'='*55}\n  [4/10] Collecting MATH (EleutherAI/hendrycks_math)\n{'='*55}")
    target      = DATASET_TARGETS[name]
    keep_levels = {"Level 1", "Level 2", "Level 3"}
    examples    = []

    # Dataset has one split per math subject — iterate all of them
    subjects = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
    ]

    for subject in subjects:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subject)
        except Exception as e:
            log.warning(f"    Could not load subject '{subject}': {e}")
            continue

        for split in ["train", "test"]:
            if split not in ds: continue
            for row in ds[split]:
                if row.get("level", "") not in keep_levels: continue
                steps, answer = clean_math_solution(row["solution"])
                if not steps or not answer: continue
                diff = MATH_DIFF_MAP.get(row.get("level", ""), "medium")
                ex   = build_example("math_lighteval", "math_hard",
                                     row["problem"], steps, answer, diff)
                if is_quality(ex["problem"], ex["solution"], ex["answer"],
                              min_sol_words=15):
                    examples.append(ex)

    random.shuffle(examples)
    examples = examples[:target]
    log_dataset_stats(name, examples)
    save_jsonl(examples, raw_path(name))

# ── 5. LogiQA ─────────────────────────────────────────────────────────────────

LOGIQA_LABELS = ["A","B","C","D"]

def format_logiqa_example(row):
    context  = row.get("context","").strip()
    question = row.get("query", row.get("question","")).strip()
    options  = row.get("options",[])
    label    = row.get("label", row.get("correct_option", 0))
    if not context or not question or not options: return None
    if isinstance(label, str):
        label = LOGIQA_LABELS.index(label.upper()) \
                if label.upper() in LOGIQA_LABELS else int(label)
    if label >= len(options): return None
    correct_text = options[label].strip()
    option_lines = "\n".join(f"{LOGIQA_LABELS[i]}) {opt.strip()}"
                             for i, opt in enumerate(options))
    problem = f"{context}\n\nQuestion: {question}\n\nOptions:\n{option_lines}"
    sents   = [s.strip() for s in re.split(r"[.!?]", context) if s.strip()]
    key_info = sents[0] if sents else context[:150]
    wrong    = [LOGIQA_LABELS[i] for i in range(len(options)) if i != label]
    steps = [
        f"Identify the key information from the passage: {key_info}.",
        f"The question asks: {question}",
        f"Evaluate the options against the passage. "
        f"Options {', '.join(wrong)} are not directly supported by the key information. "
        f"Option {LOGIQA_LABELS[label]} states: '{correct_text}', "
        f"which is consistent with the passage.",
        f"Option {LOGIQA_LABELS[label]} is correct.",
    ]
    return build_example("logiqa", "logical_deduction", problem, steps,
                         f"{LOGIQA_LABELS[label]}: {correct_text}", "medium")

def collect_logiqa():
    name = "logiqa"
    if already_collected(name): return
    log.info(f"\n{'='*55}\n  [5/10] Collecting LogiQA (GitHub raw)\n{'='*55}")
    import urllib.request, io

    target   = DATASET_TARGETS[name]
    examples = []

    # Original LogiQA data files from the authors' GitHub
    urls = {
        "train" : "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Train.txt",
        "val"   : "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Eval.txt",
        "test"  : "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt",
    }

    def parse_logiqa_txt(text):
        """
        LogiQA .txt format:
            <blank line>
            <label: 'a'/'b'/'c'/'d'>
            <context paragraph>
            <question>
            a) <option>
            b) <option>
            c) <option>
            d) <option>
        Blocks are separated by blank lines.
        """
        label_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        records   = []
        blocks    = re.split(r"\n\s*\n", text.strip())

        for block in blocks:
            lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
            if len(lines) < 7: continue
            try:
                label_char = lines[0].lower().strip()
                if label_char not in label_map: continue
                label   = label_map[label_char]
                context = lines[1]
                question = lines[2]
                opts    = []
                for line in lines[3:]:
                    m = re.match(r"^[abcdABCD][.)]\s*(.+)", line)
                    if m:
                        opts.append(m.group(1).strip())
                if len(opts) != 4: continue
                records.append({
                    "label"   : label,
                    "context" : context,
                    "question": question,
                    "options" : opts,
                })
            except Exception:
                continue
        return records

    for split_name, url in urls.items():
        try:
            log.info(f"    Downloading {split_name} from GitHub...")
            with urllib.request.urlopen(url, timeout=30) as resp:
                text = resp.read().decode("utf-8")
            records = parse_logiqa_txt(text)
            log.info(f"    Parsed {len(records)} records from {split_name}")
        except Exception as e:
            log.warning(f"    Could not download {split_name}: {e}")
            continue

        for row in records:
            context      = row["context"]
            question     = row["question"]
            options      = row["options"]
            label        = row["label"]
            correct_text = options[label].strip()

            option_lines = "\n".join(f"{LOGIQA_LABELS[i]}) {opt.strip()}"
                                     for i, opt in enumerate(options))
            problem = f"{context}\n\nQuestion: {question}\n\nOptions:\n{option_lines}"

            sents    = [s.strip() for s in re.split(r"[.!?]", context) if s.strip()]
            key_info = sents[0] if sents else context[:150]
            wrong    = [LOGIQA_LABELS[i] for i in range(len(options)) if i != label]

            steps = [
                f"Identify the key information from the passage: {key_info}.",
                f"The question asks: {question}",
                f"Evaluate the options against the passage. "
                f"Options {', '.join(wrong)} are not directly supported by the key information. "
                f"Option {LOGIQA_LABELS[label]} states: '{correct_text}', "
                f"which is consistent with the passage.",
                f"Option {LOGIQA_LABELS[label]} is correct.",
            ]
            ex = build_example("logiqa", "logical_deduction", problem, steps,
                               f"{LOGIQA_LABELS[label]}: {correct_text}", "medium")
            if is_quality(ex["problem"], ex["solution"], ex["answer"],
                          min_prob_words=30, min_sol_words=30):
                examples.append(ex)

    random.shuffle(examples)
    examples = examples[:target]
    log_dataset_stats(name, examples)
    save_jsonl(examples, raw_path(name))

# ── 6. ARC-Challenge ──────────────────────────────────────────────────────────

def format_arc_example(row, source_name, difficulty):
    question   = row["question"].strip()
    choices    = row["choices"]
    answer_key = row["answerKey"].strip()
    texts, labels = choices["text"], choices["label"]
    if not texts or not labels or not answer_key: return None
    correct_text = None
    for lbl, txt in zip(labels, texts):
        if str(lbl).strip().upper() == answer_key.upper():
            correct_text = txt.strip(); break
    if not correct_text: return None
    option_lines = "\n".join(f"{str(lbl).strip()}) {txt.strip()}"
                             for lbl, txt in zip(labels, texts))
    problem   = f"{question}\n\nOptions:\n{option_lines}"
    wrong     = [str(lbl).strip() for lbl, txt in zip(labels, texts)
                 if str(lbl).strip().upper() != answer_key.upper()]
    steps = [
        f"Read the question carefully: {question}",
        f"Consider each option. The question asks about a specific scientific concept. "
        f"Options {', '.join(wrong)} do not correctly answer what is asked.",
        f"Option {answer_key} states: '{correct_text}', which correctly answers the question.",
    ]
    return build_example(source_name, "science_reasoning", problem, steps,
                         f"{answer_key}: {correct_text}", difficulty)

def collect_arc_challenge():
    name = "arc_challenge"
    if already_collected(name): return
    log.info(f"\n{'='*55}\n  [6/10] Collecting ARC-Challenge\n{'='*55}")
    target   = DATASET_TARGETS[name]
    ds       = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    examples = []
    for split in ["train","validation","test"]:
        if split not in ds: continue
        for row in ds[split]:
            ex = format_arc_example(row, "arc_challenge", "hard")
            if ex and is_quality(ex["problem"], ex["solution"], ex["answer"],
                                  min_prob_words=10, min_sol_words=20):
                examples.append(ex)
    random.shuffle(examples)
    examples = examples[:target]
    log_dataset_stats(name, examples)
    save_jsonl(examples, raw_path(name))

# ── 7. ARC-Easy ───────────────────────────────────────────────────────────────

def collect_arc_easy():
    name = "arc_easy"
    if already_collected(name): return
    log.info(f"\n{'='*55}\n  [7/10] Collecting ARC-Easy\n{'='*55}")
    target   = DATASET_TARGETS[name]
    ds       = load_dataset("allenai/ai2_arc", "ARC-Easy")
    examples = []
    for split in ["train","validation","test"]:
        if split not in ds: continue
        for row in ds[split]:
            ex = format_arc_example(row, "arc_easy", "easy")
            if ex and is_quality(ex["problem"], ex["solution"], ex["answer"],
                                  min_prob_words=10, min_sol_words=20):
                examples.append(ex)
    random.shuffle(examples)
    examples = examples[:target]
    log_dataset_stats(name, examples)
    save_jsonl(examples, raw_path(name))

# ── 8. SciQ ───────────────────────────────────────────────────────────────────

def format_sciq_example(row):
    question       = row["question"].strip()
    correct_answer = row["correct_answer"].strip()
    support        = row.get("support","").strip()
    distractors    = [row.get(f"distractor{i}","").strip() for i in range(1,4)]
    distractors    = [d for d in distractors if d]
    if not question or not correct_answer: return None
    all_opts = distractors + [correct_answer]
    random.shuffle(all_opts)
    labels        = ["A","B","C","D"]
    correct_label = None
    option_lines  = []
    for i, opt in enumerate(all_opts[:4]):
        option_lines.append(f"{labels[i]}) {opt}")
        if opt == correct_answer:
            correct_label = labels[i]
    if not correct_label: return None
    problem = f"{question}\n\nOptions:\n" + "\n".join(option_lines)
    steps   = []
    if support and len(support.split()) >= 10:
        steps.append(f"Background knowledge: {support}")
    steps.append(f"The question asks: {question}")
    steps.append(f"Based on the supporting information, '{correct_answer}' is correct. "
                 f"Option {correct_label} states this directly.")
    return build_example("sciq", "science_reasoning", problem, steps,
                         f"{correct_label}: {correct_answer}", "easy")

def collect_sciq():
    name = "sciq"
    if already_collected(name): return
    log.info(f"\n{'='*55}\n  [8/10] Collecting SciQ\n{'='*55}")
    target   = DATASET_TARGETS[name]
    ds       = load_dataset("allenai/sciq")
    examples = []
    for split in ["train","validation","test"]:
        if split not in ds: continue
        for row in ds[split]:
            if not row.get("support","").strip(): continue
            ex = format_sciq_example(row)
            if ex and is_quality(ex["problem"], ex["solution"], ex["answer"],
                                  min_prob_words=10, min_sol_words=20):
                examples.append(ex)
    random.shuffle(examples)
    examples = examples[:target]
    log_dataset_stats(name, examples)
    save_jsonl(examples, raw_path(name))

# ── 9. OpenBookQA ─────────────────────────────────────────────────────────────

def format_openbookqa_example(row):
    question   = row["question_stem"].strip()
    choices    = row["choices"]
    answer_key = row["answerKey"].strip()
    core_fact  = row.get("fact1","").strip()
    texts, labels = choices["text"], choices["label"]
    correct_text  = None
    for lbl, txt in zip(labels, texts):
        if str(lbl).strip().upper() == answer_key.upper():
            correct_text = txt.strip(); break
    if not correct_text: return None
    option_lines = "\n".join(f"{str(lbl).strip()}) {txt.strip()}"
                             for lbl, txt in zip(labels, texts))
    problem = f"{question}\n\nOptions:\n{option_lines}"
    steps   = []
    if core_fact:
        steps.append(f"Relevant scientific fact: {core_fact}")
    steps.append(f"The question asks: {question}")
    steps.append(f"Applying the scientific fact to the options, "
                 f"option {answer_key} ('{correct_text}') is the correct answer.")
    return build_example("openbookqa", "science_reasoning", problem, steps,
                         f"{answer_key}: {correct_text}", "easy")

def collect_openbookqa():
    name = "openbookqa"
    if already_collected(name): return
    log.info(f"\n{'='*55}\n  [9/10] Collecting OpenBookQA\n{'='*55}")
    target   = DATASET_TARGETS[name]
    ds       = load_dataset("allenai/openbookqa", "main")
    examples = []
    for split in ["train","validation","test"]:
        if split not in ds: continue
        for row in ds[split]:
            ex = format_openbookqa_example(row)
            if ex and is_quality(ex["problem"], ex["solution"], ex["answer"],
                                  min_prob_words=10, min_sol_words=15):
                examples.append(ex)
    random.shuffle(examples)
    examples = examples[:target]
    log_dataset_stats(name, examples)
    save_jsonl(examples, raw_path(name))

# ── 10. OpenHermes-2.5 ────────────────────────────────────────────────────────

def extract_openhermes_pair(row):
    convs      = row.get("conversations",[])
    human_turn = None
    gpt_turn   = None
    for turn in convs:
        if turn.get("from") == "human" and human_turn is None:
            human_turn = turn.get("value","").strip()
        elif turn.get("from") == "gpt" and gpt_turn is None:
            gpt_turn = turn.get("value","").strip()
    return human_turn, gpt_turn

def is_openhermes_reasoning(problem, solution):
    if not problem or not solution: return False
    sol_lower    = solution.lower()
    has_steps    = bool(re.search(r"step\s+\d|first,|second,|third,|finally,|therefore", sol_lower))
    has_calc     = bool(re.search(r"\d+\s*[\+\-\*\/=]\s*\d+", solution))
    has_markers  = bool(re.search(r"because|since|thus|hence|as a result|which means", sol_lower))
    has_numbered = bool(re.search(r"^\d+[\.\)]\s+", solution, re.MULTILINE))
    return has_steps or has_calc or (has_markers and has_numbered)

def format_openhermes_solution(problem, solution):
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", solution)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    answer = None
    for pattern in [r"therefore[,\s]+the answer is[:\s]+([^\n.]+)",
                    r"####\s*([^\n]+)",
                    r"the answer is[:\s]+([^\n.]+)",
                    r"answer[:\s]+([^\n.]+)"]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            answer = m.group(1).strip().rstrip(".")
            text   = text[:m.start()].strip()
            break
    if not answer:
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
        answer    = sentences[-1][:80] if sentences else "See solution."
        if sentences:
            text  = ". ".join(sentences[:-1])
    numbered = re.split(r"\n\d+[\.\)]\s+", "\n" + text)
    if len(numbered) > 2:
        steps = [s.strip() for s in numbered if s.strip()]
    else:
        stepped = re.split(r"\nStep\s+\d+:?\s*", "\n" + text, flags=re.IGNORECASE)
        if len(stepped) > 2:
            steps = [s.strip() for s in stepped if s.strip()]
        else:
            paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
            steps = paras if paras else [text[:500]]
    steps = [s[:300] for s in steps[:8] if len(s.split()) >= 5]
    return steps, answer

def collect_openhermes():
    name = "openhermes"
    if already_collected(name): return
    log.info(f"\n{'='*55}\n  [10/10] Collecting OpenHermes-2.5\n{'='*55}")
    target   = DATASET_TARGETS[name]
    ds       = load_dataset("teknium/OpenHermes-2.5")
    examples = []
    for row in ds["train"]:
        category = str(row.get("category","")).lower()
        source   = str(row.get("source","")).lower()
        if not any(k in category or k in source
                   for k in OPENHERMES_KEEP_CATEGORIES):
            continue
        problem, solution = extract_openhermes_pair(row)
        if not problem or not solution: continue
        if not is_openhermes_reasoning(problem, solution): continue
        steps, answer = format_openhermes_solution(problem, solution)
        if not steps: continue
        ex = build_example("openhermes", "cs_technical",
                           problem, steps, answer, "medium")
        if is_quality(ex["problem"], ex["solution"], ex["answer"],
                      min_prob_words=10, max_prob_words=300,
                      min_sol_words=30,  max_sol_words=500):
            examples.append(ex)
        if len(examples) >= target * 3:
            break
    random.shuffle(examples)
    examples = examples[:target]
    log_dataset_stats(name, examples)
    save_jsonl(examples, raw_path(name))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — PRE-PROCESS EACH RAW FILE
# Reload each raw file, apply any dataset-specific post-processing,
# and write a clean processed file.
# ══════════════════════════════════════════════════════════════════════════════

PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def processed_path(name):
    return os.path.join(PROCESSED_DIR, f"{name}.jsonl")

def already_processed(name):
    p = processed_path(name)
    if os.path.exists(p):
        count = sum(1 for _ in open(p, encoding="utf-8"))
        log.info(f"  [{name}] already processed — {count:,} examples. Skipping.")
        return True
    return False

def normalize_answer(answer_str):
    """Standardise answer strings across all sources."""
    a = answer_str.strip()
    # Strip trailing punctuation
    a = a.rstrip(".")
    # Remove currency symbols for numeric comparison later
    a = a.replace("$","").replace(",","").strip()
    return a

def normalize_solution(solution_str):
    """
    Strip any leftover markdown and normalise whitespace.
    Ensure the termination line is always present and clean.
    """
    # Remove markdown bold/italics
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", solution_str)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Ensure termination phrase exists at the end
    if "therefore, the answer is" not in text.lower():
        text = text.rstrip() + "\n\nTherefore, the answer is [see answer field]."
    return text.strip()

def verify_step_structure(solution_str):
    """
    For most sources: require at least 2 distinct Step N: lines.
    """
    step_lines = re.findall(r"Step\s+\d+:\s*(.+)", solution_str, re.IGNORECASE)
    if len(step_lines) < 2:
        return False
    for i in range(1, len(step_lines)):
        if step_lines[i].strip() == step_lines[i-1].strip():
            return False
    return True


def verify_openhermes_structure(solution_str):
    """
    OpenHermes uses prose/numbered lists — not Step N: format.
    Accept if it has enough reasoning signal and is not degenerate.
    """
    words = solution_str.split()
    if len(words) < 30:
        return False
    # Not degenerate
    if len(words) > 20:
        ngrams = [tuple(words[i:i+5]) for i in range(len(words)-5)]
        if ngrams and len(set(ngrams)) / len(ngrams) < 0.5:
            return False
    # Must have at least one reasoning marker
    sol_lower = solution_str.lower()
    has_reasoning = bool(re.search(
        r"step\s+\d|first,|second,|third,|therefore|because|since|thus|hence|"
        r"as a result|\d+\s*[\+\-\*\/=]\s*\d+",
        sol_lower
    ))
    return has_reasoning


def reformat_openhermes_to_steps(solution_str):
    """
    Force OpenHermes prose into Step N: format during pre-processing.
    Splits by numbered list, existing steps, or paragraphs.
    """
    text = solution_str

    # Remove the termination line before reformatting
    term_match = re.search(r"\n\nTherefore, the answer is.+$", text)
    answer_line = ""
    if term_match:
        answer_line = term_match.group(0)
        text = text[:term_match.start()]

    # Try numbered list
    parts = re.split(r"\n\d+[\.\)]\s+", "\n" + text)
    if len(parts) > 2:
        steps = [p.strip() for p in parts if p.strip()]
    else:
        # Try paragraph split
        parts = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        if len(parts) >= 2:
            steps = parts
        else:
            # Sentence split as last resort
            steps = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    # Filter trivially short steps
    steps = [s for s in steps if len(s.split()) >= 5]

    if len(steps) < 2:
        return None  # Still can't make a valid chain — reject

    # Rebuild in Step N: format
    step_lines = "\n\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps[:8]))
    return step_lines + answer_line


def preprocess_dataset(name):
    if already_processed(name): return
    log.info(f"\n  [preprocess] {name}")
    raw      = load_jsonl(raw_path(name))
    cleaned  = []
    rejected = 0

    for ex in raw:
        ex["answer"]   = normalize_answer(ex["answer"])
        ex["solution"] = normalize_solution(ex["solution"])

        # Source-specific step verification
        if name == "openhermes":
            if not verify_openhermes_structure(ex["solution"]):
                rejected += 1
                continue
            # Reformat into Step N: structure
            reformatted = reformat_openhermes_to_steps(ex["solution"])
            if reformatted is None:
                rejected += 1
                continue
            ex["solution"] = reformatted
        else:
            if not verify_step_structure(ex["solution"]):
                rejected += 1
                continue

        if not is_quality(ex["problem"], ex["solution"], ex["answer"]):
            rejected += 1
            continue

        cleaned.append(ex)

    log.info(f"    Raw: {len(raw):,}  →  Clean: {len(cleaned):,}  "
             f"(rejected {rejected:,} = {100*rejected/max(1,len(raw)):.1f}%)")
    save_jsonl(cleaned, processed_path(name))
    
# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — MASTER DATASET: MERGE + DEDUP + SPLIT + SAVE
# ══════════════════════════════════════════════════════════════════════════════

def dedup(examples):
    seen = set()
    out  = []
    for ex in examples:
        key = hashlib.md5(ex["problem"].lower().strip().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out

def stratified_split(examples, val_ratio=VAL_RATIO):
    by_cat = defaultdict(list)
    for ex in examples:
        by_cat[ex["category"]].append(ex)
    train, val = [], []
    for cat, items in by_cat.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val

def print_split_stats(examples, label):
    by_cat  = defaultdict(int)
    by_src  = defaultdict(int)
    by_diff = defaultdict(int)
    sol_lens = [len(ex["solution"].split()) for ex in examples]
    for ex in examples:
        by_cat[ex["category"]]    += 1
        by_src[ex["source"]]      += 1
        by_diff[ex["difficulty"]] += 1

    log.info(f"\n  ── {label} ──")
    log.info(f"  Total          : {len(examples):,}")
    log.info(f"  Avg sol length : {sum(sol_lens)//len(sol_lens)} words")
    log.info(f"\n  By category:")
    for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        pct = 100 * n / len(examples)
        log.info(f"    {cat:<30} {n:>7,}  ({pct:.1f}%)")
    log.info(f"\n  By source:")
    for src, n in sorted(by_src.items(), key=lambda x: -x[1]):
        log.info(f"    {src:<30} {n:>7,}")
    log.info(f"\n  By difficulty:")
    for diff, n in sorted(by_diff.items(), key=lambda x: -x[1]):
        log.info(f"    {diff:<15} {n:>7,}")

def build_master():
    log.info(f"\n{'='*55}\n  PHASE 3 — Building Master Dataset\n{'='*55}")

    master_train_path = os.path.join(MASTER_DIR, "sft_train.jsonl")
    master_val_path   = os.path.join(MASTER_DIR, "sft_val.jsonl")
    sample_path       = os.path.join(MASTER_DIR, "sft_sample_20.jsonl")

    if os.path.exists(master_train_path) and os.path.exists(master_val_path):
        t = sum(1 for _ in open(master_train_path))
        v = sum(1 for _ in open(master_val_path))
        log.info(f"  Master already exists — train={t:,}  val={v:,}. Skipping.")
        return

    # Load all processed files
    all_examples = []
    for name in DATASET_TARGETS:
        p = processed_path(name)
        if not os.path.exists(p):
            log.warning(f"  WARNING: processed file missing for {name} — skipping")
            continue
        batch = load_jsonl(p)
        log.info(f"  Loaded {name:<20} {len(batch):>7,} examples")
        all_examples.extend(batch)

    log.info(f"\n  Total before dedup : {len(all_examples):,}")
    all_examples = dedup(all_examples)
    log.info(f"  Total after dedup  : {len(all_examples):,}")

    # Shuffle master before split
    random.shuffle(all_examples)

    # Stratified split
    train, val = stratified_split(all_examples, VAL_RATIO)

    # Stats
    print_split_stats(train, "Train Set")
    print_split_stats(val,   "Val Set")

    # Save
    save_jsonl(train, master_train_path)
    save_jsonl(val,   master_val_path)

    # 20-example manual inspection sample
    sample = random.sample(train, min(20, len(train)))
    save_jsonl(sample, sample_path)

    log.info(f"\n  Master dataset complete.")
    log.info(f"    Train : {len(train):,}  →  {master_train_path}")
    log.info(f"    Val   : {len(val):,}    →  {master_val_path}")
    log.info(f"    Sample: {sample_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Three explicit phases, clearly separated
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("="*55)
    log.info("  SFT Data Collection Pipeline")
    log.info(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("="*55)

    # ── PHASE 1: Collect + write each dataset ─────────────────────────────────
    log.info("\n" + "="*55)
    log.info("  PHASE 1 — Dataset Collection")
    log.info("="*55)

    collect_gsm8k()
    collect_metamath()
    collect_orca_math()
    collect_math_lighteval()
    collect_logiqa()
    collect_arc_challenge()
    collect_arc_easy()
    collect_sciq()
    collect_openbookqa()
    collect_openhermes()

    # ── PHASE 2: Pre-process each raw file ────────────────────────────────────
    log.info("\n" + "="*55)
    log.info("  PHASE 2 — Pre-processing")
    log.info("="*55)

    for name in DATASET_TARGETS:
        preprocess_dataset(name)

    # ── PHASE 3: Merge + dedup + split + save master ──────────────────────────
    build_master()

    log.info(f"\n  Pipeline complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"\n  Directory layout:")
    log.info(f"    data/sft/raw/          — one raw .jsonl per dataset")
    log.info(f"    data/sft/processed/    — cleaned, normalised per dataset")
    log.info(f"    data/sft/master/       — sft_train.jsonl, sft_val.jsonl")
    log.info("="*55)