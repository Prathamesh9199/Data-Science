# evaluate_sft.py
# Evaluates the SFT model and compares directly against the base model.
# Every section measures something SFT was specifically supposed to improve.
# Outputs everything to sft_evaluation_results.txt

import os
import re
import sys
import math
import time
import json
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from model_architecture import CustomTransformer
from config import SLMConfig

cfg    = SLMConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Tee output to file ────────────────────────────────────────────────────────

class Tee:
    def __init__(self, filepath):
        self.file   = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data); self.stdout.write(data)
    def flush(self):
        self.file.flush(); self.stdout.flush()
    def close(self):
        self.file.close()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path, label):
    print(f"  Loading [{label}] from {ckpt_path}...")
    model = CustomTransformer(cfg).to(device).to(torch.bfloat16)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
             for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    if "val_loss" in ckpt:
        print(f"    val_loss={ckpt['val_loss']:.4f}  params={params/1e6:.1f}M")
    else:
        print(f"    params={params/1e6:.1f}M")
    return model


# ── Generation primitives ─────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt,
             max_new_tokens=300, temperature=0.7, top_k=50, top_p=0.9):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    gen = ids.clone()
    for _ in range(max_new_tokens):
        ctx = gen[:, -cfg.max_seq_len:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(ctx)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1:]] = float("-inf")
        if top_p < 1.0:
            sl, si = torch.sort(logits, descending=True)
            cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            sl[cp - F.softmax(sl, dim=-1) > top_p] = float("-inf")
            logits = sl.scatter(1, si, sl)
        next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
        gen = torch.cat([gen, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(gen[0, ids.shape[1]:].tolist(), skip_special_tokens=True)


@torch.no_grad()
def continuation_log_prob(model, tokenizer, prefix, continuation):
    prefix_ids       = tokenizer.encode(prefix,       add_special_tokens=True)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    full_ids = torch.tensor([prefix_ids + continuation_ids], device=device)
    if full_ids.shape[1] > cfg.max_seq_len:
        full_ids = full_ids[:, -cfg.max_seq_len:]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids)
    log_probs  = F.log_softmax(logits, dim=-1)
    cont_start = len(prefix_ids) - 1
    cont_len   = len(continuation_ids)
    if cont_start + cont_len > log_probs.shape[1]:
        return float("-inf")
    scores = log_probs[0, cont_start:cont_start+cont_len, :].gather(
        1, full_ids[0, len(prefix_ids):len(prefix_ids)+cont_len].unsqueeze(-1)
    ).squeeze(-1)
    return scores.mean().item()


# ── Text quality helpers ──────────────────────────────────────────────────────

def repetition_rate(text, n=4):
    tokens = text.split()
    if len(tokens) < n: return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return 1 - len(set(ngrams)) / len(ngrams)

def type_token_ratio(text):
    tokens = text.lower().split()
    return len(set(tokens)) / len(tokens) if tokens else 0.0


# =============================================================================
# SECTION 1 — FORMAT COMPLIANCE
# The most basic SFT test: does the model now follow the
# Problem/Solution/Step/Therefore format it was trained on?
# =============================================================================

FORMAT_PROBES = [
    "Problem: If a factory produces 150 units per day, how many units does it produce in 6 days?\n\nSolution:",
    "Problem: A rectangle has length 12 and width 5. What is its area?\n\nSolution:",
    "Problem: All mammals are warm-blooded. A whale is a mammal. Is a whale warm-blooded?\n\nSolution:",
    "Problem: There are 3 red marbles and 5 blue marbles in a bag. How many marbles are there in total?\n\nSolution:",
    "Problem: Sara has 20 candies. She gives 7 to her friend and eats 3 herself. How many does she have left?\n\nSolution:",
    "Problem: A car travels at 60 miles per hour. How far does it travel in 2.5 hours?\n\nSolution:",
    "Problem: If log₂(x) = 4, what is x?\n\nSolution:",
    "Problem: Water boils at 100°C at sea level. What happens to water if it is heated above this temperature?\n\nSolution:",
]

def check_format(output):
    """
    Returns a dict of format compliance signals.
    Each signal is what SFT was explicitly trained to produce.
    """
    has_step        = bool(re.search(r"Step\s+\d+:", output, re.IGNORECASE))
    n_steps         = len(re.findall(r"Step\s+\d+:", output, re.IGNORECASE))
    has_termination = "therefore, the answer is" in output.lower()
    has_answer_val  = bool(re.search(
        r"therefore, the answer is\s+[\w\d\$\.\-\/]+", output, re.IGNORECASE
    ))
    # Steps are advancing: check that adjacent step content differs
    step_contents = re.findall(r"Step\s+\d+:\s*(.+?)(?=Step\s+\d+:|Therefore|$)",
                               output, re.IGNORECASE | re.DOTALL)
    step_contents = [s.strip()[:80] for s in step_contents]
    steps_advance = len(step_contents) >= 2 and \
                    all(step_contents[i] != step_contents[i-1]
                        for i in range(1, len(step_contents)))
    rep           = repetition_rate(output, n=4)
    is_degenerate = rep > 0.3

    return {
        "has_step"       : has_step,
        "n_steps"        : n_steps,
        "has_termination": has_termination,
        "has_answer_val" : has_answer_val,
        "steps_advance"  : steps_advance,
        "is_degenerate"  : is_degenerate,
        "rep_rate"       : round(rep, 3),
    }

def evaluate_format_compliance(models_dict, tokenizer):
    print("\n" + "="*65)
    print("  SECTION 1 — Format Compliance")
    print("  Did SFT teach the model to follow the Step/Therefore format?")
    print("="*65)

    results = {}
    for model_label, model in models_dict.items():
        results[model_label] = []
        print(f"\n  ── {model_label} ──")
        for prompt in FORMAT_PROBES:
            out    = generate(model, tokenizer, prompt,
                              max_new_tokens=250, temperature=0.6)
            fmt    = check_format(out)
            results[model_label].append(fmt)
            print(f"\n  Prompt : {prompt[:70]}")
            print(f"  Output : {out[:300]}")
            print(f"  steps={fmt['n_steps']}  "
                  f"termination={'✓' if fmt['has_termination'] else '✗'}  "
                  f"advances={'✓' if fmt['steps_advance'] else '✗'}  "
                  f"degen={'✗' if fmt['is_degenerate'] else '✓'}  "
                  f"rep={fmt['rep_rate']}")

    # Summary table
    print(f"\n  {'Metric':<30}", end="")
    for label in models_dict:
        print(f"  {label:>10}", end="")
    print()
    print(f"  {'-'*55}")

    metrics = ["has_step", "has_termination", "steps_advance", "is_degenerate"]
    labels  = ["Has Step N: structure", "Has Therefore termination",
               "Steps advance (no loop)", "Degenerate output"]
    invert  = {"is_degenerate"}  # lower is better for these

    for metric, label in zip(metrics, labels):
        print(f"  {label:<30}", end="")
        for model_label in models_dict:
            vals = [r[metric] for r in results[model_label]]
            rate = np.mean(vals)
            if metric in invert:
                mark = "✓" if rate < 0.2 else "✗"
            else:
                mark = "✓" if rate > 0.7 else "✗"
            print(f"  {rate:.0%} {mark:>5}", end="")
        print()

    return results


# =============================================================================
# SECTION 2 — ARITHMETIC EXECUTION
# The #1 gap from the base eval: model set up operations correctly
# but hallucinated the result. This section tests if SFT fixed it.
# =============================================================================

ARITHMETIC_PROBLEMS = [
    # (problem_in_format, expected_answer)
    ("Problem: A factory produces 150 units per day. How many units does it produce in 6 days?\n\nSolution:", "900"),
    ("Problem: A cyclist rides at 18 miles per hour. How far does she ride in 4 hours?\n\nSolution:", "72"),
    ("Problem: A printer prints 30 pages per minute. How many pages does it print in 8 minutes?\n\nSolution:", "240"),
    ("Problem: If you have 3 apples and get 2 more, how many apples do you have?\n\nSolution:", "5"),
    ("Problem: There were 5 birds on a tree. 2 flew away. How many birds are left?\n\nSolution:", "3"),
    ("Problem: A boy had 10 candies. He gave 3 to his sister and 2 to his friend. How many does he have now?\n\nSolution:", "5"),
    ("Problem: A rectangle has length 8 and width 5. What is its area?\n\nSolution:", "40"),
    ("Problem: If 3x + 7 = 22, what is x?\n\nSolution:", "5"),
    ("Problem: What is 15% of 240?\n\nSolution:", "36"),
    ("Problem: A store sells apples for $0.50 each. Maria buys 4 apples and pays with a $5 bill. How much change does she receive?\n\nSolution:", "3"),
    ("Problem: A car gets 35 miles per gallon. Gas costs $3.50 per gallon. How much does a 350-mile trip cost?\n\nSolution:", "35"),
    ("Problem: A recipe for 4 people needs 2 cups of flour. How many cups are needed for 10 people?\n\nSolution:", "5"),
]

def extract_final_answer(output):
    """
    Extract the answer after 'Therefore, the answer is'.
    Falls back to last number in output.
    """
    m = re.search(r"therefore,?\s+the answer is[:\s]+([^\n.]+)",
                  output, re.IGNORECASE)
    if m:
        # Pull first number-like token from the answer phrase
        nums = re.findall(r"[\d,\.]+", m.group(1))
        if nums:
            return nums[0].replace(",","")
    # Fallback: last standalone number in output
    nums = re.findall(r"(?<!\w)[\d,\.]+(?!\w)", output)
    if nums:
        return nums[-1].replace(",","")
    return None

def answers_match(extracted, expected):
    if extracted is None: return False
    try:
        return abs(float(extracted) - float(expected)) < 0.01
    except ValueError:
        return extracted.strip().lower() == expected.strip().lower()

def evaluate_arithmetic(models_dict, tokenizer):
    print("\n" + "="*65)
    print("  SECTION 2 — Arithmetic Execution Accuracy")
    print("  Base model set up operations correctly but got wrong answers.")
    print("  Did SFT fix the execution?")
    print("="*65)

    results = {}
    for model_label, model in models_dict.items():
        correct = 0
        results[model_label] = []
        print(f"\n  ── {model_label} ──")
        for prompt, expected in ARITHMETIC_PROBLEMS:
            out       = generate(model, tokenizer, prompt,
                                 max_new_tokens=250, temperature=0.5)
            extracted = extract_final_answer(out)
            is_correct = answers_match(extracted, expected)
            correct   += int(is_correct)
            mark       = "✓" if is_correct else "✗"
            results[model_label].append(is_correct)
            print(f"  {mark}  {prompt[:60].strip()}")
            print(f"       expected={expected}  extracted={extracted}")
            # Show first 2 lines of output for debugging
            first_lines = " | ".join(out.strip().split("\n")[:2])
            print(f"       output: {first_lines[:120]}")

        acc = correct / len(ARITHMETIC_PROBLEMS)
        print(f"\n  Accuracy: {correct}/{len(ARITHMETIC_PROBLEMS)} = {acc:.0%}")
        results[model_label + "_acc"] = acc

    print(f"\n  {'Model':<20} {'Accuracy':>10}  {'vs Base':>10}")
    print(f"  {'-'*42}")
    base_acc = results.get("BASE_acc", 0)
    for label in models_dict:
        acc  = results.get(label + "_acc", 0)
        delta = acc - base_acc
        delta_str = f"+{delta:.0%}" if delta >= 0 else f"{delta:.0%}"
        print(f"  {label:<20} {acc:>10.0%}  {delta_str:>10}")

    return results


# =============================================================================
# SECTION 3 — CHAIN-OF-THOUGHT QUALITY
# Tests whether reasoning chains now genuinely advance vs loop.
# Uses the same 8 problems from base eval for direct comparison.
# =============================================================================

COT_PROBLEMS = [
    {
        "label"   : "factory_widgets",
        "prompt"  : "Problem: A factory produces 240 widgets per day. It operates 6 days a week. How many widgets does it produce in 4 weeks?\n\nSolution:",
        "expected": "5760",
        "keywords": ["240", "6", "1440", "4", "5760"],
    },
    {
        "label"   : "jacket_discount",
        "prompt"  : "Problem: A jacket costs $80. It is 25% off. After the discount, 10% tax is applied. What is the final price?\n\nSolution:",
        "expected": "66",
        "keywords": ["80", "25", "60", "10", "66"],
    },
    {
        "label"   : "logical_chain",
        "prompt"  : "Problem: All professors at this university have a PhD. Dr. Smith is a professor here. Does Dr. Smith have a PhD?\n\nSolution:",
        "expected": "yes",
        "keywords": ["yes", "professor", "phd", "therefore", "smith"],
    },
    {
        "label"   : "pipe_rate",
        "prompt"  : "Problem: Alice can paint a fence in 3 hours. Bob can paint the same fence in 6 hours. If they work together, how long does it take?\n\nSolution:",
        "expected": "2",
        "keywords": ["1/3", "1/6", "1/2", "2", "rate"],
    },
    {
        "label"   : "container_fill",
        "prompt"  : "Problem: A container holds 50 liters. It is 40% full. How many liters must be added to make it 90% full?\n\nSolution:",
        "expected": "25",
        "keywords": ["50", "40", "20", "90", "45", "25"],
    },
    {
        "label"   : "plan_comparison",
        "prompt"  : "Problem: Plan A costs $200 upfront and $10 per month. Plan B costs $0 upfront and $25 per month. After how many months is Plan A cheaper?\n\nSolution:",
        "expected": "14",
        "keywords": ["200", "10", "25", "15", "14"],
    },
    {
        "label"   : "half_life",
        "prompt"  : "Problem: A radioactive element has a half-life of 5 years. You start with 800 grams. How much remains after 15 years?\n\nSolution:",
        "expected": "100",
        "keywords": ["800", "400", "200", "100", "5", "15"],
    },
    {
        "label"   : "code_trace",
        "prompt"  : "Problem: Consider this pseudocode:\n  x = 10\n  for i in range(3):\n      x = x * 2 - 1\nWhat is the value of x after the loop?\n\nSolution:",
        "expected": "73",
        "keywords": ["10", "19", "37", "73"],
    },
]

def score_cot(output, keywords):
    step_lines    = re.findall(r"Step\s+\d+:\s*(.+)", output, re.IGNORECASE)
    n_steps       = len(step_lines)
    has_term      = "therefore, the answer is" in output.lower()
    kw_hits       = sum(1 for k in keywords if k.lower() in output.lower())
    kw_cov        = kw_hits / len(keywords) if keywords else 0
    rep           = repetition_rate(output, n=4)
    is_degen      = rep > 0.3

    # Steps advance check
    steps_advance = False
    if len(step_lines) >= 2:
        steps_advance = all(step_lines[i].strip()[:60] != step_lines[i-1].strip()[:60]
                            for i in range(1, len(step_lines)))

    score = 0
    if not is_degen:       score += 1
    if kw_cov > 0.3:       score += 1
    if n_steps >= 2 and steps_advance: score += 1
    if has_term:           score += 1

    return {
        "score"        : score,
        "n_steps"      : n_steps,
        "steps_advance": steps_advance,
        "has_term"     : has_term,
        "kw_cov"       : round(kw_cov, 2),
        "is_degen"     : is_degen,
        "rep"          : round(rep, 3),
    }

def evaluate_cot_quality(models_dict, tokenizer):
    print("\n" + "="*65)
    print("  SECTION 3 — Chain-of-Thought Quality")
    print("  Same 8 problems as base eval — direct comparison.")
    print("="*65)

    all_results = {}
    for model_label, model in models_dict.items():
        all_results[model_label] = []
        print(f"\n  ── {model_label} ──")
        for item in COT_PROBLEMS:
            out      = generate(model, tokenizer, item["prompt"],
                                max_new_tokens=350, temperature=0.6)
            metrics  = score_cot(out, item["keywords"])
            ans      = extract_final_answer(out)
            correct  = answers_match(ans, item["expected"])
            metrics["correct"] = correct
            all_results[model_label].append(metrics)
            print(f"\n  [{item['label']}]")
            print(f"  {out[:400]}")
            print(f"  score={metrics['score']}/4  steps={metrics['n_steps']}  "
                  f"advance={'✓' if metrics['steps_advance'] else '✗'}  "
                  f"kw={metrics['kw_cov']:.0%}  "
                  f"term={'✓' if metrics['has_term'] else '✗'}  "
                  f"ans={'✓' if correct else '✗'} (got={ans}, want={item['expected']})")

    # Summary
    print(f"\n  {'Metric':<25}", end="")
    for label in models_dict:
        print(f"  {label:>10}", end="")
    print()
    print(f"  {'-'*50}")

    for metric, label in [
        ("score",         "Avg CoT score (/4)"),
        ("n_steps",       "Avg steps produced"),
        ("steps_advance", "Steps advance rate"),
        ("has_term",      "Termination rate"),
        ("kw_cov",        "Avg keyword coverage"),
        ("correct",       "Answer accuracy"),
    ]:
        print(f"  {label:<25}", end="")
        for model_label in models_dict:
            vals = [r[metric] for r in all_results[model_label]]
            mean = np.mean(vals)
            print(f"  {mean:>10.2f}", end="")
        print()

    return all_results


# =============================================================================
# SECTION 4 — ANSWER TERMINATION CONSISTENCY
# Base model had greedy answer accuracy of 12% vs sampling 38%.
# After SFT, greedy should match or beat sampling.
# =============================================================================

@torch.no_grad()
def generate_greedy(model, tokenizer, prompt, max_new_tokens=300):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    gen = ids.clone()
    for _ in range(max_new_tokens):
        ctx = gen[:, -cfg.max_seq_len:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(ctx)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen = torch.cat([gen, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(gen[0, ids.shape[1]:].tolist(), skip_special_tokens=True)

TERMINATION_PROBES = [
    ("Problem: 5 + 3 = ?\n\nSolution:", "8"),
    ("Problem: A bag has 2 red marbles and 6 blue marbles. How many marbles in total?\n\nSolution:", "8"),
    ("Problem: If today is Wednesday, what day will it be in 3 days?\n\nSolution:", "saturday"),
    ("Problem: There are 3 cats and 4 dogs in a room. How many animals are there?\n\nSolution:", "7"),
    ("Problem: A pizza is cut into 8 equal slices. 3 slices are eaten. How many remain?\n\nSolution:", "5"),
]

def evaluate_termination(models_dict, tokenizer):
    print("\n" + "="*65)
    print("  SECTION 4 — Answer Termination (Greedy vs Sampling)")
    print("  Base model: greedy=12%, sampling=38%.")
    print("  After SFT: greedy should match or exceed sampling.")
    print("="*65)

    results = {}
    for model_label, model in models_dict.items():
        greedy_correct  = 0
        sampling_correct = 0
        results[model_label] = []
        print(f"\n  ── {model_label} ──")
        for prompt, expected in TERMINATION_PROBES:
            out_greedy   = generate_greedy(model, tokenizer, prompt, max_new_tokens=200)
            out_sampling = generate(model, tokenizer, prompt,
                                    max_new_tokens=200, temperature=0.6)

            ans_g = extract_final_answer(out_greedy)
            ans_s = extract_final_answer(out_sampling)
            g_ok  = answers_match(ans_g, expected)
            s_ok  = answers_match(ans_s, expected)
            greedy_correct   += int(g_ok)
            sampling_correct += int(s_ok)

            has_term_g = "therefore, the answer is" in out_greedy.lower()
            has_term_s = "therefore, the answer is" in out_sampling.lower()

            print(f"\n  Prompt: {prompt[:60].strip()}")
            print(f"  Greedy   [term={'✓' if has_term_g else '✗'}  ans={'✓' if g_ok else '✗'}]: "
                  f"{out_greedy[:120].strip()}")
            print(f"  Sampling [term={'✓' if has_term_s else '✗'}  ans={'✓' if s_ok else '✗'}]: "
                  f"{out_sampling[:120].strip()}")

        n = len(TERMINATION_PROBES)
        results[model_label] = {
            "greedy_acc"  : greedy_correct   / n,
            "sampling_acc": sampling_correct / n,
        }
        print(f"\n  Greedy accuracy  : {greedy_correct}/{n} = {greedy_correct/n:.0%}")
        print(f"  Sampling accuracy: {sampling_correct}/{n} = {sampling_correct/n:.0%}")

    print(f"\n  {'Model':<20} {'Greedy':>10}  {'Sampling':>10}  {'G≥S?':>8}")
    print(f"  {'-'*52}")
    for label, res in results.items():
        g_s = "✓" if res["greedy_acc"] >= res["sampling_acc"] else "✗"
        print(f"  {label:<20} {res['greedy_acc']:>10.0%}  {res['sampling_acc']:>10.0%}  {g_s:>8}")

    return results


# =============================================================================
# SECTION 5 — KNOWLEDGE PRESERVATION
# SFT risk: catastrophic forgetting. Checks that base model knowledge
# encoded in weights (Section B of base eval, 87%) was not damaged.
# =============================================================================

KNOWLEDGE_COMPLETIONS = [
    # (prefix, valid_completion, invalid_completion)
    ("The derivative of sin(x) with respect to x is",
     " cos(x)", " -sin(x)"),
    ("In a right triangle, the square of the hypotenuse equals the sum of",
     " the squares of the other two sides",
     " the products of the other two sides"),
    ("According to Newton's third law of motion, for every action there is an equal and",
     " opposite reaction",
     " identical reaction in the same direction"),
    ("Atoms of the same element that have different numbers of neutrons are called",
     " isotopes", " ions"),
    ("A hash table provides average-case O(1) time complexity for insertions and lookups because each key is mapped directly to a memory location using",
     " a hash function",
     " a sorting algorithm"),
    ("Prior research has established a correlation between sleep deprivation and cognitive decline, but correlation does not imply",
     " causation", " correlation"),
    ("The value of pi is approximately",
     " 3.14159", " 2.71828"),
    ("Natural selection, as proposed by Darwin, states that organisms with traits better suited to their environment are more likely to",
     " survive and reproduce, passing on those traits",
     " mutate rapidly and change species within a generation"),
]

def evaluate_knowledge_preservation(models_dict, tokenizer):
    print("\n" + "="*65)
    print("  SECTION 5 — Knowledge Preservation")
    print("  Base model scored 87% on natural completions.")
    print("  SFT should not have degraded this. Target: ≥80%.")
    print("="*65)

    results = {}
    for model_label, model in models_dict.items():
        correct = 0
        results[model_label] = []
        print(f"\n  ── {model_label} ──")
        for prefix, valid, invalid in KNOWLEDGE_COMPLETIONS:
            s_valid   = continuation_log_prob(model, tokenizer, prefix, valid)
            s_invalid = continuation_log_prob(model, tokenizer, prefix, invalid)
            is_correct = s_valid > s_invalid
            margin     = s_valid - s_invalid
            correct   += int(is_correct)
            mark       = "✓" if is_correct else "✗"
            results[model_label].append(is_correct)
            print(f"  {mark}  {prefix[:60]}")
            print(f"       valid={s_valid:.3f}  invalid={s_invalid:.3f}  "
                  f"margin={margin:+.3f}  "
                  f"({'confident' if abs(margin) > 0.5 else 'uncertain'})")

        acc = correct / len(KNOWLEDGE_COMPLETIONS)
        results[model_label + "_acc"] = acc
        print(f"\n  Accuracy: {correct}/{len(KNOWLEDGE_COMPLETIONS)} = {acc:.0%}")

    return results


# =============================================================================
# SECTION 6 — GENERATION REGISTER
# After SFT on structured data, check the model still generates
# coherent text and hasn't lost fluency outside the SFT format.
# =============================================================================

REGISTER_PROMPTS = [
    # Free-form academic (should still work — base model was good at this)
    "The process of photosynthesis converts light energy into chemical energy. The two main stages are",
    "Gradient descent is an iterative optimization algorithm used to minimize",
    "The difference between supervised and unsupervised machine learning is",
    # Format-triggered (should now produce structured output)
    "Problem: Explain why the sky appears blue.\n\nSolution:",
    "Problem: What is the difference between a stack and a queue data structure?\n\nSolution:",
]

def evaluate_register(models_dict, tokenizer):
    print("\n" + "="*65)
    print("  SECTION 6 — Generation Register & Fluency")
    print("  Free-form prompts should still be coherent.")
    print("  Format-triggered prompts should produce structured output.")
    print("="*65)

    for model_label, model in models_dict.items():
        print(f"\n  ── {model_label} ──")
        for prompt in REGISTER_PROMPTS:
            out = generate(model, tokenizer, prompt,
                           max_new_tokens=200, temperature=0.7)
            rep = repetition_rate(out, n=4)
            ttr = type_token_ratio(out)
            is_format = prompt.startswith("Problem:")
            fmt = check_format(out)
            print(f"\n  {'[FORMAT]' if is_format else '[FREE]  '} {prompt[:70]}")
            print(f"  {out[:300]}")
            print(f"  rep={rep:.3f}  ttr={ttr:.3f}  "
                  f"{'steps='+str(fmt['n_steps'])+'  term='+('✓' if fmt['has_termination'] else '✗') if is_format else ''}")


# =============================================================================
# SECTION 7 — DIRECT BASE vs SFT COMPARISON ON BASE EVAL FAILURES
# Re-run the exact prompts that failed in the base evaluation.
# This is the clearest before/after picture.
# =============================================================================

BASE_EVAL_FAILURES = [
    # From ICL section — model copied wrong answer from prior example
    {
        "label"   : "ICL_water_freeze",
        "prompt"  : "Problem: Water is cooled below 0 degrees Celsius. What happens to it?\n\nSolution:",
        "expected": "ice",
    },
    # From ICL — correct rule, wrong number
    {
        "label"   : "ICL_sequence_5_10_15_20",
        "prompt"  : "Problem: What comes next in the sequence: 5, 10, 15, 20, ?\n\nSolution:",
        "expected": "25",
    },
    # From CoT greedy — looped infinitely
    {
        "label"   : "CoT_cats_dogs",
        "prompt"  : "Problem: There are 3 cats and 4 dogs in a room. How many animals are there in total?\n\nSolution:",
        "expected": "7",
    },
    # From MC — 0% elementary math
    {
        "label"   : "MC_train_distance",
        "prompt"  : "Problem: A train travels 60 miles per hour for 2.5 hours. How far does it travel?\n\nSolution:",
        "expected": "150",
    },
    # From reasoning — logical deduction loop
    {
        "label"   : "logic_whiskers",
        "prompt"  : "Problem: All cats are animals. Whiskers is a cat. Therefore, Whiskers is a ___.\n\nSolution:",
        "expected": "animal",
    },
    # From CoT — arithmetic hallucination
    {
        "label"   : "ICL_factory_150",
        "prompt"  : "Problem: A factory produces 150 units per day. How many units does it produce in 6 days?\n\nSolution:",
        "expected": "900",
    },
]

def evaluate_base_failures(models_dict, tokenizer):
    print("\n" + "="*65)
    print("  SECTION 7 — Direct Comparison on Base Eval Failures")
    print("  Exact prompts that failed in base evaluation.")
    print("  This is the clearest before/after signal.")
    print("="*65)

    results = {}
    for model_label, model in models_dict.items():
        correct = 0
        results[model_label] = []
        print(f"\n  ── {model_label} ──")
        for item in BASE_EVAL_FAILURES:
            out      = generate(model, tokenizer, item["prompt"],
                                max_new_tokens=250, temperature=0.6)
            ans      = extract_final_answer(out)
            # Also check if expected word appears anywhere in output
            word_hit = item["expected"].lower() in out.lower()
            is_correct = answers_match(ans, item["expected"]) or word_hit
            correct   += int(is_correct)
            fmt = check_format(out)
            mark = "✓" if is_correct else "✗"
            results[model_label].append(is_correct)
            print(f"\n  {mark}  [{item['label']}]")
            print(f"       Expected : {item['expected']}")
            print(f"       Extracted: {ans}")
            print(f"       Format   : steps={fmt['n_steps']}  "
                  f"term={'✓' if fmt['has_termination'] else '✗'}")
            print(f"       Output   : {out[:250]}")

        acc = correct / len(BASE_EVAL_FAILURES)
        results[model_label + "_acc"] = acc
        print(f"\n  Accuracy on base failures: {correct}/{len(BASE_EVAL_FAILURES)} = {acc:.0%}")

    print(f"\n  Improvement Summary:")
    print(f"  {'Problem':<30} ", end="")
    for label in models_dict:
        print(f"  {label:>10}", end="")
    print()
    print(f"  {'-'*55}")
    for i, item in enumerate(BASE_EVAL_FAILURES):
        print(f"  {item['label']:<30} ", end="")
        for label in models_dict:
            mark = "✓" if results[label][i] else "✗"
            print(f"  {mark:>10}", end="")
        print()

    return results


# =============================================================================
# FINAL REPORT
# =============================================================================

def print_final_report(format_results, arith_results, cot_results,
                       term_results, know_results, failure_results):
    print("\n" + "="*65)
    print("  SFT EVALUATION — FINAL REPORT")
    print("="*65)

    labels = list(format_results.keys())

    print(f"\n  {'Metric':<35}", end="")
    for label in labels:
        print(f"  {label:>10}", end="")
    print()
    print(f"  {'-'*60}")

    # Format compliance — has_step rate
    print(f"  {'Format: Step N: rate':<35}", end="")
    for label in labels:
        r = format_results[label]
        print(f"  {np.mean([x['has_step'] for x in r]):>10.0%}", end="")
    print()

    # Format compliance — termination rate
    print(f"  {'Format: Termination rate':<35}", end="")
    for label in labels:
        r = format_results[label]
        print(f"  {np.mean([x['has_termination'] for x in r]):>10.0%}", end="")
    print()

    # Format compliance — step advancement rate
    print(f"  {'Format: Steps advance rate':<35}", end="")
    for label in labels:
        r = format_results[label]
        print(f"  {np.mean([x['steps_advance'] for x in r]):>10.0%}", end="")
    print()

    # Arithmetic accuracy
    print(f"  {'Arithmetic accuracy':<35}", end="")
    for label in labels:
        print(f"  {arith_results.get(label+'_acc', 0):>10.0%}", end="")
    print()

    # CoT score
    print(f"  {'CoT quality score (/4)':<35}", end="")
    for label in labels:
        r = cot_results[label]
        print(f"  {np.mean([x['score'] for x in r]):>10.2f}", end="")
    print()

    # CoT answer accuracy
    print(f"  {'CoT answer accuracy':<35}", end="")
    for label in labels:
        r = cot_results[label]
        print(f"  {np.mean([x['correct'] for x in r]):>10.0%}", end="")
    print()

    # Greedy termination accuracy
    print(f"  {'Greedy answer accuracy':<35}", end="")
    for label in labels:
        print(f"  {term_results[label]['greedy_acc']:>10.0%}", end="")
    print()

    # Knowledge preservation
    print(f"  {'Knowledge preservation':<35}", end="")
    for label in labels:
        print(f"  {know_results.get(label+'_acc', 0):>10.0%}", end="")
    print()

    # Base failure recovery
    print(f"  {'Base failure recovery':<35}", end="")
    for label in labels:
        print(f"  {failure_results.get(label+'_acc', 0):>10.0%}", end="")
    print()

    # Overall verdict
    print(f"\n  {'='*55}")
    print(f"  SFT VERDICT")
    print(f"  {'='*55}")
    if "SFT" in labels:
        sft_arith  = arith_results.get("SFT_acc", 0)
        sft_format = np.mean([x['has_step'] for x in format_results.get("SFT", [])])
        sft_term   = np.mean([x['has_termination'] for x in format_results.get("SFT", [])])
        sft_know   = know_results.get("SFT_acc", 0)
        base_know  = know_results.get("BASE_acc", 0)

        print(f"\n  Format compliance  : {sft_format:.0%}  "
              f"({'✓ SFT worked' if sft_format > 0.7 else '✗ Format not learned'})")
        print(f"  Termination rate   : {sft_term:.0%}  "
              f"({'✓ Model terminates cleanly' if sft_term > 0.7 else '✗ Still not terminating'})")
        print(f"  Arithmetic accuracy: {sft_arith:.0%}  "
              f"({'✓ Fixed' if sft_arith > 0.4 else '~ Partial' if sft_arith > 0.2 else '✗ Still broken'})")
        print(f"  Knowledge preserved: {sft_know:.0%}  "
              f"({'✓ No forgetting' if sft_know >= base_know - 0.1 else '✗ Forgetting detected'})")

        if sft_format > 0.7 and sft_term > 0.7 and sft_arith > 0.3:
            print(f"\n  ★ SFT successful — proceed to DPO/GRPO preference tuning")
        elif sft_format > 0.5 or sft_term > 0.5:
            print(f"\n  ~ Partial success — consider a second SFT run with more math data")
        else:
            print(f"\n  ✗ SFT did not take — review data format and LR before next run")

    print("="*65 + "\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    tee = Tee("sft_evaluation_results.txt")
    sys.stdout = tee

    try:
        print("\n" + "="*65)
        print("  SFT Model Evaluation")
        print("  Comparing: BASE model vs SFT model")
        print("  SFT checkpoint: checkpoints/sft/sft_best_model.pt")
        print("  Base checkpoint: checkpoints/best_model.pt")
        print("="*65)

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load both models for direct comparison
        models = {
            "BASE": load_model("checkpoints/best_model.pt",          "BASE"),
            "SFT" : load_model("checkpoints/sft/sft_best_model.pt",  "SFT"),
        }

        # Print training summary from the log
        print(f"\n  SFT Training Summary:")
        print(f"  Starting val loss : 1.2930  (step 100)")
        print(f"  Final val loss    : 1.1999  (step 8000+)")
        print(f"  Improvement       : 7.2%")
        print(f"  Epochs completed  : 3")
        print(f"  Total steps       : ~8,420")
        print(f"  Training time     : ~9.1 hours")

        # Run all sections
        format_results  = evaluate_format_compliance(models, tokenizer)
        arith_results   = evaluate_arithmetic(models, tokenizer)
        cot_results     = evaluate_cot_quality(models, tokenizer)
        term_results    = evaluate_termination(models, tokenizer)
        know_results    = evaluate_knowledge_preservation(models, tokenizer)
        evaluate_register(models, tokenizer)
        failure_results = evaluate_base_failures(models, tokenizer)

        print_final_report(
            format_results, arith_results, cot_results,
            term_results, know_results, failure_results
        )

    finally:
        sys.stdout = tee.stdout
        tee.close()
        print("Results saved → sft_evaluation_results.txt")