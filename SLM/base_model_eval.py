# evaluate.py  —  SLM Base Model Evaluation
# Training data: cosmopedia, dclm-edu, dolmino, openwebmath, pes2o, stackexchange
# Evaluation: fully independent — zero overlap with any training/val split

import os
import gc
import math
import time
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from model_architecture import CustomTransformer
from config import SLMConfig

cfg    = SLMConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. MODEL LOADER
# =============================================================================

def load_model(ckpt_path="checkpoints/best_model.pt"):
    model = CustomTransformer(cfg).to(device).to(torch.bfloat16)
    print(f"  Loading : {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    # Strip torch.compile prefix
    state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
             for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params  : {total/1e6:.1f}M")
    if "val_loss" in ckpt:
        print(f"  Best val_loss : {ckpt['val_loss']:.4f}  (ppl {math.exp(ckpt['val_loss']):.2f})")
    return model


# =============================================================================
# 2. CORE PRIMITIVES
# =============================================================================

@torch.no_grad()
def sequence_log_prob(model, tokenizer, text):
    """
    Returns the mean per-token log-probability of `text`.
    Lower (more negative) = model finds it less likely.
    Used for multiple-choice scoring without any external judge.
    """
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    if ids.shape[1] < 2:
        return float("-inf")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(ids)
    log_probs = F.log_softmax(logits, dim=-1)
    # Score each token against its successor
    token_scores = log_probs[0, :-1, :].gather(
        1, ids[0, 1:].unsqueeze(-1)
    ).squeeze(-1)
    return token_scores.mean().item()


@torch.no_grad()
def continuation_log_prob(model, tokenizer, prefix, continuation):
    """
    Score only the `continuation` tokens, conditioned on `prefix`.
    Cleaner for multiple-choice: prefix = question, continuation = each answer option.
    """
    prefix_ids      = tokenizer.encode(prefix,       add_special_tokens=True)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    full_ids = torch.tensor([prefix_ids + continuation_ids], device=device)

    if full_ids.shape[1] > cfg.max_seq_len:
        full_ids = full_ids[:, -cfg.max_seq_len:]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids)

    log_probs = F.log_softmax(logits, dim=-1)
    cont_start = len(prefix_ids) - 1   # -1 because logits are shifted
    cont_len   = len(continuation_ids)

    if cont_start + cont_len > log_probs.shape[1]:
        return float("-inf")

    scores = log_probs[0, cont_start:cont_start + cont_len, :].gather(
        1, full_ids[0, len(prefix_ids):len(prefix_ids) + cont_len].unsqueeze(-1)
    ).squeeze(-1)
    return scores.mean().item()


def mc_predict(model, tokenizer, question, choices):
    """
    Multiple-choice via log-prob scoring.
    Returns (predicted_index, list_of_scores).
    """
    scores = [continuation_log_prob(model, tokenizer, question, c) for c in choices]
    return int(np.argmax(scores)), scores


@torch.no_grad()
def generate(model, tokenizer, prompt,
             max_new_tokens=256, temperature=0.7, top_k=50, top_p=0.9):
    ids  = tokenizer.encode(prompt, return_tensors="pt").to(device)
    gen  = ids.clone()
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
        probs    = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        gen      = torch.cat([gen, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(gen[0, ids.shape[1]:].tolist(), skip_special_tokens=True)


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


# =============================================================================
# 3. GENERATION QUALITY — DOMAIN-APPROPRIATE
#    Prompts match actual training distribution:
#    educational explanations, scientific text, Q&A, math exposition
# =============================================================================

GENERATION_PROMPTS = [
    # Educational / cosmopedia style
    {
        "label": "educational_explanation",
        "prompt": "Explain why the sky is blue. Include the relevant physics concept.",
        "keywords": ["scatter", "rayleigh", "wavelength", "light", "blue"]
    },
    {
        "label": "concept_explanation",
        "prompt": "What is the difference between supervised and unsupervised learning in machine learning?",
        "keywords": ["label", "data", "cluster", "pattern", "train"]
    },
    # Scientific / pes2o style
    {
        "label": "scientific_prose",
        "prompt": "The process of photosynthesis converts light energy into chemical energy. Describe the two main stages and what each produces.",
        "keywords": ["light", "dark", "atp", "glucose", "chlorophyll", "calvin"]
    },
    # StackExchange Q&A style
    {
        "label": "technical_qa",
        "prompt": "Q: What is the difference between a stack and a queue data structure?\nA:",
        "keywords": ["lifo", "fifo", "last", "first", "push", "pop", "enqueue"]
    },
    {
        "label": "programming_qa",
        "prompt": "Q: Why does floating point arithmetic sometimes give unexpected results in programming?\nA:",
        "keywords": ["binary", "precision", "float", "represent", "round"]
    },
    # OpenWebMath style
    {
        "label": "math_exposition",
        "prompt": "Explain the intuition behind the Pythagorean theorem and why a² + b² = c² holds for right triangles.",
        "keywords": ["right", "hypotenuse", "square", "area", "angle", "proof"]
    },
    # Dolmino / general web
    {
        "label": "analytical_writing",
        "prompt": "What are the main causes of inflation and how does raising interest rates help control it?",
        "keywords": ["money", "supply", "demand", "price", "rate", "spend", "borrow"]
    },
]

def keyword_coverage(text, keywords):
    t = text.lower()
    hits = sum(1 for k in keywords if k in t)
    return hits / len(keywords)

def repetition_rate(text, n=4):
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return 1 - len(set(ngrams)) / len(ngrams)

def type_token_ratio(text):
    tokens = text.lower().split()
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def evaluate_generation_quality(model, tokenizer):
    print("\n" + "="*65)
    print("  SECTION 1 — Generation Quality (Domain-Appropriate Prompts)")
    print("="*65)

    results = []
    for item in GENERATION_PROMPTS:
        out    = generate(model, tokenizer, item["prompt"],
                          max_new_tokens=200, temperature=0.7)
        rep    = repetition_rate(out, n=4)
        ttr    = type_token_ratio(out)
        kw_cov = keyword_coverage(out, item["keywords"])
        results.append({
            "label"   : item["label"],
            "rep"     : rep,
            "ttr"     : ttr,
            "kw_cov"  : kw_cov,
            "words"   : len(out.split()),
            "output"  : out,
        })
        print(f"\n  [{item['label']}]")
        print(f"  PROMPT : {item['prompt'][:90]}")
        print(f"  OUTPUT : {out[:400]}")
        print(f"  rep={rep:.3f}  ttr={ttr:.3f}  kw_coverage={kw_cov:.0%}  words={len(out.split())}")

    print("\n  --- Aggregate ---")
    print(f"  Avg repetition   : {np.mean([r['rep'] for r in results]):.3f}  (target <0.05)")
    print(f"  Avg TTR          : {np.mean([r['ttr'] for r in results]):.3f}  (target >0.55)")
    print(f"  Avg kw_coverage  : {np.mean([r['kw_cov'] for r in results]):.0%}  (target >50%)")
    return results


# =============================================================================
# 4. MULTIPLE CHOICE — LOG-PROB SCORING
#    No external judge. Model scores each option directly.
#    Covers the reasoning categories that matter for a reasoning model.
# =============================================================================

MC_BENCHMARKS = {

    "elementary_math": [
        {
            "q": "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?\n",
            "choices": ["100 miles", "120 miles", "150 miles", "180 miles"],
            "answer": 2,  # 150
        },
        {
            "q": "What is 15% of 240?\n",
            "choices": ["30", "36", "40", "24"],
            "answer": 1,  # 36
        },
        {
            "q": "A rectangle has length 8 and width 5. What is its area?\n",
            "choices": ["26", "13", "40", "45"],
            "answer": 2,  # 40
        },
        {
            "q": "If 3x + 7 = 22, what is x?\n",
            "choices": ["3", "4", "5", "6"],
            "answer": 2,  # 5
        },
        {
            "q": "What is the square root of 144?\n",
            "choices": ["11", "12", "13", "14"],
            "answer": 1,  # 12
        },
    ],

    "algebra_and_reasoning": [
        {
            "q": "If f(x) = 2x² - 3x + 1, what is f(2)?\n",
            "choices": ["1", "3", "5", "7"],
            "answer": 1,  # 3
        },
        {
            "q": "A car depreciates 20% per year. If it costs $10,000 today, what is it worth after 2 years?\n",
            "choices": ["$6,000", "$6,400", "$7,600", "$8,000"],
            "answer": 1,  # 6400
        },
        {
            "q": "Two numbers sum to 50. One is twice the other. What is the smaller number?\n",
            "choices": ["10", "15", "16.7", "25"],
            "answer": 2,  # ~16.67
        },
        {
            "q": "If log₂(x) = 5, what is x?\n",
            "choices": ["10", "25", "32", "64"],
            "answer": 2,  # 32
        },
    ],

    "logical_deduction": [
        {
            "q": "All mammals are warm-blooded. Dolphins are mammals. Which must be true?\n",
            "choices": [
                "Dolphins are fish",
                "Dolphins are warm-blooded",
                "All warm-blooded animals are mammals",
                "Dolphins breathe underwater",
            ],
            "answer": 1,
        },
        {
            "q": "If it rains, the match is cancelled. The match was not cancelled. What can we conclude?\n",
            "choices": [
                "It rained",
                "It did not rain",
                "The match was played indoors",
                "Nothing can be concluded",
            ],
            "answer": 1,  # contrapositive
        },
        {
            "q": "Some A are B. All B are C. Which statement must be true?\n",
            "choices": [
                "All A are C",
                "Some A are C",
                "No A are C",
                "All C are A",
            ],
            "answer": 1,
        },
        {
            "q": "In a race, Alice finished before Bob. Bob finished before Carol. Who finished last?\n",
            "choices": ["Alice", "Bob", "Carol", "Cannot be determined"],
            "answer": 2,
        },
        {
            "q": "If all doctors are scientists and some scientists are writers, which is definitely true?\n",
            "choices": [
                "All doctors are writers",
                "Some doctors are writers",
                "All scientists are doctors",
                "None of the above must be true",
            ],
            "answer": 3,
        },
    ],

    "scientific_reasoning": [
        {
            "q": "A scientist doubles the concentration of a reactant. The reaction rate quadruples. What is the order of the reaction with respect to that reactant?\n",
            "choices": ["Zero order", "First order", "Second order", "Third order"],
            "answer": 2,
        },
        {
            "q": "An object in space continues moving at constant velocity. Which law explains this?\n",
            "choices": [
                "Newton's Second Law",
                "Newton's First Law",
                "Newton's Third Law",
                "Law of Gravitation",
            ],
            "answer": 1,
        },
        {
            "q": "Which process converts glucose into pyruvate without using oxygen?\n",
            "choices": ["Krebs cycle", "Oxidative phosphorylation", "Glycolysis", "Photosynthesis"],
            "answer": 2,
        },
        {
            "q": "The half-life of a radioactive substance is 10 years. After 30 years, what fraction remains?\n",
            "choices": ["1/2", "1/4", "1/8", "1/16"],
            "answer": 2,  # 1/8
        },
        {
            "q": "What property of water makes it an excellent solvent for ionic compounds?\n",
            "choices": [
                "High boiling point",
                "Polar molecule with partial charges",
                "Low viscosity",
                "High surface tension",
            ],
            "answer": 1,
        },
    ],

    "computer_science": [
        {
            "q": "What is the time complexity of binary search on a sorted array of n elements?\n",
            "choices": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
            "answer": 1,
        },
        {
            "q": "Which data structure uses LIFO (Last In First Out) ordering?\n",
            "choices": ["Queue", "Stack", "Heap", "Linked List"],
            "answer": 1,
        },
        {
            "q": "In TCP/IP, which layer is responsible for end-to-end communication and error checking?\n",
            "choices": ["Network layer", "Data link layer", "Transport layer", "Application layer"],
            "answer": 2,
        },
        {
            "q": "What does SQL SELECT DISTINCT do?\n",
            "choices": [
                "Sorts the results",
                "Removes duplicate rows from results",
                "Filters rows by condition",
                "Joins two tables",
            ],
            "answer": 1,
        },
        {
            "q": "Which sorting algorithm has the best average-case time complexity?\n",
            "choices": ["Bubble sort O(n²)", "Merge sort O(n log n)", "Insertion sort O(n²)", "Selection sort O(n²)"],
            "answer": 1,
        },
    ],

    "multi_step_word_problems": [
        {
            "q": (
                "A store sells apples for $0.50 each and oranges for $0.75 each. "
                "Maria buys 4 apples and 3 oranges. She pays with a $5 bill. "
                "How much change does she receive?\n"
            ),
            "choices": ["$0.25", "$0.75", "$1.25", "$1.75"],
            "answer": 1,  # 5 - (2 + 2.25) = 0.75
        },
        {
            "q": (
                "A tank fills in 6 hours when pipe A is open. "
                "Pipe B alone fills it in 12 hours. "
                "How many hours does it take if both pipes are open?\n"
            ),
            "choices": ["3 hours", "4 hours", "5 hours", "6 hours"],
            "answer": 1,  # 4
        },
        {
            "q": (
                "A recipe for 4 people requires 2 cups of flour. "
                "You want to make it for 10 people. "
                "How many cups of flour do you need?\n"
            ),
            "choices": ["4 cups", "5 cups", "6 cups", "8 cups"],
            "answer": 1,  # 5
        },
        {
            "q": (
                "A car gets 35 miles per gallon. Gas costs $3.50 per gallon. "
                "How much does a 350-mile trip cost in fuel?\n"
            ),
            "choices": ["$25", "$30", "$35", "$40"],
            "answer": 2,  # 35
        },
    ],

    "reading_comprehension_and_inference": [
        {
            "q": (
                "Passage: 'The vaccine reduced infection rates by 85% in clinical trials. "
                "However, it required cold storage at -70°C, making distribution in rural areas challenging.'\n"
                "What is the main trade-off described?\n"
            ),
            "choices": [
                "High cost vs. high effectiveness",
                "High effectiveness vs. difficult distribution",
                "Fast production vs. slow approval",
                "Strong immunity vs. side effects",
            ],
            "answer": 1,
        },
        {
            "q": (
                "Passage: 'Company X's revenue grew 40% last year while expenses grew 60%. "
                "The CEO announced record revenues at the shareholder meeting.'\n"
                "What is misleading about the CEO's statement?\n"
            ),
            "choices": [
                "Revenue did not actually grow",
                "Revenue grew but profitability likely fell",
                "Expenses did not grow",
                "Nothing is misleading",
            ],
            "answer": 1,
        },
        {
            "q": (
                "Passage: 'Nations that adopted coal as their primary energy source in the 1800s "
                "experienced rapid industrialization but also significant public health crises.'\n"
                "What does this passage most directly support?\n"
            ),
            "choices": [
                "Coal is always harmful",
                "Industrialization requires coal",
                "Rapid development can come with health costs",
                "Public health improved during industrialization",
            ],
            "answer": 2,
        },
    ],
}


def evaluate_multiple_choice(model, tokenizer):
    print("\n" + "="*65)
    print("  SECTION 2 — Multiple Choice (Log-Prob Scoring, No External Judge)")
    print("="*65)

    category_results = {}
    grand_correct = 0
    grand_total   = 0

    for category, items in MC_BENCHMARKS.items():
        correct = 0
        print(f"\n  ── {category.upper().replace('_', ' ')} ──")
        for item in items:
            pred, scores = mc_predict(model, tokenizer, item["q"], item["choices"])
            is_correct   = (pred == item["answer"])
            correct     += int(is_correct)
            mark         = "✓" if is_correct else "✗"
            print(f"  {mark}  Q: {item['q'].strip()[:80]}")
            print(f"      Predicted: [{pred}] {item['choices'][pred]}")
            if not is_correct:
                print(f"      Correct  : [{item['answer']}] {item['choices'][item['answer']]}")
            score_str = "  ".join([f"[{i}]{s:.2f}" for i, s in enumerate(scores)])
            print(f"      Scores   : {score_str}")

        acc = correct / len(items)
        category_results[category] = {"correct": correct, "total": len(items), "acc": acc}
        print(f"  Accuracy: {correct}/{len(items)} = {acc:.0%}")
        grand_correct += correct
        grand_total   += len(items)

    print(f"\n  {'='*40}")
    print(f"  OVERALL MC ACCURACY : {grand_correct}/{grand_total} = {grand_correct/grand_total:.0%}")
    return category_results, grand_correct / grand_total


# =============================================================================
# 5. CHAIN-OF-THOUGHT EVALUATION
#    Tests whether the model can produce faithful, advancing reasoning traces.
#    Calibrated to the actual training data register (academic/technical).
# =============================================================================

COT_PROBLEMS = [
    {
        "label": "arithmetic_word_problem",
        "prompt": (
            "Q: A factory produces 240 widgets per day. It operates 6 days a week. "
            "How many widgets does it produce in 4 weeks?\n"
            "A: Let's think step by step.\n"
        ),
        "expected_answer": "5760",
        "expected_keywords": ["240", "6", "4", "week", "1440", "5760"],
    },
    {
        "label": "percentage_reasoning",
        "prompt": (
            "Q: A jacket originally costs $80. It is on sale for 25% off. "
            "After the discount, a 10% tax is applied. What is the final price?\n"
            "A: Let's think step by step.\n"
        ),
        "expected_answer": "66",
        "expected_keywords": ["80", "25", "60", "10", "66"],
    },
    {
        "label": "logical_chain",
        "prompt": (
            "Q: All professors at this university have a PhD. "
            "Dr. Smith is a professor here. "
            "Does Dr. Smith have a PhD? Explain your reasoning.\n"
            "A: Let's think step by step.\n"
        ),
        "expected_answer": "yes",
        "expected_keywords": ["yes", "professor", "phd", "therefore", "smith"],
    },
    {
        "label": "rate_problem",
        "prompt": (
            "Q: Alice can paint a fence in 3 hours. Bob can paint the same fence in 6 hours. "
            "If they work together, how long does it take?\n"
            "A: Let's think step by step.\n"
        ),
        "expected_answer": "2 hours",
        "expected_keywords": ["1/3", "1/6", "1/2", "2", "rate", "together"],
    },
    {
        "label": "multi_constraint_reasoning",
        "prompt": (
            "Q: A container holds 50 liters. It is currently 40% full. "
            "How many liters must be added to make it 90% full?\n"
            "A: Let's think step by step.\n"
        ),
        "expected_answer": "25",
        "expected_keywords": ["50", "40", "20", "90", "45", "25"],
    },
    {
        "label": "comparison_reasoning",
        "prompt": (
            "Q: Plan A costs $200 upfront and $10 per month. "
            "Plan B costs $0 upfront and $25 per month. "
            "After how many months is Plan A cheaper?\n"
            "A: Let's think step by step.\n"
        ),
        "expected_answer": "14",
        "expected_keywords": ["200", "10", "25", "15", "13", "14"],
    },
    {
        "label": "scientific_reasoning",
        "prompt": (
            "Q: A sample of a radioactive element has a half-life of 5 years. "
            "You start with 800 grams. How much remains after 15 years?\n"
            "A: Let's think step by step.\n"
        ),
        "expected_answer": "100",
        "expected_keywords": ["800", "400", "200", "100", "half", "5", "15"],
    },
    {
        "label": "code_logic_trace",
        "prompt": (
            "Q: Consider this pseudocode:\n"
            "  x = 10\n"
            "  for i in range(3):\n"
            "      x = x * 2 - 1\n"
            "What is the value of x after the loop?\n"
            "A: Let's think step by step.\n"
        ),
        "expected_answer": "69",
        "expected_keywords": ["10", "19", "37", "73", "i=0", "i=1", "i=2"],
    },
]


def analyze_cot_output(output, keywords):
    """
    Analyze a CoT output for quality signals.
    Returns a dict of metrics.
    """
    lines    = [l.strip() for l in output.strip().split("\n") if l.strip()]
    words    = output.split()

    # Step structure: does it have advancing steps?
    step_lines   = [l for l in lines if l.lower().startswith("step")]
    has_steps    = len(step_lines) >= 2

    # Answer line: does it terminate with an answer?
    has_answer   = any(
        l.lower().startswith(("answer", "therefore", "so the answer", "final", "result", "thus"))
        for l in lines[-3:]  # check last 3 lines
    )

    # Keyword coverage
    text_lower   = output.lower()
    kw_hits      = sum(1 for k in keywords if k.lower() in text_lower)
    kw_coverage  = kw_hits / len(keywords) if keywords else 0

    # Repetition
    rep          = repetition_rate(output, n=4)

    # Step advancement: are steps different from each other?
    step_unique  = len(set(step_lines)) / len(step_lines) if step_lines else 0

    # Degenerate patterns
    is_degenerate = rep > 0.5 or (len(words) > 50 and len(set(words)) < 10)

    # Overall coherence score (0-4 for CoT — we need finer granularity here)
    score = 0
    if not is_degenerate:                 score += 1
    if kw_coverage > 0.3:                score += 1
    if has_steps and step_unique > 0.5:  score += 1
    if has_answer:                       score += 1

    return {
        "score"       : score,           # 0-4
        "has_steps"   : has_steps,
        "has_answer"  : has_answer,
        "kw_coverage" : kw_coverage,
        "rep"         : rep,
        "is_degenerate": is_degenerate,
        "n_steps"     : len(step_lines),
        "step_unique" : step_unique,
    }


def evaluate_chain_of_thought(model, tokenizer):
    print("\n" + "="*65)
    print("  SECTION 3 — Chain-of-Thought Reasoning")
    print("="*65)

    all_results = []
    for item in COT_PROBLEMS:
        print(f"\n  [{item['label']}]")
        print(f"  PROMPT : {item['prompt'].strip()}")

        # Sampling run
        out_sample = generate(model, tokenizer, item["prompt"],
                              max_new_tokens=300, temperature=0.6, top_k=40)
        # Greedy run — shows model's highest-confidence reasoning path
        out_greedy = generate_greedy(model, tokenizer, item["prompt"],
                                     max_new_tokens=300)

        metrics_s = analyze_cot_output(out_sample, item["expected_keywords"])
        metrics_g = analyze_cot_output(out_greedy, item["expected_keywords"])

        ans_in_sample = item["expected_answer"].lower() in out_sample.lower()
        ans_in_greedy = item["expected_answer"].lower() in out_greedy.lower()

        print(f"\n  [Sampling output]")
        print(f"  {out_sample.strip()[:500]}")
        print(f"  score={metrics_s['score']}/4  steps={metrics_s['n_steps']}  "
              f"kw={metrics_s['kw_coverage']:.0%}  "
              f"has_answer={metrics_s['has_answer']}  "
              f"degenerate={metrics_s['is_degenerate']}  "
              f"correct_ans={'✓' if ans_in_sample else '✗'}")

        print(f"\n  [Greedy output]")
        print(f"  {out_greedy.strip()[:500]}")
        print(f"  score={metrics_g['score']}/4  steps={metrics_g['n_steps']}  "
              f"kw={metrics_g['kw_coverage']:.0%}  "
              f"has_answer={metrics_g['has_answer']}  "
              f"degenerate={metrics_g['is_degenerate']}  "
              f"correct_ans={'✓' if ans_in_greedy else '✗'}")

        all_results.append({
            "label"          : item["label"],
            "sample_score"   : metrics_s["score"],
            "greedy_score"   : metrics_g["score"],
            "ans_in_sample"  : ans_in_sample,
            "ans_in_greedy"  : ans_in_greedy,
        })

    print("\n  --- CoT Summary ---")
    print(f"  {'Problem':<30} {'Sample':>7} {'Greedy':>7} {'Ans(s)':>7} {'Ans(g)':>7}")
    print(f"  {'-'*58}")
    for r in all_results:
        print(f"  {r['label']:<30} {r['sample_score']:>5}/4  {r['greedy_score']:>5}/4  "
              f"{'✓' if r['ans_in_sample'] else '✗':>7}  {'✓' if r['ans_in_greedy'] else '✗':>7}")

    avg_sample = np.mean([r["sample_score"]  for r in all_results])
    avg_greedy = np.mean([r["greedy_score"]  for r in all_results])
    ans_sample = np.mean([r["ans_in_sample"] for r in all_results])
    ans_greedy = np.mean([r["ans_in_greedy"] for r in all_results])

    print(f"\n  Avg CoT score (sampling): {avg_sample:.2f}/4")
    print(f"  Avg CoT score (greedy)  : {avg_greedy:.2f}/4")
    print(f"  Answer accuracy (sample): {ans_sample:.0%}")
    print(f"  Answer accuracy (greedy): {ans_greedy:.0%}")
    return all_results


# =============================================================================
# 6. KNOWLEDGE PROBE — FACTUAL RECALL FROM TRAINING DOMAINS
#    Tests whether pretraining knowledge is encoded and retrievable.
# =============================================================================

KNOWLEDGE_PROBES = {
    "mathematics": [
        ("The derivative of sin(x) is", "cos(x)"),
        ("The integral of 1/x dx is", "ln"),
        ("Euler's number e is approximately", "2.71"),
        ("The sum of angles in a triangle is", "180"),
        ("A prime number has exactly", "two"),  # two factors
    ],
    "computer_science": [
        ("In Big-O notation, O(1) means", "constant"),
        ("A binary tree node has at most", "two"),
        ("HTTP status code 404 means", "found"),   # not found
        ("RAM stands for Random Access", "Memory"),
        ("Python is an interpreted", "language"),
    ],
    "science": [
        ("The speed of light in vacuum is approximately", "3"),  # 3×10^8
        ("DNA stands for Deoxyribonucleic", "Acid"),
        ("The powerhouse of the cell is the", "mitochondria"),
        ("Newton's second law: F equals", "ma"),
        ("The chemical formula for water is", "H2O"),
    ],
    "general_academic": [
        ("The French Revolution began in", "1789"),
        ("The theory of relativity was developed by", "Einstein"),
        ("Supply and demand is a fundamental concept in", "economics"),
        ("Photons are particles of", "light"),
        ("The Pythagorean theorem applies to", "right"),  # right triangles
    ],
}

def evaluate_knowledge(model, tokenizer):
    print("\n" + "="*65)
    print("  SECTION 4 — Knowledge Recall (Domain Coverage Check)")
    print("="*65)

    category_results = {}
    for category, probes in KNOWLEDGE_PROBES.items():
        correct = 0
        print(f"\n  ── {category.upper()} ──")
        for prefix, expected in probes:
            # Score completion vs a wrong alternative using log-prob
            out = generate_greedy(model, tokenizer, prefix, max_new_tokens=20)
            hit = expected.lower() in (prefix + out).lower()
            correct += int(hit)
            mark     = "✓" if hit else "✗"
            print(f"  {mark}  {prefix[:60]}")
            print(f"       Generated: {out.strip()[:80]}")
        acc = correct / len(probes)
        category_results[category] = acc
        print(f"  Accuracy: {correct}/{len(probes)} = {acc:.0%}")

    overall = np.mean(list(category_results.values()))
    print(f"\n  Overall knowledge recall: {overall:.0%}")
    return category_results


# =============================================================================
# 7. DEGENERATION STRESS TEST
#    Aggressively tests where the model breaks down.
#    Critical for understanding SFT starting point.
# =============================================================================

STRESS_PROMPTS = [
    # Long-context coherence
    {
        "label": "long_form_coherence",
        "prompt": "Write a detailed explanation of how gradient descent works in machine learning, covering the intuition, the math, and common variants like SGD, Adam, and RMSProp.",
        "max_tokens": 400,
    },
    # Abstract reasoning
    {
        "label": "abstract_reasoning",
        "prompt": "Explain the Monty Hall problem and why switching doors gives a 2/3 probability of winning.",
        "max_tokens": 300,
    },
    # Self-correction ability
    {
        "label": "error_correction",
        "prompt": "The following solution has an error. Find and fix it.\nProblem: Find the derivative of f(x) = x³ + 2x.\nSolution: f'(x) = x² + 2\nWhat is wrong and what is the correct answer?",
        "max_tokens": 150,
    },
    # Instruction following
    {
        "label": "structured_output",
        "prompt": "List exactly 3 differences between supervised and reinforcement learning. Format your answer as a numbered list.",
        "max_tokens": 200,
    },
    # Adversarial repetition trigger
    {
        "label": "repetition_resistance",
        "prompt": "The number 42 is interesting because",
        "max_tokens": 200,
    },
]

def evaluate_stress(model, tokenizer):
    print("\n" + "="*65)
    print("  SECTION 5 — Degeneration Stress Tests")
    print("="*65)

    results = []
    for item in STRESS_PROMPTS:
        out = generate(model, tokenizer, item["prompt"],
                       max_new_tokens=item["max_tokens"], temperature=0.7)
        rep  = repetition_rate(out, n=4)
        ttr  = type_token_ratio(out)
        degen = rep > 0.3
        results.append({"label": item["label"], "rep": rep, "ttr": ttr, "degen": degen})
        print(f"\n  [{item['label']}]  {'⚠ DEGENERATE' if degen else '✓ OK'}")
        print(f"  PROMPT : {item['prompt'][:90]}")
        print(f"  OUTPUT : {out.strip()[:400]}")
        print(f"  rep={rep:.3f}  ttr={ttr:.3f}")

    degen_count = sum(r["degen"] for r in results)
    print(f"\n  Degeneration failures: {degen_count}/{len(results)}")
    return results


# =============================================================================
# 8. FINAL REPORT
# =============================================================================

def print_final_report(mc_results, mc_overall, cot_results, knowledge_results):
    print("\n" + "="*65)
    print("  FINAL EVALUATION REPORT")
    print("="*65)

    print(f"\n  ── Multiple Choice Accuracy ──")
    for cat, res in mc_results.items():
        bar = "█" * int(res["acc"] * 20) + "░" * (20 - int(res["acc"] * 20))
        print(f"  {cat:<35} {bar}  {res['acc']:.0%}")
    print(f"  {'OVERALL':<35} {mc_overall:.0%}")

    avg_cot_sample = np.mean([r["sample_score"] for r in cot_results])
    avg_cot_greedy = np.mean([r["greedy_score"] for r in cot_results])
    ans_acc_sample = np.mean([r["ans_in_sample"] for r in cot_results])
    ans_acc_greedy = np.mean([r["ans_in_greedy"] for r in cot_results])
    print(f"\n  ── Chain-of-Thought ──")
    print(f"  CoT quality  (sampling) : {avg_cot_sample:.2f}/4")
    print(f"  CoT quality  (greedy)   : {avg_cot_greedy:.2f}/4")
    print(f"  Answer accuracy (sample): {ans_acc_sample:.0%}")
    print(f"  Answer accuracy (greedy): {ans_acc_greedy:.0%}")

    print(f"\n  ── Knowledge Recall ──")
    for cat, acc in knowledge_results.items():
        print(f"  {cat:<35} {acc:.0%}")

    print(f"\n  ── Reasoning Model Readiness ──")

    # Readiness verdict
    if mc_overall >= 0.65 and avg_cot_sample >= 2.5:
        print("  ★ Strong base — proceed directly to CoT SFT + DPO")
    elif mc_overall >= 0.45 and avg_cot_sample >= 1.5:
        print("  ~ Moderate base — CoT SFT required before DPO")
    else:
        print("  ✗ Weak reasoning base — intensive SFT on reasoning traces required")

    print("\n  ── SFT Priority Order (based on results) ──")
    weak_mc = sorted(mc_results.items(), key=lambda x: x[1]["acc"])
    for cat, res in weak_mc[:3]:
        print(f"  • {cat}  ({res['acc']:.0%} accuracy)")

    print("\n  ── What to watch in SFT ──")
    print("  1. CoT score should improve from current baseline")
    print("  2. Answer accuracy (greedy) is the primary north-star metric")
    print("  3. MC accuracy on math + logic should exceed 70% before DPO")
    print("  4. Degeneration rate should drop to 0/5 on stress tests")
    print("="*65 + "\n")

# =============================================================================
# SECTION A — DOMAIN PERPLEXITY ON CURATED PASSAGES
# Ground truth of what the model actually learned domain by domain.
# Passages are embedded here — zero dependency on any .bin file.
# =============================================================================

DOMAIN_PASSAGES = {

    "openwebmath": [
        # Pure mathematical prose — mirrors openwebmath training data
        """The Fundamental Theorem of Calculus establishes the relationship between 
differentiation and integration. If f is continuous on [a, b] and F is an 
antiderivative of f, then the integral from a to b of f(x)dx equals F(b) minus F(a). 
This theorem has two parts: the first guarantees the existence of antiderivatives for 
continuous functions, while the second provides a practical method for evaluating 
definite integrals without computing Riemann sums directly.""",

        """A prime number is a natural number greater than 1 that has no positive 
divisors other than 1 and itself. The prime factorization theorem, also known as the 
Fundamental Theorem of Arithmetic, states that every integer greater than 1 can be 
represented uniquely as a product of prime numbers, up to the order of the factors. 
For example, 360 equals 2 raised to the third power, times 3 squared, times 5. 
This uniqueness is what makes primes the building blocks of the integers.""",

        """To solve a quadratic equation of the form ax squared plus bx plus c equals 
zero, we can use the quadratic formula: x equals negative b plus or minus the square 
root of b squared minus 4ac, all divided by 2a. The expression under the square root, 
b squared minus 4ac, is called the discriminant. When the discriminant is positive, 
there are two distinct real roots. When it equals zero, there is exactly one real root. 
When it is negative, the roots are complex conjugates.""",
    ],

    "stackexchange": [
        # Technical Q&A prose — mirrors stackexchange distribution
        """The difference between a process and a thread lies in how they share resources. 
A process is an independent program in execution with its own memory space, file handles, 
and system resources. Threads, by contrast, exist within a process and share the same 
memory space, which makes communication between threads faster but introduces the risk 
of race conditions. Context switching between processes is more expensive than between 
threads because the operating system must save and restore more state information.""",

        """When designing a REST API, idempotency is an important property to consider. 
An HTTP method is idempotent if making the same request multiple times produces the same 
result as making it once. GET, PUT, and DELETE are idempotent, while POST is not. This 
distinction matters for retry logic: if a network failure occurs during a PUT request, 
the client can safely retry without risk of duplicate side effects. POST requests, 
however, may create duplicate resources if retried naively.""",

        """Dynamic programming solves problems by breaking them into overlapping 
subproblems and storing the results of each subproblem to avoid redundant computation. 
This technique applies when a problem has optimal substructure, meaning the optimal 
solution can be constructed from optimal solutions to its subproblems. The two main 
approaches are memoization, which is top-down with caching, and tabulation, which 
fills a table bottom-up. The Fibonacci sequence, longest common subsequence, and 
knapsack problem are classic dynamic programming examples.""",
    ],

    "cosmopedia": [
        # Educational explanatory prose
        """Photosynthesis is the process by which plants, algae, and some bacteria 
convert light energy into chemical energy stored in glucose. This process occurs 
primarily in the chloroplasts, which contain the green pigment chlorophyll. 
Photosynthesis has two main stages: the light-dependent reactions, which occur in 
the thylakoid membranes and produce ATP and NADPH, and the light-independent reactions, 
also known as the Calvin cycle, which use that energy to fix carbon dioxide into 
organic molecules in the stroma.""",

        """The water cycle, also known as the hydrological cycle, describes the 
continuous movement of water within Earth and its atmosphere. Water evaporates from 
oceans and lakes when heated by the sun, rises as water vapor, and cools to form 
clouds through condensation. When droplets in clouds become heavy enough, precipitation 
occurs in the form of rain, snow, or hail. Water then flows back to the ocean through 
rivers and groundwater, or is absorbed by plants through transpiration, completing 
the cycle.""",

        """The French Revolution, which began in 1789, fundamentally transformed 
the political and social structure of France and influenced the entire Western world. 
Driven by Enlightenment ideals of liberty, equality, and fraternity, the revolution 
overthrew the absolute monarchy and established a republic. The causes included 
financial crisis from years of war, widespread poverty among the lower classes, 
and growing resentment of aristocratic privilege. The Declaration of the Rights of 
Man and Citizen, adopted in 1789, articulated principles that would shape democratic 
governance for centuries.""",
    ],

    "dclm_edu": [
        # High-quality educational web text
        """Neural networks are computational models loosely inspired by the structure 
of the human brain. They consist of layers of interconnected nodes, called neurons, 
that transform input data through a series of linear transformations and non-linear 
activation functions. During training, the network adjusts the weights of these 
connections using backpropagation and gradient descent to minimize a loss function 
that measures the difference between predicted and actual outputs. Deep neural networks, 
which have many hidden layers, can learn hierarchical representations of data.""",

        """Supply and demand is the foundational model of market economics. The law 
of demand states that, all else being equal, as the price of a good increases, the 
quantity demanded decreases. The law of supply states that as price increases, 
producers are willing to supply more of the good. The market reaches equilibrium 
at the price where quantity supplied equals quantity demanded. When external factors 
shift either curve, such as a change in consumer income or production costs, the 
equilibrium price and quantity adjust accordingly.""",
    ],

    "dolmino": [
        # General high-quality web text
        """Climate change refers to long-term shifts in global temperatures and weather 
patterns. While some natural variation is normal, scientific evidence shows that human 
activities, particularly the burning of fossil fuels, have been the dominant driver of 
climate change since the mid-twentieth century. The release of greenhouse gases such as 
carbon dioxide and methane traps heat in the atmosphere, raising average global 
temperatures. This warming drives changes in precipitation patterns, sea level rise, 
and increased frequency of extreme weather events.""",

        """Antibiotics are compounds that kill bacteria or inhibit their growth. 
They work through several mechanisms: some, like penicillin, disrupt the synthesis 
of bacterial cell walls; others inhibit protein synthesis by binding to bacterial 
ribosomes; and some interfere with DNA replication. Critically, antibiotics are 
ineffective against viral infections such as the common cold or influenza. 
Overuse and misuse of antibiotics has led to the emergence of antibiotic-resistant 
bacteria, which represent one of the most serious threats to global public health.""",
    ],

    "pes2o": [
        # Scientific paper prose
        """Transformer architectures have become the dominant paradigm in natural 
language processing since their introduction in 2017. The core innovation is the 
self-attention mechanism, which allows each token in a sequence to attend to all 
other tokens simultaneously, capturing long-range dependencies more effectively 
than recurrent architectures. Scaled dot-product attention computes a weighted 
sum of values, where the weights are determined by the compatibility between 
queries and keys. Multi-head attention applies this mechanism in parallel across 
multiple learned subspaces.""",

        """Gradient descent is an iterative optimization algorithm used to minimize 
a differentiable objective function. At each step, the parameters are updated in 
the direction opposite to the gradient of the loss with respect to those parameters, 
scaled by a learning rate hyperparameter. Stochastic gradient descent approximates 
the true gradient using a single sample or mini-batch, introducing noise that can 
help escape local minima. Adaptive methods such as Adam maintain per-parameter 
learning rates based on estimates of first and second moments of the gradients.""",
    ],
}


@torch.no_grad()
def compute_passage_perplexity(model, tokenizer, text):
    """Compute perplexity of a model on a single text passage."""
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    if ids.shape[1] < 2:
        return float("nan")
    # Truncate to max_seq_len if needed
    ids = ids[:, :cfg.max_seq_len]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(ids)
    log_probs = F.log_softmax(logits, dim=-1)
    token_scores = log_probs[0, :-1, :].gather(
        1, ids[0, 1:].unsqueeze(-1)
    ).squeeze(-1)
    avg_nll = -token_scores.mean().item()
    return math.exp(avg_nll)


def evaluate_domain_perplexity(model, tokenizer):
    print("\n" + "="*65)
    print("  SECTION A — Domain Perplexity on Curated Passages")
    print("  (No val.bin used — passages embedded in script)")
    print("="*65)

    domain_results = {}
    all_ppls       = []

    for domain, passages in DOMAIN_PASSAGES.items():
        ppls = []
        print(f"\n  ── {domain.upper()} ──")
        for i, passage in enumerate(passages):
            ppl = compute_passage_perplexity(model, tokenizer, passage)
            ppls.append(ppl)
            token_count = len(tokenizer.encode(passage))
            print(f"  Passage {i+1}: ppl={ppl:.2f}  tokens={token_count}")

        avg_ppl = np.mean(ppls)
        std_ppl = np.std(ppls)
        domain_results[domain] = {"avg_ppl": avg_ppl, "std": std_ppl, "ppls": ppls}
        all_ppls.extend(ppls)
        print(f"  Avg: {avg_ppl:.2f}  Std: {std_ppl:.2f}")

    overall_ppl = np.mean(all_ppls)
    print(f"\n  {'='*45}")
    print(f"  {'Domain':<20} {'Avg PPL':>10}  {'Std':>8}  {'Signal'}")
    print(f"  {'-'*55}")
    for domain, res in sorted(domain_results.items(), key=lambda x: x[1]["avg_ppl"]):
        if res["avg_ppl"] < 5:
            signal = "★ Excellent"
        elif res["avg_ppl"] < 10:
            signal = "✓ Good"
        elif res["avg_ppl"] < 20:
            signal = "~ Moderate"
        else:
            signal = "✗ Weak — underfit"
        print(f"  {domain:<20} {res['avg_ppl']:>10.2f}  {res['std']:>8.2f}  {signal}")

    print(f"\n  Overall avg PPL: {overall_ppl:.2f}")
    print(f"  Interpretation: lower PPL = model learned that domain better")
    print(f"  Domains with PPL >20 are undertrained — prioritize in SFT data mix")
    return domain_results, overall_ppl


# =============================================================================
# SECTION B — NATURAL COMPLETION LOG-PROB SCORING
# MC reformatted as natural text completions — valid for base models.
# No Q/A format, no instructions. Pure "which continuation is more likely?"
# =============================================================================

NATURAL_COMPLETIONS = {

    "mathematics": [
        {
            "prefix"  : "The derivative of sin(x) with respect to x is",
            "valid"   : " cos(x)",
            "invalid" : " -sin(x)",
        },
        {
            "prefix"  : "The integral of 2x with respect to x equals",
            "valid"   : " x squared plus a constant",
            "invalid" : " 2 plus a constant",
        },
        {
            "prefix"  : "In a right triangle, the square of the hypotenuse equals the sum of",
            "valid"   : " the squares of the other two sides",
            "invalid" : " the products of the other two sides",
        },
        {
            "prefix"  : "The quadratic formula gives the roots of ax² + bx + c = 0 as x equals negative b plus or minus the square root of b squared minus 4ac, divided by",
            "valid"   : " 2a",
            "invalid" : " 4a",
        },
        {
            "prefix"  : "The value of pi is approximately",
            "valid"   : " 3.14159",
            "invalid" : " 2.71828",
        },
        {
            "prefix"  : "A function f(x) is said to be continuous at a point if the limit of f(x) as x approaches that point",
            "valid"   : " equals the value of the function at that point",
            "invalid" : " does not exist at that point",
        },
    ],

    "computer_science": [
        {
            "prefix"  : "In computer science, an algorithm with O(log n) time complexity is significantly faster than one with O(n) complexity because as input size grows, the number of operations grows",
            "valid"   : " logarithmically rather than linearly",
            "invalid" : " linearly rather than logarithmically",
        },
        {
            "prefix"  : "A hash table provides average-case O(1) time complexity for insertions and lookups because each key is mapped directly to a memory location using",
            "valid"   : " a hash function",
            "invalid" : " a sorting algorithm",
        },
        {
            "prefix"  : "Recursion is a programming technique where a function solves a problem by calling",
            "valid"   : " itself with a smaller version of the problem",
            "invalid" : " another function with a larger version of the problem",
        },
        {
            "prefix"  : "In object-oriented programming, inheritance allows a subclass to",
            "valid"   : " reuse and extend the properties and methods of a parent class",
            "invalid" : " replace the properties and methods of a parent class entirely",
        },
        {
            "prefix"  : "The TCP protocol ensures reliable data transmission by requiring the receiver to send",
            "valid"   : " acknowledgments for packets received",
            "invalid" : " retransmissions for packets received",
        },
    ],

    "science": [
        {
            "prefix"  : "Atoms of the same element that have different numbers of neutrons are called",
            "valid"   : " isotopes",
            "invalid" : " ions",
        },
        {
            "prefix"  : "According to Newton's third law of motion, for every action there is an equal and",
            "valid"   : " opposite reaction",
            "invalid" : " identical reaction in the same direction",
        },
        {
            "prefix"  : "In cellular respiration, glucose is broken down in the presence of oxygen to produce ATP, carbon dioxide, and",
            "valid"   : " water",
            "invalid" : " nitrogen",
        },
        {
            "prefix"  : "The speed of light in a vacuum is approximately 3 times 10 to the power of 8",
            "valid"   : " meters per second",
            "invalid" : " kilometers per hour",
        },
        {
            "prefix"  : "Natural selection, as proposed by Darwin, states that organisms with traits better suited to their environment are more likely to",
            "valid"   : " survive and reproduce, passing on those traits",
            "invalid" : " mutate rapidly and change species within a generation",
        },
    ],

    "logic_and_reasoning": [
        {
            "prefix"  : "If all A are B, and all B are C, then it logically follows that all A are",
            "valid"   : " C",
            "invalid" : " neither B nor C",
        },
        {
            "prefix"  : "The contrapositive of the statement 'if P then Q' is logically equivalent to the original and states 'if not Q then",
            "valid"   : " not P'",
            "invalid" : " not P and not Q'",
        },
        {
            "prefix"  : "A deductive argument is valid if the conclusion follows necessarily from the premises, and sound if it is valid and the premises are",
            "valid"   : " true",
            "invalid" : " false",
        },
        {
            "prefix"  : "The logical fallacy of affirming the consequent occurs when one incorrectly concludes that because Q is true and P implies Q,",
            "valid"   : " P must therefore be true",
            "invalid" : " Q must therefore be false",
        },
    ],

    "academic_prose_quality": [
        # Tests whether the model prefers grammatically and semantically
        # coherent academic continuations over plausible-sounding wrong ones
        {
            "prefix"  : "The results of the experiment were consistent with the hypothesis, suggesting that the proposed mechanism",
            "valid"   : " plays a significant role in the observed phenomenon",
            "invalid" : " has no relationship to the observed phenomenon whatsoever",
        },
        {
            "prefix"  : "Prior research has established a correlation between sleep deprivation and cognitive decline, but correlation does not imply",
            "valid"   : " causation",
            "invalid" : " correlation",
        },
        {
            "prefix"  : "The authors acknowledge several limitations of this study, including the small sample size, which may affect the",
            "valid"   : " generalizability of the findings",
            "invalid" : " statistical significance of the original claims",
        },
    ],
}


def evaluate_natural_completions(model, tokenizer):
    print("\n" + "="*65)
    print("  SECTION B — Natural Completion Log-Prob Scoring")
    print("  (Base-model valid: no instruction format, pure text completion)")
    print("="*65)

    category_results = {}
    grand_correct    = 0
    grand_total      = 0

    for category, items in NATURAL_COMPLETIONS.items():
        correct = 0
        print(f"\n  ── {category.upper().replace('_', ' ')} ──")
        for item in items:
            score_valid   = continuation_log_prob(
                model, tokenizer, item["prefix"], item["valid"])
            score_invalid = continuation_log_prob(
                model, tokenizer, item["prefix"], item["invalid"])
            is_correct    = score_valid > score_invalid
            correct      += int(is_correct)
            mark          = "✓" if is_correct else "✗"
            margin        = score_valid - score_invalid

            print(f"  {mark}  {item['prefix'][:70]}")
            print(f"       valid  : {item['valid']:<45} score={score_valid:.3f}")
            print(f"       invalid: {item['invalid']:<45} score={score_invalid:.3f}")
            print(f"       margin : {margin:+.3f}  {'(confident)' if abs(margin) > 0.5 else '(uncertain)'}")

        acc = correct / len(items)
        category_results[category] = {"correct": correct, "total": len(items), "acc": acc}
        print(f"  Accuracy: {correct}/{len(items)} = {acc:.0%}")
        grand_correct += correct
        grand_total   += len(items)

    overall_acc = grand_correct / grand_total
    print(f"\n  Overall: {grand_correct}/{grand_total} = {overall_acc:.0%}")
    print(f"  Random baseline: 50%  |  Target for SFT readiness: >75%")
    print(f"  Interpretation: this measures encoded knowledge, not instruction following")
    return category_results, overall_acc


# =============================================================================
# SECTION C — FEW-SHOT IN-CONTEXT LEARNING PROBE
# The only valid way to get structured output from a base model.
# We prime the context with complete examples — the model learns the
# format from context alone, no fine-tuning required.
# Tests ICL ability which emerges purely from pretraining scale.
# =============================================================================

FEW_SHOT_TASKS = [

    {
        "label": "math_word_problems",
        "description": "Arithmetic reasoning with few-shot examples",
        "shots": [
            {
                "problem": "A baker makes 12 loaves of bread per hour. How many loaves does he make in 5 hours?",
                "answer"  : "The baker makes 12 loaves per hour. In 5 hours he makes 12 times 5 equals 60 loaves."
            },
            {
                "problem": "A car travels 55 miles per hour. How far does it travel in 3 hours?",
                "answer"  : "The car travels 55 miles per hour. In 3 hours it travels 55 times 3 equals 165 miles."
            },
        ],
        "probes": [
            {
                "problem" : "A factory produces 150 units per day. How many units does it produce in 6 days?",
                "expected": "900",
            },
            {
                "problem" : "A cyclist rides at 18 miles per hour. How far does she ride in 4 hours?",
                "expected": "72",
            },
            {
                "problem" : "A printer prints 30 pages per minute. How many pages does it print in 8 minutes?",
                "expected": "240",
            },
        ],
    },

    {
        "label": "logical_syllogisms",
        "description": "Deductive reasoning with few-shot examples",
        "shots": [
            {
                "problem": "All birds have wings. A sparrow is a bird. Does a sparrow have wings?",
                "answer"  : "All birds have wings. A sparrow is a bird. Therefore, a sparrow has wings. Yes."
            },
            {
                "problem": "All metals conduct electricity. Iron is a metal. Does iron conduct electricity?",
                "answer"  : "All metals conduct electricity. Iron is a metal. Therefore, iron conducts electricity. Yes."
            },
        ],
        "probes": [
            {
                "problem" : "All squares are rectangles. Shape X is a square. Is shape X a rectangle?",
                "expected": "yes",
            },
            {
                "problem" : "No fish are mammals. A salmon is a fish. Is a salmon a mammal?",
                "expected": "no",
            },
            {
                "problem" : "All prime numbers greater than 2 are odd. 17 is a prime number greater than 2. Is 17 odd?",
                "expected": "yes",
            },
        ],
    },

    {
        "label": "pattern_completion",
        "description": "Sequence and pattern reasoning",
        "shots": [
            {
                "problem": "What comes next in the sequence: 2, 4, 6, 8, ?",
                "answer"  : "The sequence increases by 2 each time. The next number is 10."
            },
            {
                "problem": "What comes next in the sequence: 1, 3, 9, 27, ?",
                "answer"  : "The sequence multiplies by 3 each time. The next number is 81."
            },
        ],
        "probes": [
            {
                "problem" : "What comes next in the sequence: 5, 10, 15, 20, ?",
                "expected": "25",
            },
            {
                "problem" : "What comes next in the sequence: 1, 4, 9, 16, ?",
                "expected": "25",
            },
            {
                "problem" : "What comes next in the sequence: 100, 50, 25, 12.5, ?",
                "expected": "6.25",
            },
        ],
    },

    {
        "label": "cause_and_effect",
        "description": "Causal reasoning with few-shot priming",
        "shots": [
            {
                "problem": "A metal rod is heated. What happens to its length?",
                "answer"  : "When a metal rod is heated, its molecules move faster and spread apart. The rod expands and its length increases."
            },
            {
                "problem": "A plant is kept in complete darkness for a month. What happens to it?",
                "answer"  : "Without light, the plant cannot perform photosynthesis and cannot produce energy. The plant wilts and eventually dies."
            },
        ],
        "probes": [
            {
                "problem" : "Water is cooled below 0 degrees Celsius. What happens to it?",
                "expected": "ice",
            },
            {
                "problem" : "A gas is compressed into a smaller volume at constant temperature. What happens to its pressure?",
                "expected": "increase",
            },
            {
                "problem" : "An object is dropped from a height in a vacuum. What happens to its speed as it falls?",
                "expected": "increase",
            },
        ],
    },
]


def build_few_shot_prompt(task, probe):
    """Construct the full few-shot prompt for a given task and probe."""
    lines = []
    for i, shot in enumerate(task["shots"]):
        lines.append(f"Problem: {shot['problem']}")
        lines.append(f"Solution: {shot['answer']}")
        lines.append("")
    lines.append(f"Problem: {probe['problem']}")
    lines.append("Solution:")
    return "\n".join(lines)


def evaluate_few_shot_icl(model, tokenizer):
    print("\n" + "="*65)
    print("  SECTION C — Few-Shot In-Context Learning")
    print("  (Only valid way to probe structured reasoning from a base model)")
    print("="*65)

    task_results = []

    for task in FEW_SHOT_TASKS:
        print(f"\n  ── {task['label'].upper().replace('_', ' ')} ──")
        print(f"  {task['description']}")
        correct = 0

        for probe in task["probes"]:
            prompt = build_few_shot_prompt(task, probe)
            output = generate(model, tokenizer, prompt,
                              max_new_tokens=100,
                              temperature=0.3,   # low temp — we want the model's best guess
                              top_k=20)

            # Take only the first line of output — the model's direct answer
            first_line = output.strip().split("\n")[0].strip()
            hit        = probe["expected"].lower() in (first_line.lower() + output.lower()[:100])
            correct   += int(hit)
            mark       = "✓" if hit else "✗"

            print(f"\n  {mark}  Problem : {probe['problem']}")
            print(f"       Expected: {probe['expected']}")
            print(f"       Output  : {first_line[:120]}")

        acc = correct / len(task["probes"])
        task_results.append({
            "label"  : task["label"],
            "correct": correct,
            "total"  : len(task["probes"]),
            "acc"    : acc,
        })
        print(f"  Accuracy: {correct}/{len(task['probes'])} = {acc:.0%}")

    overall = np.mean([r["acc"] for r in task_results])

    print(f"\n  --- Few-Shot ICL Summary ---")
    print(f"  {'Task':<30} {'Accuracy':>10}")
    print(f"  {'-'*42}")
    for r in task_results:
        bar  = "█" * int(r["acc"] * 10) + "░" * (10 - int(r["acc"] * 10))
        print(f"  {r['label']:<30} {bar}  {r['acc']:.0%}")
    print(f"  {'OVERALL':<30} {overall:.0%}")

    print(f"\n  Interpretation:")
    print(f"  ICL accuracy measures how well the model generalises from examples")
    print(f"  in its context window. This is a direct product of pretraining quality.")
    print(f"  Strong ICL → SFT will be very effective.")
    print(f"  Weak ICL   → SFT will need more data and more epochs.")

    if overall >= 0.7:
        print(f"  Signal: ★ Strong ICL — excellent SFT candidate")
    elif overall >= 0.45:
        print(f"  Signal: ~ Moderate ICL — SFT will work but needs good data")
    else:
        print(f"  Signal: ✗ Weak ICL — base model may need more pretraining")

    return task_results, overall

# =============================================================================
# MAIN
# =============================================================================

import sys

class Tee:
    def __init__(self, filepath):
        self.file    = open(filepath, "w", encoding="utf-8")
        self.stdout  = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    def close(self):
        self.file.close()

if __name__ == "__main__":
    tee = Tee("evaluation_results.txt")
    sys.stdout = tee

    try:
        print("\n" + "="*65)
        print("  SLM Base Model Evaluation")
        print("  Training data: cosmopedia | dclm-edu | dolmino |")
        print("                 openwebmath | pes2o | stackexchange")
        print("  Evaluation   : fully independent — no val.bin used")
        print("="*65)

        model     = load_model()
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

        # --- Existing sections (language quality) ---
        gen_results             = evaluate_generation_quality(model, tokenizer)
        mc_results, mc_overall  = evaluate_multiple_choice(model, tokenizer)
        cot_results             = evaluate_chain_of_thought(model, tokenizer)
        know_results            = evaluate_knowledge(model, tokenizer)
        stress_results          = evaluate_stress(model, tokenizer)

        # --- New sections (base-model-valid) ---
        domain_ppl, overall_ppl         = evaluate_domain_perplexity(model, tokenizer)
        nat_results, nat_overall        = evaluate_natural_completions(model, tokenizer)
        icl_results, icl_overall        = evaluate_few_shot_icl(model, tokenizer)

        # --- Reports ---
        print_final_report(mc_results, mc_overall, cot_results, know_results)

        # Extended report for new sections
        print("\n" + "="*65)
        print("  EXTENDED REPORT — BASE MODEL VALIDITY SECTIONS")
        print("="*65)

        print(f"\n  Section A — Domain Perplexity")
        for domain, res in sorted(domain_ppl.items(), key=lambda x: x[1]["avg_ppl"]):
            bar = "█" * max(1, int(20 - min(res["avg_ppl"], 20)))  + "░" * min(int(res["avg_ppl"]), 20)
            print(f"  {domain:<20} ppl={res['avg_ppl']:>7.2f}  {bar}")
        print(f"  Overall avg PPL: {overall_ppl:.2f}")

        print(f"\n  Section B — Natural Completion Accuracy")
        for cat, res in nat_results.items():
            print(f"  {cat:<30} {res['correct']}/{res['total']} = {res['acc']:.0%}")
        print(f"  Overall: {nat_overall:.0%}  (random baseline=50%)")

        print(f"\n  Section C — Few-Shot ICL Accuracy")
        for r in icl_results:
            print(f"  {r['label']:<30} {r['correct']}/{r['total']} = {r['acc']:.0%}")
        print(f"  Overall: {icl_overall:.0%}")

        print(f"\n  ── Composite Readiness Score ──")
        # Weighted: domain PPL contributes inversely, others directly
        ppl_score  = max(0.0, min(1.0, (20 - overall_ppl) / 20))  # 0=PPL≥20, 1=PPL≤0
        composite  = (ppl_score * 0.35) + (nat_overall * 0.35) + (icl_overall * 0.30)
        print(f"  Domain PPL score  (35%) : {ppl_score:.2f}")
        print(f"  Natural completion(35%) : {nat_overall:.2f}")
        print(f"  Few-shot ICL      (30%) : {icl_overall:.2f}")
        print(f"  Composite score         : {composite:.2f} / 1.00")
        if composite >= 0.70:
            print(f"  Verdict: ★ Strong base — proceed to SFT")
        elif composite >= 0.50:
            print(f"  Verdict: ~ Good base — SFT will be effective")
        elif composite >= 0.35:
            print(f"  Verdict: ~ Moderate — SFT needed, consider data quality")
        else:
            print(f"  Verdict: ✗ Weak base — revisit pretraining before SFT")
        print("="*65 + "\n")

    finally:
        sys.stdout = tee.stdout
        tee.close()
        print("Results saved → evaluation_results.txt")