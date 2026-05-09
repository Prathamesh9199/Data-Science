# verify_sft_data.py
# Run before retraining. Identifies format problems in the actual training data.

import json
import re
from collections import defaultdict

def verify_sft_sample(path="data/sft/master/sft_train.jsonl", n=500):
    stats = defaultdict(int)
    bad_examples = []

    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= n: break
            ex = json.loads(line)
            sol = ex["solution"]

            has_step = bool(re.search(r"Step\s+\d+:", sol))
            has_term = "therefore, the answer is" in sol.lower()
            has_placeholder = "[see answer field]" in sol
            n_steps = len(re.findall(r"Step\s+\d+:", sol))

            stats["total"] += 1
            if has_step:      stats["has_step"] += 1
            if has_term:      stats["has_term"] += 1
            if has_placeholder: stats["has_placeholder"] += 1
            if n_steps >= 2:  stats["two_plus_steps"] += 1

            if not has_step or not has_term or has_placeholder:
                bad_examples.append({
                    "id"     : ex["id"],
                    "source" : ex["source"],
                    "problem": ex["problem"][:80],
                    "solution_preview": sol[:200],
                    "flags"  : {
                        "no_step": not has_step,
                        "no_term": not has_term,
                        "placeholder": has_placeholder,
                    }
                })

    print(f"\nVerified {stats['total']} examples from {path}")
    print(f"  Has Step N:            {stats['has_step']:>5} / {stats['total']}  "
          f"({100*stats['has_step']//stats['total']}%)")
    print(f"  Has Therefore term     {stats['has_term']:>5} / {stats['total']}  "
          f"({100*stats['has_term']//stats['total']}%)")
    print(f"  Has placeholder [BAD]  {stats['has_placeholder']:>5} / {stats['total']}  "
          f"({100*stats['has_placeholder']//stats['total']}%)")
    print(f"  Two+ distinct steps    {stats['two_plus_steps']:>5} / {stats['total']}  "
          f"({100*stats['two_plus_steps']//stats['total']}%)")
    print(f"\n  First 5 bad examples:")
    for ex in bad_examples[:5]:
        print(f"    [{ex['source']}] {ex['flags']}")
        print(f"    Problem : {ex['problem']}")
        print(f"    Solution: {ex['solution_preview']}")
        print()

if __name__ == "__main__":
    verify_sft_sample()     