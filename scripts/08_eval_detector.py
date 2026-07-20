#!/usr/bin/env python3
"""
08_eval_detector.py — AI detection bypass analysis

Tests whether community-adapted LoRA output evades AI text detectors
better than vanilla (unadapted) LLM output.

Three conditions per community pair:
  A) Real Reddit comments  (ground truth "human")
  B) Vanilla Mistral-7B    (no adapter, base model only)
  C) Community-adapted     (LoRA-blended output, alpha=1.0)

Detector: Hello-SimpleAI/chatgpt-detector-roberta
  → P(AI-generated) in [0, 1];  lower = harder to detect as AI

Key claim: Condition C scores closer to A than B does.
Community adaptation creates a statistical blind spot in off-the-shelf detectors.

Output:
  results/eval/detection_scores.csv
  results/eval/plots/detection_bypass.png

Usage:
    python3 scripts/08_eval_detector.py
    python3 scripts/08_eval_detector.py --no-vanilla   # skip base model generation
"""

import csv
import json
import argparse
import re
import random
from pathlib import Path

import torch

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data" / "formatted"
GEN_DIR     = BASE_DIR / "results" / "generated"
EVAL_DIR    = BASE_DIR / "results" / "eval"
ADAPTER_DIR = BASE_DIR / "adapters"
MODEL_ID    = "mistralai/Mistral-7B-v0.1"

PAIRS = {
    1: ("politics",  "Conservative"),
    2: ("worldnews", "Sino"),
    3: ("climate",   "climateskeptics"),
}

PAIR_TOPICS = {
    1: "Should undocumented immigrants receive a path to citizenship?",
    2: "Does China pose a threat to global democratic norms?",
    3: "Should governments impose immediate carbon taxes?",
}

N_SAMPLES   = 20   # samples per condition per pair
URL_RE      = re.compile(r'https?://\S+')
MIN_WORDS   = 15
MAX_WORDS   = 80   # cap to match generation length


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_real_comments(community: str, n: int) -> list:
    path = DATA_DIR / f"{community}.jsonl"
    comments = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            text = URL_RE.sub('', rec['messages'][1]['content']).strip()
            words = text.split()
            if MIN_WORDS <= len(words) <= MAX_WORDS:
                comments.append(' '.join(words[:MAX_WORDS]))
    random.seed(42)
    random.shuffle(comments)
    return comments[:n]


def load_adapted_texts(comm_a: str, alpha: float, n: int) -> list:
    blend_dir = GEN_DIR / f"blend_{comm_a}_{list(PAIRS.values())[next(i for i,(a,b) in PAIRS.items() if a==comm_a)-1][1]}"
    # Find blend dir for this community as A
    for pid, (a, b) in PAIRS.items():
        if a == comm_a:
            blend_dir = GEN_DIR / f"blend_{a}_{b}"
            break
    samples = sorted(blend_dir.glob(f"alpha_{alpha:.2f}_s*.txt"))
    if not samples:
        single = blend_dir / f"alpha_{alpha:.2f}.txt"
        samples = [single] if single.exists() else []
    texts = []
    for p in samples:
        raw = p.read_text().strip()
        parts = raw.split("\n\n", 1)
        text = parts[1].strip() if len(parts) > 1 else raw
        text = URL_RE.sub('', text).strip()
        words = text.split()
        if len(words) >= MIN_WORDS:
            texts.append(' '.join(words[:MAX_WORDS]))
    return texts[:n]


# ---------------------------------------------------------------------------
# Vanilla generation (base model, no adapter)
# ---------------------------------------------------------------------------

def generate_vanilla(topic: str, n: int) -> list:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cap[0] >= 8 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    model.eval()

    prompt = f"### Post:\n{topic}\n### Comment:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    texts = []
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=100, temperature=0.8, top_p=0.85,
                repetition_penalty=1.4, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        words = text.split()
        if len(words) >= MIN_WORDS:
            texts.append(' '.join(words[:MAX_WORDS]))
    del model
    torch.cuda.empty_cache()
    return texts


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def load_detector():
    from transformers import pipeline
    print("Loading AI detector (chatgpt-detector-roberta)...")
    return pipeline(
        "text-classification",
        model="Hello-SimpleAI/chatgpt-detector-roberta",
        device=0 if torch.cuda.is_available() else -1,
        truncation=True, max_length=512,
    )


def score_texts(detector, texts: list) -> list:
    """Return list of P(AI-generated) scores in [0,1]."""
    scores = []
    for text in texts:
        result = detector(text[:1000])[0]
        # label is "ChatGPT" or "Human"
        p_ai = result['score'] if result['label'] == 'ChatGPT' else 1 - result['score']
        scores.append(p_ai)
    return scores


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_detection(rows: list, out_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    pairs_data = {}
    for row in rows:
        key = row['pair']
        pairs_data.setdefault(key, {'real': [], 'vanilla': [], 'adapted': []})
        pairs_data[key][row['condition']].append(float(row['p_ai']))

    n_pairs = len(pairs_data)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5), sharey=True)
    if n_pairs == 1:
        axes = [axes]

    colors = {'real': '#2ecc71', 'vanilla': '#e74c3c', 'adapted': '#3498db'}
    labels = {'real': 'Real Reddit', 'vanilla': 'Vanilla LLM', 'adapted': 'Community-Adapted'}

    for ax, (pair, data) in zip(axes, pairs_data.items()):
        positions = [1, 2, 3]
        bplot = ax.boxplot(
            [data['real'], data['vanilla'], data['adapted']],
            positions=positions, patch_artist=True, widths=0.5,
            medianprops=dict(color='black', linewidth=2),
        )
        for patch, color in zip(bplot['boxes'], [colors['real'], colors['vanilla'], colors['adapted']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # means as dots
        for pos, cond in zip(positions, ['real', 'vanilla', 'adapted']):
            vals = data[cond]
            if vals:
                ax.plot(pos, np.mean(vals), 'D', color='black', markersize=8, zorder=5)

        ax.set_xticks(positions)
        ax.set_xticklabels(['Real\nReddit', 'Vanilla\nLLM', 'Community\nAdapted'], fontsize=10)
        ax.set_title(pair.replace('_vs_', ' vs '), fontsize=11)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel('P(AI-generated)  ↑ = detected as AI', fontsize=11)
    fig.suptitle('Detection Bypass: Community-Adapted Text vs Vanilla LLM\n'
                 'Lower score = harder to detect as AI-generated', fontsize=12)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[c], alpha=0.7, label=labels[c])
                       for c in ['real', 'vanilla', 'adapted']]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-vanilla", action="store_true",
                        help="Skip vanilla LLM generation (use cached if available)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--adapter-dir", default=None,
                        help="Path to adapter directory (default: adapters/)")
    parser.add_argument("--gen-dir", default=None,
                        help="Path to generated outputs directory (default: results/generated/)")
    parser.add_argument("--eval-dir", default=None,
                        help="Path to eval output directory (default: results/eval/)")
    args = parser.parse_args()

    global ADAPTER_DIR, GEN_DIR, EVAL_DIR
    if args.adapter_dir:
        ADAPTER_DIR = Path(args.adapter_dir)
    if args.gen_dir:
        GEN_DIR = Path(args.gen_dir)
    if args.eval_dir:
        EVAL_DIR = Path(args.eval_dir)

    random.seed(42)
    detector = load_detector()

    all_rows = []

    for pair_id, (comm_a, comm_b) in PAIRS.items():
        pair_name = f"{comm_a}_vs_{comm_b}"
        topic = PAIR_TOPICS[pair_id]
        print(f"\n{'='*60}")
        print(f"  Pair {pair_id}: {comm_a} vs {comm_b}")
        print(f"{'='*60}")

        # Condition A: real comments (from community_a — the "target" community)
        real = load_real_comments(comm_a, N_SAMPLES)
        print(f"  Real comments: n={len(real)}")
        real_scores = score_texts(detector, real)
        mean_real = sum(real_scores) / len(real_scores)
        print(f"  Real   → P(AI) mean={mean_real:.3f}")
        for s in real_scores:
            all_rows.append({'pair': pair_name, 'condition': 'real', 'p_ai': s})

        # Condition B: vanilla LLM
        vanilla_cache = EVAL_DIR / f"vanilla_{comm_a}.txt"
        if not args.no_vanilla or not vanilla_cache.exists():
            print(f"  Generating vanilla outputs (n={N_SAMPLES})...")
            vanilla = generate_vanilla(topic, N_SAMPLES)
            vanilla_cache.parent.mkdir(parents=True, exist_ok=True)
            vanilla_cache.write_text('\n---\n'.join(vanilla))
        else:
            vanilla = vanilla_cache.read_text().split('\n---\n')[:N_SAMPLES]
        print(f"  Vanilla outputs: n={len(vanilla)}")
        van_scores = score_texts(detector, vanilla)
        mean_van = sum(van_scores) / len(van_scores)
        print(f"  Vanilla → P(AI) mean={mean_van:.3f}")
        for s in van_scores:
            all_rows.append({'pair': pair_name, 'condition': 'vanilla', 'p_ai': s})

        # Condition C: community-adapted (alpha=1.0 = pure community_a)
        adapted = load_adapted_texts(comm_a, 1.0, N_SAMPLES)
        print(f"  Adapted outputs: n={len(adapted)}")
        ada_scores = score_texts(detector, adapted)
        mean_ada = sum(ada_scores) / len(ada_scores)
        print(f"  Adapted → P(AI) mean={mean_ada:.3f}")
        bypass_gain = mean_van - mean_ada
        print(f"  Detection bypass gain: {bypass_gain:+.3f} (positive = harder to detect)")
        for s in ada_scores:
            all_rows.append({'pair': pair_name, 'condition': 'adapted', 'p_ai': s})

    # Save CSV
    out_csv = EVAL_DIR / "detection_scores.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pair', 'condition', 'p_ai'])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n  Saved → {out_csv}")

    if not args.no_plots:
        plot_detection(all_rows, EVAL_DIR / "plots" / "detection_bypass.png")

    # Print summary table
    print("\n" + "="*60)
    print("  DETECTION BYPASS SUMMARY")
    print("="*60)
    print(f"  {'Pair':<30} {'Real':>8} {'Vanilla':>8} {'Adapted':>8} {'Bypass':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for pair_id, (comm_a, comm_b) in PAIRS.items():
        pair_name = f"{comm_a}_vs_{comm_b}"
        r = [float(x['p_ai']) for x in all_rows if x['pair']==pair_name and x['condition']=='real']
        v = [float(x['p_ai']) for x in all_rows if x['pair']==pair_name and x['condition']=='vanilla']
        a = [float(x['p_ai']) for x in all_rows if x['pair']==pair_name and x['condition']=='adapted']
        print(f"  {pair_name:<30} {sum(r)/len(r):>8.3f} {sum(v)/len(v):>8.3f} {sum(a)/len(a):>8.3f} {sum(v)/len(v)-sum(a)/len(a):>+8.3f}")
    print("="*60)
    print("  Bypass gain > 0 means community adaptation reduces detection rate.")


if __name__ == "__main__":
    main()
