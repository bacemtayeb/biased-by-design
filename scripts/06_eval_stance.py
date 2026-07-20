#!/usr/bin/env python3
"""
06_eval_stance.py — Stance classification across the ideology dial and debate turns

Two analyses:
  A) Ideology dial curve  — classify blend outputs at alpha 0.0→1.0, plot stance vs alpha
  B) Debate divergence    — classify each turn in a debate, plot per-community stance over rounds

Classifier: facebook/bart-large-mnli (zero-shot NLI, GPU)
Stance score: P(text entails community_a label) in [0, 1]
  → 1.0 = fully community_a stance
  → 0.0 = fully community_b stance

Output:
  results/eval/stance_alpha_{A}_{B}.csv
  results/eval/stance_debate_{A}_{B}.csv
  results/eval/plots/ideology_dial.png
  results/eval/plots/debate_divergence.png

Usage:
    python3 scripts/06_eval_stance.py
    python3 scripts/06_eval_stance.py --pair 1
    python3 scripts/06_eval_stance.py --no-plots   # CSV only, no matplotlib
"""

import json
import csv
import argparse
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
GEN_DIR    = BASE_DIR / "results" / "generated"
DEBATE_DIR = BASE_DIR / "results" / "debates"
EVAL_DIR   = BASE_DIR / "results" / "eval"

PAIRS = {
    1: ("politics",  "Conservative"),
    2: ("worldnews", "Sino"),
    3: ("climate",   "climateskeptics"),
}

# Zero-shot labels per pair.
# Label[0] = community_a stance → high score expected at alpha=1.0
# Label[1] = community_b stance → high score expected at alpha=0.0
# Rules: both labels must be positive statements (no negations); mutually exclusive framing.
# Multiple hypotheses per pair are averaged for robustness.
STANCE_LABELS = {
    1: [
        # politics (A) vs Conservative (B)
        ("the author supports progressive government policies and civil liberties",
         "the author supports conservative values, limited government, and individual rights"),
        ("this text emphasizes systemic inequality, institutional accountability, or human rights",
         "this text emphasizes personal responsibility, national security, or traditional values"),
    ],
    2: [
        # worldnews (A) vs Sino (B)
        ("the author is critical of China's government and supports Western democratic norms",
         "the author defends China's sovereignty and criticizes Western interference in Asia"),
        ("this text portrays China as a geopolitical threat or authoritarian state",
         "this text portrays Western criticism of China as biased, hypocritical, or imperialist"),
    ],
    3: [
        # climate (A) vs climateskeptics (B)
        ("the author urgently supports climate action and trusts the scientific consensus",
         "the author is skeptical of climate policy and believes it harms the economy"),
        ("this text frames climate change as an urgent crisis requiring immediate government action",
         "this text frames climate policy as economically damaging or scientifically uncertain"),
    ],
}

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_classifier():
    from transformers import pipeline
    import torch
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading zero-shot classifier (bart-large-mnli) on {'GPU' if device == 0 else 'CPU'}...")
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)


def classify(clf, text: str, labels) -> float:
    """Return stance score toward community_a (label[0]).
    labels: single (a, b) tuple OR list of (a, b) tuples — scores averaged."""
    if isinstance(labels[0], str):
        labels = [labels]
    scores = []
    for label_pair in labels:
        result = clf(text[:1024], candidate_labels=list(label_pair))
        idx = result["labels"].index(label_pair[0])
        scores.append(result["scores"][idx])
    return sum(scores) / len(scores)


def _strip_header(raw: str) -> str:
    parts = raw.strip().split("\n\n", 1)
    return parts[1].strip() if len(parts) > 1 else raw.strip()


def extract_blend_texts(blend_dir: Path, alpha: float) -> list:
    """Return list of texts for this alpha. Prefers multi-sample _s*.txt files; falls back to single."""
    samples = sorted(blend_dir.glob(f"alpha_{alpha:.2f}_s*.txt"))
    if samples:
        return [_strip_header(p.read_text()) for p in samples]
    single = blend_dir / f"alpha_{alpha:.2f}.txt"
    if single.exists():
        return [_strip_header(single.read_text())]
    return []


def latest_debate(pair_name: str) -> Path | None:
    """Return transcript.json from the most recent debate run for a pair."""
    pair_dir = DEBATE_DIR / pair_name
    if not pair_dir.exists():
        return None
    runs = sorted(pair_dir.iterdir())
    for run in reversed(runs):
        t = run / "transcript.json"
        if t.exists():
            return t
    return None


# ---------------------------------------------------------------------------
# Analysis A: ideology dial curve
# ---------------------------------------------------------------------------

def eval_alpha_curve(clf, pair_id: int, comm_a: str, comm_b: str) -> list:
    """Classify blend outputs at each alpha. Returns list of (alpha, score) dicts."""
    blend_name = f"blend_{comm_a}_{comm_b}"
    blend_path = GEN_DIR / blend_name
    if not blend_path.exists():
        print(f"  [skip] blend dir not found: {blend_path}")
        return []

    labels = STANCE_LABELS[pair_id]
    rows   = []
    first  = labels[0] if isinstance(labels[0], tuple) else labels
    print(f"\n  Pair {pair_id}: {comm_a} vs {comm_b}")
    print(f"  Labels ({len(labels) if isinstance(labels[0], tuple) else 1} hyp): '{first[0][:50]}' vs '{first[1][:50]}'")

    for alpha in ALPHAS:
        texts = extract_blend_texts(blend_path, alpha)
        if not texts:
            print(f"    alpha={alpha:.2f}  [missing]")
            continue
        scores = [classify(clf, t, labels) for t in texts]
        avg = sum(scores) / len(scores)
        print(f"    alpha={alpha:.2f}  stance={avg:.3f}  (n={len(scores)}, raw={[f'{s:.2f}' for s in scores]})")
        rows.append({"pair": f"{comm_a}_vs_{comm_b}", "alpha": alpha,
                     "stance_score": avg, "n_samples": len(scores),
                     "community_a": comm_a, "community_b": comm_b})

    return rows


# ---------------------------------------------------------------------------
# Analysis B: debate divergence
# ---------------------------------------------------------------------------

def eval_debate(clf, pair_id: int, comm_a: str, comm_b: str) -> list:
    """Classify each turn in the latest debate. Returns list of turn dicts."""
    pair_name = f"{comm_a}_vs_{comm_b}"
    transcript_path = latest_debate(pair_name)

    if transcript_path is None:
        print(f"  [skip] no debate transcript for {pair_name}")
        return []

    with open(transcript_path) as f:
        data = json.load(f)

    labels = STANCE_LABELS[pair_id]
    rows   = []

    print(f"\n  Debate {pair_id}: {pair_name}  ({len(data['turns'])} turns)")
    print(f"  Transcript: {transcript_path}")

    for turn in data["turns"]:
        text  = turn["text"]
        score = classify(clf, text, labels)
        side  = "a" if turn["community"] == comm_a else "b"
        print(f"    turn {turn['turn']:2d}  [{turn['community']:20s}]  stance={score:.3f}")
        rows.append({
            "pair":         pair_name,
            "turn":         turn["turn"],
            "community":    turn["community"],
            "side":         side,
            "stance_score": score,
        })

    return rows


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def save_csv(rows: list, path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_ideology_dial(all_alpha_rows: list, out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    pairs_seen = {}
    for row in all_alpha_rows:
        key = row["pair"]
        pairs_seen.setdefault(key, {"alphas": [], "scores": [], "a": row["community_a"], "b": row["community_b"]})
        pairs_seen[key]["alphas"].append(row["alpha"])
        pairs_seen[key]["scores"].append(row["stance_score"])

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    for i, (pair, d) in enumerate(pairs_seen.items()):
        label = f"{d['a']} ↔ {d['b']}"
        ax.plot(d["alphas"], d["scores"], marker="o", linewidth=2,
                color=colors[i % len(colors)], label=label)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Alpha  (0.0 = pure community_b  →  1.0 = pure community_a)", fontsize=11)
    ax.set_ylabel("Stance score toward community_a", fontsize=11)
    ax.set_title("Ideology Dial: Stance Shift Across Adapter Blends", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_debate_divergence(all_debate_rows: list, out_path: Path):
    import matplotlib.pyplot as plt

    pairs = {}
    for row in all_debate_rows:
        pairs.setdefault(row["pair"], []).append(row)

    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5), sharey=True)
    if n_pairs == 1:
        axes = [axes]

    colors = {"a": "#e41a1c", "b": "#377eb8"}

    for ax, (pair_name, rows) in zip(axes, pairs.items()):
        communities = {}
        for row in rows:
            communities.setdefault(row["side"], {"turns": [], "scores": [], "label": row["community"]})
            communities[row["side"]]["turns"].append(row["turn"])
            communities[row["side"]]["scores"].append(row["stance_score"])

        for side, d in communities.items():
            ax.plot(d["turns"], d["scores"], marker="o", linewidth=2,
                    color=colors[side], label=d["label"])

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(pair_name.replace("_vs_", " vs "), fontsize=11)
        ax.set_xlabel("Turn", fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Stance score toward community_a pole", fontsize=11)
    fig.suptitle("Debate Divergence: Stance per Turn by Community", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stance eval: ideology dial + debate divergence.")
    parser.add_argument("--pair", type=int, choices=[1, 2, 3], default=None,
                        help="Single pair to eval (default: all)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip matplotlib plots, output CSVs only")
    parser.add_argument("--gen-dir", default=None,
                        help="Path to generated outputs directory (default: results/generated/)")
    parser.add_argument("--debate-dir", default=None,
                        help="Path to debates directory (default: results/debates/)")
    parser.add_argument("--eval-dir", default=None,
                        help="Path to eval output directory (default: results/eval/)")
    args = parser.parse_args()

    global GEN_DIR, DEBATE_DIR, EVAL_DIR
    if args.gen_dir:
        GEN_DIR = Path(args.gen_dir)
    if args.debate_dir:
        DEBATE_DIR = Path(args.debate_dir)
    if args.eval_dir:
        EVAL_DIR = Path(args.eval_dir)

    jobs = [(pid, a, b) for pid, (a, b) in PAIRS.items()] if args.pair is None \
           else [(args.pair, *PAIRS[args.pair])]

    clf = load_classifier()

    all_alpha_rows  = []
    all_debate_rows = []

    for pair_id, comm_a, comm_b in jobs:
        print(f"\n{'='*60}")
        print(f"  Evaluating pair {pair_id}: {comm_a} vs {comm_b}")
        print(f"{'='*60}")

        alpha_rows  = eval_alpha_curve(clf, pair_id, comm_a, comm_b)
        debate_rows = eval_debate(clf, pair_id, comm_a, comm_b)

        save_csv(alpha_rows,  EVAL_DIR / f"stance_alpha_{comm_a}_{comm_b}.csv")
        save_csv(debate_rows, EVAL_DIR / f"stance_debate_{comm_a}_{comm_b}.csv")

        all_alpha_rows.extend(alpha_rows)
        all_debate_rows.extend(debate_rows)

    if not args.no_plots:
        plot_dir = EVAL_DIR / "plots"
        if all_alpha_rows:
            plot_ideology_dial(all_alpha_rows,  plot_dir / "ideology_dial.png")
        if all_debate_rows:
            plot_debate_divergence(all_debate_rows, plot_dir / "debate_divergence.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
