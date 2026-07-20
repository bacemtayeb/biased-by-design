#!/usr/bin/env python3
"""
09b_embedding_fingerprint.py — Sentence embedding community fingerprint + t-SNE

Replaces TF-IDF with sentence embeddings (all-MiniLM-L6-v2).
Produces a t-SNE plot showing:
  - Real community A texts (cluster)
  - Real community B texts (cluster)
  - Generated texts at alpha 0.0 → 1.0 (trajectory between clusters)

If generated text at alpha=1.0 clusters near real community A,
the adapter learned authentic community voice.

Output:
  results/eval/plots/embedding_tsne_{A}_{B}.png
  results/eval/embedding_similarity_{A}_{B}.csv

Usage:
    python3 scripts/09b_embedding_fingerprint.py
    python3 scripts/09b_embedding_fingerprint.py --pair 1
"""

import csv
import json
import re
import argparse
import random
from pathlib import Path

import numpy as np

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data" / "formatted"
GEN_DIR     = BASE_DIR / "results" / "generated"
EVAL_DIR    = BASE_DIR / "results" / "eval"

PAIRS = {
    1: ("politics",  "Conservative"),
    2: ("worldnews", "Sino"),
    3: ("climate",   "climateskeptics"),
}

URL_RE    = re.compile(r'https?://\S+')
MIN_WORDS = 10
N_REAL    = 150   # real samples per community for t-SNE


def clean(text: str) -> str:
    return URL_RE.sub('', text).strip()


def load_real(community: str, n: int) -> list:
    path = DATA_DIR / f"{community}.jsonl"
    texts = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            text = clean(rec['messages'][1]['content'])
            if len(text.split()) >= MIN_WORDS:
                texts.append(text)
    random.seed(42)
    random.shuffle(texts)
    return texts[:n]


def load_generated(comm_a: str, comm_b: str) -> dict:
    blend_dir = GEN_DIR / f"blend_{comm_a}_{comm_b}"
    result = {}
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        samples = sorted(blend_dir.glob(f"alpha_{alpha:.2f}_s*.txt"))
        if not samples:
            single = blend_dir / f"alpha_{alpha:.2f}.txt"
            samples = [single] if single.exists() else []
        texts = []
        for p in samples:
            raw = p.read_text().strip()
            parts = raw.split("\n\n", 1)
            text = clean(parts[1] if len(parts) > 1 else raw)
            if len(text.split()) >= MIN_WORDS:
                texts.append(text)
        result[alpha] = texts
    return result


def get_embedder():
    from sentence_transformers import SentenceTransformer
    print("Loading sentence embedder (all-MiniLM-L6-v2)...")
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed(model, texts: list) -> np.ndarray:
    return model.encode(texts, batch_size=64, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)


def cosine_sim_mean(vecs_query: np.ndarray, vecs_target: np.ndarray) -> float:
    """Mean cosine similarity between each query vec and all target vecs."""
    sims = vecs_query @ vecs_target.T   # (n_q, n_t)
    return float(sims.mean())


def eval_pair(embedder, pair_id: int, comm_a: str, comm_b: str):
    print(f"\n{'='*60}")
    print(f"  Pair {pair_id}: {comm_a} (A) vs {comm_b} (B)")
    print(f"{'='*60}")

    real_a_texts = load_real(comm_a, N_REAL)
    real_b_texts = load_real(comm_b, N_REAL)
    generated    = load_generated(comm_a, comm_b)

    print(f"  Embedding real texts ({comm_a}={len(real_a_texts)}, {comm_b}={len(real_b_texts)})...")
    emb_a = embed(embedder, real_a_texts)
    emb_b = embed(embedder, real_b_texts)

    rows = []
    all_gen_texts  = []
    all_gen_alphas = []
    all_gen_embs   = []

    print(f"\n  {'alpha':>6}  {'sim_A':>7}  {'sim_B':>7}  {'preference':>15}  {'margin':>8}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*15}  {'-'*8}")

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        texts = generated.get(alpha, [])
        if not texts:
            continue
        emb_gen = embed(embedder, texts)
        sim_a   = cosine_sim_mean(emb_gen, emb_a)
        sim_b   = cosine_sim_mean(emb_gen, emb_b)
        pref    = comm_a if sim_a >= sim_b else comm_b
        margin  = sim_a - sim_b
        print(f"  {alpha:>6.2f}  {sim_a:>7.4f}  {sim_b:>7.4f}  {pref:>15}  {margin:>+8.4f}")
        rows.append({
            'pair': f"{comm_a}_vs_{comm_b}", 'alpha': alpha,
            'sim_community_a': sim_a, 'sim_community_b': sim_b,
            'preferred_community': pref, 'margin': margin,
        })
        all_gen_texts.extend(texts)
        all_gen_alphas.extend([alpha] * len(texts))
        all_gen_embs.append(emb_gen)

    all_gen_embs = np.vstack(all_gen_embs) if all_gen_embs else np.zeros((0, emb_a.shape[1]))
    return rows, emb_a, emb_b, all_gen_embs, all_gen_alphas


def plot_tsne(comm_a: str, comm_b: str,
              emb_a: np.ndarray, emb_b: np.ndarray,
              emb_gen: np.ndarray, gen_alphas: list, out_path: Path):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from sklearn.manifold import TSNE

    n_a   = len(emb_a)
    n_b   = len(emb_b)
    n_gen = len(emb_gen)

    all_embs = np.vstack([emb_a, emb_b, emb_gen]) if n_gen > 0 else np.vstack([emb_a, emb_b])

    print(f"  Running t-SNE on {len(all_embs)} embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    coords = tsne.fit_transform(all_embs)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Real communities (background)
    ax.scatter(coords[:n_a, 0], coords[:n_a, 1],
               c='#e74c3c', alpha=0.25, s=18, label=f'Real r/{comm_a}', zorder=1)
    ax.scatter(coords[n_a:n_a+n_b, 0], coords[n_a:n_a+n_b, 1],
               c='#3498db', alpha=0.25, s=18, label=f'Real r/{comm_b}', zorder=1)

    # Generated texts — colored by alpha (0=blue → 1=red)
    if n_gen > 0:
        gen_coords = coords[n_a+n_b:]
        cmap = cm.RdYlGn
        alpha_colors = [cmap(a) for a in gen_alphas]
        sc = ax.scatter(gen_coords[:, 0], gen_coords[:, 1],
                        c=gen_alphas, cmap='RdYlGn', s=80,
                        edgecolors='black', linewidths=0.5, zorder=3,
                        label='Generated (α=0→1)')
        plt.colorbar(sc, ax=ax, label='Alpha  (0.0=pure B → 1.0=pure A)')

        # Draw trajectory: mean position per alpha
        unique_alphas = sorted(set(gen_alphas))
        mean_coords = []
        for a in unique_alphas:
            idxs = [i for i, x in enumerate(gen_alphas) if x == a]
            mean_coords.append(gen_coords[idxs].mean(axis=0))
        mean_coords = np.array(mean_coords)
        ax.plot(mean_coords[:, 0], mean_coords[:, 1], 'k--',
                linewidth=1.5, alpha=0.6, zorder=2, label='Ideology trajectory')
        for a, mc in zip(unique_alphas, mean_coords):
            ax.annotate(f'α={a:.2f}', mc, fontsize=9, fontweight='bold',
                        xytext=(4, 4), textcoords='offset points')

    ax.set_title(f't-SNE: Community Embedding Space\n'
                 f'r/{comm_a} ↔ r/{comm_b}  —  Generated text ideology trajectory',
                 fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.set_xlabel('t-SNE dim 1', fontsize=10)
    ax.set_ylabel('t-SNE dim 2', fontsize=10)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--gen-dir", default=None,
                        help="Path to generated outputs directory (default: results/generated/)")
    parser.add_argument("--eval-dir", default=None,
                        help="Path to eval output directory (default: results/eval/)")
    args = parser.parse_args()

    global GEN_DIR, EVAL_DIR
    if args.gen_dir:
        GEN_DIR = Path(args.gen_dir)
    if args.eval_dir:
        EVAL_DIR = Path(args.eval_dir)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"])

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn", "-q"])

    embedder = get_embedder()

    jobs = [(pid, a, b) for pid, (a, b) in PAIRS.items()] if args.pair is None \
           else [(args.pair, *PAIRS[args.pair])]

    for pair_id, comm_a, comm_b in jobs:
        rows, emb_a, emb_b, emb_gen, gen_alphas = eval_pair(
            embedder, pair_id, comm_a, comm_b)

        csv_path = EVAL_DIR / f"embedding_similarity_{comm_a}_{comm_b}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved → {csv_path}")

        plot_tsne(comm_a, comm_b, emb_a, emb_b, emb_gen, gen_alphas,
                  EVAL_DIR / "plots" / f"tsne_{comm_a}_{comm_b}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
