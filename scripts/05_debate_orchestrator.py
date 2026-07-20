#!/usr/bin/env python3
"""
05_debate_orchestrator.py — Multi-agent debate between opposing community adapters

Two community-conditioned adapters take turns responding to each other on a topic.
History of the last HISTORY_WINDOW turns is fed into each prompt so each agent
"sees" what the other said.

Output per run:
  results/debates/{A}_vs_{B}/{timestamp}/transcript.json   (structured, for auto-eval)
  results/debates/{A}_vs_{B}/{timestamp}/transcript.txt    (human-readable, for survey)

Usage:
    python3 scripts/05_debate_orchestrator.py --pair 1
    python3 scripts/05_debate_orchestrator.py --all-pairs
    python3 scripts/05_debate_orchestrator.py --community-a climate --community-b climateskeptics
    python3 scripts/05_debate_orchestrator.py --pair 2 --rounds 6 --topic "Is China a threat?"
"""

import gc
import os
import re
import json
import hashlib
import argparse
from datetime import datetime
from pathlib import Path

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR    = Path(__file__).parent.parent
ADAPTER_DIR = BASE_DIR / "adapters"
RESULTS_DIR = BASE_DIR / "results" / "debates"
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

# Google News search queries per pair — optimised to return relevant recent headlines
PAIR_SEARCH_QUERIES = {
    1: "US immigration policy 2025",
    2: "China geopolitics Taiwan 2025",
    3: "carbon tax climate policy 2025",
}

DEFAULT_ROUNDS  = 5   # rounds = turns per agent; total turns = rounds * 2
HISTORY_WINDOW  = 2   # previous turns included in each prompt
MAX_NEW_TOKENS  = 120
MAX_RETRIES     = 4   # regeneration attempts before accepting best effort

GEN_CONFIG = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.8,
    top_p=0.85,
    repetition_penalty=1.4,
    do_sample=True,
)

# ---------------------------------------------------------------------------
# Output filter
# ---------------------------------------------------------------------------

import re

_URL_RE       = re.compile(r'https?://|www\.|reddit\.com/r/|redd\.it/')
_META_RE      = re.compile(
    r'\[\+\d+\]|\[-\d+\]'          # score badges [+1] [-1]
    r'|\(\+\d+\|-\d+\)'            # score parens (+2|-1)
    r'|▽|▲'                        # vote arrows
    r'|\*\*[A-Za-z_]+\*\*:'        # **Username**: patterns
    r'|^\s*[>#*\-]{3,}'            # markdown headers / list spam
    r'|submission score'
    r'|View\]|Link\]',
    re.MULTILINE,
)
_REMOVED_RE   = re.compile(
    r'removed by reddit|deleted|this post was|this sub|this comment has been',
    re.IGNORECASE,
)
_MIN_WORDS    = 15


def is_noisy(text: str) -> bool:
    if len(text.split()) < _MIN_WORDS:
        return True
    if _URL_RE.search(text):
        return True
    if _REMOVED_RE.search(text):
        return True
    if _META_RE.search(text):
        return True
    return False


# ---------------------------------------------------------------------------
# RAG — news context fetch
# ---------------------------------------------------------------------------


def fetch_news_context(topic: str, cache_dir: Path,
                       search_query: str | None = None) -> dict | None:
    """Fetch top headline + excerpt from Google News RSS. Returns None on failure."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key  = hashlib.md5(topic.encode()).hexdigest()[:10]
    cache_file = cache_dir / f"news_{cache_key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    try:
        import feedparser
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "feedparser", "-q"],
                       capture_output=True)
        import feedparser

    import urllib.parse
    query = search_query or topic
    url   = ("https://news.google.com/rss/search?q="
             + urllib.parse.quote(query)
             + "&hl=en-US&gl=US&ceid=US:en")
    try:
        feed = feedparser.parse(url)
    except Exception:
        return None

    if not feed.entries:
        return None

    import html as _html

    # Build candidate list from top 10 results
    candidates = []
    for entry in feed.entries[:10]:
        t = _html.unescape(entry.get('title', ''))
        s = _html.unescape(re.sub(r'<[^>]+>', '', entry.get('summary', '')))
        candidates.append((t, s))

    # Semantic re-ranking: embed topic + headlines, pick highest cosine sim
    # This makes retrieval query-aware, satisfying the RAG definition
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        topic_vec = _embedder.encode([topic], normalize_embeddings=True)
        cand_vecs = _embedder.encode([c[0] for c in candidates], normalize_embeddings=True)
        best_idx  = int(np.argmax((topic_vec @ cand_vecs.T)[0]))
    except Exception:
        best_idx = 0

    title, summary = candidates[best_idx]
    words       = summary.split()
    raw_excerpt = ' '.join(words[:50]) if len(words) > 5 else ''
    title_core  = title.split(' - ')[0].lower().strip()
    excerpt     = '' if title_core in raw_excerpt.lower() else raw_excerpt
    result      = {'headline': title, 'excerpt': excerpt}
    cache_file.write_text(json.dumps(result))
    return result


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(topic: str, history: list, news: dict | None = None) -> str:
    """
    history: list of (community_name, text) from oldest to newest.
    news: dict with 'headline' and 'excerpt' keys, or None.
    """
    parts = []
    if news:
        parts.append(f"Recent news: {news['headline']}")
        if news.get('excerpt'):
            parts.append(news['excerpt'])
        parts.append("")
    parts.append(topic)

    recent = history[-HISTORY_WINDOW:]
    if recent:
        parts.append("")
        parts.extend(f"{speaker}: {text}" for speaker, text in recent)

    return f"### Post:\n" + "\n".join(parts) + "\n### Comment:\n"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_base_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    use_bf16 = cap[0] >= 8

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------

def run_debate(
    community_a: str,
    community_b: str,
    topic: str,
    n_rounds: int,
    news: dict | None = None,
) -> list:
    """
    Runs n_rounds of debate (each round = one turn per agent = 2 total turns).
    Returns list of turn dicts.
    Base model is loaded fresh and released after this call.
    """
    from peft import PeftModel

    for name in [community_a, community_b]:
        p = ADAPTER_DIR / name
        if not p.exists():
            raise FileNotFoundError(f"Adapter not found: {p}. Run 03_train_adapters.py first.")

    print(f"\n{'='*60}")
    print(f"  Debate : {community_a}  vs  {community_b}")
    print(f"  Topic  : {topic}")
    print(f"  Rounds : {n_rounds}  ({n_rounds * 2} total turns)")
    if news:
        print(f"  News   : {news['headline']}")
    print(f"{'='*60}\n")

    print("Loading base model...")
    base_model, tokenizer = build_base_model()

    peft_model = PeftModel.from_pretrained(
        base_model, str(ADAPTER_DIR / community_a), adapter_name="a"
    )
    peft_model.load_adapter(str(ADAPTER_DIR / community_b), adapter_name="b")

    history = []  # (community_name, text)
    turns   = []
    speakers = [("a", community_a), ("b", community_b)]

    for round_idx in range(n_rounds):
        for adapter_key, community in speakers:
            peft_model.set_adapter(adapter_key)
            peft_model.eval()

            prompt = build_prompt(topic, history, news=news)
            inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)

            response = None
            for attempt in range(MAX_RETRIES):
                with torch.no_grad():
                    out = peft_model.generate(
                        **inputs,
                        pad_token_id=tokenizer.eos_token_id,
                        **GEN_CONFIG,
                    )
                candidate = tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                if "\n\n" in candidate:
                    candidate = candidate.split("\n\n")[0].strip()
                if not is_noisy(candidate):
                    response = candidate
                    break
                if attempt == 0:
                    print(f"  [noisy output, retrying...]")

            if response is None:
                # Accept last candidate rather than drop the turn
                response = candidate
                print(f"  [WARNING: noisy output kept after {MAX_RETRIES} retries]")

            turn_num = round_idx * 2 + (1 if adapter_key == "a" else 2)
            print(f"[Turn {turn_num}] {community}:\n{response}\n")

            turns.append({
                "turn":      turn_num,
                "community": community,
                "side":      adapter_key,   # "a" or "b"
                "text":      response,
            })
            history.append((community, response))

    # Release GPU memory before next pair
    del peft_model, base_model
    torch.cuda.empty_cache()
    gc.collect()

    return turns


# ---------------------------------------------------------------------------
# Mode B: Echo chamber (same adapter vs itself)
# ---------------------------------------------------------------------------

def run_echo_chamber(community: str, topic: str, n_rounds: int,
                     news: dict | None = None) -> list:
    """Same adapter on both sides — models real echo chamber dynamics."""
    from peft import PeftModel

    p = ADAPTER_DIR / community
    if not p.exists():
        raise FileNotFoundError(f"Adapter not found: {p}")

    print(f"\n{'='*60}")
    print(f"  Echo chamber : {community}  vs  {community}")
    print(f"  Topic        : {topic}")
    print(f"  Rounds       : {n_rounds}")
    if news:
        print(f"  News         : {news['headline']}")
    print(f"{'='*60}\n")

    base_model, tokenizer = build_base_model()
    peft_model = PeftModel.from_pretrained(base_model, str(p), adapter_name="echo")
    peft_model.set_adapter("echo")
    peft_model.eval()

    history, turns = [], []
    labels = ["Agent-1", "Agent-2"]

    for round_idx in range(n_rounds):
        for agent_idx in range(2):
            prompt = build_prompt(topic, history, news=news)
            inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)

            with torch.no_grad():
                out = peft_model.generate(
                    **inputs, pad_token_id=tokenizer.eos_token_id, **GEN_CONFIG
                )
            text = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            if "\n\n" in text:
                text = text.split("\n\n")[0].strip()

            label    = labels[agent_idx]
            turn_num = round_idx * 2 + agent_idx + 1
            print(f"[Turn {turn_num}] {label}:\n{text}\n")
            turns.append({"turn": turn_num, "community": community,
                          "agent": label, "text": text})
            history.append((label, text))

    del peft_model, base_model
    torch.cuda.empty_cache()
    gc.collect()
    return turns


def save_echo_transcript(turns: list, community: str, topic: str,
                         news: dict | None = None) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir   = RESULTS_DIR / f"{community}_echo" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"mode": "echo_chamber", "community": community, "topic": topic,
            "news_context": news, "timestamp": timestamp, "turns": turns}
    (out_dir / "transcript.json").write_text(json.dumps(data, indent=2))
    lines = [f"Topic: {topic} [Echo Chamber: {community}]"]
    if news:
        lines.append(f"News:  {news['headline']}")
    lines += [f"{'─'*60}", ""]
    for t in turns:
        lines += [f"[{t['agent']}]", t["text"], ""]
    (out_dir / "transcript.txt").write_text("\n".join(lines))
    print(f"Saved → {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Mode C: Coordinated campaign (N comments, same adapter, no dialogue)
# ---------------------------------------------------------------------------

def run_campaign(community: str, topic: str, n: int = 20,
                 news: dict | None = None) -> list:
    """Generate N independent comments from the same adapter — simulates astroturfing flood."""
    from peft import PeftModel

    p = ADAPTER_DIR / community
    if not p.exists():
        raise FileNotFoundError(f"Adapter not found: {p}")

    print(f"\n{'='*60}")
    print(f"  Campaign : {community}  ×{n} comments")
    print(f"  Topic    : {topic}")
    if news:
        print(f"  News     : {news['headline']}")
    print(f"{'='*60}\n")

    base_model, tokenizer = build_base_model()
    peft_model = PeftModel.from_pretrained(base_model, str(p), adapter_name="camp")
    peft_model.set_adapter("camp")
    peft_model.eval()

    prompt = build_prompt(topic, [], news=news)
    inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)
    comments = []

    for i in range(n):
        with torch.no_grad():
            out = peft_model.generate(
                **inputs, pad_token_id=tokenizer.eos_token_id, **GEN_CONFIG
            )
        text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        if "\n\n" in text:
            text = text.split("\n\n")[0].strip()
        print(f"  [{i+1:02d}] {text[:80]}...")
        comments.append(text)

    del peft_model, base_model
    torch.cuda.empty_cache()
    gc.collect()
    return comments


def save_campaign(comments: list, community: str, topic: str,
                  news: dict | None = None) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir   = RESULTS_DIR / f"{community}_campaign" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"mode": "campaign", "community": community, "topic": topic,
            "news_context": news, "timestamp": timestamp,
            "n": len(comments), "comments": comments}
    (out_dir / "campaign.json").write_text(json.dumps(data, indent=2))
    lines = [f"Topic: {topic} [Campaign: {community} ×{len(comments)}]"]
    if news:
        lines.append(f"News: {news['headline']}")
    lines += [f"{'─'*60}", ""]
    for i, c in enumerate(comments, 1):
        lines += [f"[Comment {i:02d}]", c, ""]
    (out_dir / "campaign.txt").write_text("\n".join(lines))
    print(f"Saved → {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_transcript(
    turns: list,
    topic: str,
    community_a: str,
    community_b: str,
    news: dict | None = None,
) -> Path:
    pair_name = f"{community_a}_vs_{community_b}"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir   = RESULTS_DIR / pair_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON — used by auto-eval script
    data = {
        "pair":        pair_name,
        "community_a": community_a,
        "community_b": community_b,
        "topic":       topic,
        "news_context": news,
        "timestamp":   timestamp,
        "n_turns":     len(turns),
        "turns":       turns,
    }
    (out_dir / "transcript.json").write_text(json.dumps(data, indent=2))

    # Human-readable TXT — used for survey
    lines = [f"Topic: {topic}"]
    if news:
        lines.append(f"News:  {news['headline']}")
        if news.get('excerpt'):
            lines.append(f"       {news['excerpt']}")
    lines += [f"{'─'*60}", ""]
    for t in turns:
        lines.append(f"[{t['community']}]")
        lines.append(t["text"])
        lines.append("")
    (out_dir / "transcript.txt").write_text("\n".join(lines))

    print(f"Saved → {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate 1v1 debates between community-conditioned LoRA adapters."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pair", type=int, choices=[1, 2, 3],
                     help="Predefined pair (1=politics/Conservative, 2=worldnews/Sino, 3=climate/climateskeptics)")
    src.add_argument("--all-pairs", action="store_true",
                     help="Run all 3 predefined pairs sequentially")
    src.add_argument("--community-a", metavar="A",
                     help="Custom community A (speaks first)")

    parser.add_argument("--community-b", metavar="B",
                        help="Custom community B — required with --community-a")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help=f"Rounds per debate (default: {DEFAULT_ROUNDS}; total turns = rounds×2)")
    parser.add_argument("--topic", default=None,
                        help="Override topic for all pairs")
    parser.add_argument("--adapter-dir", default=None,
                        help="Path to adapter directory (default: adapters/)")
    parser.add_argument("--results-dir", default=None,
                        help="Path to debate results directory (default: results/debates/)")
    parser.add_argument("--no-rag", action="store_true",
                        help="Disable RAG news context (use topic only)")
    parser.add_argument("--echo-chamber", action="store_true",
                        help="Mode B: same adapter debates itself (echo chamber)")
    parser.add_argument("--campaign", type=int, default=0, metavar="N",
                        help="Mode C: generate N independent comments (coordinated campaign)")

    args = parser.parse_args()

    global ADAPTER_DIR, RESULTS_DIR
    if args.adapter_dir:
        ADAPTER_DIR = Path(args.adapter_dir)
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    if args.community_a and not args.community_b:
        parser.error("--community-a requires --community-b")

    news_cache_dir = BASE_DIR / "results" / "news_cache"

    if args.all_pairs:
        jobs = [(pid, a, b) for pid, (a, b) in PAIRS.items()]
    elif args.pair:
        a, b = PAIRS[args.pair]
        jobs = [(args.pair, a, b)]
    else:
        jobs = [(None, args.community_a, args.community_b)]

    for pair_id, comm_a, comm_b in jobs:
        topic    = args.topic or PAIR_TOPICS.get(pair_id, PAIR_TOPICS[1])
        # Custom topic overrides predefined pair search query
        search_q = topic if args.topic else (PAIR_SEARCH_QUERIES.get(pair_id) if pair_id else None)
        news     = None if args.no_rag else fetch_news_context(topic, news_cache_dir, search_q)

        if args.echo_chamber:
            for comm in [comm_a, comm_b]:
                turns = run_echo_chamber(comm, topic, args.rounds, news=news)
                save_echo_transcript(turns, comm, topic, news=news)
        elif args.campaign:
            for comm in [comm_a, comm_b]:
                comments = run_campaign(comm, topic, n=args.campaign, news=news)
                save_campaign(comments, comm, topic, news=news)
        else:
            turns = run_debate(comm_a, comm_b, topic, args.rounds, news=news)
            save_transcript(turns, topic, comm_a, comm_b, news=news)

    print("\nDone.")


if __name__ == "__main__":
    main()
