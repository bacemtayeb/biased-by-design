#!/usr/bin/env python3
"""
02_format_dataset.py
Format raw Arctic Shift JSONL dumps into silent-conditioned training pairs.

Input:  dialogue-2/cleaned/data/r_{community}_{posts,comments}.jsonl
Output: data/formatted/{community}.jsonl

Training format (silent — no community label):
    <|user|>
    {attribute_tokens}
    {post_title}[. {post_body}]
    <|assistant|>
    {comment_body}

Filters:
  - post score >= MIN_POST_SCORE
  - comment score >= MIN_COMMENT_SCORE
  - comment body len >= MIN_COMMENT_LEN chars
  - top-level comments only (parent_id starts with t3_)
  - no [deleted] / [removed]
  - top comment per post (by score)
  - up to TARGET_PAIRS per community (highest post score first)
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).parent.parent
RAW_DIR       = BASE_DIR / "dialogue-2/cleaned/data"
OUT_DIR       = BASE_DIR / "data/formatted"
TARGET_PAIRS  = 5000
MAX_PER_POST  = 5         # max comments taken per post
MIN_POST_SCORE    = 5
MIN_COMMENT_SCORE = 5
MIN_COMMENT_LEN   = 50   # characters

COMMUNITIES = [
    "politics",
    "Conservative",
    "worldnews",
    "Sino",
    "climate",
    "climateskeptics",
]

# ---------------------------------------------------------------------------
# Attribute token detection
# ---------------------------------------------------------------------------

def detect_attributes(text: str) -> list:
    tokens = []
    words  = text.split()
    n      = len(words)

    # Length
    if n < 30:
        tokens.append("[SHORT]")
    elif n > 100:
        tokens.append("[LONG]")

    # Factual: numbers, percentages, citation language
    if re.search(r'\d+%|\d+\.\d+|\b\d{4}\b', text) or \
       re.search(r'\b(according to|study|research|data|percent|statistics|source|report|cited)\b', text, re.I):
        tokens.append("[FACTUAL]")

    # Sarcastic
    if '/s' in text or \
       re.search(r'\b(yeah right|oh great|totally|obviously|shocking|wow so|sure sure)\b', text, re.I):
        tokens.append("[SARCASTIC]")

    # Aggressive
    if re.search(r'\b(idiot|stupid|liar|pathetic|disgusting|moron|ridiculous|nonsense|garbage|trash|wrong|lies?)\b',
                 text, re.I) or text.count('!') >= 3:
        tokens.append("[AGGRESSIVE]")

    # Emotional (but not already flagged aggressive)
    if "[AGGRESSIVE]" not in tokens and (
       re.search(r'\b(outrage|horrifying|devastating|incredible|terrible|awful|wonderful|love|hate|fear|angry|sad|scared|furious)\b',
                 text, re.I) or text.count('!') >= 1):
        tokens.append("[EMOTIONAL]")

    # Stealth: hedged/balanced framing, no strong sentiment
    if "[AGGRESSIVE]" not in tokens and "[EMOTIONAL]" not in tokens and \
       re.search(r'\b(however|on the other hand|while|although|nuanced|complex|it depends|fair point|both)\b',
                 text, re.I):
        tokens.append("[STEALTH]")

    # Fallback
    if not tokens:
        tokens.append("[FACTUAL]" if n > 50 else "[EMOTIONAL]")

    return tokens

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def is_deleted(text: str) -> bool:
    return not text or text.strip() in ("[deleted]", "[removed]", "")


def _iter_jsonl(paths):
    """Yield parsed JSON objects from one or more JSONL files."""
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _get_paths(base: Path) -> list:
    """Return base path + extension path if it exists."""
    stem = base.stem          # e.g. r_politics_posts
    ext_path = base.parent / f"{stem}_extensions.jsonl"
    paths = [base]
    if ext_path.exists():
        paths.append(ext_path)
    return paths


def load_posts(path: Path) -> dict:
    """post_id -> post dict, filtered. Reads base + extension file if present."""
    posts = {}
    for p in _iter_jsonl(_get_paths(path)):
        if p.get("score", 0) < MIN_POST_SCORE:
            continue
        title = p.get("title", "").strip()
        if is_deleted(title) or len(title) < 10:
            continue
        posts[p["id"]] = p
    return posts


def load_comments(path: Path) -> dict:
    """post_id -> list[comment], top-level only, filtered, sorted score desc. Reads base + extension."""
    comments = defaultdict(list)
    for c in _iter_jsonl(_get_paths(path)):
        if not c.get("parent_id", "").startswith("t3_"):
            continue
        if c.get("score", 0) < MIN_COMMENT_SCORE:
            continue
        body = c.get("body", "").strip()
        if is_deleted(body) or len(body) < MIN_COMMENT_LEN:
            continue
        post_id = c.get("link_id", "").replace("t3_", "")
        comments[post_id].append(c)

    for pid in comments:
        comments[pid].sort(key=lambda x: x.get("score", 0), reverse=True)
    return comments


def format_pair(post: dict, comment: dict) -> dict:
    title   = post.get("title", "").strip()
    body    = post.get("selftext", "").strip()
    c_body  = comment.get("body", "").strip()

    attributes = detect_attributes(c_body)
    attr_str   = "".join(attributes)

    # user content: attribute tokens, then post
    user_content = f"{attr_str}\n{title}"
    if body and not is_deleted(body):
        user_content += f". {body}"

    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": c_body},
        ],
        "metadata": {
            "subreddit":     post.get("subreddit", ""),
            "post_id":       post["id"],
            "comment_id":    comment["id"],
            "post_score":    post.get("score", 0),
            "comment_score": comment.get("score", 0),
            "attributes":    attributes,
        },
    }

# ---------------------------------------------------------------------------
# Per-community pipeline
# ---------------------------------------------------------------------------

def process_community(community: str, target: int):
    posts_path    = RAW_DIR / f"r_{community}_posts.jsonl"
    comments_path = RAW_DIR / f"r_{community}_comments.jsonl"

    if not posts_path.exists() or not comments_path.exists():
        print(f"[{community}] MISSING files — skipping")
        return

    print(f"[{community}] Loading posts ...")
    posts = load_posts(posts_path)
    print(f"[{community}]   {len(posts):,} posts passed filters")

    print(f"[{community}] Loading comments ...")
    comments = load_comments(comments_path)
    print(f"[{community}]   {len(comments):,} posts have qualifying top-level comments")

    # Sort posts by score desc, take top MAX_PER_POST comments per post, stop at target
    pairs = []
    for post_id, post in sorted(posts.items(), key=lambda x: x[1].get("score", 0), reverse=True):
        if post_id not in comments:
            continue
        for comment in comments[post_id][:MAX_PER_POST]:
            pairs.append(format_pair(post, comment))
            if len(pairs) >= target:
                break
        if len(pairs) >= target:
            break

    if len(pairs) < target:
        print(f"[{community}]   WARNING: only {len(pairs):,} pairs available (target was {target:,})")

    print(f"[{community}]   {len(pairs):,} pairs formed")

    # Attribute distribution
    attr_counts = defaultdict(int)
    for p in pairs:
        for a in p["metadata"]["attributes"]:
            attr_counts[a] += 1
    print(f"[{community}]   attributes: { dict(sorted(attr_counts.items())) }")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{community}.jsonl"
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"[{community}]   Saved → {out_path}\n")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global MAX_PER_POST
    parser = argparse.ArgumentParser(description="Format Reddit data for silent adapter training.")
    parser.add_argument("--community", default=None,
                        help="Single community to process (default: all)")
    parser.add_argument("--target", type=int, default=TARGET_PAIRS,
                        help=f"Target pairs per community (default: {TARGET_PAIRS})")
    parser.add_argument("--max-per-post", type=int, default=MAX_PER_POST,
                        help=f"Max comments taken per post (default: {MAX_PER_POST})")
    args = parser.parse_args()

    MAX_PER_POST = args.max_per_post
    communities = [args.community] if args.community else COMMUNITIES
    for c in communities:
        process_community(c, args.target)

    print("Done.")


if __name__ == "__main__":
    main()
