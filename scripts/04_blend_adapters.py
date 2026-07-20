#!/usr/bin/env python3
"""
04_blend_adapters.py — LoRA adapter interpolation (ideology dial)

Blends two community LoRA adapters at alpha values across a spectrum.
  alpha=1.0 → pure A  |  alpha=0.0 → pure B  |  alpha=0.5 → midpoint

Blend step (CPU-only): loads raw adapter weights, arithmetic interpolation,
saves blended adapter to disk.

Generation step (GPU): loads base model once, swaps named adapters to verify
ideology shift across the spectrum. Skip with --no-generate.

Usage:
    python3 scripts/04_blend_adapters.py --pair 1
    python3 scripts/04_blend_adapters.py --pair 2 --no-generate
    python3 scripts/04_blend_adapters.py --all-pairs
    python3 scripts/04_blend_adapters.py --community-a climate --community-b climateskeptics

Output: adapters/blended_{A}_{B}_alpha_{X:.2f}/
"""

import os
import shutil
import argparse
from pathlib import Path

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR    = Path(__file__).parent.parent
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

DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

GEN_CONFIG = dict(
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.85,
    repetition_penalty=1.4,
    do_sample=True,
)

# ---------------------------------------------------------------------------
# Weight I/O
# ---------------------------------------------------------------------------

def load_adapter_weights(adapter_path: Path) -> dict:
    sf = adapter_path / "adapter_model.safetensors"
    pt = adapter_path / "adapter_model.bin"
    if sf.exists():
        from safetensors.torch import load_file
        return {k: v.float() for k, v in load_file(str(sf)).items()}
    if pt.exists():
        return {k: v.float() for k, v in torch.load(pt, map_location="cpu").items()}
    raise FileNotFoundError(f"No adapter weights in {adapter_path}")


def save_adapter_weights(weights: dict, output_dir: Path, config_src: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in weights.items()},
                  str(output_dir / "adapter_model.safetensors"))
    except ImportError:
        torch.save(weights, output_dir / "adapter_model.bin")
    shutil.copy(config_src / "adapter_config.json", output_dir / "adapter_config.json")
    # Copy tokenizer files if present (saved by 03_train_adapters.py)
    for fname in ["tokenizer_config.json", "tokenizer.model",
                  "tokenizer.json", "special_tokens_map.json"]:
        src = config_src / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)


# ---------------------------------------------------------------------------
# Blend
# ---------------------------------------------------------------------------

def blend(community_a: str, community_b: str, alphas: list) -> list:
    path_a = ADAPTER_DIR / community_a
    path_b = ADAPTER_DIR / community_b

    for p, name in [(path_a, community_a), (path_b, community_b)]:
        if not p.exists():
            raise FileNotFoundError(f"Adapter not found: {p}. Train with 03_train_adapters.py first.")

    print(f"\nBlending: {community_a} (α=1.0) ↔ {community_b} (α=0.0)")
    weights_a = load_adapter_weights(path_a)
    weights_b = load_adapter_weights(path_b)

    if set(weights_a) != set(weights_b):
        raise ValueError(
            f"Adapter weight keys don't match — were {community_a} and {community_b} "
            "trained with the same LoRA config?"
        )

    output_dirs = []
    for alpha in alphas:
        blended = {k: alpha * weights_a[k] + (1.0 - alpha) * weights_b[k]
                   for k in weights_a}
        out_dir = ADAPTER_DIR / f"blended_{community_a}_{community_b}_alpha_{alpha:.2f}"
        save_adapter_weights(blended, out_dir, path_a)
        print(f"  α={alpha:.2f} → {out_dir.name}")
        output_dirs.append((alpha, out_dir))

    return output_dirs  # list of (alpha, Path)


# ---------------------------------------------------------------------------
# Generation
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


def generate_samples(adapter_pairs: list, topic: str, community_a: str, community_b: str,
                     n_samples: int = 1, gen_dir: Path = None):
    """Load base model once, swap named adapters for each alpha, generate n_samples per alpha."""
    out_dir = (gen_dir or BASE_DIR / "results" / "generated") / f"blend_{community_a}_{community_b}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {n_samples} sample(s) per alpha — topic: {topic!r}")
    print("Loading base model (once)...")
    base_model, tokenizer = build_base_model()

    from peft import PeftModel

    def adapter_key(a: float) -> str:
        return f"alpha_{a:.2f}".replace(".", "_")

    # Load first adapter to create PeftModel wrapper
    first_alpha, first_dir = adapter_pairs[0]
    peft_model = PeftModel.from_pretrained(
        base_model, str(first_dir), adapter_name=adapter_key(first_alpha)
    )

    # Load remaining adapters as named adapters
    for alpha, adapter_dir in adapter_pairs[1:]:
        peft_model.load_adapter(str(adapter_dir), adapter_name=adapter_key(alpha))

    prompt = f"### Post:\n{topic}\n### Comment:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)

    print(f"\n{'─'*60}")
    print(f"  α=1.0 = pure {community_a}   |   α=0.0 = pure {community_b}")
    print(f"{'─'*60}")

    for alpha, _ in adapter_pairs:
        peft_model.set_adapter(adapter_key(alpha))
        peft_model.eval()

        if alpha == 1.0:
            label = f"α=1.0  pure {community_a}"
        elif alpha == 0.0:
            label = f"α=0.0  pure {community_b}"
        else:
            label = f"α={alpha:.2f} blend"

        for sample_idx in range(n_samples):
            with torch.no_grad():
                out = peft_model.generate(
                    **inputs,
                    pad_token_id=tokenizer.eos_token_id,
                    **GEN_CONFIG,
                )
            generated = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            print(f"\n[{label}  s{sample_idx:02d}]\n{generated}")

            if n_samples > 1:
                out_file = out_dir / f"alpha_{alpha:.2f}_s{sample_idx:02d}.txt"
            else:
                out_file = out_dir / f"alpha_{alpha:.2f}.txt"
            out_file.write_text(f"topic: {topic}\nlabel: {label}\n\n{generated}\n")

    print(f"\n{'─'*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Blend community LoRA adapters across an ideology spectrum."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pair", type=int, choices=[1, 2, 3],
                     help="Predefined pair (1=politics/Conservative, 2=worldnews/Sino, 3=climate/climateskeptics)")
    src.add_argument("--all-pairs", action="store_true",
                     help="Run all 3 predefined pairs sequentially")
    src.add_argument("--community-a", metavar="A",
                     help="Custom community A (alpha=1.0 end)")

    parser.add_argument("--community-b", metavar="B",
                        help="Custom community B (alpha=0.0 end) — required with --community-a")
    parser.add_argument("--alphas", default=None,
                        help="Comma-separated alpha values, e.g. 0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--no-generate", action="store_true",
                        help="Blend weights only — skip sample generation (no GPU needed)")
    parser.add_argument("--skip-blend", action="store_true",
                        help="Skip weight blending — use existing blended adapters on disk")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Samples to generate per alpha (default: 1; use 5+ for reliable eval)")
    parser.add_argument("--topic", default=None,
                        help="Override the generation topic for all pairs")
    parser.add_argument("--adapter-dir", default=None,
                        help="Path to adapter directory (default: adapters/)")
    parser.add_argument("--gen-dir", default=None,
                        help="Path to generated outputs directory (default: results/generated/)")

    args = parser.parse_args()

    global ADAPTER_DIR
    if args.adapter_dir:
        ADAPTER_DIR = Path(args.adapter_dir)
    gen_dir = Path(args.gen_dir) if args.gen_dir else None

    if args.community_a and not args.community_b:
        parser.error("--community-a requires --community-b")

    alphas = ([float(a) for a in args.alphas.split(",")]
              if args.alphas else DEFAULT_ALPHAS)

    # Build list of (pair_id_or_None, comm_a, comm_b) to process
    jobs = []
    if args.all_pairs:
        jobs = [(pid, a, b) for pid, (a, b) in PAIRS.items()]
    elif args.pair:
        a, b = PAIRS[args.pair]
        jobs = [(args.pair, a, b)]
    else:
        jobs = [(None, args.community_a, args.community_b)]

    for pair_id, comm_a, comm_b in jobs:
        if args.skip_blend:
            adapter_pairs = []
            for alpha in alphas:
                out_dir = ADAPTER_DIR / f"blended_{comm_a}_{comm_b}_alpha_{alpha:.2f}"
                if not out_dir.exists():
                    raise FileNotFoundError(
                        f"Blended adapter missing: {out_dir}. Run without --skip-blend first."
                    )
                adapter_pairs.append((alpha, out_dir))
            print(f"\nUsing existing blended adapters for {comm_a} ↔ {comm_b}")
        else:
            adapter_pairs = blend(comm_a, comm_b, alphas)

        if not args.no_generate:
            topic = args.topic
            if topic is None:
                topic = PAIR_TOPICS.get(pair_id, PAIR_TOPICS[1])
            generate_samples(adapter_pairs, topic, comm_a, comm_b,
                             n_samples=args.n_samples, gen_dir=gen_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
