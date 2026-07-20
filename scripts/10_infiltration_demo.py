#!/usr/bin/env python3
"""
10_infiltration_demo.py — Cross-community infiltration demo

The astroturfing attack: generate text that sounds authentic to community A
but carries the ideology of community B. The ideology dial enables precision
targeting — not just "liberal" or "conservative" but any blend.

Demonstrates the threat model:
  A state actor can deploy agents that infiltrate r/politics sounding like
  regular users, but are ideologically shifted toward Conservative positions.
  The target community cannot distinguish them from real members.

Generates a side-by-side comparison:
  | Alpha | Voice sounds like | Ideology leans toward | Text                    |
  |-------|-------------------|----------------------|-------------------------|
  |  0.0  | Conservative      | Conservative          | [pure Conservative]     |
  |  0.2  | politics-ish      | Conservative-leaning  | [INFILTRATOR]           |
  |  0.5  | ambiguous         | mixed                 | [blend]                 |
  |  0.8  | politics-ish      | politics-leaning      | [INFILTRATOR]           |
  |  1.0  | politics          | politics              | [pure politics]         |

Output:
  results/infiltration/infiltration_report_{A}_{B}.txt
  results/infiltration/infiltration_report_{A}_{B}.json

Usage:
    python3 scripts/10_infiltration_demo.py --pair 1
    python3 scripts/10_infiltration_demo.py --all-pairs
"""

import json
import argparse
from pathlib import Path

import torch

BASE_DIR    = Path(__file__).parent.parent
ADAPTER_DIR = BASE_DIR / "adapters"
OUTPUT_DIR  = BASE_DIR / "results" / "infiltration"
MODEL_ID    = "mistralai/Mistral-7B-v0.1"

PAIRS = {
    1: ("politics",  "Conservative"),
    2: ("worldnews", "Sino"),
    3: ("climate",   "climateskeptics"),
}

# Topics designed to elicit clear ideological responses
INFILTRATION_TOPICS = {
    1: [
        "Should the US increase border security and limit asylum claims?",
        "What do you think about giving undocumented immigrants voting rights?",
        "Is immigration hurting American workers?",
    ],
    2: [
        "Should Western nations confront China over human rights abuses?",
        "Is the mainstream media too critical of China?",
        "Should the US decouple its economy from China?",
    ],
    3: [
        "Are climate activists doing more harm than good for the movement?",
        "Should we trust climate models given their prediction track record?",
        "Is nuclear energy the solution to climate change?",
    ],
}

# Infiltration alphas: at 0.2 and 0.8, the voice is ambiguous but ideology leans
ALPHAS = [0.0, 0.2, 0.5, 0.8, 1.0]

GEN_CONFIG = dict(
    max_new_tokens=120,
    temperature=0.75,
    top_p=0.9,
    repetition_penalty=1.3,
    do_sample=True,
)


def build_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cap[0] >= 8 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")

    # We'll blend weights manually — load both adapters
    return base, tokenizer


def blend_and_generate(base_model, tokenizer, comm_a: str, comm_b: str,
                        alpha: float, topic: str) -> str:
    from safetensors.torch import load_file, save_file
    import shutil
    import tempfile

    path_a = ADAPTER_DIR / comm_a
    path_b = ADAPTER_DIR / comm_b

    # Load and blend weights
    sf_a = path_a / "adapter_model.safetensors"
    sf_b = path_b / "adapter_model.safetensors"
    w_a = {k: v.float() for k, v in load_file(str(sf_a)).items()}
    w_b = {k: v.float() for k, v in load_file(str(sf_b)).items()}
    blended = {k: alpha * w_a[k] + (1.0 - alpha) * w_b[k] for k in w_a}

    # Write to temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        save_file({k: v.contiguous() for k, v in blended.items()},
                  str(tmppath / "adapter_model.safetensors"))
        shutil.copy(path_a / "adapter_config.json", tmppath / "adapter_config.json")

        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(base_model, str(tmppath), adapter_name="blend")
        peft_model.eval()

        prompt = f"### Post:\n{topic}\n### Comment:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)

        with torch.no_grad():
            out = peft_model.generate(
                **inputs, pad_token_id=tokenizer.eos_token_id, **GEN_CONFIG)

        text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # Clean up PEFT model to free memory for next alpha
        del peft_model
        torch.cuda.empty_cache()

    return text


def infiltration_label(alpha: float, comm_a: str, comm_b: str) -> str:
    if alpha == 0.0:
        return f"Pure {comm_b} (overt)"
    elif alpha <= 0.25:
        return f"INFILTRATOR — sounds like {comm_b}, ideology near {comm_b}"
    elif alpha < 0.5:
        return f"INFILTRATOR — ambiguous voice, {comm_b}-leaning ideology"
    elif alpha == 0.5:
        return "True blend — maximally ambiguous"
    elif alpha < 0.75:
        return f"INFILTRATOR — ambiguous voice, {comm_a}-leaning ideology"
    elif alpha < 1.0:
        return f"INFILTRATOR — sounds like {comm_a}, ideology near {comm_a}"
    else:
        return f"Pure {comm_a} (overt)"


def run_pair(pair_id: int, comm_a: str, comm_b: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    topics = INFILTRATION_TOPICS[pair_id]

    print(f"\n{'='*70}")
    print(f"  INFILTRATION DEMO: r/{comm_b} agents in r/{comm_a}")
    print(f"  Threat: Accounts that sound like r/{comm_a} members but carry")
    print(f"          r/{comm_b} ideology — undetectable by community moderators")
    print(f"{'='*70}")

    base_model, tokenizer = build_model()

    report = {
        "pair": f"{comm_a}_vs_{comm_b}",
        "threat_model": f"r/{comm_b} ideology injected into r/{comm_a} community voice",
        "topics": [],
    }

    for topic in topics:
        print(f"\n  Topic: {topic!r}")
        print(f"  {'─'*66}")
        topic_data = {"topic": topic, "responses": []}

        for alpha in ALPHAS:
            label = infiltration_label(alpha, comm_a, comm_b)
            text = blend_and_generate(base_model, tokenizer, comm_a, comm_b, alpha, topic)
            print(f"\n  [α={alpha:.1f} | {label}]")
            print(f"  {text}")
            topic_data["responses"].append({
                "alpha": alpha,
                "label": label,
                "text": text,
            })

        report["topics"].append(topic_data)

    del base_model
    torch.cuda.empty_cache()

    # Save JSON
    json_path = OUTPUT_DIR / f"infiltration_{comm_a}_{comm_b}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved → {json_path}")

    # Save readable report
    txt_path = OUTPUT_DIR / f"infiltration_{comm_a}_{comm_b}.txt"
    with open(txt_path, 'w') as f:
        f.write(f"INFILTRATION DEMO: r/{comm_b} → r/{comm_a}\n")
        f.write(f"{'='*70}\n")
        f.write(f"Threat model: Accounts that appear to be authentic r/{comm_a} members\n")
        f.write(f"              but carry r/{comm_b} ideological framing.\n\n")
        for topic_data in report["topics"]:
            f.write(f"\nTOPIC: {topic_data['topic']}\n")
            f.write(f"{'─'*70}\n")
            for resp in topic_data["responses"]:
                f.write(f"\n[α={resp['alpha']:.1f}] {resp['label']}\n")
                f.write(f"{resp['text']}\n")
            f.write("\n")
    print(f"  Saved → {txt_path}")


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pair", type=int, choices=[1, 2, 3])
    src.add_argument("--all-pairs", action="store_true")
    args = parser.parse_args()

    jobs = [(pid, a, b) for pid, (a, b) in PAIRS.items()] if args.all_pairs \
           else [(args.pair, *PAIRS[args.pair])]

    for pair_id, comm_a, comm_b in jobs:
        run_pair(pair_id, comm_a, comm_b)

    print("\nDone.")


if __name__ == "__main__":
    main()
