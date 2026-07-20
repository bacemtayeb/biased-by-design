#!/usr/bin/env python3
"""
03_train_adapters.py  — Experiment 1 only (community LoRA adapters)
Train one community-conditioned LoRA adapter on Mistral-7B-v0.1 BASE.

Key fixes over trainer_0.py:
  - Response-only loss via DataCollatorForCompletionOnlyLM (response_template only,
    avoids known TRL bug when instruction_template is used with BOS token)
  - 4-bit QLoRA (BitsAndBytes nf4, double quant)
  - LoRA r=16 targeting all 4 attention projections
  - Eval split + EarlyStoppingCallback
  - Gradient clipping 1.0
  - OOM guard: auto-reduces batch size if <15GB VRAM detected
  - Checkpoint resume: resumes from latest checkpoint if one exists

Usage (single community):
    python3 scripts/03_train_adapters.py --community politics

Usage (SLURM array — all 6 in parallel):
    sbatch scripts/slurm_train_array.sh

Output: adapters/{community}/
"""

import os
import json
import argparse
import subprocess
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data/formatted"
ADAPTER_DIR = BASE_DIR / "adapters"
MODEL_ID    = "mistralai/Mistral-7B-v0.1"

COMMUNITIES = [
    "politics",        # SLURM array 0
    "Conservative",    # SLURM array 1
    "worldnews",       # SLURM array 2
    "Sino",            # SLURM array 3
    "climate",         # SLURM array 4
    "climateskeptics", # SLURM array 5
]

# Response template for DataCollatorForCompletionOnlyLM.
# IMPORTANT: pass token IDs computed in context, not the raw string.
# Mistral tokenizer encodes "###" differently at string start (token 774)
# vs after a newline (token 27332). Using the string directly causes
# template search to fail → all labels = -100 → loss = 0 → no training.
# Fix: encode "\n### Comment:\n" without BOS, drop the leading space token [0].
RESPONSE_TEMPLATE_STR = "\n### Comment:\n"

MAX_SEQ_LENGTH = 512
EVAL_RATIO     = 0.1

# Default training config — may be downgraded by OOM guard
BATCH_SIZE    = 4
GRAD_ACCUM    = 4   # effective batch = 16

# Auto-detect precision: bf16 on Ampere+ (compute >= 8.0: A100, H100, RTX 3090+, RTX 4090)
#                        fp16 on Volta/Turing (V100, T4, RTX 20/30 series compute < 8.0)
_cap     = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
USE_BF16 = _cap[0] >= 8
USE_FP16 = not USE_BF16

# ---------------------------------------------------------------------------
# OOM guard
# ---------------------------------------------------------------------------

def get_gpu_vram_mb() -> int:
    """Return VRAM in MB for GPU 0. Returns 0 if nvidia-smi unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        return int(out.stdout.strip().split("\n")[0])
    except Exception:
        return 0


def safe_batch_config():
    """
    Scale batch size to available VRAM.
    Down: <15GB  → batch=1, accum=16  (tight, e.g. RTX 5070)
    Base: 15-39GB → batch=4, accum=4  (V100 16GB, RTX 4090 24GB)
    Up:   40-79GB → batch=8, accum=2  (A100 40GB)
    Max:  ≥80GB  → batch=16, accum=1 (H100/H200)
    """
    vram = get_gpu_vram_mb()
    print(f"  GPU VRAM: {vram:,} MB")
    if vram > 0 and vram < 15_000:
        print("  Config: batch=1, accum=16 (low VRAM)")
        return 1, 16
    elif vram >= 80_000:
        print("  Config: batch=16, accum=1 (H100/H200 — max throughput)")
        return 16, 1
    elif vram >= 40_000:
        print("  Config: batch=8, accum=2 (A100/large GPU)")
        return 8, 2
    else:
        print("  Config: batch=4, accum=4 (standard)")
        return BATCH_SIZE, GRAD_ACCUM

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

import re as _re
_URL_RE = _re.compile(r'https?://\S+')


def format_example(record: dict) -> str | None:
    msgs = record["messages"]
    user_content      = msgs[0]["content"]
    assistant_content = msgs[1]["content"]
    # strip URLs — noisy for style learning
    assistant_content = _URL_RE.sub('', assistant_content).strip()
    # skip very short comments
    if len(assistant_content.split()) < 10:
        return None
    return (
        f"### Post:\n{user_content}\n"
        f"### Comment:\n{assistant_content}"
    )


def load_community_dataset(community: str):
    path = DATA_DIR / f"{community}.jsonl"
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                text = format_example(json.loads(line))
                if text is not None:
                    texts.append(text)

    n_eval   = max(50, int(len(texts) * EVAL_RATIO))
    train_ds = Dataset.from_dict({"text": texts[:-n_eval]})
    eval_ds  = Dataset.from_dict({"text": texts[-n_eval:]})

    print(f"  Train: {len(train_ds):,}  |  Eval: {len(eval_ds):,}")
    return train_ds, eval_ds

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Flash Attention 2: 2-4x faster on Ampere+ (H100, A100, RTX 4090)
    # Skipped on Volta (V100) — not supported, falls back automatically
    fa2_kwargs = {}
    if USE_BF16:
        try:
            import flash_attn  # noqa: F401
            fa2_kwargs["attn_implementation"] = "flash_attention_2"
            print("  Flash Attention 2: enabled")
        except ImportError:
            print("  Flash Attention 2: not installed — using default attention")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        **fa2_kwargs,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

# ---------------------------------------------------------------------------
# Checkpoint resume helper
# ---------------------------------------------------------------------------

def find_last_checkpoint(output_dir: Path):
    """Return path to latest checkpoint dir, or None."""
    checkpoints = sorted(output_dir.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1]))
    if checkpoints:
        ckpt = checkpoints[-1]
        print(f"  Resuming from checkpoint: {ckpt}")
        return str(ckpt)
    return None

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(community: str, reset: bool = False):
    print(f"\n{'='*60}")
    print(f"  Community : {community}")
    print(f"  Experiment: 1 — community LoRA adapters")
    print(f"{'='*60}")

    batch_size, grad_accum = safe_batch_config()
    train_ds, eval_ds      = load_community_dataset(community)
    model, tokenizer       = build_model_and_tokenizer()

    # Compute response template token IDs in context (not standalone).
    # Drop the leading space/BOS artifact [0] from the encoding.
    resp_ids = tokenizer.encode(RESPONSE_TEMPLATE_STR, add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=resp_ids,
        tokenizer=tokenizer,
        mlm=False,
    )

    output_dir = ADAPTER_DIR / community
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_ckpt = find_last_checkpoint(output_dir) if not reset else None

    args = SFTConfig(
        output_dir=str(output_dir),
        # Training
        num_train_epochs=2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        # Precision — auto-detected at startup (fp16 for V100, bf16 for Ampere+)
        fp16=USE_FP16,
        bf16=USE_BF16,
        # Eval & saving
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_steps=25,
        report_to="none",
        # SFT
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n  Adapter saved → {output_dir}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--community", default=None, choices=COMMUNITIES)
    parser.add_argument("--reset", action="store_true",
                        help="Ignore existing checkpoints, train from scratch")
    args = parser.parse_args()

    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_id is not None:
        community = COMMUNITIES[int(slurm_id)]
        print(f"SLURM array task {slurm_id} → {community}")
    elif args.community:
        community = args.community
    else:
        parser.error("Provide --community or set SLURM_ARRAY_TASK_ID")

    train(community, reset=args.reset)


if __name__ == "__main__":
    main()
