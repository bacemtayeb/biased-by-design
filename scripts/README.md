# Scripts Directory

Core pipeline for training, blending, and evaluating community-adapted language models.

## Pipeline Overview

```
┌─────────────────┐
│ 02_format       │  Convert raw Reddit data to training format
│ _dataset.py     │  (silent conditioning, no community labels)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 03_train        │  LoRA fine-tuning per community
│ _adapters.py    │  (v1: r=16, 4 modules, 2 epochs)
│                 │
│ 03_train        │  v2: r=32, 7 modules, 4 epochs
│ _adapters_v2.py │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 04_blend        │  Adapter interpolation (ideology dial)
│ _adapters.py    │  α ∈ [0.0, 0.25, 0.5, 0.75, 1.0]
└────────┬────────┘
         │
         ├─────────────────────┬──────────────────┬──────────────────┐
         ▼                     ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐
│ 06_eval      │  │ 08_eval          │  │ 09b_embedding    │  │ 10_          │
│ _stance.py   │  │ _detector.py     │  │ _fingerprint.py  │  │ infiltration │
│              │  │                  │  │                  │  │ _demo.py     │
│ NLI stance   │  │ AI detection     │  │ Sentence embed   │  │              │
│ measurement  │  │ bypass analysis  │  │ + t-SNE plots    │  │ α=0.8 attack │
└──────────────┘  └──────────────────┘  └──────────────────┘  └──────────────┘
                           │
                           ▼
                  ┌──────────────┐
                  │ 11_scale     │
                  │ _analysis.py │
                  │              │
                  │ Cost vs IRA  │
                  └──────────────┘
```

## Script Descriptions

### Data Preparation

**`02_format_dataset.py`**
- Converts raw Reddit data (posts + comments) to training format
- Implements **silent conditioning** (no community label in prompt)
- Output: `data/formatted/{community}.jsonl`

### Training

**`03_train_adapters.py`** (v1)
- LoRA fine-tuning using TRL `SFTTrainer`
- Config: r=16, α=32, 4 attention modules (q/k/v/o_proj)
- 2 epochs, 5k samples per community
- Output: `adapters/{community}/`

**`03_train_adapters_v2.py`** (v2)
- Higher-capacity version: r=32, α=64, 7 modules
- 4 epochs, 10k samples (politics/worldnews/Sino), 5k others
- Output: `adapters_v2/{community}/`

### Blending & Generation

**`04_blend_adapters.py`**
- Linear interpolation of LoRA weights
- Generates text at 5 blend ratios: α = 0.0, 0.25, 0.5, 0.75, 1.0
- Output: `results/generated/blend_{A}_{B}/alpha_*.txt`

**`05_debate_orchestrator.py`**
- Multi-agent debate generation
- Modes: cross-community, echo chamber, coordinated campaign
- Output: `results/debates/`

### Evaluation

**`06_eval_stance.py`**
- NLI-based stance measurement using `facebook/bart-large-mnli`
- Tracks ideology shift across α spectrum
- Output: `results/eval/stance_alpha_*.csv`, `plots/ideology_dial.png`

**`08_eval_detector.py`**
- AI detection bypass analysis
- Model: `chatgpt-detector-roberta` (HuggingFace)
- Compares: real Reddit, vanilla LLM, community-adapted
- Output: `results/eval/detection_scores.csv`, `plots/detection_bypass.png`

**`09b_embedding_fingerprint.py`**
- Sentence embedding analysis using `sentence-transformers`
- Cosine distance matrix + t-SNE visualization
- Output: `results/eval/embedding_similarity_*.csv`, `plots/tsne_*.png`

**`10_infiltration_demo.py`**
- Generates α=0.8 "infiltrator" text
- Sounds like Community A, carries Community B ideology
- Output: `results/infiltration/{pair}_{topic}.txt`

**`11_scale_analysis.py`**
- Cost/throughput comparison vs IRA 2016
- Calculates: training cost, generation speed, cost-per-comment
- Output: `results/eval/scale_analysis.txt`, `plots/scale_cost.png`

## Usage Examples

### Full Pipeline (v1 adapters)

```bash
# 1. Format training data
python scripts/02_format_dataset.py --community politics

# 2. Train adapter
python scripts/03_train_adapters.py --community politics

# 3. Generate blended outputs (all pairs)
python scripts/04_blend_adapters.py --all-pairs

# 4. Run all evals
python scripts/06_eval_stance.py
python scripts/08_eval_detector.py
python scripts/09b_embedding_fingerprint.py
python scripts/10_infiltration_demo.py
python scripts/11_scale_analysis.py
```

### Training All 6 Communities

```bash
# Sequential (local)
for comm in politics Conservative worldnews Sino climate climateskeptics; do
    python scripts/03_train_adapters.py --community $comm
done

# Parallel (multiple cloud GPU instances, one community each)
python scripts/03_train_adapters.py --community $COMMUNITY
```

### Evaluation Only (Using Existing Adapters)

```bash
# Assumes adapters/{community}/ already exist
python scripts/04_blend_adapters.py --pair 1  # politics ↔ Conservative
python scripts/06_eval_stance.py
python scripts/08_eval_detector.py
```

## Configuration

All scripts support command-line args. Common options:

```bash
--community {politics,Conservative,worldnews,Sino,climate,climateskeptics}
--pair {1,2,3}  # 1=pol/Con, 2=wn/Sino, 3=clim/skeptic
--all-pairs     # Run all 3 pairs
--output-dir DIR
```

See `python scripts/<script>.py --help` for full options.

## Dependencies

Requires packages from `requirements.txt`:
- `transformers`, `peft`, `trl`, `accelerate`, `bitsandbytes` (training)
- `sentence-transformers`, `scikit-learn` (evaluation)
- `matplotlib`, `seaborn` (plots)

## Hardware Requirements

- **Minimum:** 16GB GPU (RTX 5070, Tesla V100) — auto-reduces batch size
- **Recommended:** 24GB GPU (RTX 4090, A5000) — optimal batch size
- **Training time:** ~26 min/community (RTX 4090), ~1.5 hrs (V100)

All scripts use 4-bit quantization (QLoRA) to fit in consumer GPUs.

## Outputs

All results written to `../results/`:
- `eval/` — CSV metrics, plots
- `infiltration/` — infiltration demos
- `debates/` — multi-agent outputs
- `generated/` — blended text samples

See `../results/README.md` for details.
