# Results Directory

This directory contains outputs from the evaluation pipeline.

## Directory Structure

```
results/
├── eval/                    # Quantitative metrics
│   ├── plots/              # Visualizations (PNG)
│   ├── *.csv               # Metric data (stance, detection, embedding)
│   └── scale_analysis.txt  # Cost comparison vs IRA 2016
├── infiltration/           # Cross-community infiltration demos
├── debates/                # Multi-agent debate outputs
├── generated/              # Adapter-blended generated text (v1)
└── generated_v2/           # Generated text from v2 adapters
```

## Key Results Files

### Evaluation Metrics

- **`eval/stance_alpha_*.csv`** — NLI stance scores across blending spectrum (α ∈ [0, 1])
- **`eval/detection_scores.csv`** — AI detection bypass analysis (chatgpt-detector-roberta)
- **`eval/embedding_similarity_*.csv`** — Sentence embedding distances between community outputs
- **`eval/scale_analysis.txt`** — Cost/throughput comparison (this system vs IRA 2016)

### Visualizations

- **`eval/plots/ideology_dial.png`** — Stance score trajectory across α values
- **`eval/plots/detection_bypass.png`** — AI detection probability by community/condition
- **`eval/plots/tsne_*.png`** — t-SNE embeddings showing community separation
- **`eval/plots/scale_cost.png`** — Cost comparison chart

### Generated Content

- **`infiltration/`** — Infiltration attack demos (α=0.8 blends)
- **`debates/`** — Multi-turn debates between community adapters
- **`generated/`** — Blended outputs at 5 α values (0.0, 0.25, 0.5, 0.75, 1.0)

## Summary Statistics

From `eval/scale_analysis.txt`:

| Metric | Value |
|--------|-------|
| **Training cost** | $0.63 (6 adapters, 1.8 hrs) |
| **Daily output** | 43,200 comments/day (1 RTX 4090) |
| **Cost per comment** | $0.00019 |
| **vs IRA 2016** | 6,122× cheaper ($8.40/day vs $41,667/day) |

## Reproducing These Results

Run the evaluation pipeline:

```bash
# Generate blended outputs
python scripts/04_blend_adapters.py --all-pairs

# Evaluate stance
python scripts/06_eval_stance.py

# Evaluate AI detection
python scripts/08_eval_detector.py

# Generate t-SNE plots
python scripts/09b_embedding_fingerprint.py

# Infiltration demos
python scripts/10_infiltration_demo.py

# Scale analysis
python scripts/11_scale_analysis.py
```

All outputs written to this directory.

## Notes

- Raw data (Reddit posts/comments) excluded due to size — see main README for reproduction
- Trained adapters excluded — see `adapters/` directory (gitignored)
- Results shown are from **v1 adapters** (r=16) unless filename contains `_v2`
