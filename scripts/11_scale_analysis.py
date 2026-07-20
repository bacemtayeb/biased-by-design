#!/usr/bin/env python3
"""
11_scale_analysis.py — Scale and cost analysis for astroturfing operations

Computes:
  - Training cost per community adapter
  - Cost per generated comment at scale
  - Comparison to human-operated influence campaigns
  - Projection to state-actor scale

Output:
  results/eval/scale_analysis.txt
  results/eval/plots/scale_cost.png
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
EVAL_DIR = BASE_DIR / "results" / "eval"
LOG_DIR  = BASE_DIR / "logs"

# Empirical measurements from this run
TRAINING_STATS = {
    "gpu":              "NVIDIA RTX 4090 (24GB)",
    "gpu_cost_per_hr":  0.35,        # vast.ai spot price USD/hr
    "communities":      6,
    "samples_per_comm": 4300,        # ~average after filtering
    "train_time_per_comm_min": 18,   # empirical from retrain.log
    "total_train_time_min":    108,
}

GENERATION_STATS = {
    "samples_per_alpha": 5,
    "alphas_per_pair":   5,
    "pairs":             3,
    "time_per_sample_s": 2.0,        # empirical from pipeline
    "total_gen_samples": 75,         # 5 × 5 × 3
}

# IRA (Internet Research Agency) 2016 for comparison
IRA_STATS = {
    "employees":        1000,
    "posts_per_day":    35000,
    "cost_per_month_usd": 1_250_000,
    "source": "Senate Intelligence Committee report, 2019",
}


def compute_costs():
    train_hr  = TRAINING_STATS["total_train_time_min"] / 60
    train_cost = train_hr * TRAINING_STATS["gpu_cost_per_hr"]

    # Generation throughput
    sec_per_comment = GENERATION_STATS["time_per_sample_s"]
    comments_per_hr = 3600 / sec_per_comment
    comments_per_day = comments_per_hr * 24
    cost_per_day = TRAINING_STATS["gpu_cost_per_hr"] * 24

    # Scale projections
    target_comments = 35000  # match IRA daily output
    gpu_hours_needed = target_comments / comments_per_hr
    gpu_cost_to_match_ira = gpu_hours_needed * TRAINING_STATS["gpu_cost_per_hr"]

    return {
        "train_hours":      round(train_hr, 2),
        "train_cost_usd":   round(train_cost, 2),
        "comments_per_hour": int(comments_per_hr),
        "comments_per_day": int(comments_per_day),
        "cost_per_day_usd": round(cost_per_day, 2),
        "gpu_hours_to_match_ira": round(gpu_hours_needed, 1),
        "cost_to_match_ira_daily": round(gpu_cost_to_match_ira, 2),
        "cost_ratio_vs_ira": round(IRA_STATS["cost_per_month_usd"] / (gpu_cost_to_match_ira * 30), 0),
    }


def write_report(costs: dict):
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║         ASTROTURFING SCALE ANALYSIS: AI vs Human Operations     ║
╚══════════════════════════════════════════════════════════════════╝

TRAINING COST (one-time setup)
─────────────────────────────
  Hardware:          {TRAINING_STATS['gpu']}
  Cloud cost:        ~${TRAINING_STATS['gpu_cost_per_hr']:.2f}/hr (vast.ai spot)
  Communities:       {TRAINING_STATS['communities']} adapters
  Training time:     {costs['train_hours']:.1f} hours total
  TOTAL TRAIN COST:  ${costs['train_cost_usd']:.2f}

GENERATION THROUGHPUT
─────────────────────
  Speed:             {costs['comments_per_hour']:,} targeted comments/hour
  Daily capacity:    {costs['comments_per_day']:,} comments/day (1 GPU)
  Daily GPU cost:    ${costs['cost_per_day_usd']:.2f}
  Cost per comment:  ${costs['cost_per_day_usd'] / costs['comments_per_day']:.5f}

COMPARISON: This System vs Internet Research Agency (IRA, 2016)
────────────────────────────────────────────────────────────────
  IRA daily output:  {IRA_STATS['posts_per_day']:,} posts/day
  IRA monthly cost:  ${IRA_STATS['cost_per_month_usd']:,} (1,000 employees)
  IRA source:        {IRA_STATS['source']}

  Our system output: {costs['comments_per_day']:,} targeted comments/day (1 GPU)
  GPU-hrs to match:  {costs['gpu_hours_to_match_ira']} hours (~{int(costs['gpu_hours_to_match_ira']//24)} days continuous)
  Cost to match IRA: ${costs['cost_to_match_ira_daily']:.2f}/day vs ${IRA_STATS['cost_per_month_usd']/30:,.0f}/day

  COST REDUCTION:    {int(costs['cost_ratio_vs_ira']):,}x cheaper than human-operated campaign

KEY DIFFERENTIATORS vs IRA-style operations
────────────────────────────────────────────
  1. Precision targeting:   ideology dial enables sub-community positioning
                            (not just "liberal" or "conservative" — any blend)
  2. Authenticity:          community adapters match linguistic fingerprints
                            of real users — harder to detect than generic LLM
  3. Scalability:           marginal cost ~$0 per additional comment
  4. Deniability:           no human operators to expose/arrest/deport
  5. Speed:                 rapid deployment for breaking news/events

THREAT SEVERITY ASSESSMENT
───────────────────────────
  Low-resource state actor (e.g. ~$10K budget):
    → Train adapters on 50+ communities ($50)
    → Generate 1M targeted comments/month (~$250 GPU cost)
    → Cover every major political subreddit with authentic-seeming voices

  Advanced state actor:
    → Real-time adapter fine-tuning on trending topics
    → Continuous ideological repositioning during events
    → Cross-platform deployment (Reddit, Twitter, Facebook)
"""
    out_path = EVAL_DIR / "scale_analysis.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(report)
    print(f"  Saved → {out_path}")
    return report


def plot_scale(costs: dict, out_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Cost comparison bar
    ax = axes[0]
    systems = ['IRA\n(human)', 'This system\n(1 GPU)']
    daily_costs = [IRA_STATS['cost_per_month_usd'] / 30, costs['cost_per_day_usd']]
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(systems, daily_costs, color=colors, alpha=0.8, width=0.5)
    ax.set_ylabel('Daily operational cost (USD)', fontsize=11)
    ax.set_title('Cost to Generate 35,000+ Targeted\nComments per Day', fontsize=11)
    ax.set_yscale('log')
    for bar, cost in zip(bars, daily_costs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                f'${cost:,.0f}/day', ha='center', va='bottom', fontsize=11, fontweight='bold')
    reduction = int(costs['cost_ratio_vs_ira'])
    ax.text(0.5, 0.5, f'{reduction:,}×\ncheaper', transform=ax.transAxes,
            ha='center', va='center', fontsize=18, fontweight='bold',
            color='#27ae60', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: Throughput at scale (GPU count)
    ax2 = axes[1]
    n_gpus = [1, 4, 16, 64, 256]
    comments_per_day = [costs['comments_per_day'] * n for n in n_gpus]
    gpu_costs = [costs['cost_per_day_usd'] * n for n in n_gpus]

    color = '#3498db'
    ax2.plot(n_gpus, comments_per_day, 'o-', color=color, linewidth=2, markersize=8)
    ax2.axhline(IRA_STATS['posts_per_day'], color='#e74c3c', linestyle='--',
                linewidth=1.5, label=f'IRA daily output ({IRA_STATS["posts_per_day"]:,})')
    ax2.set_xlabel('Number of GPUs', fontsize=11)
    ax2.set_ylabel('Comments per day', fontsize=11)
    ax2.set_title('Scale: Comments/Day vs GPU Count\n(RTX 4090, $0.35/hr each)', fontsize=11)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Annotate cost per GPU tier
    for n, c, cost in zip(n_gpus, comments_per_day, gpu_costs):
        ax2.annotate(f'${cost:.0f}/day', (n, c),
                     textcoords='offset points', xytext=(5, 5), fontsize=8, color='gray')

    fig.suptitle('AI-Powered Astroturfing: Scale and Cost Analysis', fontsize=13, fontweight='bold')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main():
    costs = compute_costs()
    write_report(costs)

    import sys
    if '--no-plots' not in sys.argv:
        plot_scale(costs, EVAL_DIR / "plots" / "scale_cost.png")

    # Save JSON for report generation
    (EVAL_DIR / "scale_costs.json").write_text(json.dumps(costs, indent=2))
    print("\nDone.")


if __name__ == "__main__":
    main()
