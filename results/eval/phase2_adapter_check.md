# Phase 2 — Adapter Sanity Check Results
> Date: 2026-05-27 ~19:30 UTC
> Script: `scripts/check_adapters.py`
> Run on: vast.ai RTX 4090

## Pairwise Cosine Distance Matrix

Higher = more diverged. All values well above 0.05 "too similar" threshold.

```
                    politics  Conservative  worldnews   Sino    climate  climateskeptics
politics               0.558         0.681      0.668  0.685      0.652            0.640
Conservative           0.681         0.607      0.669  0.697      0.654            0.653
worldnews              0.668         0.669      0.566  0.676      0.640            0.630
Sino                   0.685         0.697      0.676  0.578      0.669            0.676
climate                0.652         0.654      0.640  0.669      0.575            0.636
climateskeptics        0.640         0.653      0.630  0.676      0.636            0.481
```

## Interpretation

- Cross-community distances: 0.630–0.697 — adapters clearly diverged
- No pairs flagged as too similar (threshold < 0.05)
- Diagonal non-zero = expected artifact (mean of unit vectors ≠ unit vector)

### Noteworthy pairs
| Pair | Distance | Notes |
|------|----------|-------|
| Sino ↔ Conservative | 0.697 | Most diverged — geopolitical vs. domestic US |
| politics ↔ Conservative | 0.681 | Strong divergence despite same domain |
| worldnews ↔ climateskeptics | 0.630 | Lowest cross-pair — closest outputs |

### Per-community internal consistency (diagonal)
| Community | Self-dist | Interpretation |
|-----------|-----------|----------------|
| climateskeptics | 0.481 | Most consistent style across topics |
| climate | 0.575 | Consistent |
| Sino | 0.578 | Consistent |
| worldnews | 0.566 | Consistent |
| politics | 0.558 | Consistent |
| Conservative | 0.607 | Most varied across prompts |

## Test Prompts Used

1. "Should social media companies be required to moderate political speech?"
2. "Is China's rise a threat to global stability?"
3. "Should governments impose carbon taxes to fight climate change?"

## Verdict

**All 6 adapters verified. Divergence confirmed. Proceed to Phase 3.**
