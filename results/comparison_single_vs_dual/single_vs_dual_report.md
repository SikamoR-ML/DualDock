# Single vs Dual Comparative Analysis

- Single run: `/Users/sikamor/Downloads/DualDock_results/single_b1_13782_20260220_155823`
- Dual run: `/Users/sikamor/Downloads/DualDock_results/runs/dual_b1_vs_b2_overnight_20260221_031257`
- Output dir: `/Users/sikamor/Downloads/DualDock_results/comparison_single_vs_dual`
- Single post-sampling available: yes

## 1) Descriptive Stats

| Dataset | N | Mean | Median | Std | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Single base total_reward | 470 | 0.1467 | 0.1075 | 0.1174 | 0.0251 | 0.6757 |
| Single post total_reward | 1200 | 0.1637 | 0.1196 | 0.1303 | 0.0233 | 0.8439 |
| Dual base total_reward | 95 | 0.0101 | 0.0000 | 0.0204 | 0.0000 | 0.0806 |
| Single base targetA | 470 | 0.1467 | 0.1075 | 0.1174 | 0.0251 | 0.6757 |
| Single post targetA | 1200 | 0.1637 | 0.1196 | 0.1303 | 0.0233 | 0.8439 |
| Dual base targetA | 95 | 0.1284 | 0.1089 | 0.0821 | 0.0292 | 0.4294 |
| Dual base offTargetB | 95 | 0.1490 | 0.1179 | 0.1054 | 0.0251 | 0.5531 |

## 2) Selectivity in Dual Run

- Positive selectivity margin (targetA - offTargetB > 0): 33/95 (34.7%)
- Dual margin mean: -0.0206, median: -0.0097

## 3) Statistical Comparison (Single base vs Dual base)

- TargetA Mann-Whitney (two-sided): U=23140.0, p=5.746e-01
- TargetA Cohen's d (single - dual): 0.163
- TargetA Cliff's delta (single vs dual): 0.037
- TargetA mean diff (single - dual): 0.0182 (95% CI -0.0021..0.0379)
- Total-reward Mann-Whitney (two-sided): U=43679.0, p=4.517e-49
- Total-reward Cohen's d (single - dual): 1.272
- Total-reward Cliff's delta (single vs dual): 0.957
- Total-reward mean diff (single - dual): 0.1366 (95% CI 0.1254..0.1478)

## 4) Practical Thresholds (`best_total_reward`)

- Single base >= 0.2: 98 (20.9%)
- Dual base >= 0.2: 0 (0.0%)
- Single post >= 0.2: 321 (26.8%)

## 5) Top-K Comparison (`best_total_reward`)

- Top-10 mean: single_base=0.6017, dual_base=0.0627, single_post=0.7312
- Top-25 mean: single_base=0.5072, dual_base=0.0372, single_post=0.6449
- Top-50 mean: single_base=0.4230, dual_base=0.0191, single_post=0.5779

## 6) Interpretation for Presentation

1. On this pair of runs, SingleDock shows clearly higher potency-oriented reward than DualDock.
2. DualDock optimization is harder because reward requires selectivity; many candidates get near-zero total reward when off-target score matches or exceeds target-A score.
3. This dual run is a short overnight setting (12 RL steps, no post-sampling), so it should be treated as a smoke/diagnostic run, not the final dual benchmark.
4. Fair final judgment needs matched budgets: same RL steps and post-sampling policy for single and dual.
