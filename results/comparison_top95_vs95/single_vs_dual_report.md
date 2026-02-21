# Single vs Dual Comparative Analysis

- Single run: `/Users/sikamor/Downloads/DualDock_results/single_b1_13782_20260220_155823`
- Dual run: `/Users/sikamor/Downloads/DualDock_results/runs/dual_b1_vs_b2_overnight_20260221_031257`
- Comparison source: `base`
- Single input file: `/Users/sikamor/Downloads/DualDock_results/single_b1_13782_20260220_155823/results/ranked_ligands.jsonl`
- Dual input file: `/Users/sikamor/Downloads/DualDock_results/runs/dual_b1_vs_b2_overnight_20260221_031257/results/ranked_ligands.jsonl`
- Output dir: `/Users/sikamor/projects/DualDock/results/comparison_top95_vs95`
- Single post-sampling available: yes
- Comparison subset: top-95 rows from each run using source=`base`

## 1) Descriptive Stats

| Dataset | N | Mean | Median | Std | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Single base total_reward | 95 | 0.3399 | 0.3090 | 0.1194 | 0.2056 | 0.6757 |
| Single post total_reward | 95 | 0.5009 | 0.4865 | 0.1071 | 0.3717 | 0.8439 |
| Dual base total_reward | 95 | 0.0101 | 0.0000 | 0.0204 | 0.0000 | 0.0806 |
| Single base targetA | 95 | 0.3399 | 0.3090 | 0.1194 | 0.2056 | 0.6757 |
| Single post targetA | 95 | 0.5009 | 0.4865 | 0.1071 | 0.3717 | 0.8439 |
| Dual base targetA | 95 | 0.1284 | 0.1089 | 0.0821 | 0.0292 | 0.4294 |
| Dual base offTargetB | 95 | 0.1490 | 0.1179 | 0.1054 | 0.0251 | 0.5531 |

## 2) Selectivity in Dual Run

- Positive selectivity margin (targetA - offTargetB > 0): 33/95 (34.7%)
- Dual margin mean: -0.0206, median: -0.0097

## 3) Statistical Comparison (Single base vs Dual base)

- TargetA Mann-Whitney (two-sided): U=8518.0, p=4.237e-26
- TargetA Cohen's d (single - dual): 2.065
- TargetA Cliff's delta (single vs dual): 0.888
- TargetA mean diff (single - dual): 0.2116 (95% CI 0.1824..0.2420)
- Total-reward Mann-Whitney (two-sided): U=9025.0, p=8.579e-34
- Total-reward Cohen's d (single - dual): 3.852
- Total-reward Cliff's delta (single vs dual): 1.000
- Total-reward mean diff (single - dual): 0.3299 (95% CI 0.3066..0.3554)

## 4) Practical Thresholds (`best_total_reward`)

- Single base >= 0.2: 95 (100.0%)
- Dual base >= 0.2: 0 (0.0%)
- Single post >= 0.2: 95 (100.0%)

## 5) Top-K Comparison (`best_total_reward`)

- Top-10 mean: single_base=0.6017, dual_base=0.0627, single_post=0.7312
- Top-25 mean: single_base=0.5072, dual_base=0.0372, single_post=0.6449
- Top-50 mean: single_base=0.4230, dual_base=0.0191, single_post=0.5779

## 6) Interpretation for Presentation

1. On this pair of runs, SingleDock shows clearly higher potency-oriented reward than DualDock.
2. DualDock optimization is harder because reward requires selectivity; many candidates get near-zero total reward when off-target score matches or exceeds target-A score.
3. This dual run is a short overnight setting (12 RL steps, no post-sampling), so it should be treated as a smoke/diagnostic run, not the final dual benchmark.
4. Fair final judgment needs matched budgets: same RL steps and post-sampling policy for single and dual.
