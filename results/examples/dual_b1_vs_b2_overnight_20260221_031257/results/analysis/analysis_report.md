# DualDock Run Statistical Report

- Run directory: `/Users/sikamor/Downloads/DualDock_results/runs/dual_b1_vs_b2_overnight_20260221_031257`
- Analysis directory: `/Users/sikamor/Downloads/DualDock_results/runs/dual_b1_vs_b2_overnight_20260221_031257/results/analysis`
- Post-sampling available: no

## 1) Descriptive Statistics (`best_total_reward`)

| Group | N | Mean | Median | Std | Min | Max | Q1 | Q3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base RL | 95 | 0.0101 | 0.0000 | 0.0204 | 0.0000 | 0.0806 | 0.0000 | 0.0066 |

## 2) Practical Thresholds (Base only)

| Threshold | Base count (%) |
|---:|---:|
| >= 0.2 | 0 (0.0%) |
| >= 0.3 | 0 (0.0%) |
| >= 0.4 | 0 (0.0%) |
| >= 0.5 | 0 (0.0%) |
| >= 0.6 | 0 (0.0%) |
| >= 0.7 | 0 (0.0%) |
| >= 0.8 | 0 (0.0%) |

## 3) Top-K Quality (Base only)

- Top-10 mean reward: base=0.0627
- Top-50 mean reward: base=0.0191
- Top-100 mean reward: base=0.0101

## 4) RL Optimization Dynamics

- Steps parsed: 12
- First/last score: step 1=0.010, step 12=0.010
- Best step score: step 2=0.030

## 5) Scientific Interpretation

1. This run has no post-sampling output, so the report summarizes base RL quality only.
2. The run can be used for optimization diagnostics and baseline quality, but cannot support post-vs-base uplift claims.
