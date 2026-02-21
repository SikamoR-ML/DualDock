# DualDock

DualDock is an orchestration layer for docking-guided molecular generation with:
- **REINVENT4** as the RL generator
- **Boltz-2** as the structure/affinity scorer

This repository is intentionally focused on **reproducible experiment orchestration**.  
It does not vendor full REINVENT4 or Boltz source trees.

## Repository Layout

```text
DualDock/
  dualdock/                                   # Core runner + ranking
    run_experiment.py
    rank_results.py
    boltz_single_wrapper.py                   # compatibility entrypoint
    boltz_dual_wrapper.py                     # compatibility entrypoint

  integrations/
    reinvent4/
      external_process/                       # Active REINVENT ExternalProcess wrappers
        boltz_single_wrapper.py
        boltz_dual_wrapper.py

  configs/
    hardware/
      RTX_3080.toml
      A6000.toml
    experiments/                              # Only final sample configs
      single_b1_3080.toml
      dual_b1_vs_b2_3080.toml
      single_b1_a6000.toml
      dual_b1_vs_b2_a6000.toml

  assets/
    targets/adrenoreceptors/                  # Example Boltz YAML targets
    proteins/adrenoreceptors/                 # Example protein files
    msa/adrenoreceptors/                      # Example MSA files
    templates/                                # Reusable YAML templates

  priors/
    reinvent.prior

  scripts/
    run_reinvent_dualdock.sh
    check_portability.sh
    analyze_dualdock_results.py
    compare_single_dual_runs.py

  boltz/                                      # empty placeholder (local clone, gitignored)
  reinvent4/                                  # empty placeholder (local clone, gitignored)
  results/                                    # output placeholder
```

## What Was Finalized

- Core execution logic is in `dualdock/`.
- REINVENT/Boltz integration wrappers are moved to `integrations/reinvent4/external_process/`.
- Only the requested sample experiment configs remain (single/dual for 3080 and A6000).
- Example targets are included under `assets/targets/adrenoreceptors/`.
- Local runtime noise is excluded via `.gitignore`.

## Prerequisites

- Python 3.11+
- A working REINVENT4 installation
- A working Boltz installation (`boltz` CLI)
- CUDA-compatible environment for GPU runs (3080/A6000 profiles)

## 1) Environment Setup

Create a local env file (not committed):

```bash
cp .env.example .env
set -a; source .env; set +a
```

Required values in `.env`:
- `REINVENT4_BIN` (or `REINVENT4_ROOT`)
- `BOLTZ_BIN` (or `BOLTZ_ROOT`)

Optional:
- `PYTHON`

## 2) Quick Portability Check

```bash
bash scripts/check_portability.sh
```

This checks paths, wrappers, and required sample configs.

## 3) Run a Sample Experiment

Example (dual, RTX 3080):

```bash
python -m dualdock.run_experiment \
  --config configs/experiments/dual_b1_vs_b2_3080.toml \
  --run-name dual_3080_sample
```

or with helper script:

```bash
bash scripts/run_reinvent_dualdock.sh configs/experiments/dual_b1_vs_b2_3080.toml dual_3080_sample
```

## 4) What You Edit for Your Own Project

### A. Executables
Edit in `.env` or directly in config `[executables]`:
- `reinvent4_bin`
- `boltz_bin`
- `python_bin`

### B. Hardware Profile
Choose one profile in config:
- `hardware.profile = "RTX_3080"`
- `hardware.profile = "A6000"`

Then adjust `configs/hardware/*.toml` only if your machine differs:
- `cuda_visible_devices`
- `gpu_ids` / lock settings (A6000)
- `num_workers`, `timeout_sec`, sampling steps

### C. Target Files
In `[paths]` edit:
- `target_a_template`
- `target_b_template` (for dual mode)

Current examples use:
- `assets/targets/adrenoreceptors/b1.yaml`
- `assets/targets/adrenoreceptors/b2.yaml`

If adding your own targets, keep `__SMILES__` placeholder in YAML templates.

### D. RL and Scoring Behavior
In config sections:
- `[rl]` for batch size, steps, sigma, checkpoints
- `[scoring]` for QED/external weights
- `[selectivity]` for dual-objective reward logic
- `[boltz]` for inference runtime and quality/speed tradeoffs

### E. Prior/Agent
In `[paths]`:
- `prior_file`
- `agent_file`

Default points to `priors/reinvent.prior`.

## 5) Config Guide (Current Final Samples)

- `configs/experiments/single_b1_3080.toml`:
  - Single-target sample for 3080.
- `configs/experiments/dual_b1_vs_b2_3080.toml`:
  - Dual-selectivity sample for 3080.
- `configs/experiments/single_b1_a6000.toml`:
  - Single-target sample for A6000.
- `configs/experiments/dual_b1_vs_b2_a6000.toml`:
  - Dual-selectivity sample for A6000.

## 6) Integration Notes

- REINVENT calls wrappers through `ExternalProcess`.
- Active wrapper modules:
  - `integrations.reinvent4.external_process.boltz_single_wrapper`
  - `integrations.reinvent4.external_process.boltz_dual_wrapper`
- Compatibility entrypoints remain in `dualdock/boltz_*_wrapper.py`.

## 7) Outputs

Each run writes to `runs/<run_name>/`:
- generated configs
- logs
- per-molecule traces (`per_molecule_scores.jsonl`)
- ranked CSV/JSONL outputs
- reproducibility bundle

## 8) Recommended GitHub Push Checklist

1. `bash scripts/check_portability.sh`
2. `git status` (no runtime artifacts)
3. Verify `.env` is not tracked
4. Verify `runs/`, checkpoints, logs are not tracked
5. Commit and push

## 9) Included Example Results

This repository includes exported run artifacts for reproducible inspection:

- Single run:
  - `results/examples/single_b1_13782_20260220_155823`
- Dual run:
  - `results/examples/dual_b1_vs_b2_overnight_20260221_031257`
- Cross-run comparison (single vs dual):
  - `results/comparison_single_vs_dual`

These snapshots intentionally exclude heavy runtime-only files such as checkpoints, tensorboard event logs, and cache databases.
