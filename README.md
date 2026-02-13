# DualDock (single-target vs double-target docking)

Goal: empirically test whether double-target docking improves the probability of obtaining hits for both targets vs single-target.

## Repo structure
- configs/ - experiment configs (YAML)
- src/ - pipeline code
- scripts/ - runner scripts
- results/ - run artifacts (ignored by git)
- env/ - conda environments

## Mac (dev/analysis) quickstart
1) Create env:
   conda env create -f env/environment-mac.yml
   conda activate dualdock-mac
2) Run a test run:
   ./scripts/run_local.sh "" analyze demo_run
3) Inspect:
   cat results/demo_run/metadata.json

## Compute backends (planned)
- Windows/WSL + RTX 3080
- Linux GPU server (A6000), same pipeline interface
