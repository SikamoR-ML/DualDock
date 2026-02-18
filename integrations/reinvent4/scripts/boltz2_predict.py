#!/usr/bin/env python3
"""
REINVENT4 ExternalProcess predictor wrapper for Boltz-2.

Contract:
- Input:  SMILES lines on stdin (one SMILES per line).
- Output: JSON on stdout in the form:
    {"payload": {"boltz2_score": [float, float, ...]}}
  where the list length must match the number of input SMILES.

Notes:
- Boltz `predict` expects DATA to be a YAML file or a directory containing YAML files.
- This wrapper generates per-ligand YAML files in a temp folder, runs `boltz predict`,
  then reads affinity JSON outputs and maps them to [0, 1] scores.
"""

import sys, json, os, math, shutil, tempfile, subprocess
from pathlib import Path
from typing import List, Optional


# -----------------------------
# Configuration via environment
# -----------------------------
# Path to `boltz` executable (must be in PATH by default)
BOLTZ_BIN: str = os.environ.get("BOLTZ_BIN", "boltz")

# Compute settings
BOLTZ_ACCEL: str = os.environ.get("BOLTZ_ACCEL", "gpu")   # gpu|cpu|tpu
BOLTZ_DEVICES: str = os.environ.get("BOLTZ_DEVICES", "1") # number of devices (string for CLI)

# Runtime guard
TIMEOUT_SEC: int = int(os.environ.get("BOLTZ_TIMEOUT", "300"))

# Fast settings for RL / smoke tests (defaults are heavy and feel "hung")
RECYCLE: str = os.environ.get("BOLTZ_RECYCLE", "1")
SAMP_STEPS: str = os.environ.get("BOLTZ_STEPS", "30")
DIFF_SAMPLES: str = os.environ.get("BOLTZ_DIFF", "1")
AFF_STEPS: str = os.environ.get("BOLTZ_AFF_STEPS", "30")
AFF_DIFF: str = os.environ.get("BOLTZ_AFF_DIFF", "1")

# Protein definition source (choose ONE):
# A) Provide protein sequence directly
PROTEIN_SEQ: str = os.environ.get("BOLTZ_PROTEIN_SEQ", "").strip()

# B) Provide a YAML template file path with a __SMILES__ placeholder
#    Example template should include protein definition and ligand with smiles: "__SMILES__"
YAML_TEMPLATE_PATH: str = os.environ.get("BOLTZ_YAML_TEMPLATE", "").strip()


# -----------------------------
# Helpers
# -----------------------------
def _sigmoid(x: float) -> float:
    """Map any real value to (0, 1). You can replace this with a domain-specific transform."""
    return 1.0 / (1.0 + math.exp(-x))


def _read_smiles_from_stdin() -> List[str]:
    """Read SMILES from stdin, strip blanks."""
    raw = sys.stdin.read()
    return [s.strip() for s in raw.splitlines() if s.strip()]


def _load_template_text(path_str: str) -> Optional[str]:
    """Load YAML template file if provided."""
    if not path_str:
        return None
    p = Path(path_str).expanduser().resolve()
    return p.read_text(encoding="utf-8")


def _make_yaml_from_seq(seq: str, smi: str) -> str:
    """
    Minimal Boltz YAML for protein+ligand affinity.
    - Uses msa: empty for speed and to avoid server calls.
    """
    # NOTE: keep single quotes around SMILES to avoid YAML parsing issues.
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {seq}\n"
        "      msa: empty\n"
        "  - ligand:\n"
        "      id: B\n"
        f"      smiles: '{smi}'\n"
        "properties:\n"
        "  - affinity:\n"
        "      binder: B\n"
    )


def _make_yaml_from_template(template_text: str, smi: str) -> str:
    """Replace __SMILES__ placeholder in a user-provided template."""
    if "__SMILES__" not in template_text:
        raise RuntimeError("YAML template must contain __SMILES__ placeholder")
    # Wrap SMILES in quotes in the template itself (recommended).
    return template_text.replace("__SMILES__", smi)


def _run_boltz_predict(in_dir: Path, out_dir: Path) -> int:
    """Run `boltz predict` on a directory of YAML files. Returns process return code."""
    cmd = [
        BOLTZ_BIN, "predict", str(in_dir),
        "--out_dir", str(out_dir),
        "--accelerator", BOLTZ_ACCEL,
        "--devices", str(BOLTZ_DEVICES),
        "--model", "boltz2",
        "--override",
        "--recycling_steps", str(RECYCLE),
        "--sampling_steps", str(SAMP_STEPS),
        "--diffusion_samples", str(DIFF_SAMPLES),
        "--sampling_steps_affinity", str(AFF_STEPS),
        "--diffusion_samples_affinity", str(AFF_DIFF),
        "--num_workers", "0",
    ]

    # IMPORTANT: suppress Boltz stdout so this wrapper can print ONLY JSON to stdout.
    # Otherwise REINVENT ExternalProcess JSON parsing will break.
    p = subprocess.run(
        cmd,
        timeout=TIMEOUT_SEC,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,  # change to sys.stderr if you want to see Boltz logs
        check=False,
    )
    return p.returncode


def _read_affinity_scores(out_dir: Path, names: List[str], in_dir_name: str) -> List[float]:
    pred_root = _find_predictions_root(out_dir, in_dir_name)
    """
    Read Boltz affinity outputs and map to scores.
    Expected file:
      <pred_root>/<name>/affinity_<name>.json
    where pred_root is usually:
      out_dir/boltz_results_<in_dir_name>/predictions
    """
    pred_root = _find_predictions_root(out_dir, in_dir_name)

    scores: List[float] = []
    for name in names:
        aff_path = pred_root / name / f"affinity_{name}.json"
        if not aff_path.exists():
            scores.append(0.0)
            continue

        data = json.loads(aff_path.read_text(encoding="utf-8"))
        aff = float(data.get("affinity_pred_value", 0.0))
        scores.append(_sigmoid(aff))

    return scores


def _find_predictions_root(out_dir: Path, in_dir_name: str) -> Path:
    """
    Boltz writes results under:
      out_dir/boltz_results_<in_dir_name>/predictions/...
    but we keep it robust by falling back to glob search.
    """
    # Primary expected location
    primary = out_dir / f"boltz_results_{in_dir_name}" / "predictions"
    if primary.exists():
        return primary

    # Fallback: first predictions folder found under out_dir/boltz_results_*/predictions
    hits = sorted(out_dir.glob("boltz_results_*/predictions"))
    if hits:
        return hits[0]

    # If nothing found, return the primary path (caller will handle missing)
    return primary


def main() -> None:
    smilies = _read_smiles_from_stdin()

    # If REINVENT asks to score an empty batch, return empty payload.
    if not smilies:
        print(json.dumps({"payload": {"boltz2_score": []}}))
        return

    # Validate protein/template configuration.
    template_text = _load_template_text(YAML_TEMPLATE_PATH)
    if template_text is None and not PROTEIN_SEQ:
        raise RuntimeError("Set BOLTZ_PROTEIN_SEQ or BOLTZ_YAML_TEMPLATE")

    # Create a temp workspace for Boltz inputs/outputs.
    tmp_root = Path(tempfile.mkdtemp(prefix="boltz_batch_"))
    in_dir = tmp_root / "in"
    out_dir = tmp_root / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    names: List[str] = []

    try:
        # Write one YAML per SMILES so we can retrieve per-ligand affinity outputs reliably.
        for i, smi in enumerate(smilies):
            name = f"lig_{i:04d}"
            names.append(name)
            yaml_path = in_dir / f"{name}.yaml"

            if template_text is not None:
                yaml_text = _make_yaml_from_template(template_text, smi)
            else:
                yaml_text = _make_yaml_from_seq(PROTEIN_SEQ, smi)

            yaml_path.write_text(yaml_text, encoding="utf-8")

        # Run Boltz and handle failures gracefully (do not crash the RL loop).
        try:
            rc = _run_boltz_predict(in_dir, out_dir)
            if rc != 0:
                scores = [0.0] * len(smilies)
            else:
                scores = _read_affinity_scores(out_dir, names, in_dir.name)
        except subprocess.TimeoutExpired:
            # Timeout: return zeros, do not kill REINVENT run.
            scores = [0.0] * len(smilies)

        # Emit the required JSON payload for REINVENT ExternalProcess.
        print(json.dumps({"payload": {"boltz2_score": scores}}))

    finally:
        # Debug option: keep temp folder to inspect Boltz outputs.
        keep_tmp = os.environ.get("BOLTZ_KEEP_TMP", "0") == "1"
        if keep_tmp:
            print(f"[DEBUG] Keeping tmp dir: {tmp_root}", file=sys.stderr)
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
