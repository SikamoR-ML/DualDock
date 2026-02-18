#!/usr/bin/env python3
"""
REINVENT4 ExternalProcess wrapper for Boltz-2 via YAML templates.

Contract (REINVENT ExternalProcess):
- stdin:  SMILES (one per line)
- stdout: {"payload": {"boltz2_score": [float, ...]}}
  length MUST match number of input SMILES.

Boltz expects DATA to be:
- a YAML file, or
- a directory containing YAML files
We use a directory of per-ligand YAMLs for stable mapping.

Modes (set via CLI args, env as fallback):
- Single:
    --mode single --template <path/to/template.yaml>
    env fallback: BOLTZ_YAML_TEMPLATE
  score = affinity_score(target)

- DualDock:
    --mode dual --target <target_template.yaml> --offtarget <offtarget_template.yaml>
    env fallback: DUALDOCK_TARGET_TEMPLATE and DUALDOCK_OFFTARGET_TEMPLATE
  score = min(target, 1 - off_target)

stdout MUST stay clean (JSON only). Logs go to stderr only in debug mode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import re
import tempfile
from pathlib import Path
from typing import Optional


# -----------------------------
# Config via environment (fallback)
# -----------------------------
BOLTZ_BIN: str = os.environ.get("BOLTZ_BIN", "boltz")
BOLTZ_ACCEL: str = os.environ.get("BOLTZ_ACCEL", "gpu")      # gpu|cpu|tpu
BOLTZ_DEVICES: str = os.environ.get("BOLTZ_DEVICES", "1")

BOLTZ_DEBUG: bool = os.environ.get("BOLTZ_DEBUG", "0") == "1"
TIMEOUT_SEC: int = int(os.environ.get("BOLTZ_TIMEOUT", "300"))

RECYCLE: str = os.environ.get("BOLTZ_RECYCLE", "1")
SAMP_STEPS: str = os.environ.get("BOLTZ_STEPS", "30")
DIFF_SAMPLES: str = os.environ.get("BOLTZ_DIFF", "1")
AFF_STEPS: str = os.environ.get("BOLTZ_AFF_STEPS", "30")
AFF_DIFF: str = os.environ.get("BOLTZ_AFF_DIFF", "1")

KEEP_TMP: bool = os.environ.get("BOLTZ_KEEP_TMP", "0") == "1"

# env fallbacks for paths
ENV_SINGLE_TEMPLATE: str = os.environ.get("BOLTZ_YAML_TEMPLATE", "").strip()
ENV_TARGET_TEMPLATE: str = os.environ.get("DUALDOCK_TARGET_TEMPLATE", "").strip()
ENV_OFFTARGET_TEMPLATE: str = os.environ.get("DUALDOCK_OFFTARGET_TEMPLATE", "").strip()


# -----------------------------
# CLI
# -----------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--mode", choices=["auto", "single", "dual"], default="auto")
    p.add_argument("--template", default=None, help="single-target YAML template path")
    p.add_argument("--target", default=None, help="dual target YAML template path")
    p.add_argument("--offtarget", default=None, help="dual off-target YAML template path")
    p.add_argument("--config", default=None, help="targets TOML config path (relative to repo root allowed)")
    return p.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def _dbg(msg: str) -> None:
    if BOLTZ_DEBUG:
        print(f"[dualdock_predict] {msg}", file=sys.stderr, flush=True)


def _emit(scores: list[float]) -> None:
    print(json.dumps({"payload": {"boltz2_score": scores}}))


def _read_smiles() -> list[str]:
    raw = sys.stdin.read()
    return [s.strip() for s in raw.splitlines() if s.strip()]


def _load_text(path_str: str) -> str:
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"Template not found: {p}")
    return p.read_text(encoding="utf-8")


def _yaml_from_template(template_text: str, smi: str) -> str:
    if "__SMILES__" not in template_text:
        raise RuntimeError("Template must contain __SMILES__ placeholder")
    return template_text.replace("__SMILES__", smi)


def _sigmoid(x: float) -> float:
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _run_boltz_predict(in_dir: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        BOLTZ_BIN, "predict",
        str(in_dir),
        "--out_dir", str(out_dir),
        "--model", "boltz2",
        "--accelerator", BOLTZ_ACCEL,
        "--devices", BOLTZ_DEVICES,
        "--recycle", RECYCLE,
        "--sampling_steps", SAMP_STEPS,
        "--diffusion_samples", DIFF_SAMPLES,
        "--affinity_steps", AFF_STEPS,
        "--affinity_diffusion_samples", AFF_DIFF,
    ]

    _dbg("Running boltz predict")
    _dbg("CMD: " + " ".join(cmd))

    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,  # keep stdout clean
            stderr=(sys.stderr if BOLTZ_DEBUG else subprocess.DEVNULL),
            timeout=TIMEOUT_SEC,
            check=False,
            text=True,
        )
        _dbg(f"boltz rc={p.returncode}")
        return int(p.returncode)
    except subprocess.TimeoutExpired:
        _dbg(f"boltz timeout after {TIMEOUT_SEC}s for {in_dir}")
        return 124
    except Exception as e:
        _dbg(f"boltz failed: {e}")
        return 1


def _find_predictions_root(out_dir: Path, in_dir_name: str) -> Path:
    primary = out_dir / f"boltz_results_{in_dir_name}" / "predictions"
    if primary.exists():
        return primary

    hits = sorted(out_dir.glob("boltz_results_*/predictions"))
    if hits:
        return hits[0]

    return primary


def _read_scores(out_dir: Path, names: list[str], in_dir_name: str) -> list[float]:
    pred_root = _find_predictions_root(out_dir, in_dir_name)
    if not pred_root.exists():
        _dbg(f"predictions root not found: {pred_root}")
        return [0.0] * len(names)

    scores: list[float] = []
    for name in names:
        aff_path = pred_root / name / f"affinity_{name}.json"
        if not aff_path.exists():
            _dbg(f"missing affinity: {aff_path}")
            scores.append(0.0)
            continue

        try:
            data = json.loads(aff_path.read_text(encoding="utf-8"))
            v0 = float(data.get("affinity_pred_value", 0.0))
            v1 = float(data.get("affinity_pred_value1", v0))
            v2 = float(data.get("affinity_pred_value2", v0))
            mean_v = (v0 + v1 + v2) / 3.0
            scores.append(_sigmoid(mean_v / 3.0))
        except Exception as e:
            _dbg(f"failed parse {aff_path}: {e}")
            scores.append(0.0)

    return scores


def _score_with_template(smiles: list[str], template_text: str, tag: str) -> tuple[list[float], Path]:
    tmp_root = Path(tempfile.mkdtemp(prefix=f"boltz_{tag}_"))
    in_dir = tmp_root / "in"
    out_dir = tmp_root / "out"
    in_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    for i, smi in enumerate(smiles):
        name = f"lig_{i:05d}"
        names.append(name)
        (in_dir / f"{name}.yaml").write_text(_yaml_from_template(template_text, smi), encoding="utf-8")

    rc = _run_boltz_predict(in_dir, out_dir)
    if rc != 0:
        return ([0.0] * len(smiles), tmp_root)

    return (_read_scores(out_dir, names, in_dir.name), tmp_root)


def _combine(target: list[float], offtarget: list[float]) -> list[float]:
    n = min(len(target), len(offtarget))
    out: list[float] = []
    for i in range(n):
        t = float(target[i])
        o = float(offtarget[i])
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        if o < 0.0:
            o = 0.0
        elif o > 1.0:
            o = 1.0
        out.append(min(t, 1.0 - o))

    if len(out) < len(target):
        out.extend([0.0] * (len(target) - len(out)))
    return out


def _cleanup_tmp(tmp_paths: list[Path]) -> None:
    if KEEP_TMP:
        for p in tmp_paths:
            _dbg(f"KEEP TMP: {p}")
    else:
        for p in tmp_paths:
            shutil.rmtree(p, ignore_errors=True)


def _repo_root() -> Path:
    # scripts/dualdock_predict.py -> repo root is parent of scripts/
    return Path(__file__).resolve().parents[1]


def _strip_inline_comment(s: str) -> str:
    # remove inline comments starting with '#', but keep content before it
    if "#" in s:
        return s.split("#", 1)[0].rstrip()
    return s


def _parse_targets_toml(path: Path) -> dict:
    """
    Minimal TOML parser for our needs.
    Supports:
      mode = "dual" | "single" | "auto"
      [single] template = "..."
      [dual] target = "..." ; offtarget = "..."
    """
    cfg = {"mode": "auto", "single": {}, "dual": {}}
    section: Optional[str] = None

    text = path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = _strip_inline_comment(raw).strip()
        if not line:
            continue

        m = re.match(r"^\[(?P<section>[A-Za-z0-9_]+)\]\s*$", line)
        if m:
            sec = m.group("section").strip()
            section = sec if sec in ("single", "dual") else None
            continue

        if "=" not in line:
            continue

        key, val = [x.strip() for x in line.split("=", 1)]

        # accept "..." or '...'
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]

        if section is None:
            if key == "mode":
                cfg["mode"] = val.strip()
        else:
            cfg[section][key] = val

    return cfg


def _abs_from_repo(root: Path, p: str) -> str:
    if not p:
        return ""
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((root / pp).resolve())


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = _parse_args()
    smiles = _read_smiles()
    if not smiles:
        _emit([])
        return

    # Priority: CLI > ENV fallback
    single_template = (args.template if args.template is not None else ENV_SINGLE_TEMPLATE).strip()
    target_template = (args.target if args.target is not None else ENV_TARGET_TEMPLATE).strip()
    offtarget_template = (args.offtarget if args.offtarget is not None else ENV_OFFTARGET_TEMPLATE).strip()
       # Apply --config (relative to repo root allowed)
    root = _repo_root()
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = (root / cfg_path).resolve()

        cfg = _parse_targets_toml(cfg_path)

        # If args.mode is auto, allow config to define mode
        cfg_mode = str(cfg.get("mode", "auto")).strip() or "auto"
        if args.mode == "auto" and cfg_mode in ("single", "dual", "auto"):
            args.mode = cfg_mode

        if args.mode == "single":
            t = str(cfg.get("single", {}).get("template", "")).strip()
            if t:
                single_template = _abs_from_repo(root, t)
        elif args.mode == "dual":
            t = str(cfg.get("dual", {}).get("target", "")).strip()
            o = str(cfg.get("dual", {}).get("offtarget", "")).strip()
            if t:
                target_template = _abs_from_repo(root, t)
            if o:
                offtarget_template = _abs_from_repo(root, o)
        else:
            # auto: prefer dual if both present in config, else single
            t = str(cfg.get("dual", {}).get("target", "")).strip()
            o = str(cfg.get("dual", {}).get("offtarget", "")).strip()
            s = str(cfg.get("single", {}).get("template", "")).strip()
            if t and o:
                target_template = _abs_from_repo(root, t)
                offtarget_template = _abs_from_repo(root, o)
            elif s:
                single_template = _abs_from_repo(root, s)

    # Mode selection
    if args.mode == "single":
        target_template = ""
        offtarget_template = ""
    elif args.mode == "dual":
        single_template = ""
    # auto:
    # - if both target/offtarget are present -> dual
    # - else -> single

    dual_mode = bool(target_template and offtarget_template)

    try:
        if dual_mode:
            t_text = _load_text(target_template)
            o_text = _load_text(offtarget_template)

            t_scores, t_tmp = _score_with_template(smiles, t_text, "target")
            o_scores, o_tmp = _score_with_template(smiles, o_text, "offtarget")
            scores = _combine(t_scores, o_scores)

            _emit(scores)
            _cleanup_tmp([t_tmp, o_tmp])
        else:
            if not single_template:
                raise RuntimeError(
                    "No templates provided. Use CLI:\n"
                    "  single: --mode single --template <path>\n"
                    "  dual:   --mode dual --target <path> --offtarget <path>\n"
                    "Or env fallback:\n"
                    "  BOLTZ_YAML_TEMPLATE or DUALDOCK_TARGET_TEMPLATE/DUALDOCK_OFFTARGET_TEMPLATE"
                )

            tmpl = _load_text(single_template)
            scores, tmp = _score_with_template(smiles, tmpl, "single")
            _emit(scores)
            _cleanup_tmp([tmp])

    except Exception as e:
        _dbg(f"wrapper error: {e}")
        _emit([0.0] * len(smiles))


if __name__ == "__main__":
    main()
