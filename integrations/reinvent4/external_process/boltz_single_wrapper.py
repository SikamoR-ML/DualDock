#!/usr/bin/env python3
"""Boltz-2 single-target wrapper for REINVENT ExternalProcess."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


DEFAULT_CONFIG_ENV = "DUALDOCK_WRAPPER_CONFIG"


class WrapperError(RuntimeError):
    """Configuration/runtime error in wrapper."""


class WrapperLogger:
    def __init__(self, log_path: Optional[Path], debug: bool = False):
        self._debug = bool(debug)
        self._fh = None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = log_path.open("a", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
        if self._debug:
            sys.stderr.write(line)
            sys.stderr.flush()
        if self._fh is not None:
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


@dataclass
class InputMolecule:
    index: int
    smiles: str
    canonical_smiles: Optional[str]
    valid: bool
    reason: str


@dataclass
class TargetScores:
    raw_scores: List[float]
    norm_scores: List[float]
    reasons: List[str]
    canonical_smiles: List[str]


class ScoreCache:
    def __init__(self, db_path: Path, logger: WrapperLogger):
        self._db_path = db_path
        self._logger = logger
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), timeout=30)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS boltz_score_cache (
                namespace TEXT NOT NULL,
                target_id TEXT NOT NULL,
                canonical_smiles TEXT NOT NULL,
                raw_score REAL NOT NULL,
                norm_score REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(namespace, target_id, canonical_smiles)
            )
            """
        )
        self._conn.commit()

    def get_many(
        self,
        namespace: str,
        target_id: str,
        canonical_smiles: Sequence[str],
    ) -> Dict[str, Tuple[float, float]]:
        if not canonical_smiles:
            return {}
        placeholders = ",".join("?" for _ in canonical_smiles)
        query = (
            "SELECT canonical_smiles, raw_score, norm_score FROM boltz_score_cache "
            "WHERE namespace = ? AND target_id = ? AND canonical_smiles IN ("
            + placeholders
            + ")"
        )
        rows = self._conn.execute(query, [namespace, target_id, *canonical_smiles]).fetchall()
        out: Dict[str, Tuple[float, float]] = {}
        for canonical, raw_score, norm_score in rows:
            out[str(canonical)] = (float(raw_score), float(norm_score))
        return out

    def put_many(
        self,
        namespace: str,
        target_id: str,
        values: Dict[str, Tuple[float, float]],
    ) -> None:
        if not values:
            return
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        rows = [
            (namespace, target_id, canonical, float(raw), float(norm), now)
            for canonical, (raw, norm) in values.items()
        ]
        self._conn.executemany(
            """
            INSERT INTO boltz_score_cache(namespace, target_id, canonical_smiles, raw_score, norm_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, target_id, canonical_smiles)
            DO UPDATE SET raw_score = excluded.raw_score,
                          norm_score = excluded.norm_score,
                          updated_at = excluded.updated_at
            """,
            rows,
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "dualdock").is_dir() and (parent / "configs").is_dir():
            return parent
    return current.parents[3]


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise WrapperError(f"Wrapper config not found: {config_path}")
    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    if config_path.suffix.lower() == ".toml":
        if tomllib is None:
            raise WrapperError("tomllib unavailable; use JSON config or Python>=3.11")
        return tomllib.loads(config_path.read_text(encoding="utf-8"))
    raise WrapperError(f"Unsupported wrapper config format: {config_path.suffix}")


def _resolve_path(value: str, base_dirs: Sequence[Path]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for base in base_dirs:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (base_dirs[0] / path).resolve()


def _validate_smiles(smiles: Sequence[str], require_rdkit: bool) -> List[InputMolecule]:
    if require_rdkit and Chem is None:
        raise WrapperError(
            "RDKit is required for SMILES validation/canonicalization. "
            "Install RDKit or set validation.require_rdkit=false in wrapper config."
        )

    out: List[InputMolecule] = []
    for idx, smi in enumerate(smiles):
        if not smi:
            out.append(InputMolecule(idx, smi, None, False, "empty_smiles"))
            continue

        if Chem is None:
            out.append(InputMolecule(idx, smi, smi, True, "rdkit_disabled"))
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(InputMolecule(idx, smi, None, False, "invalid_smiles"))
            continue

        canonical = Chem.MolToSmiles(mol, canonical=True)
        out.append(InputMolecule(idx, smi, canonical, True, "ok"))
    return out


def _extract_numeric_mean(data: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    values: List[float] = []
    for key in keys:
        if key not in data:
            continue
        try:
            values.append(float(data[key]))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / float(len(values))


def _normalize_score(raw_score: float, norm_cfg: Dict[str, Any]) -> float:
    method = str(norm_cfg.get("method", "identity")).strip().lower()
    clip_min = float(norm_cfg.get("clip_min", 0.0))
    clip_max = float(norm_cfg.get("clip_max", 1.0))
    invert = bool(norm_cfg.get("invert", False))

    value = float(raw_score)

    if method == "sigmoid":
        midpoint = float(norm_cfg.get("midpoint", 0.0))
        slope = float(norm_cfg.get("slope", 1.0))
        z = slope * (value - midpoint)
        if z >= 0:
            ez = math.exp(-z)
            value = 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            value = ez / (1.0 + ez)
    elif method == "minmax":
        low = float(norm_cfg.get("low", 0.0))
        high = float(norm_cfg.get("high", 1.0))
        if high <= low:
            raise WrapperError("Normalization minmax requires high > low")
        value = (value - low) / (high - low)
    elif method in {"identity", "clip"}:
        pass
    else:
        raise WrapperError(f"Unsupported normalization method: {method}")

    if invert:
        value = 1.0 - value

    if value < clip_min:
        value = clip_min
    if value > clip_max:
        value = clip_max

    return float(value)


def _rewrite_template_paths(template_text: str, template_path: Path, repo_root: Path, logger: WrapperLogger) -> str:
    # Rewrite msa/template relative paths to absolute to keep templates portable.
    pattern = re.compile(r"^(\s*(msa|template)\s*:\s*)(['\"]?)([^'\"\n]+)\3\s*$", re.MULTILINE)

    def repl(match: re.Match[str]) -> str:
        prefix = match.group(1)
        quote = match.group(3) or ""
        raw_value = match.group(4).strip()
        if raw_value in {"", "empty", "none", "null"}:
            return match.group(0)

        candidate = Path(raw_value)
        if candidate.is_absolute():
            return match.group(0)

        resolved = _resolve_path(raw_value, [template_path.parent, repo_root])
        if not resolved.exists():
            logger.log(f"WARNING template path does not exist yet, keeping as-is: {raw_value}")
            return match.group(0)

        return f"{prefix}{quote}{resolved.as_posix()}{quote}"

    return pattern.sub(repl, template_text)


def _render_yaml(template_text: str, smiles: str) -> str:
    if "__SMILES__" not in template_text:
        raise WrapperError("Target template must contain __SMILES__ placeholder")
    return template_text.replace("__SMILES__", smiles)


def _find_predictions_root(out_dir: Path, input_dir_name: str) -> Path:
    primary = out_dir / f"boltz_results_{input_dir_name}" / "predictions"
    if primary.exists():
        return primary
    candidates = sorted(out_dir.glob("boltz_results_*/predictions"))
    if candidates:
        return candidates[0]
    return primary


def _read_boltz_scores(
    out_dir: Path,
    names: Sequence[str],
    input_dir_name: str,
    score_mode: str,
) -> Dict[str, Tuple[Optional[float], str]]:
    pred_root = _find_predictions_root(out_dir, input_dir_name)
    output: Dict[str, Tuple[Optional[float], str]] = {}

    for name in names:
        affinity_path = pred_root / name / f"affinity_{name}.json"
        if not affinity_path.exists():
            output[name] = (None, "missing_affinity_json")
            continue

        try:
            data = json.loads(affinity_path.read_text(encoding="utf-8"))
        except Exception as exc:
            output[name] = (None, f"invalid_affinity_json:{type(exc).__name__}")
            continue

        if score_mode == "pred":
            keys = ["affinity_pred_value", "affinity_pred_value1", "affinity_pred_value2"]
        else:
            keys = [
                "affinity_probability_binary",
                "affinity_probability_binary1",
                "affinity_probability_binary2",
            ]

        mean_value = _extract_numeric_mean(data, keys)
        if mean_value is None:
            output[name] = (None, "missing_score_keys")
        else:
            output[name] = (float(mean_value), "ok")

    return output


def _run_boltz_batch(
    boltz_cfg: Dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    logger: WrapperLogger,
) -> Tuple[int, str]:
    boltz_bin = str(boltz_cfg.get("bin", "")).strip()
    if not boltz_bin:
        raise WrapperError("boltz.bin is empty")

    cmd = [
        boltz_bin,
        "predict",
        str(input_dir),
        "--out_dir",
        str(output_dir),
        "--model",
        str(boltz_cfg.get("model", "boltz2")),
        "--accelerator",
        str(boltz_cfg.get("accelerator", "gpu")),
        "--devices",
        str(boltz_cfg.get("devices", "1")),
        "--override",
        "--recycling_steps",
        str(boltz_cfg.get("recycling_steps", 1)),
        "--sampling_steps",
        str(boltz_cfg.get("sampling_steps", 30)),
        "--diffusion_samples",
        str(boltz_cfg.get("diffusion_samples", 1)),
        "--sampling_steps_affinity",
        str(boltz_cfg.get("sampling_steps_affinity", 30)),
        "--diffusion_samples_affinity",
        str(boltz_cfg.get("diffusion_samples_affinity", 1)),
        "--num_workers",
        str(boltz_cfg.get("num_workers", 0)),
    ]

    timeout = int(boltz_cfg.get("timeout_sec", 900))
    logger.log(f"Boltz command: {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )

    stderr_tail = proc.stderr[-2000:] if proc.stderr else ""
    logger.log(f"Boltz rc={proc.returncode}; stderr_tail={stderr_tail}")
    return int(proc.returncode), stderr_tail


def _score_uncached_canonicals(
    canonical_smiles: Sequence[str],
    template_path: Path,
    template_text: str,
    boltz_cfg: Dict[str, Any],
    norm_cfg: Dict[str, Any],
    fallback_score: float,
    repo_root: Path,
    logger: WrapperLogger,
) -> Dict[str, Tuple[float, float, str, bool]]:
    """Return canonical -> (raw_score, norm_score, reason, can_cache)."""
    if not canonical_smiles:
        return {}

    tmp_root = Path(tempfile.mkdtemp(prefix="dualdock_boltz_batch_"))
    input_dir = tmp_root / "in"
    output_dir = tmp_root / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    keep_tmp = bool(boltz_cfg.get("keep_tmp", False))
    score_mode = str(boltz_cfg.get("score_mode", "prob")).strip().lower()
    if score_mode not in {"prob", "pred"}:
        raise WrapperError(f"Unsupported boltz.score_mode={score_mode}; expected prob|pred")

    result: Dict[str, Tuple[float, float, str, bool]] = {}
    names: List[str] = []
    name_to_smiles: Dict[str, str] = {}

    try:
        for idx, canonical in enumerate(canonical_smiles):
            name = f"lig_{idx:05d}"
            names.append(name)
            name_to_smiles[name] = canonical
            yaml_body = _render_yaml(template_text, canonical)
            yaml_body = _rewrite_template_paths(yaml_body, template_path, repo_root, logger)
            (input_dir / f"{name}.yaml").write_text(yaml_body, encoding="utf-8")

        try:
            rc, stderr_tail = _run_boltz_batch(boltz_cfg, input_dir, output_dir, logger)
        except subprocess.TimeoutExpired:
            rc = 124
            stderr_tail = "timeout"

        if rc != 0:
            reason = f"boltz_failed_rc_{rc}"
            if stderr_tail:
                reason = f"{reason}:{stderr_tail[:120]}"
            logger.log(f"Boltz failed for batch of {len(canonical_smiles)} molecules: {reason}")
            for canonical in canonical_smiles:
                norm_fallback = _normalize_score(fallback_score, norm_cfg)
                result[canonical] = (fallback_score, norm_fallback, reason, False)
            return result

        parsed = _read_boltz_scores(output_dir, names, input_dir.name, score_mode)
        for name in names:
            canonical = name_to_smiles[name]
            raw_score, reason = parsed.get(name, (None, "missing_parser_output"))
            if raw_score is None:
                norm_fallback = _normalize_score(fallback_score, norm_cfg)
                result[canonical] = (fallback_score, norm_fallback, f"fallback:{reason}", False)
                continue

            norm_score = _normalize_score(float(raw_score), norm_cfg)
            result[canonical] = (float(raw_score), norm_score, "ok", True)

        return result

    finally:
        if keep_tmp:
            logger.log(f"Keeping temporary Boltz workspace: {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)


def score_target_batch(
    smiles: Sequence[str],
    config: Dict[str, Any],
    target_id: str,
    logger: WrapperLogger,
) -> TargetScores:
    repo_root_value = str(config.get("repo_root", "")).strip()
    repo_root = Path(repo_root_value).resolve() if repo_root_value else _repo_root()

    targets_cfg = config.get("targets", {})
    if target_id not in targets_cfg:
        raise WrapperError(f"Target '{target_id}' not defined in wrapper config targets")

    target_cfg = targets_cfg[target_id]
    template_value = str(target_cfg.get("template", "")).strip()
    if not template_value:
        raise WrapperError(f"targets.{target_id}.template is empty")

    template_path = _resolve_path(template_value, [repo_root, Path.cwd()])
    if not template_path.exists():
        raise WrapperError(f"Target template not found: {template_path}")

    template_text = template_path.read_text(encoding="utf-8")

    fallback_cfg = config.get("fallback", {})
    fallback_score = float(fallback_cfg.get("score", 0.05))
    invalid_smiles_score = float(fallback_cfg.get("invalid_smiles_score", 0.0))

    validation_cfg = config.get("validation", {})
    require_rdkit = bool(validation_cfg.get("require_rdkit", True))

    mols = _validate_smiles(smiles, require_rdkit=require_rdkit)

    raw_scores = [invalid_smiles_score for _ in mols]
    norm_scores = [invalid_smiles_score for _ in mols]
    reasons = ["invalid_smiles" for _ in mols]
    canonical_out = ["" for _ in mols]

    valid_canonical_set: List[str] = []
    for mol in mols:
        if mol.valid and mol.canonical_smiles:
            canonical_out[mol.index] = mol.canonical_smiles
            if mol.canonical_smiles not in valid_canonical_set:
                valid_canonical_set.append(mol.canonical_smiles)
        else:
            reasons[mol.index] = mol.reason

    norm_cfg = target_cfg.get("normalization", {})

    cache_cfg = config.get("cache", {})
    cache_db_value = str(cache_cfg.get("db_path", "")).strip()
    cache_namespace = str(cache_cfg.get("namespace", "default")).strip() or "default"

    scored: Dict[str, Tuple[float, float, str]] = {}
    cache_hits: Dict[str, Tuple[float, float]] = {}

    cache_obj: Optional[ScoreCache] = None
    try:
        if cache_db_value:
            cache_path = _resolve_path(cache_db_value, [repo_root, Path.cwd()])
            cache_obj = ScoreCache(cache_path, logger)
            cache_hits = cache_obj.get_many(cache_namespace, target_id, valid_canonical_set)
            logger.log(
                f"target={target_id}: cache hits {len(cache_hits)}/{len(valid_canonical_set)} "
                f"(namespace={cache_namespace})"
            )

        uncached = [canonical for canonical in valid_canonical_set if canonical not in cache_hits]

        boltz_cfg = dict(config.get("boltz", {}))
        boltz_bin_value = str(boltz_cfg.get("bin", "")).strip() or os.environ.get("BOLTZ_BIN", "").strip()
        if not boltz_bin_value:
            boltz_bin_value = "boltz"
        boltz_cfg["bin"] = boltz_bin_value

        uncached_scores = _score_uncached_canonicals(
            uncached,
            template_path=template_path,
            template_text=template_text,
            boltz_cfg=boltz_cfg,
            norm_cfg=norm_cfg,
            fallback_score=fallback_score,
            repo_root=repo_root,
            logger=logger,
        )

        to_cache: Dict[str, Tuple[float, float]] = {}
        for canonical, (raw_score, norm_score, reason, can_cache) in uncached_scores.items():
            scored[canonical] = (raw_score, norm_score, reason)
            if can_cache:
                to_cache[canonical] = (raw_score, norm_score)

        if cache_obj is not None and to_cache:
            cache_obj.put_many(cache_namespace, target_id, to_cache)

        for mol in mols:
            if not mol.valid or not mol.canonical_smiles:
                raw_scores[mol.index] = invalid_smiles_score
                norm_scores[mol.index] = invalid_smiles_score
                reasons[mol.index] = mol.reason
                continue

            canonical = mol.canonical_smiles
            if canonical in cache_hits:
                raw_score, norm_score = cache_hits[canonical]
                raw_scores[mol.index] = raw_score
                norm_scores[mol.index] = norm_score
                reasons[mol.index] = "cache_hit"
                continue

            if canonical in scored:
                raw_score, norm_score, reason = scored[canonical]
                raw_scores[mol.index] = raw_score
                norm_scores[mol.index] = norm_score
                reasons[mol.index] = reason
                continue

            raw_scores[mol.index] = fallback_score
            norm_scores[mol.index] = _normalize_score(fallback_score, norm_cfg)
            reasons[mol.index] = "fallback:missing_internal_score"

    finally:
        if cache_obj is not None:
            cache_obj.close()

    return TargetScores(
        raw_scores=[float(x) for x in raw_scores],
        norm_scores=[float(x) for x in norm_scores],
        reasons=reasons,
        canonical_smiles=canonical_out,
    )


def _read_smiles_from_stdin() -> List[str]:
    raw = sys.stdin.read()
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _next_batch_index(counter_file: Optional[Path]) -> int:
    if counter_file is None:
        return int(time.time())

    counter_file.parent.mkdir(parents=True, exist_ok=True)
    counter_file.touch(exist_ok=True)

    with counter_file.open("r+", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

        raw = fh.read().strip()
        current = int(raw) if raw else 0
        next_value = current + 1

        fh.seek(0)
        fh.truncate(0)
        fh.write(str(next_value))
        fh.flush()

        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    return next_value


def _append_trace(
    config: Dict[str, Any],
    smiles: List[str],
    target_scores: TargetScores,
    target_id: str,
) -> None:
    logging_cfg = config.get("logging", {})
    trace_jsonl_value = str(logging_cfg.get("trace_jsonl", "")).strip()
    if not trace_jsonl_value:
        return

    trace_jsonl = Path(trace_jsonl_value).expanduser().resolve()
    trace_counter_value = str(logging_cfg.get("trace_counter", "")).strip()
    trace_counter = Path(trace_counter_value).expanduser().resolve() if trace_counter_value else None

    batch_index = _next_batch_index(trace_counter)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_id = str(config.get("run_id", "dualdock_single")).strip() or "dualdock_single"

    trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with trace_jsonl.open("a", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

        for idx, smi in enumerate(smiles):
            rec = {
                "run_id": run_id,
                "timestamp": timestamp,
                "batch_index": batch_index,
                "molecule_index": idx,
                "smiles": smi,
                "canonical_smiles": target_scores.canonical_smiles[idx],
                "targetA_raw": float(target_scores.raw_scores[idx]),
                "targetA_score": float(target_scores.norm_scores[idx]),
                "offTargetB_raw": 0.0,
                "offTargetB_score": 0.0,
                "off_target_penalty": 0.0,
                "total_reward": float(target_scores.norm_scores[idx]),
                "aggregation_mode": "single",
                "targetA_reason": target_scores.reasons[idx],
                "offTargetB_reason": "single_mode",
                "target_id": target_id,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _ensure_array_lengths(payload: Dict[str, Any], expected_len: int) -> None:
    for key, value in payload.items():
        if isinstance(value, list) and len(value) != expected_len:
            raise WrapperError(
                f"Payload length mismatch for '{key}': expected {expected_len}, got {len(value)}"
            )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DualDock single-target Boltz wrapper")
    parser.add_argument(
        "--config",
        default=os.environ.get(DEFAULT_CONFIG_ENV, ""),
        help="Path to wrapper runtime config (.json/.toml)",
    )
    parser.add_argument(
        "--target-id",
        default=os.environ.get("DUALDOCK_TARGET_ID", "target_a"),
        help="Target key from config.targets (default: target_a)",
    )
    parser.add_argument(
        "--output-key",
        default="target_score",
        help="Payload key containing normalized target score",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    if not args.config:
        sys.stderr.write(
            "ERROR: wrapper config path is required. Pass --config or set "
            f"{DEFAULT_CONFIG_ENV}.\n"
        )
        return 2

    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)

    log_value = str(config.get("logging", {}).get("wrapper_log", "")).strip()
    logger = WrapperLogger(Path(log_value).expanduser().resolve() if log_value else None, debug=False)

    smiles = _read_smiles_from_stdin()
    if not smiles:
        print(json.dumps({"payload": {args.output_key: [], "canonical_smiles": [], "reasons": []}}))
        logger.close()
        return 0

    try:
        result = score_target_batch(smiles, config=config, target_id=args.target_id, logger=logger)
        _append_trace(config=config, smiles=smiles, target_scores=result, target_id=args.target_id)
        payload: Dict[str, Any] = {
            args.output_key: result.norm_scores,
            f"{args.output_key}_raw": result.raw_scores,
            "canonical_smiles": result.canonical_smiles,
            "reasons": result.reasons,
            # compatibility aliases
            "target_score": result.norm_scores,
            "targetA_score": result.norm_scores,
            "total_reward": result.norm_scores,
            "boltz2_score": result.norm_scores,
        }
        _ensure_array_lengths(payload, len(smiles))
        print(json.dumps({"payload": payload}))
        return 0

    except WrapperError as exc:
        logger.log(f"WrapperError: {exc}")
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2
    except Exception as exc:  # pragma: no cover
        logger.log(f"Unhandled exception: {type(exc).__name__}: {exc}")
        sys.stderr.write(f"ERROR: unhandled exception {type(exc).__name__}: {exc}\n")
        return 2
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
