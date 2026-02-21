#!/usr/bin/env python3
"""Unified DualDock experiment runner (REINVENT4 + Boltz-2)."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


ALLOWED_DOCKING_MODES = {"single", "dual"}


class GpuLease:
    def __init__(self, gpu_id: int, lock_file: Path, fd: int):
        self.gpu_id = int(gpu_id)
        self.lock_file = lock_file
        self._fd = fd
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            os.close(self._fd)
        except OSError:
            pass
        try:
            self.lock_file.unlink(missing_ok=True)
        except OSError:
            pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_toml(path: Path) -> Dict[str, Any]:
    if tomllib is None:
        raise RuntimeError("tomllib unavailable; use Python 3.11+")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _toml_quote(value: str) -> str:
    return json.dumps(value)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return _toml_quote(str(value))


def _find_command(candidate: str) -> Optional[str]:
    if not candidate:
        return None

    expanded = os.path.expanduser(candidate)
    if os.path.sep in expanded:
        path = Path(expanded)
        if path.exists():
            return str(path.resolve())
        return None

    return shutil.which(expanded)


def _resolve_reinvent_bin(config: Dict[str, Any]) -> str:
    exec_cfg = config.get("executables", {})
    from_cfg = str(exec_cfg.get("reinvent4_bin", "")).strip()
    from_env = os.environ.get("REINVENT4_BIN", "").strip()

    for candidate in [from_cfg, from_env, "reinvent"]:
        resolved = _find_command(candidate)
        if resolved:
            return resolved

    root_cfg = str(exec_cfg.get("reinvent4_root", "")).strip()
    root_env = os.environ.get("REINVENT4_ROOT", "").strip()
    for root in [root_cfg, root_env]:
        if not root:
            continue
        root_path = Path(os.path.expanduser(root)).resolve()
        for rel in ["reinvent", "bin/reinvent"]:
            candidate = root_path / rel
            if candidate.exists():
                return str(candidate)

    raise RuntimeError(
        "REINVENT executable not found. Set executables.reinvent4_bin in experiment config "
        "or REINVENT4_BIN/REINVENT4_ROOT environment variables."
    )


def _resolve_boltz_bin(config: Dict[str, Any]) -> str:
    exec_cfg = config.get("executables", {})
    boltz_cfg = config.get("boltz", {})

    from_cfg = str(exec_cfg.get("boltz_bin", "")).strip() or str(boltz_cfg.get("bin", "")).strip()
    from_env = os.environ.get("BOLTZ_BIN", "").strip()

    for candidate in [from_cfg, from_env, "boltz"]:
        resolved = _find_command(candidate)
        if resolved:
            return resolved

    root_cfg = str(exec_cfg.get("boltz_root", "")).strip()
    root_env = os.environ.get("BOLTZ_ROOT", "").strip()
    for root in [root_cfg, root_env]:
        if not root:
            continue
        root_path = Path(os.path.expanduser(root)).resolve()
        for rel in ["boltz", "bin/boltz"]:
            candidate = root_path / rel
            if candidate.exists():
                return str(candidate)

    raise RuntimeError(
        "Boltz executable not found. Set executables.boltz_bin in experiment config "
        "or BOLTZ_BIN/BOLTZ_ROOT environment variables."
    )


def _selected_env_snapshot(env: Dict[str, str]) -> Dict[str, str]:
    prefixes = ("BOLTZ", "REINVENT", "CUDA", "CONDA", "PYTHON", "DUALDOCK")
    return {key: value for key, value in env.items() if key.startswith(prefixes)}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _extract_pid_from_lock(lock_path: Path) -> Optional[int]:
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
        pid = int(data.get("pid", 0))
        return pid if pid > 0 else None
    except Exception:
        pass
    try:
        pid = int(raw)
        return pid if pid > 0 else None
    except Exception:
        return None


def _parse_gpu_ids(hardware_cfg: Dict[str, Any]) -> List[int]:
    value = hardware_cfg.get("gpu_ids", "")
    if isinstance(value, str):
        raw = value.strip()
        if raw:
            ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
            if ids:
                return ids
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        ids = [int(x) for x in value]
        if ids:
            return ids
    gpu_count = int(hardware_cfg.get("gpu_count", 1))
    if gpu_count <= 0:
        raise RuntimeError("hardware.gpu_count must be >= 1 for auto GPU allocation")
    return list(range(gpu_count))


def _try_acquire_gpu_lock(
    gpu_id: int,
    lock_dir: Path,
    run_name: str,
    repo_root: Path,
) -> Optional[GpuLease]:
    lock_file = lock_dir / f"gpu{gpu_id}.lock"
    lock_dir.mkdir(parents=True, exist_ok=True)

    pid = _extract_pid_from_lock(lock_file) if lock_file.exists() else None
    if pid is not None and not _pid_alive(pid):
        try:
            lock_file.unlink(missing_ok=True)
        except OSError:
            pass

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock_file), flags, 0o644)
    except FileExistsError:
        return None

    payload = {
        "pid": os.getpid(),
        "run_name": run_name,
        "repo_root": str(repo_root),
        "gpu_id": int(gpu_id),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    os.write(fd, (json.dumps(payload) + "\n").encode("utf-8"))
    return GpuLease(gpu_id=gpu_id, lock_file=lock_file, fd=fd)


def _acquire_gpu_lease(
    hardware_cfg: Dict[str, Any],
    repo_root: Path,
    run_name: str,
) -> GpuLease:
    lock_dir_raw = str(hardware_cfg.get("gpu_lock_dir", "/tmp/dualdock_gpu_locks")).strip()
    lock_dir = Path(lock_dir_raw).expanduser()
    if not lock_dir.is_absolute():
        lock_dir = (repo_root / lock_dir).resolve()

    gpu_ids = _parse_gpu_ids(hardware_cfg)
    wait_sec = int(hardware_cfg.get("gpu_lock_wait_sec", 600))
    poll_sec = float(hardware_cfg.get("gpu_lock_poll_sec", 3.0))
    if wait_sec < 0:
        raise RuntimeError("hardware.gpu_lock_wait_sec must be >= 0")
    if poll_sec <= 0:
        raise RuntimeError("hardware.gpu_lock_poll_sec must be > 0")

    deadline = time.monotonic() + float(wait_sec)
    while True:
        for gpu_id in gpu_ids:
            lease = _try_acquire_gpu_lock(
                gpu_id=int(gpu_id),
                lock_dir=lock_dir,
                run_name=run_name,
                repo_root=repo_root,
            )
            if lease is not None:
                return lease

        if time.monotonic() >= deadline:
            raise RuntimeError(
                "No free GPU lock available. "
                f"gpu_ids={gpu_ids}, lock_dir={lock_dir}, wait_sec={wait_sec}. "
                "Either wait for another run to finish or increase gpu_lock_wait_sec."
            )
        time.sleep(poll_sec)


def _run_cmd(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _git_snapshot(repo_root: Path) -> Dict[str, Any]:
    rc_head, out_head, err_head = _run_cmd(["git", "rev-parse", "HEAD"], repo_root)
    rc_status, out_status, err_status = _run_cmd(["git", "status", "--short"], repo_root)
    return {
        "head_rc": rc_head,
        "head": out_head,
        "head_err": err_head,
        "status_rc": rc_status,
        "status": out_status,
        "status_err": err_status,
    }


def _build_stage_chunks(max_steps: int, checkpoint_every: int) -> List[int]:
    if checkpoint_every <= 0 or checkpoint_every >= max_steps:
        return [max_steps]

    chunks: List[int] = []
    remaining = max_steps
    while remaining > 0:
        part = min(checkpoint_every, remaining)
        chunks.append(part)
        remaining -= part
    return chunks


def _find_latest_checkpoint(checkpoints_dir: Path) -> Path:
    checkpoints = [p for p in checkpoints_dir.glob("*.chkpt") if p.is_file()]
    if not checkpoints:
        raise RuntimeError(
            f"No checkpoints found in {checkpoints_dir}. "
            "Check RL stage settings (checkpoint_every_steps/chkpt_file)."
        )
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def _render_sampling_toml(
    config: Dict[str, Any],
    run_dir: Path,
    checkpoint_path: Path,
    sampled_csv_path: Path,
) -> str:
    rl_cfg = config.get("rl", {})
    post_cfg = config.get("postprocess", {})

    num_smiles = int(post_cfg.get("num_smiles", 2000))
    if num_smiles <= 0:
        raise RuntimeError("postprocess.num_smiles must be > 0")

    sample_strategy = str(post_cfg.get("sample_strategy", "multinomial")).strip() or "multinomial"
    temperature = float(post_cfg.get("temperature", 1.0))
    unique_molecules = bool(post_cfg.get("unique_molecules", True))
    randomize_smiles = bool(post_cfg.get("randomize_smiles", False))
    target_smiles_path = str(post_cfg.get("target_smiles_path", "")).strip()

    sampling_tb_dir = (run_dir / "tb_logs_sampling").resolve()
    json_out_config = (run_dir / "configs" / "sampling_rendered.json").resolve()

    lines: List[str] = [
        "### Auto-generated by dualdock.run_experiment",
        "run_type = \"sampling\"",
        f"device = {_toml_quote(str(rl_cfg.get('device', 'cuda:0')))}",
        f"tb_logdir = {_toml_quote(sampling_tb_dir.as_posix())}",
        f"json_out_config = {_toml_quote(json_out_config.as_posix())}",
        "",
        "[parameters]",
        f"model_file = {_toml_quote(checkpoint_path.as_posix())}",
        f"num_smiles = {_toml_value(num_smiles)}",
        f"output_file = {_toml_quote(sampled_csv_path.as_posix())}",
        f"sample_strategy = {_toml_quote(sample_strategy)}",
        f"temperature = {_toml_value(temperature)}",
        f"unique_molecules = {_toml_value(unique_molecules)}",
        f"randomize_smiles = {_toml_value(randomize_smiles)}",
    ]
    if target_smiles_path:
        lines.append(f"target_smiles_path = {_toml_quote(target_smiles_path)}")
    lines.append("")
    return "\n".join(lines)


def _read_sampled_smiles(sampled_csv_path: Path, max_smiles: int = 0) -> List[str]:
    if not sampled_csv_path.exists():
        raise RuntimeError(f"Sampling output CSV not found: {sampled_csv_path}")

    with sampled_csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise RuntimeError(f"Sampling output has no header: {sampled_csv_path}")

        smiles_col = None
        for candidate in ("SMILES", "smiles", "RDKit_SMILES (REINVENT)"):
            if candidate in reader.fieldnames:
                smiles_col = candidate
                break
        if smiles_col is None:
            smiles_col = reader.fieldnames[0]

        smiles: List[str] = []
        for row in reader:
            smi = str(row.get(smiles_col, "")).strip()
            if not smi:
                continue
            smiles.append(smi)
            if max_smiles > 0 and len(smiles) >= max_smiles:
                break
    return smiles


def _iter_batches(values: Sequence[str], batch_size: int) -> Sequence[List[str]]:
    if batch_size <= 0:
        raise RuntimeError("postprocess.score_batch_size must be > 0")
    return [list(values[i : i + batch_size]) for i in range(0, len(values), batch_size)]


def _score_sampled_smiles_with_wrapper(
    smiles: Sequence[str],
    docking_mode: str,
    python_bin: str,
    wrapper_cfg_path: Path,
    repo_root: Path,
    env: Dict[str, str],
    batch_size: int,
) -> int:
    if docking_mode == "single":
        wrapper_cmd = [
            python_bin,
            "-m",
            "integrations.reinvent4.external_process.boltz_single_wrapper",
            "--config",
            str(wrapper_cfg_path),
            "--target-id",
            "target_a",
            "--output-key",
            "target_score",
        ]
    else:
        wrapper_cmd = [
            python_bin,
            "-m",
            "integrations.reinvent4.external_process.boltz_dual_wrapper",
            "--config",
            str(wrapper_cfg_path),
        ]

    total_scored = 0
    for batch in _iter_batches(smiles, batch_size=batch_size):
        payload_text = "\n".join(batch) + "\n"
        proc = subprocess.run(
            wrapper_cmd,
            cwd=str(repo_root),
            env=env,
            input=payload_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Post-sampling wrapper scoring failed "
                f"(code={proc.returncode}). stderr: {proc.stderr.strip()}"
            )
        try:
            parsed = json.loads(proc.stdout)
            payload = parsed.get("payload", {})
            expected = len(batch)
            for key, value in payload.items():
                if isinstance(value, list) and len(value) != expected:
                    raise RuntimeError(
                        f"Post-sampling wrapper payload mismatch for '{key}': "
                        f"expected {expected}, got {len(value)}"
                    )
        except Exception as exc:
            raise RuntimeError(
                "Failed parsing post-sampling wrapper JSON output. "
                f"stdout_tail={proc.stdout[-300:]!r}; error={type(exc).__name__}: {exc}"
            ) from exc
        total_scored += len(batch)
    return total_scored


def _run_post_sampling_and_scoring(
    config: Dict[str, Any],
    run_dir: Path,
    repo_root: Path,
    reinvent_bin: str,
    python_bin: str,
    docking_mode: str,
    wrapper_cfg: Dict[str, Any],
    env: Dict[str, str],
) -> Dict[str, Any]:
    post_cfg = config.get("postprocess", {})
    enabled = bool(post_cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    checkpoints_dir = (run_dir / "checkpoints").resolve()
    checkpoint_path = _find_latest_checkpoint(checkpoints_dir)

    post_dir = (run_dir / "results" / "post_sampling").resolve()
    post_dir.mkdir(parents=True, exist_ok=True)

    sampled_csv_path = (post_dir / "sampled_smiles.csv").resolve()
    sampled_csv_path.parent.mkdir(parents=True, exist_ok=True)

    sampling_toml = _render_sampling_toml(
        config=config,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        sampled_csv_path=sampled_csv_path,
    )
    sampling_cfg_path = (run_dir / "configs" / "post_sampling.toml").resolve()
    sampling_cfg_path.write_text(sampling_toml + "\n", encoding="utf-8")

    sampling_cmd = [reinvent_bin, "-l", str((run_dir / "logs" / "post_sampling_reinvent.log").resolve()), str(sampling_cfg_path)]
    with (run_dir / "logs" / "post_sampling_stdout.log").open("w", encoding="utf-8") as stdout_log, (
        run_dir / "logs" / "post_sampling_stderr.log"
    ).open("w", encoding="utf-8") as stderr_log:
        proc = subprocess.run(
            sampling_cmd,
            cwd=str(repo_root),
            env=env,
            stdout=stdout_log,
            stderr=stderr_log,
            check=False,
        )
    sampling_rc = int(proc.returncode)
    if sampling_rc != 0:
        raise RuntimeError(
            "Post-training sampling failed "
            f"(code={sampling_rc}). See {run_dir / 'logs' / 'post_sampling_reinvent.log'}"
        )

    max_smiles = int(post_cfg.get("max_sampled_smiles", 0))
    sampled_smiles = _read_sampled_smiles(sampled_csv_path, max_smiles=max_smiles)
    if not sampled_smiles:
        raise RuntimeError(
            f"Sampling produced no SMILES in {sampled_csv_path}. "
            "Cannot run post-training scoring."
        )

    post_wrapper_cfg = copy.deepcopy(wrapper_cfg)
    post_wrapper_cfg["run_id"] = str(post_wrapper_cfg.get("run_id", "dualdock")) + "_post"
    post_wrapper_cfg["logging"] = {
        "wrapper_log": str((run_dir / "logs" / "post_wrapper.log").resolve().as_posix()),
        "trace_jsonl": str((run_dir / "logs" / "post_sampling_scores.jsonl").resolve().as_posix()),
        "trace_counter": str((run_dir / "logs" / "post_trace_counter.txt").resolve().as_posix()),
    }

    post_wrapper_cfg_path = (run_dir / "configs" / "post_wrapper_runtime.json").resolve()
    post_wrapper_cfg_path.write_text(json.dumps(post_wrapper_cfg, indent=2), encoding="utf-8")

    post_trace = Path(post_wrapper_cfg["logging"]["trace_jsonl"]).resolve()
    if post_trace.exists():
        post_trace.unlink()

    score_batch_size = int(post_cfg.get("score_batch_size", 128))
    scored_count = _score_sampled_smiles_with_wrapper(
        smiles=sampled_smiles,
        docking_mode=docking_mode,
        python_bin=python_bin,
        wrapper_cfg_path=post_wrapper_cfg_path,
        repo_root=repo_root,
        env=env,
        batch_size=score_batch_size,
    )

    rank_by = str(post_cfg.get("rank_by", config.get("output", {}).get("rank_by", "best_total_reward")))
    make_plots = bool(post_cfg.get("plots", config.get("output", {}).get("plots", False)))
    rank_cmd = [
        python_bin,
        "-m",
        "dualdock.rank_results",
        "--trace-jsonl",
        str(post_trace),
        "--out-dir",
        str(post_dir),
        "--rank-by",
        rank_by,
    ]
    if make_plots:
        rank_cmd.append("--plots")

    rank_proc = subprocess.run(
        rank_cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    post_rank_rc = int(rank_proc.returncode)
    if post_rank_rc != 0:
        raise RuntimeError(
            f"Post-training ranking failed (code={post_rank_rc}). stderr: {rank_proc.stderr.strip()}"
        )

    return {
        "enabled": True,
        "checkpoint_used": str(checkpoint_path),
        "sampling_config": str(sampling_cfg_path),
        "sampling_csv": str(sampled_csv_path),
        "sampled_smiles_count": len(sampled_smiles),
        "scored_smiles_count": scored_count,
        "post_trace_jsonl": str(post_trace),
        "ranked_csv": str((post_dir / "ranked_ligands.csv").resolve()),
        "ranked_jsonl": str((post_dir / "ranked_ligands.jsonl").resolve()),
        "scored_csv": str((post_dir / "scored_molecules.csv").resolve()),
        "scored_jsonl": str((post_dir / "scored_molecules.jsonl").resolve()),
        "rank_by": rank_by,
        "sampling_return_code": sampling_rc,
        "rank_return_code": post_rank_rc,
    }


def _get_docking_mode(config: Dict[str, Any]) -> str:
    mode = str(config.get("docking", {}).get("mode", "dual")).strip().lower()
    if mode not in ALLOWED_DOCKING_MODES:
        raise RuntimeError(f"docking.mode must be one of {sorted(ALLOWED_DOCKING_MODES)}; got '{mode}'")
    return mode


def _validate_optimizer(config: Dict[str, Any]) -> None:
    optimizer_cfg = config.get("optimizer", {})
    optimizer_name = str(optimizer_cfg.get("name", "adam")).strip().lower()
    if optimizer_name != "adam":
        raise RuntimeError(
            "This REINVENT4 integration currently supports only Adam optimizer via learning_strategy.rate. "
            "Set optimizer.name='adam'."
        )


def _render_scoring_toml(
    config: Dict[str, Any],
    wrapper_cfg_path: Path,
    python_bin: str,
    docking_mode: str,
) -> str:
    scoring = config.get("scoring", {})
    qed_weight = float(scoring.get("qed_weight", 0.2))
    external_weight = float(scoring.get("external_weight", 0.8))

    if docking_mode == "single":
        external_property = str(scoring.get("external_property", "target_score"))
        wrapper_name = "SingleDock Target A"
        args = (
            f"-m integrations.reinvent4.external_process.boltz_single_wrapper --config {wrapper_cfg_path.as_posix()} "
            "--target-id target_a --output-key target_score"
        )
    else:
        external_property = str(scoring.get("external_property", "total_reward"))
        wrapper_name = "DualDock Selectivity"
        args = f"-m integrations.reinvent4.external_process.boltz_dual_wrapper --config {wrapper_cfg_path.as_posix()}"

    lines = [
        "### Auto-generated by dualdock.run_experiment",
        "",
        "[[component]]",
        "[component.QED]",
        "",
        "[[component.QED.endpoint]]",
        f"name = {_toml_quote('QED Score')}",
        f"weight = {_toml_value(qed_weight)}",
        "",
        "[[component]]",
        "[component.ExternalProcess]",
        "",
        "[[component.ExternalProcess.endpoint]]",
        f"name = {_toml_quote(wrapper_name)}",
        f"weight = {_toml_value(external_weight)}",
        "",
        "[component.ExternalProcess.endpoint.params]",
        f"executable = {_toml_quote(python_bin)}",
        f"args = {_toml_quote(args)}",
        f"property = {_toml_quote(external_property)}",
        "",
    ]
    return "\n".join(lines)


def _render_staged_learning_toml(
    config: Dict[str, Any],
    scoring_path: Path,
    run_dir: Path,
    prior_file: Path,
    agent_file: Path,
) -> str:
    rl = config.get("rl", {})

    max_steps = int(rl.get("max_steps", 100))
    min_steps = int(rl.get("min_steps", 20))
    max_score = float(rl.get("max_score", 1.0))
    checkpoint_every = int(rl.get("checkpoint_every_steps", max_steps))
    stage_chunks = _build_stage_chunks(max_steps=max_steps, checkpoint_every=checkpoint_every)

    checkpoint_prefix = str(rl.get("checkpoint_prefix", "agent")).strip() or "agent"
    reward_strategy = str(rl.get("reward_strategy", "dap")).strip() or "dap"

    results_dir = (run_dir / "results").resolve()
    checkpoints_dir = (run_dir / "checkpoints").resolve()
    tb_dir = (run_dir / "tb_logs").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_prefix = str((results_dir / str(rl.get("summary_csv_prefix", "staged_learning"))).as_posix())
    json_out_config = str((run_dir / "configs" / "staged_learning_rendered.json").as_posix())

    lines: List[str] = [
        "### Auto-generated by dualdock.run_experiment",
        "run_type = \"staged_learning\"",
        f"device = {_toml_quote(str(rl.get('device', 'cuda:0')))}",
        f"tb_logdir = {_toml_quote(tb_dir.as_posix())}",
        f"json_out_config = {_toml_quote(json_out_config)}",
        "",
        "[parameters]",
        f"summary_csv_prefix = {_toml_quote(summary_csv_prefix)}",
        f"use_checkpoint = {_toml_value(bool(rl.get('use_checkpoint', False)))}",
        f"purge_memories = {_toml_value(bool(rl.get('purge_memories', False)))}",
        f"prior_file = {_toml_quote(prior_file.as_posix())}",
        f"agent_file = {_toml_quote(agent_file.as_posix())}",
        f"batch_size = {_toml_value(int(rl.get('batch_size', 8)))}",
        f"unique_sequences = {_toml_value(bool(rl.get('unique_sequences', True)))}",
        f"randomize_smiles = {_toml_value(bool(rl.get('randomize_smiles', True)))}",
        f"tb_isim = {_toml_value(bool(rl.get('tb_isim', False)))}",
        "",
        "[learning_strategy]",
        f"type = {_toml_quote(reward_strategy)}",
        f"sigma = {_toml_value(int(rl.get('sigma', 128)))}",
        f"rate = {_toml_value(float(rl.get('learning_rate', 0.0001)))}",
        "",
    ]

    for idx, chunk_steps in enumerate(stage_chunks, start=1):
        chkpt_name = f"{checkpoint_prefix}_stage{idx:02d}.chkpt"
        chkpt_path = (checkpoints_dir / chkpt_name).as_posix()

        lines.extend(
            [
                "[[stage]]",
                f"chkpt_file = {_toml_quote(chkpt_path)}",
                "termination = \"simple\"",
                f"max_score = {_toml_value(max_score)}",
                f"min_steps = {_toml_value(min(min_steps, chunk_steps))}",
                f"max_steps = {_toml_value(chunk_steps)}",
                "",
                "[stage.scoring]",
                f"type = {_toml_quote(str(config.get('scoring', {}).get('aggregation', 'geometric_mean')))}",
                f"filename = {_toml_quote(scoring_path.as_posix())}",
                "filetype = \"toml\"",
                "",
            ]
        )

    return "\n".join(lines)


def _prepare_wrapper_runtime_config(
    config: Dict[str, Any],
    run_dir: Path,
    repo_root: Path,
    boltz_bin: str,
    docking_mode: str,
) -> Dict[str, Any]:
    paths = config.get("paths", {})

    target_a_template = str(paths.get("target_a_template", "")).strip()
    target_b_template = str(paths.get("target_b_template", "")).strip()

    if not target_a_template:
        raise RuntimeError("Experiment config must define paths.target_a_template")
    if docking_mode == "dual" and not target_b_template:
        raise RuntimeError("Dual mode requires paths.target_b_template")

    logging_dir = run_dir / "logs"
    cache_dir = run_dir / "cache"
    logging_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    boltz_cfg = dict(config.get("boltz", {}))
    boltz_cfg["bin"] = boltz_bin

    targets: Dict[str, Dict[str, Any]] = {
        "target_a": {
            "name": str(paths.get("target_a_name", "target_a")),
            "template": target_a_template,
            "normalization": config.get("selectivity", {}).get("normalization", {}).get("target_a", {}),
        }
    }
    if docking_mode == "dual":
        targets["target_b"] = {
            "name": str(paths.get("target_b_name", "target_b")),
            "template": target_b_template,
            "normalization": config.get("selectivity", {}).get("normalization", {}).get("target_b", {}),
        }

    return {
        "run_id": str(config.get("experiment", {}).get("name", run_dir.name)),
        "repo_root": repo_root.as_posix(),
        "docking": {"mode": docking_mode},
        "targets": targets,
        "selectivity": {
            "mode": str(config.get("selectivity", {}).get("mode", "weighted_sum")),
            "weight_a": float(config.get("selectivity", {}).get("weight_a", 0.5)),
            "weight_b": float(config.get("selectivity", {}).get("weight_b", 0.5)),
            "reward_shift": float(config.get("selectivity", {}).get("reward_shift", 0.0)),
            "reward_clip_min": float(config.get("selectivity", {}).get("reward_clip_min", 0.0)),
            "reward_clip_max": float(config.get("selectivity", {}).get("reward_clip_max", 1.0)),
            "penalty": {
                "off_target_threshold": float(
                    config.get("selectivity", {}).get("penalty", {}).get("off_target_threshold", 1.1)
                ),
                "off_target_penalty": float(
                    config.get("selectivity", {}).get("penalty", {}).get("off_target_penalty", 0.0)
                ),
            },
        },
        "fallback": {
            "score": float(config.get("fallback", {}).get("score", 0.05)),
            "invalid_smiles_score": float(config.get("fallback", {}).get("invalid_smiles_score", 0.0)),
        },
        "validation": {
            "require_rdkit": bool(config.get("validation", {}).get("require_rdkit", True)),
        },
        "boltz": {
            "bin": boltz_bin,
            "model": str(boltz_cfg.get("model", "boltz2")),
            "score_mode": str(boltz_cfg.get("score_mode", "prob")),
            "timeout_sec": int(boltz_cfg.get("timeout_sec", 900)),
            "accelerator": str(boltz_cfg.get("accelerator", "gpu")),
            "devices": str(boltz_cfg.get("devices", "1")),
            "num_workers": int(boltz_cfg.get("num_workers", 0)),
            "recycling_steps": int(boltz_cfg.get("recycling_steps", 1)),
            "sampling_steps": int(boltz_cfg.get("sampling_steps", 30)),
            "diffusion_samples": int(boltz_cfg.get("diffusion_samples", 1)),
            "sampling_steps_affinity": int(boltz_cfg.get("sampling_steps_affinity", 30)),
            "diffusion_samples_affinity": int(boltz_cfg.get("diffusion_samples_affinity", 1)),
            "keep_tmp": bool(boltz_cfg.get("keep_tmp", False)),
        },
        "cache": {
            "db_path": str((cache_dir / "boltz_cache.sqlite").as_posix()),
            "namespace": str(config.get("experiment", {}).get("name", "dualdock")),
        },
        "logging": {
            "wrapper_log": str((logging_dir / "wrapper.log").as_posix()),
            "trace_jsonl": str((logging_dir / "per_molecule_scores.jsonl").as_posix()),
            "trace_counter": str((logging_dir / "trace_counter.txt").as_posix()),
        },
    }


def _save_repro_bundle(
    run_dir: Path,
    config: Dict[str, Any],
    env: Dict[str, str],
    repo_root: Path,
    source_config: Path,
    profile_path: Optional[Path],
) -> None:
    repro_dir = run_dir / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)

    env_info = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "selected_env": _selected_env_snapshot(env),
    }
    (repro_dir / "environment.json").write_text(json.dumps(env_info, indent=2), encoding="utf-8")

    git_info = _git_snapshot(repo_root)
    (repro_dir / "git.json").write_text(json.dumps(git_info, indent=2), encoding="utf-8")

    (repro_dir / "merged_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    shutil.copy2(source_config, repro_dir / "experiment_config.toml")
    if profile_path and profile_path.exists():
        shutil.copy2(profile_path, repro_dir / "hardware_profile.toml")


def _prepare_config(config_path: Path, repo_root: Path) -> Tuple[Dict[str, Any], Optional[Path]]:
    exp_cfg = _load_toml(config_path)

    profile_name = str(exp_cfg.get("hardware", {}).get("profile", "")).strip()
    profile_path = None
    profile_cfg: Dict[str, Any] = {}

    if profile_name:
        profile_path = (repo_root / "configs" / "hardware" / f"{profile_name}.toml").resolve()
        if not profile_path.exists():
            raise RuntimeError(f"Hardware profile not found: {profile_path}")
        profile_cfg = _load_toml(profile_path)

    return _deep_merge(profile_cfg, exp_cfg), profile_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DualDock selective RL experiment")
    parser.add_argument("--config", required=True, help="Path to experiment TOML config")
    parser.add_argument("--run-name", default="", help="Override run folder name")
    parser.add_argument("--dry-run", action="store_true", help="Render configs without launching REINVENT")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()

    config_path = _resolve_path(args.config, repo_root)
    if not config_path.exists():
        raise RuntimeError(f"Experiment config not found: {config_path}")

    config, profile_path = _prepare_config(config_path, repo_root)
    docking_mode = _get_docking_mode(config)
    _validate_optimizer(config)

    experiment = config.get("experiment", {})
    experiment_name = str(experiment.get("name", "dualdock")).strip() or "dualdock"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() if args.run_name else f"{experiment_name}_{timestamp}"

    output_root = str(config.get("output", {}).get("root", "runs")).strip() or "runs"
    run_dir = _resolve_path(output_root, repo_root) / run_name

    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    reinvent_bin = _resolve_reinvent_bin(config)
    boltz_bin = _resolve_boltz_bin(config)
    python_bin = str(config.get("executables", {}).get("python_bin", "")).strip() or sys.executable
    python_bin = str(Path(python_bin).expanduser().resolve())

    paths = config.get("paths", {})
    prior_path = _resolve_path(str(paths.get("prior_file", "priors/reinvent.prior")), repo_root)
    agent_path = _resolve_path(str(paths.get("agent_file", str(prior_path.as_posix()))), repo_root)

    if not prior_path.exists():
        raise RuntimeError(f"Prior file not found: {prior_path}")
    if not agent_path.exists():
        raise RuntimeError(f"Agent file not found: {agent_path}")

    wrapper_cfg = _prepare_wrapper_runtime_config(
        config,
        run_dir=run_dir,
        repo_root=repo_root,
        boltz_bin=boltz_bin,
        docking_mode=docking_mode,
    )
    wrapper_cfg_path = (run_dir / "configs" / "wrapper_runtime.json").resolve()
    wrapper_cfg_path.write_text(json.dumps(wrapper_cfg, indent=2), encoding="utf-8")

    scoring_toml = _render_scoring_toml(
        config,
        wrapper_cfg_path=wrapper_cfg_path,
        python_bin=python_bin,
        docking_mode=docking_mode,
    )
    scoring_path = (run_dir / "configs" / "stage_scoring.toml").resolve()
    scoring_path.write_text(scoring_toml + "\n", encoding="utf-8")

    staged_toml = _render_staged_learning_toml(
        config,
        scoring_path=scoring_path,
        run_dir=run_dir,
        prior_file=prior_path,
        agent_file=agent_path,
    )
    staged_path = (run_dir / "configs" / "staged_learning.toml").resolve()
    staged_path.write_text(staged_toml + "\n", encoding="utf-8")

    cmd = [reinvent_bin, "-l", str((run_dir / "logs" / "reinvent.log").resolve()), str(staged_path)]
    (run_dir / "reproducibility").mkdir(parents=True, exist_ok=True)
    (run_dir / "reproducibility" / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["BOLTZ_BIN"] = boltz_bin
    env["DUALDOCK_WRAPPER_CONFIG"] = str(wrapper_cfg_path)
    env["PYTHONPATH"] = str(repo_root) if not env.get("PYTHONPATH") else str(repo_root) + os.pathsep + env["PYTHONPATH"]
    hardware_cfg = dict(config.get("hardware", {}))
    cuda_visible = str(hardware_cfg.get("cuda_visible_devices", "")).strip()
    auto_gpu = cuda_visible.lower() == "auto" or bool(hardware_cfg.get("auto_select_gpu", False))
    gpu_lease: Optional[GpuLease] = None

    try:
        if auto_gpu and not args.dry_run:
            gpu_lease = _acquire_gpu_lease(
                hardware_cfg=hardware_cfg,
                repo_root=repo_root,
                run_name=run_name,
            )
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_lease.gpu_id)
            env["DUALDOCK_ALLOCATED_GPU_ID"] = str(gpu_lease.gpu_id)
        elif cuda_visible and cuda_visible.lower() != "auto":
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible

        _save_repro_bundle(
            run_dir=run_dir,
            config=config,
            env=env,
            repo_root=repo_root,
            source_config=config_path,
            profile_path=profile_path,
        )

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "run_dir": str(run_dir),
                        "docking_mode": docking_mode,
                        "reinvent_bin": reinvent_bin,
                        "boltz_bin": boltz_bin,
                        "staged_config": str(staged_path),
                        "scoring_config": str(scoring_path),
                        "wrapper_config": str(wrapper_cfg_path),
                        "command": cmd,
                        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
                        "gpu_auto_allocation_enabled": auto_gpu,
                        "postprocess_enabled": bool(config.get("postprocess", {}).get("enabled", False)),
                    },
                    indent=2,
                )
            )
            return 0

        with (run_dir / "logs" / "reinvent_stdout.log").open("w", encoding="utf-8") as stdout_log, (
            run_dir / "logs" / "reinvent_stderr.log"
        ).open("w", encoding="utf-8") as stderr_log:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=stdout_log,
                stderr=stderr_log,
                check=False,
            )
        rc = int(proc.returncode)

        rank_by = str(config.get("output", {}).get("rank_by", "best_total_reward"))
        make_plots = bool(config.get("output", {}).get("plots", False))

        trace_jsonl = Path(wrapper_cfg["logging"]["trace_jsonl"]).resolve()
        rank_cmd = [
            python_bin,
            "-m",
            "dualdock.rank_results",
            "--trace-jsonl",
            str(trace_jsonl),
            "--out-dir",
            str((run_dir / "results").resolve()),
            "--rank-by",
            rank_by,
        ]
        if make_plots:
            rank_cmd.append("--plots")

        rank_rc = 0
        rank_stdout, rank_stderr = "", ""
        if trace_jsonl.exists():
            rank_proc = subprocess.run(
                rank_cmd,
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            rank_rc = int(rank_proc.returncode)
            rank_stdout = rank_proc.stdout
            rank_stderr = rank_proc.stderr

        postprocess_summary: Dict[str, Any] = {"enabled": False}
        if rc == 0 and rank_rc == 0:
            postprocess_summary = _run_post_sampling_and_scoring(
                config=config,
                run_dir=run_dir,
                repo_root=repo_root,
                reinvent_bin=reinvent_bin,
                python_bin=python_bin,
                docking_mode=docking_mode,
                wrapper_cfg=wrapper_cfg,
                env=env,
            )

        manifest = {
            "run_dir": str(run_dir),
            "docking_mode": docking_mode,
            "reinvent_return_code": rc,
            "rank_return_code": rank_rc,
            "trace_jsonl": str(trace_jsonl),
            "ranked_csv": str((run_dir / "results" / "ranked_ligands.csv").resolve()),
            "ranked_jsonl": str((run_dir / "results" / "ranked_ligands.jsonl").resolve()),
            "checkpoints_dir": str((run_dir / "checkpoints").resolve()),
            "staged_summary_prefix": str((run_dir / "results").resolve() / "staged_learning"),
            "ranking_criterion": rank_by,
            "reward_strategy": str(config.get("rl", {}).get("reward_strategy", "dap")),
            "optimizer": str(config.get("optimizer", {}).get("name", "adam")),
            "rank_stdout": rank_stdout,
            "rank_stderr": rank_stderr,
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
            "allocated_gpu_id": env.get("DUALDOCK_ALLOCATED_GPU_ID", ""),
            "postprocess": postprocess_summary,
        }
        (run_dir / "results" / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if rc != 0:
            raise RuntimeError(
                textwrap.dedent(
                    f"""
                    REINVENT failed (code={rc}).
                    Logs:
                    - {(run_dir / 'logs' / 'reinvent.log').resolve()}
                    - {(run_dir / 'logs' / 'reinvent_stdout.log').resolve()}
                    - {(run_dir / 'logs' / 'reinvent_stderr.log').resolve()}
                    """
                ).strip()
            )

        if rank_rc != 0:
            raise RuntimeError(f"Ranking postprocess failed (code={rank_rc}). stderr: {rank_stderr}")

        print(json.dumps(manifest, indent=2))
        return 0
    finally:
        if gpu_lease is not None:
            gpu_lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
