#!/usr/bin/env python3
"""Boltz-2 dual-target wrapper with selectivity reward shaping."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None

from integrations.reinvent4.external_process.boltz_single_wrapper import (
    DEFAULT_CONFIG_ENV,
    WrapperError,
    WrapperLogger,
    _ensure_array_lengths,
    _load_config,
    _read_smiles_from_stdin,
    score_target_batch,
)


def _clip(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


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


def _compute_reward(
    target_a: List[float],
    target_b: List[float],
    selectivity_cfg: Dict[str, Any],
) -> Dict[str, List[float]]:
    mode = str(selectivity_cfg.get("mode", "weighted_sum")).strip().lower()
    weight_a = float(selectivity_cfg.get("weight_a", 0.5))
    weight_b = float(selectivity_cfg.get("weight_b", 0.5))
    reward_shift = float(selectivity_cfg.get("reward_shift", 0.0))
    reward_clip_min = float(selectivity_cfg.get("reward_clip_min", 0.0))
    reward_clip_max = float(selectivity_cfg.get("reward_clip_max", 1.0))

    penalty_cfg = selectivity_cfg.get("penalty", {})
    off_target_threshold = float(penalty_cfg.get("off_target_threshold", 1.1))
    off_target_penalty = float(penalty_cfg.get("off_target_penalty", 0.0))

    rewards: List[float] = []
    penalties: List[float] = []

    for a_score, b_score in zip(target_a, target_b):
        if mode == "weighted_sum":
            reward = (weight_a * float(a_score)) - (weight_b * float(b_score))
        elif mode == "maximin":
            reward = min(float(a_score), 1.0 - float(b_score))
        else:
            raise WrapperError(
                f"Unsupported selectivity.mode='{mode}'. Use weighted_sum or maximin."
            )

        penalty = 0.0
        if float(b_score) >= off_target_threshold:
            penalty = off_target_penalty

        reward = reward - penalty + reward_shift
        reward = _clip(reward, reward_clip_min, reward_clip_max)

        rewards.append(float(reward))
        penalties.append(float(penalty))

    return {"rewards": rewards, "penalties": penalties, "mode": mode}


def _append_trace(
    trace_jsonl: Optional[Path],
    smiles: List[str],
    canonical_smiles: List[str],
    target_a_raw: List[float],
    target_a_norm: List[float],
    target_b_raw: List[float],
    target_b_norm: List[float],
    reward: List[float],
    penalties: List[float],
    reasons_a: List[str],
    reasons_b: List[str],
    run_id: str,
    mode: str,
    counter_file: Optional[Path],
) -> None:
    if trace_jsonl is None:
        return

    batch_index = _next_batch_index(counter_file)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with trace_jsonl.open("a", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

        for idx in range(len(smiles)):
            record = {
                "run_id": run_id,
                "timestamp": timestamp,
                "batch_index": batch_index,
                "molecule_index": idx,
                "smiles": smiles[idx],
                "canonical_smiles": canonical_smiles[idx],
                "targetA_raw": float(target_a_raw[idx]),
                "targetA_score": float(target_a_norm[idx]),
                "offTargetB_raw": float(target_b_raw[idx]),
                "offTargetB_score": float(target_b_norm[idx]),
                "off_target_penalty": float(penalties[idx]),
                "total_reward": float(reward[idx]),
                "aggregation_mode": mode,
                "targetA_reason": reasons_a[idx],
                "offTargetB_reason": reasons_b[idx],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DualDock dual-target Boltz wrapper")
    parser.add_argument(
        "--config",
        default=os.environ.get(DEFAULT_CONFIG_ENV, ""),
        help="Path to wrapper runtime config (.json/.toml)",
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

    logging_cfg = config.get("logging", {})
    wrapper_log = str(logging_cfg.get("wrapper_log", "")).strip()
    logger = WrapperLogger(Path(wrapper_log).expanduser().resolve() if wrapper_log else None, debug=False)

    smiles = _read_smiles_from_stdin()
    if not smiles:
        print(
            json.dumps(
                {
                    "payload": {
                        "targetA_score": [],
                        "offTargetB_score": [],
                        "total_reward": [],
                        "offTargetPenalty": [],
                        "boltz2_score": [],
                        "boltz2_dual_score": [],
                    }
                }
            )
        )
        logger.close()
        return 0

    try:
        target_a = score_target_batch(smiles, config=config, target_id="target_a", logger=logger)
        target_b = score_target_batch(smiles, config=config, target_id="target_b", logger=logger)

        reward_bundle = _compute_reward(
            target_a=target_a.norm_scores,
            target_b=target_b.norm_scores,
            selectivity_cfg=config.get("selectivity", {}),
        )

        rewards = reward_bundle["rewards"]
        penalties = reward_bundle["penalties"]
        mode = str(reward_bundle["mode"])

        payload = {
            "targetA_score": target_a.norm_scores,
            "offTargetB_score": target_b.norm_scores,
            "total_reward": rewards,
            "offTargetPenalty": penalties,
            "targetA_raw": target_a.raw_scores,
            "offTargetB_raw": target_b.raw_scores,
            "canonical_smiles": target_a.canonical_smiles,
            "targetA_reason": target_a.reasons,
            "offTargetB_reason": target_b.reasons,
            # compatibility aliases
            "boltz2_score": rewards,
            "boltz2_dual_score": rewards,
        }

        _ensure_array_lengths(payload, len(smiles))

        trace_path_value = str(logging_cfg.get("trace_jsonl", "")).strip()
        trace_counter_value = str(logging_cfg.get("trace_counter", "")).strip()
        run_id = str(config.get("run_id", "dualdock_run")).strip() or "dualdock_run"

        _append_trace(
            trace_jsonl=Path(trace_path_value).expanduser().resolve() if trace_path_value else None,
            smiles=smiles,
            canonical_smiles=target_a.canonical_smiles,
            target_a_raw=target_a.raw_scores,
            target_a_norm=target_a.norm_scores,
            target_b_raw=target_b.raw_scores,
            target_b_norm=target_b.norm_scores,
            reward=rewards,
            penalties=penalties,
            reasons_a=target_a.reasons,
            reasons_b=target_b.reasons,
            run_id=run_id,
            mode=mode,
            counter_file=Path(trace_counter_value).expanduser().resolve() if trace_counter_value else None,
        )

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
