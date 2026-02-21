#!/usr/bin/env python3
"""Postprocess DualDock trace into ranked ligand tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _build_ranked(records: List[Dict[str, Any]], rank_by: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        canonical = str(rec.get("canonical_smiles", "")).strip()
        if not canonical:
            canonical = str(rec.get("smiles", "")).strip()
        grouped[canonical].append(rec)

    ranked_rows: List[Dict[str, Any]] = []
    for canonical, rows in grouped.items():
        reward_values = [float(r.get("total_reward", 0.0)) for r in rows]
        target_a_values = [float(r.get("targetA_score", 0.0)) for r in rows]
        target_b_values = [float(r.get("offTargetB_score", 0.0)) for r in rows]

        best_row = max(rows, key=lambda x: float(x.get("total_reward", 0.0)))

        ranked_rows.append(
            {
                "canonical_smiles": canonical,
                "best_total_reward": float(max(reward_values)) if reward_values else 0.0,
                "mean_total_reward": float(mean(reward_values)) if reward_values else 0.0,
                "best_targetA_score": float(max(target_a_values)) if target_a_values else 0.0,
                "best_offTargetB_score": float(min(target_b_values)) if target_b_values else 0.0,
                "best_off_target_penalty": float(best_row.get("off_target_penalty", 0.0)),
                "occurrences": len(rows),
                "best_batch_index": int(best_row.get("batch_index", 0)),
                "best_smiles": str(best_row.get("smiles", "")),
                "ranking_criterion": rank_by,
            }
        )

    if rank_by == "mean_total_reward":
        ranked_rows.sort(key=lambda r: float(r["mean_total_reward"]), reverse=True)
    else:
        ranked_rows.sort(key=lambda r: float(r["best_total_reward"]), reverse=True)

    for idx, row in enumerate(ranked_rows, start=1):
        row["rank"] = idx

    return ranked_rows


def _maybe_plot(records: List[Dict[str, Any]], ranked: List[Dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    rewards = [float(r.get("total_reward", 0.0)) for r in records]
    if rewards:
        fig = plt.figure(figsize=(7, 4))
        plt.hist(rewards, bins=30)
        plt.title("Reward distribution")
        plt.xlabel("total_reward")
        plt.ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "reward_hist.png", dpi=150)
        plt.close(fig)

    if ranked:
        x = [float(r.get("best_targetA_score", 0.0)) for r in ranked]
        y = [float(r.get("best_offTargetB_score", 0.0)) for r in ranked]
        c = [float(r.get("best_total_reward", 0.0)) for r in ranked]

        fig = plt.figure(figsize=(6, 5))
        scatter = plt.scatter(x, y, c=c, cmap="viridis", s=20)
        plt.colorbar(scatter, label="best_total_reward")
        plt.xlabel("best_targetA_score")
        plt.ylabel("best_offTargetB_score")
        plt.title("Selectivity landscape")
        fig.tight_layout()
        fig.savefig(out_dir / "selectivity_scatter.png", dpi=150)
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank DualDock trace outputs")
    parser.add_argument("--trace-jsonl", required=True, help="Path to per-molecule trace JSONL")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--rank-by",
        default="best_total_reward",
        choices=["best_total_reward", "mean_total_reward"],
        help="Ranking criterion",
    )
    parser.add_argument("--plots", action="store_true", help="Generate optional plots")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    trace_path = Path(args.trace_jsonl).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    records = _read_jsonl(trace_path)

    all_scores_columns = [
        "run_id",
        "timestamp",
        "batch_index",
        "molecule_index",
        "smiles",
        "canonical_smiles",
        "targetA_raw",
        "targetA_score",
        "offTargetB_raw",
        "offTargetB_score",
        "off_target_penalty",
        "total_reward",
        "aggregation_mode",
        "targetA_reason",
        "offTargetB_reason",
    ]

    ranked = _build_ranked(records, rank_by=args.rank_by)

    ranked_columns = [
        "rank",
        "canonical_smiles",
        "best_smiles",
        "best_total_reward",
        "mean_total_reward",
        "best_targetA_score",
        "best_offTargetB_score",
        "best_off_target_penalty",
        "occurrences",
        "best_batch_index",
        "ranking_criterion",
    ]

    _write_csv(out_dir / "scored_molecules.csv", records, all_scores_columns)
    _write_jsonl(out_dir / "scored_molecules.jsonl", records)

    _write_csv(out_dir / "ranked_ligands.csv", ranked, ranked_columns)
    _write_jsonl(out_dir / "ranked_ligands.jsonl", ranked)

    (out_dir / "ranking_criterion.txt").write_text(args.rank_by + "\n", encoding="utf-8")

    if args.plots:
        _maybe_plot(records, ranked, out_dir / "plots")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
