#!/usr/bin/env python3
"""Compare SingleDock and DualDock run outputs with stats and plots."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover
    mannwhitneyu = None


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def top_n_rows(rows: List[dict], n: int) -> List[dict]:
    if n <= 0:
        return rows
    if len(rows) < n:
        raise RuntimeError(f"Requested top-{n}, but only {len(rows)} rows are available.")
    if rows and "rank" in rows[0]:
        ordered = sorted(rows, key=lambda r: int(r.get("rank", 10**9)))
        return ordered[:n]
    return rows[:n]


def extract_scores(rows: Iterable[dict], key: str) -> np.ndarray:
    return np.asarray([float(r.get(key, 0.0) or 0.0) for r in rows], dtype=float)


def parse_reinvent_step_scores(log_path: Path) -> List[Tuple[int, float]]:
    if not log_path.exists():
        return []
    pattern = re.compile(r"Score:\s*([0-9]+\.[0-9]+).*Step:\s*([0-9]+)")
    out: List[Tuple[int, float]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                out.append((int(m.group(2)), float(m.group(1))))
    return out


def summary(arr: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
    }


def topk_mean(scores: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    return float(np.mean(np.sort(scores)[::-1][:k])) if k > 0 else float("nan")


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return float("nan")
    var_a = np.var(a_arr, ddof=1)
    var_b = np.var(b_arr, ddof=1)
    pooled = ((len(a_arr) - 1) * var_a + (len(b_arr) - 1) * var_b) / (len(a_arr) + len(b_arr) - 2)
    if pooled <= 0:
        return float("nan")
    return (np.mean(a_arr) - np.mean(b_arr)) / math.sqrt(pooled)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    gt = 0
    lt = 0
    for x in a_arr:
        gt += int(np.sum(x > b_arr))
        lt += int(np.sum(x < b_arr))
    denom = len(a_arr) * len(b_arr)
    return (gt - lt) / denom if denom else float("nan")


def bootstrap_ci_diff_mean(
    a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 42
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        diffs[i] = float(np.mean(a_s) - np.mean(b_s))
    return float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def count_threshold(arr: np.ndarray, t: float) -> Tuple[int, float]:
    cnt = int(np.sum(arr >= t))
    return cnt, 100.0 * cnt / len(arr) if len(arr) else 0.0


def plot_hist(
    single_base: np.ndarray,
    dual_base: np.ndarray,
    out: Path,
    single_post: np.ndarray | None = None,
) -> None:
    lo = min(single_base.min(), dual_base.min(), single_post.min() if single_post is not None else single_base.min())
    hi = max(single_base.max(), dual_base.max(), single_post.max() if single_post is not None else single_base.max())
    bins = np.linspace(lo, hi, 40)
    fig, ax = plt.subplots(figsize=(8.3, 5), dpi=140)
    ax.hist(single_base, bins=bins, density=True, alpha=0.45, label="Single base", color="#2c7fb8")
    if single_post is not None:
        ax.hist(single_post, bins=bins, density=True, alpha=0.45, label="Single post", color="#d95f0e")
    ax.hist(dual_base, bins=bins, density=True, alpha=0.45, label="Dual base", color="#31a354")
    ax.set_xlabel("best_total_reward")
    ax.set_ylabel("Density")
    ax.set_title("Reward Distribution Comparison")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_total_reward_hist_compare.png")
    plt.close(fig)


def plot_ecdf_targeta(
    single_base_ta: np.ndarray,
    dual_base_ta: np.ndarray,
    out: Path,
    single_post_ta: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 5), dpi=140)
    for arr, label, color in [
        (single_base_ta, "Single base targetA", "#2c7fb8"),
        (single_post_ta, "Single post targetA", "#d95f0e") if single_post_ta is not None else None,
        (dual_base_ta, "Dual base targetA", "#31a354"),
    ]:
        if arr is None:
            continue
        x = np.sort(arr)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=label, linewidth=2, color=color)
    ax.set_xlabel("best_targetA_score")
    ax.set_ylabel("ECDF")
    ax.set_title("Target-A Potency Distribution")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "02_targetA_ecdf_compare.png")
    plt.close(fig)


def plot_dual_scatter_target_vs_offtarget(dual_ta: np.ndarray, dual_ot: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=140)
    ax.scatter(dual_ta, dual_ot, s=22, alpha=0.7, color="#31a354")
    lo = min(dual_ta.min(), dual_ot.min())
    hi = max(dual_ta.max(), dual_ot.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4, color="#333333", label="targetA = offTargetB")
    ax.set_xlabel("best_targetA_score")
    ax.set_ylabel("best_offTargetB_score")
    ax.set_title("Dual Run: Potency vs Off-target")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "03_dual_target_vs_offtarget_scatter.png")
    plt.close(fig)


def plot_selectivity_margin_hist(dual_margin: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 5), dpi=140)
    bins = np.linspace(dual_margin.min(), dual_margin.max(), 40)
    ax.hist(dual_margin, bins=bins, density=True, alpha=0.7, color="#31a354")
    ax.axvline(0.0, linestyle="--", color="#111111", linewidth=1.5, label="margin = 0")
    ax.set_xlabel("targetA - offTargetB (margin)")
    ax.set_ylabel("Density")
    ax.set_title("Dual Selectivity Margin Distribution")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "04_dual_selectivity_margin_hist.png")
    plt.close(fig)


def plot_topk_curves(
    single_base: np.ndarray,
    dual_base: np.ndarray,
    out: Path,
    single_post: np.ndarray | None = None,
) -> None:
    k_max = min(90, len(single_base), len(dual_base))
    k = np.arange(1, k_max + 1)
    fig, ax = plt.subplots(figsize=(8.3, 5), dpi=140)

    sb = np.sort(single_base)[::-1]
    ax.plot(k, np.cumsum(sb[:k_max]) / k, label="Single base", color="#2c7fb8", linewidth=2)

    if single_post is not None:
        sp = np.sort(single_post)[::-1]
        k_max_post = min(k_max, len(sp))
        ax.plot(
            np.arange(1, k_max_post + 1),
            np.cumsum(sp[:k_max_post]) / np.arange(1, k_max_post + 1),
            label="Single post",
            color="#d95f0e",
            linewidth=2,
        )

    db = np.sort(dual_base)[::-1]
    ax.plot(k, np.cumsum(db[:k_max]) / k, label="Dual base", color="#31a354", linewidth=2)

    ax.set_xlabel("Top-K")
    ax.set_ylabel("Mean best_total_reward in Top-K")
    ax.set_title("Top-K Reward Comparison")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "05_topk_reward_compare.png")
    plt.close(fig)


def plot_step_scores(
    single_steps: List[Tuple[int, float]], dual_steps: List[Tuple[int, float]], out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 5), dpi=140)
    if single_steps:
        x, y = zip(*single_steps)
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=3.5, color="#2c7fb8", label="Single run")
    if dual_steps:
        x, y = zip(*dual_steps)
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=3.5, color="#31a354", label="Dual run")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Batch score")
    ax.set_title("Training Dynamics: Single vs Dual")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "06_reinvent_step_score_compare.png")
    plt.close(fig)


def fmt_row(name: str, stats: Dict[str, float]) -> str:
    return (
        f"| {name} | {int(stats['n'])} | {stats['mean']:.4f} | {stats['median']:.4f} | {stats['std']:.4f} | "
        f"{stats['min']:.4f} | {stats['max']:.4f} |"
    )


def run_compare(
    single_run: Path,
    dual_run: Path,
    out_dir: Path,
    top_n: int | None = None,
    source: str = "base",
) -> None:
    if source == "base":
        single_rows_path = single_run / "results" / "ranked_ligands.jsonl"
        dual_rows_path = dual_run / "results" / "ranked_ligands.jsonl"
    else:
        single_rows_path = single_run / "results" / "post_sampling" / "ranked_ligands.jsonl"
        dual_rows_path = dual_run / "results" / "post_sampling" / "ranked_ligands.jsonl"

    if not single_rows_path.exists():
        raise RuntimeError(f"Single input file not found for source='{source}': {single_rows_path}")
    if not dual_rows_path.exists():
        raise RuntimeError(f"Dual input file not found for source='{source}': {dual_rows_path}")

    single_base_rows = read_jsonl(single_rows_path)
    dual_base_rows = read_jsonl(dual_rows_path)

    single_post_path = single_run / "results" / "post_sampling" / "ranked_ligands.jsonl"
    has_single_post = source == "base" and single_post_path.exists()
    single_post_rows = read_jsonl(single_post_path) if has_single_post else []

    if top_n is not None:
        single_base_rows = top_n_rows(single_base_rows, top_n)
        dual_base_rows = top_n_rows(dual_base_rows, top_n)
        if has_single_post and len(single_post_rows) >= top_n:
            single_post_rows = top_n_rows(single_post_rows, top_n)

    s_base_total = extract_scores(single_base_rows, "best_total_reward")
    s_base_ta = extract_scores(single_base_rows, "best_targetA_score")
    d_base_total = extract_scores(dual_base_rows, "best_total_reward")
    d_base_ta = extract_scores(dual_base_rows, "best_targetA_score")
    d_base_ot = extract_scores(dual_base_rows, "best_offTargetB_score")
    d_margin = d_base_ta - d_base_ot

    s_post_total = extract_scores(single_post_rows, "best_total_reward") if has_single_post else None
    s_post_ta = extract_scores(single_post_rows, "best_targetA_score") if has_single_post else None

    out_dir.mkdir(parents=True, exist_ok=True)

    plot_hist(s_base_total, d_base_total, out_dir, s_post_total)
    plot_ecdf_targeta(s_base_ta, d_base_ta, out_dir, s_post_ta)
    plot_dual_scatter_target_vs_offtarget(d_base_ta, d_base_ot, out_dir)
    plot_selectivity_margin_hist(d_margin, out_dir)
    plot_topk_curves(s_base_total, d_base_total, out_dir, s_post_total)

    s_steps = parse_reinvent_step_scores(single_run / "logs" / "reinvent.log")
    d_steps = parse_reinvent_step_scores(dual_run / "logs" / "reinvent.log")
    plot_step_scores(s_steps, d_steps, out_dir)

    ta_u, ta_p = (float("nan"), float("nan"))
    total_u, total_p = (float("nan"), float("nan"))
    if mannwhitneyu is not None:
        ta_u, ta_p = mannwhitneyu(s_base_ta, d_base_ta, alternative="two-sided")
        total_u, total_p = mannwhitneyu(s_base_total, d_base_total, alternative="two-sided")

    ta_d = cohens_d(s_base_ta, d_base_ta)
    total_d = cohens_d(s_base_total, d_base_total)
    ta_delta = cliffs_delta(s_base_ta, d_base_ta)
    total_delta = cliffs_delta(s_base_total, d_base_total)
    ta_dm, ta_lo, ta_hi = bootstrap_ci_diff_mean(s_base_ta, d_base_ta)
    total_dm, total_lo, total_hi = bootstrap_ci_diff_mean(s_base_total, d_base_total)

    pos_margin_cnt, pos_margin_pct = count_threshold(d_margin, 0.0)
    s_base_ge_02_cnt, s_base_ge_02_pct = count_threshold(s_base_total, 0.2)
    d_base_ge_02_cnt, d_base_ge_02_pct = count_threshold(d_base_total, 0.2)

    report = out_dir / "single_vs_dual_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Single vs Dual Comparative Analysis\n\n")
        f.write(f"- Single run: `{single_run}`\n")
        f.write(f"- Dual run: `{dual_run}`\n")
        f.write(f"- Comparison source: `{source}`\n")
        f.write(f"- Single input file: `{single_rows_path}`\n")
        f.write(f"- Dual input file: `{dual_rows_path}`\n")
        f.write(f"- Output dir: `{out_dir}`\n")
        f.write(f"- Single post-sampling available: {'yes' if has_single_post else 'no'}\n")
        if top_n is not None:
            f.write(f"- Comparison subset: top-{top_n} rows from each run using source=`{source}`\n")
        f.write("\n")

        f.write("## 1) Descriptive Stats\n\n")
        f.write("| Dataset | N | Mean | Median | Std | Min | Max |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        single_tag = "Single base" if source == "base" else "Single post-sampling"
        dual_tag = "Dual base" if source == "base" else "Dual post-sampling"
        f.write(fmt_row(f"{single_tag} total_reward", summary(s_base_total)) + "\n")
        if has_single_post and s_post_total is not None:
            f.write(fmt_row("Single post total_reward", summary(s_post_total)) + "\n")
        f.write(fmt_row(f"{dual_tag} total_reward", summary(d_base_total)) + "\n")
        f.write(fmt_row(f"{single_tag} targetA", summary(s_base_ta)) + "\n")
        if has_single_post and s_post_ta is not None:
            f.write(fmt_row("Single post targetA", summary(s_post_ta)) + "\n")
        f.write(fmt_row(f"{dual_tag} targetA", summary(d_base_ta)) + "\n")
        f.write(fmt_row(f"{dual_tag} offTargetB", summary(d_base_ot)) + "\n\n")

        f.write("## 2) Selectivity in Dual Run\n\n")
        f.write(
            f"- Positive selectivity margin (targetA - offTargetB > 0): "
            f"{pos_margin_cnt}/{len(d_margin)} ({pos_margin_pct:.1f}%)\n"
        )
        f.write(f"- Dual margin mean: {np.mean(d_margin):.4f}, median: {np.median(d_margin):.4f}\n\n")

        f.write(f"## 3) Statistical Comparison ({single_tag} vs {dual_tag})\n\n")
        f.write(f"- TargetA Mann-Whitney (two-sided): U={ta_u:.1f}, p={ta_p:.3e}\n")
        f.write(f"- TargetA Cohen's d (single - dual): {ta_d:.3f}\n")
        f.write(f"- TargetA Cliff's delta (single vs dual): {ta_delta:.3f}\n")
        f.write(f"- TargetA mean diff (single - dual): {ta_dm:.4f} (95% CI {ta_lo:.4f}..{ta_hi:.4f})\n")
        f.write(f"- Total-reward Mann-Whitney (two-sided): U={total_u:.1f}, p={total_p:.3e}\n")
        f.write(f"- Total-reward Cohen's d (single - dual): {total_d:.3f}\n")
        f.write(f"- Total-reward Cliff's delta (single vs dual): {total_delta:.3f}\n")
        f.write(
            f"- Total-reward mean diff (single - dual): {total_dm:.4f} "
            f"(95% CI {total_lo:.4f}..{total_hi:.4f})\n\n"
        )

        f.write("## 4) Practical Thresholds (`best_total_reward`)\n\n")
        f.write(
            f"- Single base >= 0.2: {s_base_ge_02_cnt} ({s_base_ge_02_pct:.1f}%)\n"
            f"- Dual base >= 0.2: {d_base_ge_02_cnt} ({d_base_ge_02_pct:.1f}%)\n"
        )
        if has_single_post and s_post_total is not None:
            sp_cnt, sp_pct = count_threshold(s_post_total, 0.2)
            f.write(f"- Single post >= 0.2: {sp_cnt} ({sp_pct:.1f}%)\n")
        f.write("\n")

        f.write("## 5) Top-K Comparison (`best_total_reward`)\n\n")
        for k in [10, 25, 50]:
            line = (
                f"- Top-{k} mean: single_base={topk_mean(s_base_total, k):.4f}, "
                f"dual_base={topk_mean(d_base_total, k):.4f}"
            )
            if has_single_post and s_post_total is not None:
                line += f", single_post={topk_mean(s_post_total, k):.4f}"
            f.write(line + "\n")
        f.write("\n")

        f.write("## 6) Interpretation for Presentation\n\n")
        f.write(
            "1. On this pair of runs, SingleDock shows clearly higher potency-oriented reward than DualDock.\n"
        )
        f.write(
            "2. DualDock optimization is harder because reward requires selectivity; many candidates get near-zero "
            "total reward when off-target score matches or exceeds target-A score.\n"
        )
        f.write(
            "3. This dual run is a short overnight setting (12 RL steps, no post-sampling), so it should be treated "
            "as a smoke/diagnostic run, not the final dual benchmark.\n"
        )
        f.write(
            "4. Fair final judgment needs matched budgets: same RL steps and post-sampling policy for single and dual.\n"
        )

    print(f"Saved report: {report}")
    print("Saved plots:")
    for p in sorted(out_dir.glob("*.png")):
        print(f"- {p}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Single and Dual DualDock runs.")
    parser.add_argument("--single-run", required=True, type=Path, help="Path to single run directory")
    parser.add_argument("--dual-run", required=True, type=Path, help="Path to dual run directory")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for report and plots")
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Use equal top-N rows from single and dual base ranked outputs (e.g., 1000).",
    )
    parser.add_argument(
        "--source",
        choices=["base", "post"],
        default="base",
        help="Which ranked source to compare: base RL ranked_ligands or post_sampling ranked_ligands.",
    )
    args = parser.parse_args()
    run_compare(args.single_run, args.dual_run, args.out_dir, args.top_n, args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
