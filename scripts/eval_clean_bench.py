import json, numpy as np, os

#!/usr/bin/env python3
import json
import os
import numpy as np

TARGETS = ["b1_full", "b1_nomsa", "b1_notemplate"]
BASE = "/home/sikamor/projects/DualDock/bench/clean"

def load_jsonl(path: str) -> np.ndarray:
    xs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            xs.extend(obj["payload"]["boltz2_score"])
    return np.array(xs, dtype=float)

def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / tp[-1] if tp[-1] else tp
    fpr = fp / fp[-1] if fp[-1] else fp
    return float(np.trapz(tpr, fpr))

def main() -> None:
    best = None
    rows = []

    for t in TARGETS:
        p_path = os.path.join(BASE, f"pos_scores_{t}.jsonl")
        n_path = os.path.join(BASE, f"neg_scores_{t}.jsonl")
        if not (os.path.exists(p_path) and os.path.exists(n_path)):
            print(t, "missing files")
            continue

        pos = load_jsonl(p_path)
        neg = load_jsonl(n_path)

        A = auc(pos, neg)
        Aneg = auc(-pos, -neg)
        if A >= Aneg:
            best_auc, sign = A, "+score"
        else:
            best_auc, sign = Aneg, "-score"

        row = (
            t, len(pos), len(neg),
            float(np.median(pos)), float(pos.mean()),
            float(np.median(neg)), float(neg.mean()),
            A, Aneg, best_auc, sign
        )
        rows.append(row)

        if best is None or best_auc > best[0]:
            best = (best_auc, sign, t)

    print("target npos nneg pos_med pos_mean neg_med neg_mean auc auc(-score) best_auc best_sign")
    for r in rows:
        print("%-13s %4d %4d  % .6f % .6f  % .6f % .6f   %.3f   %.3f     %.3f   %s" % r)

    if best:
        print("\nBEST:", best[2], "use", best[1], "best_auc", round(best[0], 3))

if __name__ == "__main__":
    main()

