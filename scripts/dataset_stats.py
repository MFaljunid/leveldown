"""Dataset statistics table for the paper (after filtering and the temporal leave-one-out split).

    python scripts/dataset_stats.py                      (all datasets whose raw files exist)
    python scripts/dataset_stats.py --datasets ml-1m gowalla

Outputs results/paper/datasets.{csv,tex} and prints the table.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from leveldown.data import ROOT, activity_groups, group_labels, read_ratings, tie_stats, k_core  # noqa: E402
from run_gate import load_split  # noqa: E402

DS = {"ml-100k": "ML-100K", "ml-1m": "ML-1M", "gowalla": "Gowalla"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DS))
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    rows = []
    for ds in args.datasets:
        try:
            split = load_split(ds, cfg)
        except FileNotFoundError as e:
            print(f"skip {ds}: {e}")
            continue
        n = np.array([len(t) for t in split.train]) + 2          # train + val + test
        inter = int(n.sum())
        g, ntr = activity_groups(split.train)
        ranges = "; ".join(f"{int(ntr[g == q].min())}-{int(ntr[g == q].max())}" for q in range(4))
        d = cfg["datasets"][ds]
        df = read_ratings(ds, d["raw_dir"])
        if d.get("core"):
            df = k_core(df, d["core"], d["core"])
        ties = tie_stats(df, cfg["min_interactions"])
        row = {"dataset": DS[ds], "users": split.n_users, "items": split.n_items, "interactions": inter,
               "density": inter / (split.n_users * split.n_items), "mean_per_user": n.mean(),
               "median_per_user": float(np.median(n)), "activity_quartiles_train": ranges,
               "ties_at_last_ts": float((ties > 1).mean())}
        try:
            lab, names = group_labels(split, "gender", ds, d["raw_dir"])
            row["female_share"] = float((lab == 0).mean())
        except (ValueError, FileNotFoundError):
            row["female_share"] = None
        rows.append(row)
    if not rows:
        raise SystemExit("no dataset found")
    out = ROOT / "results" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "datasets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [r"\begin{tabular}{lrrrrrll}", r"\toprule",
             r"Dataset & Users & Items & Interactions & Density & Median/user & Activity quartiles (\#train) & Female \\",
             r"\midrule"]
    for r in rows:
        fem = "-" if r["female_share"] is None else f"{r['female_share']:.1%}".replace("%", r"\%")
        lines.append(f"{r['dataset']} & {r['users']:,} & {r['items']:,} & {r['interactions']:,} & "
                     f"{r['density']:.4f} & {r['median_per_user']:.0f} & {r['activity_quartiles_train']} & {fem} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "datasets.tex").write_text("\n".join(lines) + "\n")
    for r in rows:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"written to {out}")