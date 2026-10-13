"""How often do several interactions share a user's LAST timestamp, by activity group?

    python scripts/check_ties.py --dataset ml-100k

With tie_break="item", the test item is then the highest item id among those, which may bias the
held-out item (e.g. towards later-catalogued, less popular items) for users who rate in bursts.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from leveldown.data import ROOT, activity_groups, load, read_ratings, tie_stats  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml-100k", "ml-1m"])
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    raw = cfg["datasets"][args.dataset]["raw_dir"]
    df = read_ratings(args.dataset, raw)
    ties = tie_stats(df, cfg["min_interactions"])
    split = load(args.dataset, raw, cfg["min_interactions"])
    groups, n = activity_groups(split.train)
    t = ties.sort_index().to_numpy()               # users in the same (sorted original id) order as the split
    print(f"{args.dataset}: users {len(t)}")
    print(f"{'group':>6}{'#train':>8}{'share with tie at last ts':>28}{'mean #items at last ts':>25}")
    for g in range(4):
        m = groups == g
        print(f"{g:>6}{n[m].mean():>8.1f}{(t[m] > 1).mean():>28.1%}{t[m].mean():>25.1f}")