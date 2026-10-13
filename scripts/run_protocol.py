"""C4: how protocol choices change the measured group gaps (one backbone, no intervention).

    python scripts/run_protocol.py --dataset ml-100k

Grid: timestamp tie-breaking {item id, random} x evaluation {full ranking, sampled 99 negatives}.
For each tie-breaking variant iALS is tuned on validation (full ranking), then the TEST item is evaluated
both ways. Reported per grouping (activity; gender when available): per-group NDCG@10, worst / best group,
gap and relative gap, overall NDCG@10, and the same for ItemPop.
Outputs: results/<dataset>/<timestamp>_protocol/ (output.txt, config.yaml, protocol.json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from leveldown.data import ROOT, group_labels  # noqa: E402
from leveldown.models import IALS, ItemPop, evaluate, evaluate_sampled  # noqa: E402
from leveldown.runlog import Run  # noqa: E402
from run_gate import load_split, tune  # noqa: E402


def main(dataset, cfg, log=print):
    k, seed = cfg["eval"]["k"], cfg["seed"]
    groupings = ["activity"] + (["gender"] if dataset in ("ml-100k", "ml-1m") else [])
    out = {}
    for tie in ("item", "random"):
        split = load_split(dataset, cfg, tie)
        log(f"\n===== tie-breaking: {tie}  ({dataset}: users={split.n_users} items={split.n_items})")
        m, hp = tune("ials", IALS, cfg["grids"]["ials"], split, k, lambda *_: None)
        log(f"  iALS selected {hp}")
        pop = ItemPop().fit(split.train, split.n_items)
        evals = {}
        for mname, model in (("ials", m), ("itempop", pop)):
            evals[(mname, "full")] = evaluate(model, split, "test", k)[1]
            evals[(mname, "sampled99")] = evaluate_sampled(model, split, "test", k, 99, seed)[1]
        for gname in groupings:
            labels, names = group_labels(split, gname, dataset, cfg["datasets"][dataset]["raw_dir"])
            log(f"  -- grouping {gname}")
            log(f"     {'model':<8}{'protocol':<11}{'overall':>9}" + "".join(f"{n:>9}" for n in names)
                + f"{'worst':>8}{'best':>7}{'gap':>9}{'rel.gap':>9}")
            for (mname, prot), nd in evals.items():
                gm = [nd[labels == i].mean() for i in range(len(names))]
                lo, hi = int(np.argmin(gm)), int(np.argmax(gm))
                gap = gm[hi] - gm[lo]
                rel = gap / gm[lo] if gm[lo] > 0 else float("inf")
                log(f"     {mname:<8}{prot:<11}{nd.mean():>9.4f}" + "".join(f"{v:>9.4f}" for v in gm)
                    + f"{names[lo]:>8}{names[hi]:>7}{gap:>9.4f}{rel:>9.0%}")
                out[f"{tie}|{gname}|{mname}|{prot}"] = {"overall": nd.mean(), "groups": dict(zip(names, gm)),
                                                         "worst": names[lo], "best": names[hi], "gap": gap,
                                                         "rel_gap": rel, "hp": hp if mname == "ials" else {}}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml-100k", "ml-1m", "gowalla"])
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    cfg["results_dir"] = f"results/{args.dataset}"
    with Run(cfg, "protocol") as run:
        res = main(args.dataset, cfg)
        run.path("protocol.json").write_text(json.dumps(res, indent=2, default=float))