"""L2 (pre-registered): where a group has headroom (model below a simple reference for that group),
can a simple intervention level that group UP?

    python scripts/run_headroom.py --dataset ml-100k

Models tuned exactly as in run_gate.py. Headroom diagnostic (VALIDATION): per activity group, model
NDCG@10 vs ItemPop NDCG@10. Intervention: for the most active group only,
    score(u, i) = z_u(model score) + beta * z(item train popularity)
(z_u = standardised over items for that user), beta chosen on that group's VALIDATION NDCG@10.
TEST: per-group NDCG@10 before/after; paired bootstrap 95% CI of the group-3 change; overall change.
Outputs: results/<dataset>/<timestamp>_headroom/ (output.txt, config.yaml, headroom.json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from leveldown.data import ROOT, activity_groups, load  # noqa: E402
from leveldown.models import EASE, IALS, ItemPop, evaluate, interaction_matrix  # noqa: E402
from leveldown.runlog import Run  # noqa: E402
from run_gate import tune  # noqa: E402

BETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


class Blend:
    """Scores of `base`, with z_u(base) + beta * z(pop) for users in `target` (others: base unchanged)."""

    def __init__(self, base, pop, target, beta):
        self.base, self.beta = base, beta
        self.zpop = (pop - pop.mean()) / (pop.std() or 1.0)
        self.target = set(int(u) for u in target)

    def scores(self, users):
        S = self.base.scores(users).astype(float)
        if self.beta == 0:
            return S
        for j, u in enumerate(users):
            if int(u) in self.target:
                row = S[j]
                S[j] = (row - row.mean()) / (row.std() or 1.0) + self.beta * self.zpop
        return S


def paired_ci(a, b, B=1000, seed=0):
    d = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main(dataset, cfg, log=print):
    k, B, seed = cfg["eval"]["k"], cfg["eval"]["bootstrap"], cfg["seed"]
    split = load(dataset, cfg["datasets"][dataset]["raw_dir"], cfg["min_interactions"],
                 cfg.get("tie_break", "item"), cfg["seed"])
    groups, _ = activity_groups(split.train)
    pop = interaction_matrix(split.train, split.n_items).sum(axis=0)
    hi_g = int(groups.max())
    target = np.where(groups == hi_g)[0]
    log(f"{dataset}: users={split.n_users} items={split.n_items}  target group {hi_g} ({len(target)} users)")

    itempop = ItemPop().fit(split.train, split.n_items)
    _, pop_val = evaluate(itempop, split, "val", k)
    log("\n== tuning (as run_gate.py)")
    models = {"ials": tune("ials", IALS, cfg["grids"]["ials"], split, k, log)[0],
              "ease": tune("ease", EASE, cfg["grids"]["ease"], split, k, log)[0]}

    out = {}
    for name, m in models.items():
        _, base_val = evaluate(m, split, "val", k)
        log(f"\n===== {name}")
        log("headroom diagnostic (VALIDATION NDCG@10): group: model vs ItemPop")
        for g in range(4):
            mg, pg = base_val[groups == g].mean(), pop_val[groups == g].mean()
            log(f"   group {g}: {mg:.4f} vs {pg:.4f}  {'HEADROOM' if pg > mg else ''}")

        best_beta, best_v = 0.0, -1.0
        for beta in BETAS:
            _, v = evaluate(Blend(m, pop, target, beta), split, "val", k)
            val_g = v[target].mean()
            log(f"   beta {beta:<5}: validation NDCG@10 of group {hi_g} = {val_g:.4f}")
            if val_g > best_v:
                best_beta, best_v = beta, val_g
        log(f"   -> selected beta {best_beta} (on validation, group {hi_g} only)")

        _, t0 = evaluate(m, split, "test", k)
        _, t1 = evaluate(Blend(m, pop, target, best_beta), split, "test", k)
        log(f"TEST NDCG@10 by group (before -> after)")
        for g in range(4):
            mm = groups == g
            log(f"   group {g}: {t0[mm].mean():.4f} -> {t1[mm].mean():.4f}")
        d, lo, hi = paired_ci(t0[target], t1[target], B, seed)
        dall = t1.mean() - t0.mean()
        ok = lo > 0 and d > 0.005 and dall >= 0
        log(f"   group {hi_g} change {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | overall {t0.mean():.4f} -> {t1.mean():.4f} "
            f"({dall:+.4f})  ->  {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
        out[name] = {"beta": best_beta, "group_before": [float(t0[groups == g].mean()) for g in range(4)],
                     "group_after": [float(t1[groups == g].mean()) for g in range(4)],
                     "target_change": [d, lo, hi], "overall_change": float(dall), "supported": bool(ok)}
    log(f"\n== L2 DECISION: iALS {'SUPPORTED' if out['ials']['supported'] else 'NOT SUPPORTED'} | "
        f"EASE {'SUPPORTED' if out['ease']['supported'] else 'NOT SUPPORTED'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml-100k", "ml-1m"])
    ap.add_argument("--tie-break", default="item", choices=["item", "random"])
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    cfg["results_dir"] = f"results/{args.dataset}"
    cfg["tie_break"] = args.tie_break
    with Run(cfg, "headroom") as run:
        res = main(args.dataset, cfg)
        run.path("headroom.json").write_text(json.dumps(res, indent=2, default=float))