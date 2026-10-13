"""C2' confirmation: is the activity-group accuracy gap explained by the popularity of the held-out item?

    python scripts/run_gate.py --dataset ml-100k
    python scripts/run_gate.py --dataset ml-1m

Temporal leave-one-out, full ranking. Models tuned on the validation item (NDCG@10): iALS (primary),
EASE (secondary), ItemPop (reference). Then, for each model, the gap between the most- and the
least-accurate activity group is decomposed with the test item's train popularity.
Outputs: results/<dataset>/<timestamp>_gate/ (output.txt, config.yaml, gate.json)
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from leveldown.analysis import decompose  # noqa: E402
from leveldown.data import ROOT, activity_groups, load  # noqa: E402
from leveldown.models import EASE, IALS, ItemKNN, ItemPop, evaluate, item_popularity  # noqa: E402
from leveldown.runlog import Run  # noqa: E402


def grid(d):
    keys = list(d)
    return [dict(zip(keys, v)) for v in itertools.product(*(d[k] for k in keys))]


def tune(name, cls, g, split, k, log, seed=None):
    """Grid search on the validation item. If `seed` is given and the model takes a seed, it is added to
    every hyper-parameter dict (so all later refits of the selected model use the same training seed)."""
    import inspect
    takes_seed = "seed" in inspect.signature(cls.__init__).parameters
    best = None
    for hp in grid(g):
        if seed is not None and takes_seed:
            hp = dict(hp, seed=int(seed))
        m = cls(**hp).fit(split.train, split.n_items)
        _, nd = evaluate(m, split, "val", k)
        log(f"  {name} {hp}: val NDCG@{k} {nd.mean():.4f}")
        if best is None or nd.mean() > best[0]:
            best = (nd.mean(), hp, m)
    log(f"  -> selected {name} {best[1]} (val NDCG@{k} {best[0]:.4f})")
    return best[2], best[1]


BACKBONES = {"ials": IALS, "ease": EASE, "knn": ItemKNN}


def build_backbones(split, cfg, k, log, seed=None, names=None):
    """Tune every configured backbone on validation. Returns {name: (cls, hp, fitted model)}."""
    out = {}
    for name in (names or cfg.get("backbones", ["ials", "ease"])):
        if name == "ease" and split.n_items > cfg.get("ease_max_items", 12000):
            log(f"  ease skipped: {split.n_items} items > ease_max_items ({cfg.get('ease_max_items', 12000)})")
            continue
        model, hp = tune(name, BACKBONES[name], cfg["grids"][name], split, k, log, seed)
        out[name] = (BACKBONES[name], hp, model)
    return out


def load_split(dataset, cfg, tie_break="random"):
    d = cfg["datasets"][dataset]
    return load(dataset, d["raw_dir"], cfg["min_interactions"], tie_break, cfg["seed"], d.get("core"))


def main(dataset, cfg, log=print):
    k, B, seed = cfg["eval"]["k"], cfg["eval"]["bootstrap"], cfg["seed"]
    split = load_split(dataset, cfg, cfg.get("tie_break", "random"))
    groups, n_train = activity_groups(split.train)
    pop = item_popularity(split.train, split.n_items)
    tpop = np.log1p(pop[split.test])
    log(f"{dataset}: users={split.n_users} items={split.n_items}  temporal LOO, FULL RANKING, k={k}")

    log("\n== tuning on the validation item")
    models = {"itempop": (ItemPop().fit(split.train, split.n_items), {})}
    for name, (_, hp, m) in build_backbones(split, cfg, k, log).items():
        models[name] = (m, hp)

    log("\n== TEST by activity group (0 = least active)")
    top10 = np.quantile(np.log1p(pop), 0.9)
    log(f"{'group':>6}{'#train':>8}{'test-item log pop':>19}{'test in top-10%':>17}")
    for gg in range(4):
        m = groups == gg
        log(f"{gg:>6}{n_train[m].mean():>8.1f}{tpop[m].mean():>19.3f}{(tpop[m] >= top10).mean():>17.1%}")

    res = {}
    log(f"\n{'model':<9}{'HR@k':>8}{'NDCG@k':>9}" + "".join(f"{'NDCG g' + str(g):>10}" for g in range(4)))
    per = {}
    for name, (m, hp) in models.items():
        hr, nd = evaluate(m, split, "test", k)
        per[name] = (hr, nd)
        gm = [nd[groups == g].mean() for g in range(4)]
        log(f"{name:<9}{hr.mean():>8.4f}{nd.mean():>9.4f}" + "".join(f"{v:>10.4f}" for v in gm))
        res[name] = {"hp": hp, "HR": hr.mean(), "NDCG": nd.mean(), "NDCG_by_group": gm}

    log("\n== decomposition of the gap (least- vs most-accurate group) by test-item popularity")
    for name in [n for n in models if n != "itempop"]:
        hr, nd = per[name]
        gm = [nd[groups == g].mean() for g in range(4)]
        lo, hi = int(np.argmax(gm)), int(np.argmin(gm))
        for mname, y in (("NDCG", nd), ("HR", hr)):
            d = decompose(y, groups, tpop, lo, hi, B=B, seed=seed)
            log(f"  {name:<5}{mname:<5} group {hi} - group {lo}: raw {d['raw']:+.4f} | adj {d['adjusted']:+.4f} "
                f"(share {d['share']:.1%} [{d['share_ci'][0]:.1%}, {d['share_ci'][1]:.1%}]) | matched "
                f"{d['matched']:+.4f} (ratio {abs(d['matched']) / abs(d['raw']):.0%}, coverage {d['coverage']:.0%})")
            res[name][f"decomp_{mname}"] = dict(d, lo=lo, hi=hi)

    g = cfg["gate"]
    p = res[g["primary_model"]]["decomp_NDCG"]
    beats_pop = res[g["primary_model"]]["NDCG"] > res["itempop"]["NDCG"]
    ok = beats_pop and p["share"] >= g["min_share"] and abs(p["matched"]) <= g["max_matched_ratio"] * abs(p["raw"])
    log(f"\n== C2' DECISION ({g['primary_model']}, NDCG@{k}): beats ItemPop {beats_pop} | share {p['share']:.1%} "
        f"(need >= {g['min_share']:.0%}) | matched/raw {abs(p['matched']) / abs(p['raw']):.0%} "
        f"(need <= {g['max_matched_ratio']:.0%})  ->  {'CONFIRMED' if ok else 'NOT CONFIRMED'}")
    res["decision"] = {"confirmed": bool(ok), "beats_itempop": bool(beats_pop)}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml-100k", "ml-1m", "gowalla"])
    ap.add_argument("--tie-break", default="random", choices=["item", "random"],
                    help="random = the protocol from L3b on (item = original L1 runs)")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    cfg["results_dir"] = f"results/{args.dataset}"
    cfg["tie_break"] = args.tie_break
    with Run(cfg, "gate") as run:
        out = main(args.dataset, cfg)
        run.path("gate.json").write_text(json.dumps(out, indent=2, default=float))