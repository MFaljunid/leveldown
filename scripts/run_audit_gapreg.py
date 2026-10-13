"""C1 audit, intervention 3 (in-processing regulariser): BPR-MF with a gap penalty on the groups' training
losses,  L = BPR + lambda * |L_adv - L_dis|.

    python scripts/run_audit_gapreg.py --dataset ml-100k
    python scripts/run_audit_gapreg.py --dataset ml-1m --groups gender

Protocol: temporal leave-one-out with RANDOM tie-breaking, full ranking, NDCG@10.
Base BPR-MF tuned on validation (grid grids.bpr). Advantaged / disadvantaged group = best / worst group of the
BASE model on VALIDATION; the penalty acts on the training-loss gap between these two groups.
lambda in --lambdas (the first must be 0 = base). For every lambda, test per-group NDCG and the decomposition
of the gap change (same definitions as run_audit_reweight.py):
  closure  = |gap_base| - |gap_w|                       (> 0: narrower gap)
  up       = max(0, d_dis)   down = max(0, -d_adv)      (d_* = change of group mean vs base)
  LD share = down / (up + down)                         (share of the closure obtained by lowering
                                                         the advantaged group)
Practitioner selection (as typically done): lambda with the smallest VALIDATION gap among settings whose
overall validation NDCG is >= 95% of the base model's.
Verdict for the selected w (paired bootstrap 95% CIs on test):
  LEVELING DOWN if the gap narrows, d_adv < 0 (CI) and d_dis is not > 0 (CI)
  LEVELING UP   if the gap narrows, d_dis > 0 (CI) and d_adv is not < 0 (CI)
  MIXED         if both CIs exclude 0 in their direction;  NO EFFECT otherwise.
Outputs: results/<dataset>/<timestamp>_audit_gapreg_<groups>/ (output.txt, config.yaml, audit.json)
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
from leveldown.models import BPRMF, evaluate  # noqa: E402
from leveldown.runlog import Run  # noqa: E402
from run_gate import load_split, tune  # noqa: E402


def paired_ci(a, b, B=1000, seed=0):
    d = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def verdict(closure, dadv, ddis):
    down_sig = dadv[2] < 0
    up_sig = ddis[1] > 0
    if closure <= 0:
        return "NO NARROWING"
    if down_sig and up_sig:
        return "MIXED"
    if down_sig:
        return "LEVELING DOWN"
    if up_sig:
        return "LEVELING UP"
    return "NO SIGNIFICANT EFFECT"


def main(dataset, cfg, weights, grouping="activity", model_seed=None, log=print):
    k, B, seed = cfg["eval"]["k"], cfg["eval"]["bootstrap"], cfg["seed"]
    split = load_split(dataset, cfg)
    groups, gnames = group_labels(split, grouping, dataset, cfg["datasets"][dataset]["raw_dir"])
    G = len(gnames)
    log(f"{dataset}: users={split.n_users} items={split.n_items}  groups={grouping} "
        f"({', '.join(f'{n}: {int((groups == i).sum())}' for i, n in enumerate(gnames))})  "
        f"random tie-breaking, FULL RANKING, NDCG@{k}")
    log("\n== tuning base BPR-MF on validation")
    _, hp_b = tune("bpr", BPRMF, cfg["grids"]["bpr"], split, k, log, model_seed)
    bases = {"bpr": (BPRMF, hp_b)}
    out = {}
    for name, (cls, hp) in bases.items():
        _, v0 = evaluate(cls(**hp).fit(split.train, split.n_items), split, "val", k)
        gv = [v0[groups == g].mean() for g in range(G)]
        adv, dis = int(np.argmax(gv)), int(np.argmin(gv))
        log(f"\n===== {name} {hp}: advantaged group {gnames[adv]}, disadvantaged group {gnames[dis]} (validation)")
        rows, tests = [], {}
        for w in weights:
            m = cls(**hp, gap_lambda=float(w), gap_groups=(groups, adv, dis)).fit(split.train, split.n_items)
            _, v = evaluate(m, split, "val", k)
            _, t = evaluate(m, split, "test", k)
            tests[w] = t
            rows.append({"w": w, "val_all": v.mean(), "val_gap": v[groups == dis].mean() - v[groups == adv].mean(),
                         "test_groups": [t[groups == g].mean() for g in range(G)], "test_all": t.mean()})
        t0 = tests[weights[0]]
        gap0 = t0[groups == dis].mean() - t0[groups == adv].mean()
        log(f"{'lambda':>6}{'val all':>9}{'val gap':>9}" + "".join(f"{'test ' + n:>9}" for n in gnames)
            + f"{'test all':>9}{'gap':>9}{'closure':>9}{'d_adv':>9}{'d_dis':>9}{'LD share':>9}")
        for r in rows:
            t = tests[r["w"]]
            gap = t[groups == dis].mean() - t[groups == adv].mean()
            dadv = t[groups == adv].mean() - t0[groups == adv].mean()
            ddis = t[groups == dis].mean() - t0[groups == dis].mean()
            closure = abs(gap0) - abs(gap)
            up, down = max(0.0, ddis), max(0.0, -dadv)
            ld = down / (up + down) if closure > 0 and (up + down) > 0 else float("nan")
            r.update(gap=gap, closure=closure, d_adv=dadv, d_dis=ddis, ld_share=ld)
            log(f"{r['w']:>6g}{r['val_all']:>9.4f}{r['val_gap']:>+9.4f}" + "".join(f"{x:>9.4f}" for x in r["test_groups"])
                + f"{r['test_all']:>9.4f}{gap:>+9.4f}{closure:>+9.4f}{dadv:>+9.4f}{ddis:>+9.4f}"
                + (f"{ld:>9.0%}" if ld == ld else f"{'-':>9}"))
        base_val = rows[0]["val_all"]
        eligible = [r for r in rows if r["val_all"] >= 0.95 * base_val]
        sel = min(eligible, key=lambda r: abs(r["val_gap"]))
        t = tests[sel["w"]]
        dadv = paired_ci(t0[groups == adv], t[groups == adv], B, seed)
        ddis = paired_ci(t0[groups == dis], t[groups == dis], B, seed)
        dall = paired_ci(t0, t, B, seed)
        vd = verdict(sel["closure"], dadv, ddis)
        log(f"selected lambda = {sel['w']:g} (smallest validation gap with >= 95% of base validation NDCG)")
        log(f"   test: d_adv {dadv[0]:+.4f} [{dadv[1]:+.4f}, {dadv[2]:+.4f}] | d_dis {ddis[0]:+.4f} "
            f"[{ddis[1]:+.4f}, {ddis[2]:+.4f}] | d_all {dall[0]:+.4f} [{dall[1]:+.4f}, {dall[2]:+.4f}] | "
            f"gap {gap0:+.4f} -> {sel['gap']:+.4f}  ->  {vd}")
        out[name] = {"hp": hp, "lambdas": weights, "groups": gnames, "adv": gnames[adv], "dis": gnames[dis], "rows": rows, "selected_w": sel["w"],
                     "d_adv": dadv, "d_dis": ddis, "d_all": dall, "verdict": vd}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml-100k", "ml-1m", "gowalla"])
    ap.add_argument("--lambdas", default="0,0.05,0.1,0.2,0.5")
    ap.add_argument("--groups", default="activity", choices=["activity", "gender"])
    ap.add_argument("--seed", type=int, default=None, help="model training seed (default: the models' own default, 0)")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    cfg["results_dir"] = f"results/{args.dataset}"
    weights = [float(x) for x in args.lambdas.split(",")]
    if weights[0] != 0.0:
        raise SystemExit("--lambdas must start with 0 (the base model)")
    with Run(cfg, f"audit_gapreg_{args.groups}" + (f"_s{args.seed}" if args.seed is not None else "")) as run:
        res = main(args.dataset, cfg, weights, args.groups, args.seed)
        res["model_seed"] = args.seed
        run.path("audit.json").write_text(json.dumps(res, indent=2, default=float))