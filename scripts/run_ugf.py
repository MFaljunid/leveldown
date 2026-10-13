"""C1 audit, intervention 2: UGF re-ranking (Li et al., WWW 2021) — ORACLE vs LEAK-FREE.

    python scripts/run_ugf.py --dataset ml-100k

UGF: from each user's base top-N list, select K items maximising the sum of base scores subject to
|mean M(adv) - mean M(dis)| <= eps. Settings of the paper: advantaged = top 5% users by #train
interactions, eps = half of the base gap. The paper assumes the active group is the better-served one;
here the better-served group is determined from the data (test for ORACLE, validation for LEAK-FREE),
and "adv" below always means the better-served group. With one held-out item per user, F1@K = 2*hit/(1+K), so the
constraint is a constraint on HR@K and the 0-1 programme is solved exactly by a greedy over unit
actions (promote a disadvantaged user's held-out item into the top-K at cost s[K-1]-s[r]; demote an
advantaged user's held-out item out of the top-K at cost s[r]-s[K]; cheapest per unit of gap first).
  ORACLE    : constraint computed with the TEST labels, as in the public implementation
              (src/model.py builds the constraint from the 'label' column of the test rank file).
  LEAK-FREE : test labels unknown. Without labels the base top-K is already each user's best list by
              the model's own scores, so no list can be improved; the only way to meet the constraint is
              to degrade the advantaged group. Implemented as: advantaged users' lists skip their first r
              items, r = smallest value whose VALIDATION gap is <= eps_val (half of the base validation gap).
Outputs: results/<dataset>/<timestamp>_ugf/ (output.txt, config.yaml, ugf.json)
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
from leveldown.runlog import Run  # noqa: E402
from run_audit_reweight import paired_ci, verdict  # noqa: E402
from run_gate import build_backbones, load_split  # noqa: E402


def topn(model, split, which, N, chunk=512):
    """Per user: indices and scores of the top-N items (known positives excluded) and the rank of the
    held-out item within it (N if outside)."""
    target = split.test if which == "test" else split.val
    other = split.val if which == "test" else split.test
    n = split.n_users
    idx = np.zeros((n, N), int)
    sc = np.zeros((n, N))
    rank = np.full(n, N)
    for s in range(0, n, chunk):
        users = np.arange(s, min(s + chunk, n))
        S = model.scores(users).astype(float)
        for j, u in enumerate(users):
            row = S[j]
            row[np.append(split.train[u], other[u]).astype(int)] = -np.inf
            top = np.argpartition(-row, N)[:N]
            top = top[np.argsort(-row[top], kind="stable")]
            idx[u], sc[u] = top, row[top]
            hit = np.where(top == target[u])[0]
            if len(hit):
                rank[u] = int(hit[0])
    return idx, sc, rank


def metrics_from_positions(pos, K):
    """pos: position of the held-out item in the final top-K list (K or more = miss)."""
    hr = (pos < K).astype(float)
    nd = np.where(pos < K, 1.0 / np.log2(np.minimum(pos, K - 1) + 2), 0.0)
    return hr, nd


def oracle_ugf(sc, rank, adv_mask, K, eps):
    """Greedy exact solution with TEST labels. Returns final positions of the held-out item."""
    n = len(rank)
    pos = np.where(rank < K, rank, K + 1000)
    n_adv, n_dis = adv_mask.sum(), (~adv_mask).sum()
    gap = (pos[adv_mask] < K).mean() - (pos[~adv_mask] < K).mean()
    actions = []
    for u in range(n):
        r = rank[u]
        if not adv_mask[u] and K <= r < sc.shape[1]:         # promote into top-K
            actions.append(((sc[u, K - 1] - sc[u, r]) * n_dis, u, "promote"))
        if adv_mask[u] and r < K:                            # demote out of top-K
            actions.append(((sc[u, r] - sc[u, K]) * n_adv, u, "demote"))
    actions.sort(key=lambda a: a[0])
    applied = 0
    for _, u, kind in actions:
        if gap <= eps:
            break
        if kind == "promote":
            # the held-out item enters; within the final list items are ranked by original score,
            # so it lands at the last position (its score is the lowest of the selected items)
            pos[u] = K - 1
            gap -= 1.0 / n_dis
        else:
            pos[u] = K + 1000
            gap -= 1.0 / n_adv
        applied += 1
    return pos, gap, applied


def shifted_positions(rank, adv_mask, r, K):
    """Advantaged users' lists = their top-N items r .. r+K-1 (re-ranked by score: relative order kept)."""
    pos = np.where(rank < K, rank, K + 1000).astype(int)
    a = adv_mask & (rank >= r) & (rank < r + K)
    pos[adv_mask] = K + 1000
    pos[a] = rank[a] - r
    return pos


def main(dataset, cfg, top_frac, N, grouping="activity", model_seed=None, names=None, log=print):
    K, B, seed = cfg["eval"]["k"], cfg["eval"]["bootstrap"], cfg["seed"]
    split = load_split(dataset, cfg)
    if grouping == "activity":
        n_train = np.array([len(t) for t in split.train])
        adv_mask = n_train >= np.quantile(n_train, 1 - top_frac)
        gname = f"active-{top_frac:.0%}"
    else:
        lab, _ = group_labels(split, "gender", dataset, cfg["datasets"][dataset]["raw_dir"])
        adv_mask = lab == 1
        gname = "male"
    log(f"{dataset}: users={split.n_users}  group A = {gname} ({adv_mask.sum()} users) vs rest, "
        f"K={K}, N={N}, random tie-breaking, full ranking")
    log("\n== tuning base models (as run_gate.py)")
    models = {name: m for name, (_, _, m) in build_backbones(split, cfg, K, log, model_seed, names).items()}
    out = {}
    for name, m in models.items():
        _, sc_t, rk_t = topn(m, split, "test", N)
        _, sc_v, rk_v = topn(m, split, "val", N)
        base_pos = np.where(rk_t < K, rk_t, K + 1000)
        hr0, nd0 = metrics_from_positions(base_pos, K)
        vhr0 = (np.where(rk_v < K, rk_v, K + 1000) < K).astype(float)
        a_t = hr0[adv_mask].mean() >= hr0[~adv_mask].mean()      # is the active group better on TEST?
        a_v = vhr0[adv_mask].mean() >= vhr0[~adv_mask].mean()    # ... on VALIDATION?
        log(f"\n===== {name}: base test HR@{K} {gname} {hr0[adv_mask].mean():.4f} rest {hr0[~adv_mask].mean():.4f} "
            f"overall {hr0.mean():.4f}; better-served group: {gname if a_t else 'rest'} (test), "
            f"{gname if a_v else 'rest'} (validation)")
        res = {}
        runs = []
        # ORACLE (test labels)
        better_t = adv_mask if a_t else ~adv_mask
        g0 = hr0[better_t].mean() - hr0[~better_t].mean()
        pos_o, _, n_act = oracle_ugf(sc_t, rk_t, better_t, K, abs(g0) / 2)
        runs.append(("ORACLE (test labels, as public code)", pos_o, better_t))
        # LEAK-FREE (validation)
        better_v = adv_mask if a_v else ~adv_mask
        vg0 = vhr0[better_v].mean() - vhr0[~better_v].mean()
        r_sel = 0
        for r in range(0, N - K + 1):
            vh = (shifted_positions(rk_v, better_v, r, K) < K).astype(float)
            if abs(vh[better_v].mean() - vh[~better_v].mean()) <= abs(vg0) / 2:
                r_sel = r
                break
        runs.append((f"LEAK-FREE (validation; better-served lists skip top {r_sel})",
                     shifted_positions(rk_t, better_v, r_sel, K), better_v))
        for label, pos, better in runs:
            hr, nd = metrics_from_positions(pos, K)
            gap0 = hr0[better].mean() - hr0[~better].mean()
            gap = hr[better].mean() - hr[~better].mean()
            dadv = paired_ci(hr0[better], hr[better], B, seed)
            ddis = paired_ci(hr0[~better], hr[~better], B, seed)
            dall = paired_ci(hr0, hr, B, seed)
            vd = verdict(abs(gap0) - abs(gap), dadv, ddis)
            log(f"  {label}")
            log(f"     HR@{K}: better {hr0[better].mean():.4f}->{hr[better].mean():.4f}  worse "
                f"{hr0[~better].mean():.4f}->{hr[~better].mean():.4f}  overall {hr0.mean():.4f}->{hr.mean():.4f} "
                f"(gap {gap0:+.4f} -> {gap:+.4f}) | NDCG overall {nd0.mean():.4f}->{nd.mean():.4f}")
            log(f"     d_better {dadv[0]:+.4f} [{dadv[1]:+.4f}, {dadv[2]:+.4f}] | d_worse {ddis[0]:+.4f} "
                f"[{ddis[1]:+.4f}, {ddis[2]:+.4f}] | d_all {dall[0]:+.4f} [{dall[1]:+.4f}, {dall[2]:+.4f}]  ->  {vd}")
            res[label.split(" ")[0]] = {"hr_better": hr[better].mean(), "hr_worse": hr[~better].mean(),
                                        "hr_all": hr.mean(), "gap_base": gap0, "gap": gap, "d_better": dadv,
                                        "d_worse": ddis, "d_all": dall, "verdict": vd}
        res["oracle_actions"], res["leakfree_shift"] = int(n_act), int(r_sel)
        out[name] = res
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml-100k", "ml-1m", "gowalla"])
    ap.add_argument("--top-frac", type=float, default=0.05, help="advantaged share (paper: 5%%)")
    ap.add_argument("--N", type=int, default=100, help="length of the base list to re-rank from")
    ap.add_argument("--groups", default="activity", choices=["activity", "gender"])
    ap.add_argument("--seed", type=int, default=None, help="model training seed (default: the models' own default, 0)")
    ap.add_argument("--backbones", default=None, help="comma list, e.g. ials (default: all in the config)")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    cfg["results_dir"] = f"results/{args.dataset}"
    with Run(cfg, f"ugf_{args.groups}" + (f"_s{args.seed}" if args.seed is not None else "")) as run:
        res = main(args.dataset, cfg, args.top_frac, args.N, args.groups, args.seed,
                   args.backbones.split(",") if args.backbones else None)
        res["model_seed"] = args.seed
        run.path("ugf.json").write_text(json.dumps(res, indent=2, default=float))