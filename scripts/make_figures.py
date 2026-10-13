"""Paper figures and the protocol / target-difficulty tables (run after aggregate.py).

    python scripts/make_figures.py

Inputs (latest run per dataset): results/<ds>/*_audit_* (via results/paper/audit_all_settings.csv),
*_ugf* (results/paper/ugf.csv), *_gate (gate.json, random tie-breaking only), *_protocol (protocol.json).
Outputs in results/paper/:
  F1_teaser.pdf                 ML-1M gap regulariser (seed 0): advantaged / disadvantaged / overall vs lambda
  F2_leveling_map.pdf           every narrowing setting of every audit: change of the advantaged group (x)
                                vs change of the disadvantaged group (y); UGF leak-free and ORACLE added
  F3_ugf.pdf                    UGF: change of the worse-served group, ORACLE vs LEAK-FREE
  F4_personalization_gain.pdf   per activity group: iALS and ItemPop test NDCG@10 (gate runs)
  T6_protocol.tex               relative gap and worst group under tie-breaking x evaluation protocol
  T7_target.tex                 test-item popularity decomposition of the activity gap
"""
import csv
import json
import re
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "paper"
DS = {"ml-100k": "ML-100K", "ml-1m": "ML-1M", "gowalla": "Gowalla"}
COL = {"ml-100k": "#1b9e77", "ml-1m": "#d95f02", "gowalla": "#7570b3"}
MARK = {"audit_reweight": "o", "audit_gapreg": "s"}
INT = {"audit_reweight": "Reweighting", "audit_gapreg": "Loss-gap penalty"}
BB = {"ials": "iALS", "ease": "EASE", "knn": "ItemKNN", "bpr": "BPR-MF"}


def latest(ds, pattern, fname, need_random=False):
    best = None
    d = RES / ds
    if not d.exists():
        return None
    for run in sorted(d.iterdir()):
        if re.match(r"^\d{8}-\d{6}_" + pattern + "$", run.name) and (run / fname).exists():
            if need_random:
                c = run / "config.yaml"
                tb = yaml.safe_load(c.read_text(encoding="utf-8")).get("tie_break", "item") if c.exists() else "item"
                if tb != "random":
                    continue
            best = run
    return None if best is None else json.loads((best / fname).read_text())


def read_csv(name):
    p = OUT / name
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def f1(plt):
    rows = [r for r in read_csv("audit_all_settings.csv")
            if r["dataset"] == "ml-1m" and r["intervention"] == "audit_gapreg" and r["attribute"] == "activity"
            and r["seed"] == "0"]
    if not rows:
        return "F1 skipped (no ML-1M gap-regulariser run)"
    rows.sort(key=lambda r: float(r["setting"]))
    x = np.arange(len(rows))
    adv = [float(r["test_act0"]) for r in rows]
    dis = [float(r["test_act3"]) for r in rows]
    allv = [float(r["test_all"]) for r in rows]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(x, adv, "-o", color="#2166ac", label="least active (advantaged)")
    ax.plot(x, dis, "-o", color="#b2182b", label="most active (disadvantaged)")
    ax.plot(x, allv, "--", color="k", lw=1, label="all users")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(r['setting']):g}" for r in rows])
    ax.set_xlabel(r"gap-penalty weight $\lambda$")
    ax.set_ylabel("test NDCG@10")
    i = 1 if len(rows) > 1 else 0
    ax.plot([x[i]], [dis[i]], "o", ms=11, mfc="none", mec="k", mew=0.8)
    ax.plot([x[i]], [adv[i]], "o", ms=11, mfc="none", mec="k", mew=0.8)
    note = (f"\u03bb = {float(rows[i]['setting']):g}: gap {adv[0] - dis[0]:.4f} \u2192 {adv[i] - dis[i]:.4f}, "
            f"disadvantaged {dis[0]:.4f} \u2192 {dis[i]:.4f}")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("ML-1M, BPR-MF with a loss-gap penalty\n" + note, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "F1_teaser.pdf")
    plt.close(fig)
    return "F1_teaser.pdf"


def f2(plt):
    """Single-column figure (3.4 in wide): color = dataset, marker = intervention, two compact legends
    below the plot, no title (the caption describes it)."""
    from matplotlib.lines import Line2D
    rows = read_csv("audit_all_settings.csv")
    if not rows:
        return "F2 skipped"
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    for r in rows:
        if float(r["closure"]) <= 0:
            continue                                  # only settings that narrow the gap
        ds, kind = r["dataset"], r["intervention"]
        ax.scatter(float(r["d_adv"]), float(r["d_dis"]), s=12, c=COL[ds], marker=MARK[kind], alpha=0.8,
                   edgecolors="none", zorder=3)
    for r in read_csv("ugf.csv"):
        ds = r["dataset"]
        ax.scatter(float(r["LEAK-FREE_d_better"]), float(r["LEAK-FREE_d_worse"]), s=26, c=COL[ds], marker="^",
                   edgecolors="k", linewidths=0.5, zorder=4)
        ax.scatter(float(r["ORACLE_d_better"]), float(r["ORACLE_d_worse"]), s=26, facecolors="none",
                   edgecolors=COL[ds], linewidths=1.0, marker="^", zorder=4)
    ax.axhline(0, color="grey", lw=0.6, zorder=2)
    ax.axvline(0, color="grey", lw=0.6, zorder=2)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    xpad, ypad = 0.04 * (xl[1] - xl[0]), 0.06 * (yl[1] - yl[0])
    ax.set_xlim(xl[0] - xpad, max(xl[1], 0) + xpad)
    ax.set_ylim(yl[0] - ypad, yl[1] + ypad)
    X0, X1 = ax.get_xlim()
    Y0, Y1 = ax.get_ylim()
    ax.axhspan(0, Y1, color="#e0f3db", alpha=0.6, zorder=0)
    ax.text(X0 + 0.03 * (X1 - X0), Y1 - 0.03 * (Y1 - Y0), "worse-served group gains", fontsize=7, va="top",
            color="#4d4d4d")
    ax.text(X0 + 0.03 * (X1 - X0), Y0 + 0.03 * (Y1 - Y0), "both groups lose", fontsize=7, va="bottom",
            color="#4d4d4d")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("change of the better-served group", fontsize=8)
    ax.set_ylabel("change of the worse-served group", fontsize=8)
    ds_handles = [Line2D([], [], ls="", marker="o", ms=5, color=COL[d], label=DS[d]) for d in DS]
    kind_handles = [Line2D([], [], ls="", marker="o", ms=4.5, color="#555555", label="reweighting"),
                    Line2D([], [], ls="", marker="s", ms=4.5, color="#555555", label="loss-gap penalty"),
                    Line2D([], [], ls="", marker="^", ms=5.5, color="#555555", markeredgecolor="k",
                           label="UGF leak-free"),
                    Line2D([], [], ls="", marker="^", ms=5.5, color="none", markeredgecolor="#555555",
                           label="UGF oracle (test labels)")]
    leg1 = fig.legend(handles=ds_handles, loc="lower center", ncol=3, fontsize=7, frameon=False,
                      bbox_to_anchor=(0.55, 0.075), handletextpad=0.2, columnspacing=0.9)
    fig.legend(handles=kind_handles, loc="lower center", ncol=2, fontsize=7, frameon=False,
               bbox_to_anchor=(0.55, 0.0), handletextpad=0.2, columnspacing=0.9)
    fig.add_artist(leg1)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(OUT / "F2_leveling_map.pdf")
    plt.close(fig)
    return "F2_leveling_map.pdf"


def f3(plt):
    rows = read_csv("ugf.csv")
    if not rows:
        return "F3 skipped"
    labels = [f"{DS[r['dataset']]}\n{BB.get(r['backbone'], r['backbone'])}\n{r['attribute']}" for r in rows]
    o = [float(r["ORACLE_d_worse"]) for r in rows]
    lf = [float(r["LEAK-FREE_d_worse"]) for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(rows)), 3.8))
    ax.bar(x - 0.2, o, 0.4, label="ORACLE (test labels, as public code)", color="#fdae61")
    ax.bar(x + 0.2, lf, 0.4, label="LEAK-FREE", color="#2c7bb6")
    ax.plot(x + 0.2, lf, "o", ms=3, color="#2c7bb6")
    top = max(max(o), 1e-9)
    for i, r in enumerate(rows):
        lo, hi = (float(v) for v in r["ORACLE_d_worse_ci"].strip("[]").split(","))
        if lo > 0:
            ax.text(x[i] - 0.2, o[i] + 0.015 * top, "*", ha="center", fontsize=9)
        ax.text(x[i] + 0.2, lf[i] + 0.015 * top, f"{lf[i]:.3f}" if abs(lf[i]) >= 5e-4 else "0", ha="center",
                fontsize=5.5, color="#2c7bb6")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("change of the worse-served group\n(test HR@10)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("UGF re-ranking: the 'lift' requires test labels (* = 95% CI excludes 0)", fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.28)
    fig.savefig(OUT / "F3_ugf.pdf")
    plt.close(fig)
    return "F3_ugf.pdf"


def f4(plt):
    data = {ds: latest(ds, "gate", "gate.json", need_random=True) for ds in DS}
    data = {k: v for k, v in data.items() if v}
    if not data:
        return "F4 skipped (no gate runs with random tie-breaking)"
    fig, axes = plt.subplots(1, len(data), figsize=(3.0 * len(data), 2.8), squeeze=False)
    for ax, (ds, g) in zip(axes[0], data.items()):
        ials = g["ials"]["NDCG_by_group"]
        pop = g["itempop"]["NDCG_by_group"]
        x = np.arange(len(ials))
        ax.bar(x - 0.2, ials, 0.4, label="iALS", color="#5e3c99")
        ax.bar(x + 0.2, pop, 0.4, label="ItemPop", color="#b2abd2")
        for i in x:
            ax.text(i, max(ials[i], pop[i]) + 0.04 * max(ials), f"{ials[i] / pop[i]:.1f}×", ha="center", fontsize=6)
        ax.set_ylim(0, max(max(ials), max(pop)) * 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels(["Q1\n(least)", "Q2", "Q3", "Q4\n(most)"], fontsize=7)
        ax.set_xlabel("activity quartile", fontsize=7)
        ax.set_title(DS[ds], fontsize=8)
        ax.set_ylabel("test NDCG@10", fontsize=7)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("Personalization gain over popularity shrinks with activity", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "F4_personalization_gain.pdf")
    plt.close(fig)
    return "F4_personalization_gain.pdf"


def t6():
    lines = [r"\begin{tabular}{llllrrll}", r"\toprule",
             r"Dataset & Grouping & Tie-break & Evaluation & Overall & Rel. gap & Worst & Best \\", r"\midrule"]
    n = 0
    for ds in DS:
        p = latest(ds, "protocol", "protocol.json")
        if not p:
            continue
        for key, v in p.items():
            tie, grouping, model, prot = key.split("|")
            if model != "ials":
                continue
            lines.append(f"{DS[ds]} & {grouping} & {tie} & {'full ranking' if prot == 'full' else 'sampled (99)'} & "
                         f"{v['overall']:.4f} & {v['rel_gap']:.0%} & {v['worst']} & {v['best']} \\\\".replace("%", r"\%"))
            n += 1
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "T6_protocol.tex").write_text("\n".join(lines) + "\n")
    return f"T6_protocol.tex ({n} rows)"


def t7():
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Dataset & Model & Raw gap & Adjusted & Share explained [95\% CI] & Matched / raw \\", r"\midrule"]
    n = 0
    for ds in DS:
        g = latest(ds, "gate", "gate.json", need_random=True)
        if not g:
            continue
        for bb, r in g.items():
            if bb in ("itempop", "decision") or "decomp_NDCG" not in r:
                continue
            d = r["decomp_NDCG"]
            lines.append(f"{DS[ds]} & {BB.get(bb, bb)} & {d['raw']:+.4f} & {d['adjusted']:+.4f} & "
                         f"{d['share']:.0%} [{d['share_ci'][0]:.0%}, {d['share_ci'][1]:.0%}] & "
                         f"{abs(d['matched']) / abs(d['raw']):.0%} \\\\".replace("%", r"\%"))
            n += 1
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "T7_target.tex").write_text("\n".join(lines) + "\n")
    return f"T7_target.tex ({n} rows)"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for fn in (f1, f2, f3, f4):
        print(fn(plt))
    print(t6())
    print(t7())
    print(f"written to {OUT}")