"""Composite audit figures with one shared legend (for the paper).

    python scripts/make_grid.py

Outputs results/paper/F5_audit_grid_activity.pdf (3 x 3) and F6_audit_grid_gender.pdf (2 x 3), built from
the seed-0 audit runs (latest run per configuration, as in aggregate.py). Panel letters (a), (b), ...
are drawn in the figure, so the LaTeX file needs a single \\includegraphics per figure.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from aggregate import collect  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "paper"
DS = {"ml-100k": "ML-100K", "ml-1m": "ML-1M", "gowalla": "Gowalla"}
KIND = {"audit_reweight": "reweighting", "audit_gapreg": "gap penalty"}
BB = {"ials": "iALS", "ease": "EASE", "knn": "ItemKNN", "bpr": "BPR-MF"}

ACTIVITY = [("ml-100k", "audit_reweight", "ials"), ("ml-100k", "audit_reweight", "ease"), ("ml-100k", "audit_gapreg", "bpr"),
            ("ml-1m", "audit_reweight", "ials"), ("ml-1m", "audit_reweight", "ease"), ("ml-1m", "audit_gapreg", "bpr"),
            ("gowalla", "audit_reweight", "ials"), ("gowalla", "audit_reweight", "knn"), ("gowalla", "audit_gapreg", "bpr")]
GENDER = [("ml-100k", "audit_reweight", "ials"), ("ml-100k", "audit_reweight", "ease"), ("ml-100k", "audit_gapreg", "bpr"),
          ("ml-1m", "audit_reweight", "ials"), ("ml-1m", "audit_reweight", "ease"), ("ml-1m", "audit_gapreg", "bpr")]

STYLE_ACT = {0: dict(color="#2166ac", ls="-", lw=1.6, marker="o", ms=3.5),     # Q1, least active
             1: dict(color="#7f7f7f", ls=":", lw=1.2, marker="o", ms=2.5),
             2: dict(color="#bdbdbd", ls=":", lw=1.2, marker="o", ms=2.5),
             3: dict(color="#b2182b", ls="-", lw=1.6, marker="o", ms=3.5)}     # Q4, most active
STYLE_GEN = {"F": dict(color="#7b3294", ls="-", lw=1.6, marker="o", ms=3.5),
             "M": dict(color="#e08214", ls="-", lw=1.6, marker="s", ms=3.5)}
STYLE_ALL = dict(color="k", ls="--", lw=1.0)


def panel_data(runs, ds, kind, attr, bb):
    hit = runs.get((ds, kind, attr, 0))
    if not hit:
        return None
    r = hit[1].get(bb)
    if not isinstance(r, dict) or "rows" not in r:
        return None
    xs = [row["w"] for row in r["rows"]]
    G = np.array([row["test_groups"] for row in r["rows"]])
    allv = [row["test_all"] for row in r["rows"]]
    names = r.get("groups") or ([f"act{i}" for i in range(G.shape[1])] if attr == "activity" else ["F", "M"])
    adv = r.get("adv")
    adv = names[adv] if isinstance(adv, int) else adv
    return xs, G, allv, names, adv


def draw(spec, attr, ncols, fname, legend_handles, size):
    runs = collect(RES)
    nrows = int(np.ceil(len(spec) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False)
    for k, (ds, kind, bb) in enumerate(spec):
        ax = axes[k // ncols][k % ncols]
        letter = chr(ord("a") + k)
        ax.set_title(f"({letter}) {DS[ds]}, {KIND[kind]}, {BB[bb]}", fontsize=7.5, loc="left")
        d = panel_data(runs, ds, kind, attr, bb)
        if d is None:
            ax.text(0.5, 0.5, "not run", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="grey")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        xs, G, allv, names, adv = d
        pos = np.arange(len(xs))
        if attr == "activity":
            for gi in range(G.shape[1]):
                ax.plot(pos, G[:, gi], **STYLE_ACT[gi])
        else:
            for gi, gname in enumerate(names):
                ax.plot(pos, G[:, gi], **STYLE_GEN[gname])
            ax.set_title(f"({letter}) {DS[ds]}, {KIND[kind]}, {BB[bb]}\nadvantaged on validation: "
                         f"{'female' if adv == 'F' else 'male'}", fontsize=7.5, loc="left")
        ax.plot(pos, allv, **STYLE_ALL)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("weight $w$" if kind == "audit_reweight" else r"penalty $\lambda_{\mathrm{gap}}$", fontsize=7.5)
        if k % ncols == 0:
            ax.set_ylabel("test NDCG@10", fontsize=7.5)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname)
    plt.close(fig)
    return fname


if __name__ == "__main__":
    act_legend = [Line2D([], [], **STYLE_ACT[0], label="$Q_1$ least active (advantaged)"),
                  Line2D([], [], **STYLE_ACT[1], label="$Q_2$"),
                  Line2D([], [], **STYLE_ACT[2], label="$Q_3$"),
                  Line2D([], [], **STYLE_ACT[3], label="$Q_4$ most active (disadvantaged)"),
                  Line2D([], [], **STYLE_ALL, label="all users")]
    gen_legend = [Line2D([], [], **STYLE_GEN["F"], label="female"),
                  Line2D([], [], **STYLE_GEN["M"], label="male"),
                  Line2D([], [], **STYLE_ALL, label="all users")]
    print(draw(ACTIVITY, "activity", 3, "F5_audit_grid_activity.pdf", act_legend, (7.2, 6.4)))
    print(draw(GENDER, "gender", 3, "F6_audit_grid_gender.pdf", gen_legend, (7.2, 4.4)))
    print(f"written to {OUT}")