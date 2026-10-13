"""Collect every audit and UGF run into paper tables (CSV + LaTeX) and figures.

    python scripts/aggregate.py
    python scripts/aggregate.py --no-figures

Reads results/<dataset>/<timestamp>_<run>/{audit.json, ugf.json}. Run names:
  audit_reweight[_activity|_gender][_sK], audit_gapreg_<groups>[_sK], ugf[_activity|_gender][_sK]
(no grouping suffix = activity; no seed suffix = the models' default seed 0). If a configuration was run
several times, the latest run is used. Outputs go to results/paper/:
  audit_selected.csv / .tex   selected setting per (dataset, intervention, attribute, backbone): d_adv, d_dis,
                              d_all as mean ± sd over seeds, verdicts per seed, max LD share over settings
  audit_all_settings.csv      every setting of every run (for plots / appendix)
  ugf.csv / .tex              ORACLE vs LEAK-FREE: d_better, d_worse, d_all (seed 0 / first run)
  fig_<dataset>_<intervention>_<attribute>_<backbone>.pdf   per-group test NDCG@10 vs w / lambda (seed 0)
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_RE = re.compile(r"^\d{8}-\d{6}_(audit_reweight|audit_gapreg|ugf)(?:_(activity|gender))?(?:_s(\d+))?$")
NAMES = {"audit_reweight": "Reweighting", "audit_gapreg": "Loss-gap penalty", "ugf": "UGF"}
DS_NAMES = {"ml-100k": "ML-100K", "ml-1m": "ML-1M", "gowalla": "Gowalla"}
BB_NAMES = {"ials": "iALS", "ease": "EASE", "knn": "ItemKNN", "bpr": "BPR-MF"}


def collect(results_dir):
    """{(dataset, kind, attribute, seed): (folder, json)} keeping the latest run per key."""
    runs = {}
    for ds_dir in sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name in DS_NAMES):
        for run in sorted(ds_dir.iterdir()):
            m = RUN_RE.match(run.name)
            if not m:
                continue
            kind, attr, seed = m.group(1), m.group(2) or "activity", int(m.group(3) or 0)
            f = run / ("ugf.json" if kind == "ugf" else "audit.json")
            if not f.exists():
                continue
            runs[(ds_dir.name, kind, attr, seed)] = (run, json.loads(f.read_text()))   # sorted -> latest wins
    return runs


def fmt(mean, sd, n):
    return f"{mean:+.4f}" if n == 1 else f"{mean:+.4f} ± {sd:.4f}"


def audit_tables(runs, out):
    groups = defaultdict(list)            # (ds, kind, attr, backbone) -> list of (seed, result)
    allrows = []
    for (ds, kind, attr, seed), (folder, js) in runs.items():
        if kind == "ugf":
            continue
        for bb, r in js.items():
            if not isinstance(r, dict) or "rows" not in r:
                continue
            groups[(ds, kind, attr, bb)].append((seed, r))
            for row in r["rows"]:
                allrows.append({"dataset": ds, "intervention": kind, "attribute": attr, "backbone": bb,
                                "seed": seed, "setting": row["w"], "val_all": row["val_all"],
                                "val_gap": row["val_gap"], "test_all": row["test_all"], "gap": row["gap"],
                                "closure": row["closure"], "d_adv": row["d_adv"], "d_dis": row["d_dis"],
                                "ld_share": row.get("ld_share"),
                                **{f"test_{n}": v for n, v in zip(r.get("groups", [f"g{i}" for i in range(len(row["test_groups"]))]),
                                                                  row["test_groups"])}})
    with open(out / "audit_all_settings.csv", "w", newline="") as f:
        keys = sorted({k for r in allrows for k in r}, key=lambda k: (k.startswith("test_"), k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(allrows)

    table = []
    for (ds, kind, attr, bb), items in sorted(groups.items(), key=lambda x: (list(DS_NAMES).index(x[0][0]), x[0][1:])):
        items.sort()
        da = np.array([r["d_adv"][0] for _, r in items])
        dd = np.array([r["d_dis"][0] for _, r in items])
        dl = np.array([r["d_all"][0] for _, r in items])
        lds = [row.get("ld_share") for _, r in items for row in r["rows"]
               if row.get("ld_share") is not None and row.get("ld_share") == row.get("ld_share")]
        table.append({"dataset": ds, "intervention": kind, "attribute": attr, "backbone": bb, "seeds": len(items),
                      "selected": ",".join(f"{r['selected_w']:g}" for _, r in items),
                      "d_adv": fmt(da.mean(), da.std(ddof=1) if len(da) > 1 else 0, len(da)),
                      "d_dis": fmt(dd.mean(), dd.std(ddof=1) if len(dd) > 1 else 0, len(dd)),
                      "d_all": fmt(dl.mean(), dl.std(ddof=1) if len(dl) > 1 else 0, len(dl)),
                      "dis_up_significant": sum(r["d_dis"][1] > 0 for _, r in items),
                      "dis_down_significant": sum(r["d_dis"][2] < 0 for _, r in items),
                      "verdicts": "; ".join(r["verdict"] for _, r in items),
                      "max_ld_share": f"{max(lds):.0%}" if lds else "-",
                      "min_ld_share": f"{min(lds):.0%}" if lds else "-"})
    with open(out / "audit_selected.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]) if table else ["empty"])
        w.writeheader()
        w.writerows(table)

    short = {"LEVELING DOWN": "LD", "LEVELING UP": "LU", "MIXED": "MIX", "NO NARROWING": "none",
             "NO SIGNIFICANT EFFECT": "n.s."}
    lines = [r"\begin{tabular}{lllllrrrcc}", r"\toprule",
             r"Dataset & Intervention & Attr. & Model & Sel. & $\Delta$adv & $\Delta$dis & $\Delta$all & Verdicts & LD share \\",
             r"\midrule"]
    for t in table:
        verdicts = "/".join(short.get(v.strip(), v.strip()) for v in t["verdicts"].split(";"))
        lds = t["min_ld_share"] if t["min_ld_share"] == t["max_ld_share"] else f"{t['min_ld_share']}--{t['max_ld_share']}"
        lines.append(f"{DS_NAMES[t['dataset']]} & {NAMES[t['intervention']]} & {t['attribute']} & "
                     f"{BB_NAMES.get(t['backbone'], t['backbone'])} & {t['selected']} & "
                     f"{t['d_adv'].replace('±', '$\\pm$')} & {t['d_dis'].replace('±', '$\\pm$')} & "
                     f"{t['d_all'].replace('±', '$\\pm$')} & {verdicts} & {lds.replace('%', r'\%')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "audit_selected.tex").write_text("\n".join(lines) + "\n")
    return table


def ugf_tables(runs, out):
    table = []
    for (ds, kind, attr, seed), (folder, js) in sorted(runs.items(), key=lambda x: (list(DS_NAMES).index(x[0][0]), x[0][1:])):
        if kind != "ugf" or seed != min(s for (d, k, a, s) in runs if (d, k, a) == (ds, kind, attr)):
            continue
        for bb, r in js.items():
            if not isinstance(r, dict) or "ORACLE" not in r:
                continue
            row = {"dataset": ds, "attribute": attr, "backbone": bb}
            for v in ("ORACLE", "LEAK-FREE"):
                x = r[v]
                for key in ("d_better", "d_worse", "d_all"):
                    row[f"{v}_{key}"] = x[key][0]
                    row[f"{v}_{key}_ci"] = f"[{x[key][1]:+.4f}, {x[key][2]:+.4f}]"
                row[f"{v}_verdict"] = x["verdict"]
                row[f"{v}_hr_all"] = x["hr_all"]
            table.append(row)
    if not table:
        return table
    with open(out / "ugf.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    lines = [r"\begin{tabular}{lllrrrrrr}", r"\toprule",
             r" & & & \multicolumn{3}{c}{ORACLE (test labels)} & \multicolumn{3}{c}{LEAK-FREE} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"Dataset & Attr. & Model & $\Delta$better & $\Delta$worse & $\Delta$all & $\Delta$better & $\Delta$worse & $\Delta$all \\",
             r"\midrule"]
    for t in table:
        def c(v, k):
            val, ci = t[f"{v}_{k}"], t[f"{v}_{k}_ci"]
            lo, hi = (float(x) for x in ci.strip("[]").split(","))
            star = "$^{*}$" if lo > 0 or hi < 0 else ""
            return f"{val:+.4f}{star}"
        lines.append(f"{DS_NAMES[t['dataset']]} & {t['attribute']} & {BB_NAMES.get(t['backbone'], t['backbone'])} & "
                     + " & ".join(c(v, k) for v in ("ORACLE", "LEAK-FREE") for k in ("d_better", "d_worse", "d_all"))
                     + r" \\")
    lines += [r"\bottomrule", r"\multicolumn{9}{l}{\footnotesize $^{*}$ paired bootstrap 95\% CI excludes 0.}\\",
              r"\end{tabular}"]
    (out / "ugf.tex").write_text("\n".join(lines) + "\n")
    return table


def figures(runs, out, no_titles=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: figures skipped)")
        return 0
    n = 0
    for (ds, kind, attr, seed), (folder, js) in runs.items():
        if kind == "ugf" or seed != 0:
            continue
        for bb, r in js.items():
            if not isinstance(r, dict) or "rows" not in r:
                continue
            xs = [row["w"] for row in r["rows"]]
            G = np.array([row["test_groups"] for row in r["rows"]])
            names = r.get("groups") or ([f"act{i}" for i in range(G.shape[1])] if attr == "activity" else ["F", "M"])
            adv, dis = r.get("adv"), r.get("dis")
            adv = names[adv] if isinstance(adv, int) else adv          # older runs stored group indices
            dis = names[dis] if isinstance(dis, int) else dis
            # paper notation: activity quartiles Q1 (least active) .. Q4 (most active)
            rename = {f"act{i}": f"Q{i + 1}" for i in range(4)}
            names = [rename.get(n, n) for n in names]
            adv, dis = rename.get(adv, adv), rename.get(dis, dis)
            fig, ax = plt.subplots(figsize=(5.2, 3.0))
            pos = range(len(xs))
            for gi, gname in enumerate(names):
                style = "-" if gname in (adv, dis) else ":"
                label = gname + (" (adv.)" if gname == adv else " (disadv.)" if gname == dis else "")
                ax.plot(pos, G[:, gi], style, marker="o", ms=3, label=label)
            ax.plot(pos, [row["test_all"] for row in r["rows"]], "k--", lw=1, label="all users")
            ax.set_xticks(list(pos))
            ax.set_xticklabels([f"{x:g}" for x in xs])
            ax.set_xlabel("w" if kind == "audit_reweight" else r"$\lambda$")
            ax.set_ylabel("test NDCG@10")
            if not no_titles:
                ax.set_title(f"{DS_NAMES[ds]} · {NAMES[kind]} · {BB_NAMES.get(bb, bb)} · {attr}", fontsize=8)
            ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      title="adv./disadv.:\nset on validation", title_fontsize=5.5)
            fig.tight_layout()
            fig.savefig(out / f"fig_{ds}_{kind}_{attr}_{bb}.pdf")
            plt.close(fig)
            n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-titles", action="store_true",
                    help="omit panel titles (the LaTeX subcaptions already name each panel)")
    args = ap.parse_args()
    res = (ROOT / args.results).resolve()
    out = res / "paper"
    out.mkdir(parents=True, exist_ok=True)
    runs = collect(res)
    print(f"runs found: {len(runs)}")
    for (ds, kind, attr, seed), (folder, _) in sorted(runs.items()):
        print(f"  {ds:<8} {kind:<15} {attr:<9} seed {seed}: {folder.name}")
    a = audit_tables(runs, out)
    u = ugf_tables(runs, out)
    nf = 0 if args.no_figures else figures(runs, out, args.no_titles)
    print(f"\naudit rows: {len(a)} | UGF rows: {len(u)} | figures: {nf}")
    print(f"written to {out}")
    if a:
        print("\n== selected settings (mean ± sd over seeds)")
        for t in a:
            print(f"  {t['dataset']:<8} {t['intervention']:<15} {t['attribute']:<9} {t['backbone']:<5} seeds={t['seeds']} "
                  f"sel={t['selected']:<12} d_adv {t['d_adv']:<18} d_dis {t['d_dis']:<18} d_all {t['d_all']:<18} "
                  f"dis up(sig)={t['dis_up_significant']} down(sig)={t['dis_down_significant']} LD {t['min_ld_share']}-{t['max_ld_share']}")