"""Rebuild Tables 3-6 of the paper from the results archive (no GPU needed).

Usage:
    python analysis/make_paper_tables.py path/to/CIMD_results [--out results/paper_tables]

Table 2 (test mIoU) combines the WHU-OPT-SAR test table from notebook 01 with the DFC2020
rows of Table 3; both are in results/paper_tables/Table2_test_mIoU.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def auroc(scores, labels):
    labels = np.asarray(labels, bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(np.asarray(scores)).rank().values
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def capture(scores, labels, frac=0.3):
    labels = np.asarray(labels, bool)
    top = np.argsort(-np.asarray(scores))[:int(frac * len(labels))]
    return float(labels[top].sum() / max(labels.sum(), 1))


def forecast_rows(path, dataset, rng, n_boot=1000, thr=0.20):
    d = pd.read_csv(path)
    d["fail"] = d["recall_drop"] > thr
    grp = ["seed", "removed", "class"]
    d["class_avg"] = d.groupby(grp)["D_pred"].transform("mean")
    d["oracle"] = d.groupby(grp)["recall_drop"].transform("mean")
    rows = []
    for removed, g in d.groupby("removed"):
        g = g.reset_index(drop=True)
        gain = auroc(g.D_pred, g.fail) - auroc(g.class_avg, g.fail)
        boots = []
        for _ in range(n_boot):
            s = g.iloc[rng.integers(0, len(g), len(g))]
            boots.append(auroc(s.D_pred, s.fail) - auroc(s.class_avg, s.fail))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append([dataset, removed, f"{g.fail.mean():.1%}", round(auroc(g.D_pred, g.fail), 3),
                     round(auroc(g.class_avg, g.fail), 3), round(auroc(g.oracle, g.fail), 3),
                     f"{gain:+.3f} ({lo:.3f}, {hi:.3f})",
                     f"{capture(g.D_pred, g.fail):.0%} / {capture(g.class_avg, g.fail):.0%}"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default="results/paper_tables")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    R, out = Path(a.results_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dfc = R / "DFC2020"

    # Table 3: all methods on DFC2020
    ev = pd.read_csv(dfc / "tables/E3_eval_cache.csv")
    ev["Robust"] = ev[["FULL_mIoU", "OPT_mIoU", "SAR_mIoU"]].mean(axis=1)
    paired = pd.read_csv(dfc / "tables/E3_paired_vs_A1.csv").set_index("experiment")
    agg = pd.read_csv(dfc / "tables/E3_all_methods_mean_std.csv", header=[0, 1], index_col=0)
    rows = []
    for e in ["A0F", "A1", "A2", "A3", "B_KD", "B_REC", "B_GATE"]:
        x = ev[ev.experiment == e]
        if x.empty:
            continue
        delta = "–" if e == "A1" else f"{paired.loc[e, 'Robust_mean_minus_A1']:+.3f} ({int(paired.loc[e, 'seeds_better_robust'])}/3)"
        rows.append([e, *(f"{x[c].mean():.3f}" for c in ["FULL_mIoU", "OPT_mIoU", "SAR_mIoU", "Robust"]), delta,
                     f"{agg.loc[e, ('AUC_optical_degraded', 'mean')]:.3f}", f"{agg.loc[e, ('AUC_sar_degraded', 'mean')]:.3f}"])
    pd.DataFrame(rows, columns=["Method", "Full input", "Optical only (SAR removed)", "SAR only (optical removed)",
                                "Robust mean", "Delta vs A1 (seeds better)", "Degradation AUC optical corrupted",
                                "Degradation AUC SAR corrupted"]).to_csv(out / "Table3_DFC2020_all_methods.csv", index=False)

    # Table 4: class dependency on DFC2020
    s = pd.read_csv(dfc / "tables/A3_dependency_summary.csv")
    c = pd.read_csv(dfc / "tables/failure_forecast_classes_A3.csv")
    rows = []
    for cl in s["class"].unique():
        a_ = s[s["class"] == cl].set_index("modality")
        b_ = c[c["class"] == cl].set_index("removed")
        rows.append([cl, round(a_.loc["optical", "D_true"], 2), round(a_.loc["optical", "D_pred"], 2),
                     round(b_.loc["optical", "IoU_drop"], 3), round(a_.loc["sar", "D_true"], 2),
                     round(a_.loc["sar", "D_pred"], 2), round(b_.loc["sar", "IoU_drop"], 3)])
    pd.DataFrame(rows, columns=["Class", "D optical", "D_pred optical", "IoU drop optical removed", "D SAR",
                                "D_pred SAR", "IoU drop SAR removed"]).sort_values("D optical", ascending=False) \
        .to_csv(out / "Table4_DFC2020_class_dependency.csv", index=False)

    # Table 5: failure forecast, both datasets
    # one fresh generator per dataset, as in forecast_statistics.py, so the intervals match it exactly
    rows = forecast_rows(R / "WHU-OPT-SAR/instances/failure_forecast_instances_A3.csv", "WHU-OPT-SAR",
                         np.random.default_rng(a.seed))
    rows += forecast_rows(dfc / "instances/failure_forecast_instances_A3.csv", "DFC2020", np.random.default_rng(a.seed))
    pd.DataFrame(rows, columns=["Dataset", "Sensor removed", "Failure rate", "AUROC D_pred", "AUROC class-average D",
                                "AUROC oracle class prior", "Gain (95% CI)",
                                "Caught top 30%: D_pred / class average"]).to_csv(out / "Table5_failure_forecast.csv", index=False)

    # Table 6: predicted dependency vs predictive uncertainty (DFC2020)
    e1 = pd.read_csv(dfc / "instances/E1_forecast_vs_uncertainty_instances.csv")
    e1["fail"] = e1["recall_drop"] > 0.20
    e1["class_avg_D"] = e1.groupby(["seed", "removed", "class"])["D_pred"].transform("mean")
    e1["rank_avg"] = (e1.groupby(["seed", "removed"])["D_pred"].rank(pct=True)
                      + e1.groupby(["seed", "removed"])["U_entropy"].rank(pct=True))
    names = {"D_pred": "D_pred (predicted dependency)", "class_avg_D": "Class-average D_pred",
             "U_maxprob": "1 - max softmax", "U_entropy": "Normalised entropy", "U_margin": "1 - top-two margin",
             "rank_avg": "Rank average of D_pred and entropy"}
    pd.DataFrame([[v] + [round(auroc(g[k], g.fail), 3) for _, g in e1.groupby("removed")] for k, v in names.items()],
                 columns=["Score", "Optical removed", "SAR removed"]).to_csv(out / "Table6_DFC2020_uncertainty_baselines.csv", index=False)
    print("Tables written to", out)


if __name__ == "__main__":
    main()
