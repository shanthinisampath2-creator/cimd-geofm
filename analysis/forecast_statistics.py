"""Statistics for the CIMD failure forecast (no GPU needed).

Reads a `failure_forecast_instances_A3.csv` file written by the notebooks and reports, for each
removed sensor:
  * failure-detection AUROC of D_pred, of the class-average D, and of an oracle class prior;
  * the AUROC gain of D_pred over the class-average D with a 95 % bootstrap interval;
  * the share of failures caught when the top-k % highest-risk instances are flagged.

Usage:
    python analysis/forecast_statistics.py results/DFC2020/failure_forecast_instances_A3.csv
    python analysis/forecast_statistics.py <csv> --threshold 0.2 --bootstrap 1000 --out stats.csv

Note: the bootstrap resamples class instances, not whole patches, so the intervals are slightly
optimistic when several classes come from the same patch.
"""
import argparse

import numpy as np
import pandas as pd


def auroc(scores, labels):
    labels = np.asarray(labels, bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(np.asarray(scores)).rank().values
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def capture(scores, labels, frac):
    labels = np.asarray(labels, bool)
    k = int(frac * len(labels))
    top = np.argsort(-np.asarray(scores))[:k]
    return float(labels[top].sum() / max(labels.sum(), 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--threshold", type=float, default=0.20, help="recall drop that counts as a failure")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional CSV to save the summary")
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    d["fail"] = d["recall_drop"] > a.threshold
    group = ["seed", "removed", "class"] if "seed" in d.columns else ["removed", "class"]
    d["class_avg_D"] = d.groupby(group)["D_pred"].transform("mean")
    d["oracle_prior"] = d.groupby(group)["recall_drop"].transform("mean")
    rng = np.random.default_rng(a.seed)

    rows = []
    for removed, g in d.groupby("removed"):
        g = g.reset_index(drop=True)
        gain = auroc(g.D_pred, g.fail) - auroc(g.class_avg_D, g.fail)
        boots = []
        for _ in range(a.bootstrap):
            s = g.iloc[rng.integers(0, len(g), len(g))]
            boots.append(auroc(s.D_pred, s.fail) - auroc(s.class_avg_D, s.fail))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        row = {"removed": removed, "instances": len(g), "failure_rate": g.fail.mean(),
               "AUROC_D_pred": auroc(g.D_pred, g.fail), "AUROC_class_avg_D": auroc(g.class_avg_D, g.fail),
               "AUROC_oracle_prior": auroc(g.oracle_prior, g.fail),
               "gain_vs_class_avg": gain, "gain_CI95_low": lo, "gain_CI95_high": hi}
        for frac in (0.1, 0.2, 0.3):
            row[f"caught_top{int(frac * 100)}_D_pred"] = capture(g.D_pred, g.fail, frac)
            row[f"caught_top{int(frac * 100)}_class_avg"] = capture(g.class_avg_D, g.fail, frac)
        rows.append(row)

    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(out.round(3).T.to_string(header=False))
    if a.out:
        out.to_csv(a.out, index=False)
        print("saved:", a.out)


if __name__ == "__main__":
    main()
