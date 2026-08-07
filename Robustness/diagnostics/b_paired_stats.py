#!/usr/bin/env python3
"""
b_paired_stats.py -- paired-significance analysis for the speech and tumor comparisons.
Runs ONLY if the saved prediction files exist; otherwise refuses. Nothing here
simulates, imputes, or re-runs models.

  (A) Wilcoxon signed-rank over the five FSDD folds: reservoir vs RF, reservoir
      vs SVM. Input: fsdd_fold_scores.csv with columns fold,reservoir,rf,svm
      (per-fold error or accuracy, 5 rows).
  (B) McNemar (exact binomial on discordant pairs) for the 48 paired tumor
      predictions: reservoir vs LDA. Input: tumor_paired_preds.csv with columns
      sample,truth,reservoir,lda (48 rows).

    python3 b_paired_stats.py --fsdd fsdd_fold_scores.csv --tumor tumor_paired_preds.csv

HONESTY GATE: refuses on missing/short files; prints exact test statistics with
sample sizes; makes no significance claim beyond the computed p-values.
"""
import argparse, csv, math, os, sys

def need(path, min_rows, cols):
    if not path:
        return None
    if not os.path.isfile(path):
        sys.exit(f"REFUSING: {path} not found. Do not fabricate predictions.")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < min_rows or any(c not in rows[0] for c in cols):
        sys.exit(f"REFUSING: {path} malformed (need >= {min_rows} rows, cols {cols}).")
    return rows

def wilcoxon(x, y):
    from scipy.stats import wilcoxon as W
    stat, p = W(x, y, alternative="two-sided", zero_method="wilcox", method="exact")
    return stat, p

def mcnemar(rows):
    b = sum(1 for r in rows if r["reservoir"] == r["truth"] and r["lda"] != r["truth"])
    c = sum(1 for r in rows if r["lda"] == r["truth"] and r["reservoir"] != r["truth"])
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = sum(math.comb(n, k) for k in range(min(b, c) + 1)) * 2 / 2 ** n
    return b, c, min(1.0, p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsdd"); ap.add_argument("--tumor")
    a = ap.parse_args()
    if not (a.fsdd or a.tumor):
        sys.exit("Nothing to do: pass --fsdd and/or --tumor.")
    f = need(a.fsdd, 5, ["fold", "reservoir", "rf", "svm"])
    if f:
        res = [float(r["reservoir"]) for r in f]
        for base in ("rf", "svm"):
            stat, p = wilcoxon(res, [float(r[base]) for r in f])
            print(f"Wilcoxon reservoir vs {base.upper()} (n={len(f)} folds): W={stat}, p={p:.4f}")
    t = need(a.tumor, 48, ["sample", "truth", "reservoir", "lda"])
    if t:
        b, c, p = mcnemar(t)
        print(f"McNemar reservoir vs LDA (n={len(t)}): discordant b={b}, c={c}, exact p={p:.4f}")

if __name__ == "__main__":
    main()
