#!/usr/bin/env python3
"""
b6_persistence_check.py -- independent verification for II-5 (NRMSE convention,
financial task, Supplement S.8).

Computes the persistence-predictor (yhat_t = y_{t-1}) NRMSE on daily closes for
^GSPC, AAPL, ^IXIC over the paper's protocol (rolling two-year-train / one-year-
test, test years 2016-2023) under two candidate normalizations of the test-window
target: range (max-min) and standard deviation. The persistence NRMSE is invariant
under any affine pre-normalization of the series, so the unknown injection scaling
drops out.

Finding to check against S.8: "Both models operate well below the persistence
baseline." Reported model NRMSE: GSPC 0.075+-0.058, AAPL 0.059+-0.025,
IXIC 0.079+-0.070.

HONESTY GATE: this script downloads current Yahoo data, which may differ from the
deposited extracts by provider revisions. The conclusion is robust to revisions
(the two conventions differ by ~4x) but the deposited extracts remain the
reference; rerun this script against them via --csv to remove the caveat.

    python3 b6_persistence_check.py                # fetch current data
    python3 b6_persistence_check.py --csv GSPC.csv AAPL.csv IXIC.csv
"""
import argparse, sys
import numpy as np

def load_yahoo():
    import yfinance as yf
    out = {}
    for tk in ["^GSPC", "AAPL", "^IXIC"]:
        s = yf.download(tk, start="2014-01-01", end="2024-01-01",
                        auto_adjust=False, progress=False)["Close"].squeeze()
        out[tk] = s
    return out

def load_csv(paths):
    import csv, datetime
    out = {}
    for p in paths:
        dates, close = [], []
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    dates.append(datetime.date.fromisoformat(row["Date"][:10]))
                    close.append(float(row["Close"]))
                except (KeyError, ValueError):
                    continue
        if len(close) < 500:
            sys.exit(f"REFUSING: {p} too short or malformed.")
        out[p] = (np.array([d.year for d in dates]), np.array(close))
    return out

def stats(years, vals):
    per_r, per_sd = [], []
    for ty in range(2016, 2024):
        t = vals[years == ty]
        if len(t) < 200:
            continue
        rmse = np.sqrt(np.mean((t[1:] - t[:-1]) ** 2))
        per_r.append(rmse / (t.max() - t.min()))
        per_sd.append(rmse / t.std())
    return np.array(per_r), np.array(per_sd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="*")
    a = ap.parse_args()
    if a.csv:
        data = load_csv(a.csv)
        items = [(k, v[0], v[1]) for k, v in data.items()]
    else:
        data = load_yahoo()
        items = [(k, np.array(s.index.year), s.values.astype(float)) for k, s in data.items()]
    print(f"{'series':22} {'persistence range-norm':>24} {'persistence SD-norm':>22}")
    for name, years, vals in items:
        r, sd = stats(years, vals)
        print(f"{name:22} {r.mean():>12.4f} ± {r.std():.4f} {sd.mean():>12.4f} ± {sd.std():.4f}")
    print("\nReported model NRMSE (S.8): GSPC 0.075±0.058, AAPL 0.059±0.025, IXIC 0.079±0.070")
    print("If range-normalized: models sit ABOVE persistence -> S.8 claim false.")
    print("If SD-normalized:   models sit well below persistence -> S.8 claim true.")

if __name__ == "__main__":
    main()
