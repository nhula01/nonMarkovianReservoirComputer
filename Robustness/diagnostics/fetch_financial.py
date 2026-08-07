#!/usr/bin/env python3
"""
fetch_financial.py -- documented fetch procedure for the financial series
(GSPC, AAPL, IXIC; 2014-01-01 to 2024-01-01, daily closes, Yahoo Finance).

This DOCUMENTS how the deposited extracts were obtained. It deliberately requires
the adjustment setting on the command line rather than defaulting, because that
setting is a fact about the original download and must not be guessed:

    python3 fetch_financial.py --auto-adjust {true|false} --outdir extracts_redownload/

HONESTY GATE: output goes to a separate directory and is NEVER claimed to match
MANIFEST.sha256 -- providers revise historical series. Reproduction of the paper's
numbers is from the deposited extracts, verified against the manifest, not from
this script's output.
"""
import argparse, sys

TICKERS = {"^GSPC": "GSPC.csv", "AAPL": "AAPL.csv", "^IXIC": "IXIC.csv"}
START, END = "2014-01-01", "2024-01-01"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto-adjust", required=True, choices=["true", "false"],
                    help="MUST match the original download setting (recorded in repo README)")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not installed; pip install yfinance")
    import os
    os.makedirs(a.outdir, exist_ok=True)
    for tk, name in TICKERS.items():
        df = yf.download(tk, start=START, end=END, auto_adjust=(a.auto_adjust == "true"),
                         progress=False)
        df.to_csv(os.path.join(a.outdir, name))
        print(f"fetched {tk} -> {name} ({len(df)} rows)  [NOT the deposited extract]")

if __name__ == "__main__":
    main()
