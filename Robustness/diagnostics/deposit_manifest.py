#!/usr/bin/env python3
"""
deposit_manifest.py -- SHA-256 manifest for the deposited financial CSV extracts.

Run against the EXISTING extracts used for the paper (never against a re-download):

    python3 deposit_manifest.py path/to/GSPC.csv path/to/AAPL.csv path/to/IXIC.csv

Emits MANIFEST.sha256 plus a ready-to-paste Data Availability block recording file
names, sizes, row counts, date coverage, and checksums.

HONESTY GATE: this script only hashes files it is given. It makes no claim that a
re-download reproduces these hashes (providers revise series; that is the point of
depositing the extracts). It refuses empty or missing files rather than guessing.
"""
import csv, hashlib, os, sys, datetime

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def coverage(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise SystemExit(f"REFUSING: {path} has no data rows.")
    dates = [r[0] for r in rows[1:] if r and r[0]]
    return len(rows) - 1, min(dates), max(dates)

def detect_adjustment(path):
    """Infer the original download's auto_adjust setting from the extract itself.
    yfinance with auto_adjust=False writes both 'Close' and 'Adj Close' columns;
    with auto_adjust=True it writes no 'Adj Close' column. Read off the artifact,
    never guessed."""
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    has_adj = any(h.strip().lower() == "adj close" for h in header)
    return "auto_adjust=False (raw Close + Adj Close columns present)" if has_adj \
        else "auto_adjust=True (no Adj Close column)"

def main(paths):
    if not paths:
        raise SystemExit("usage: deposit_manifest.py CSV [CSV ...]")
    lines, block = [], []
    for p in paths:
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            raise SystemExit(f"REFUSING: {p} missing or empty. Do not fabricate.")
        digest = sha256(p)
        n, d0, d1 = coverage(p)
        adj = detect_adjustment(p)
        mtime = datetime.date.fromtimestamp(os.path.getmtime(p)).isoformat()
        lines.append(f"{digest}  {os.path.basename(p)}")
        block.append(f"%   {os.path.basename(p)}: {n} rows, {d0} to {d1}, sha256 {digest}")
        block.append(f"%     adjustment (detected from columns): {adj}")
        block.append(f"%     file mtime {mtime} (candidate download date; verify against records)")
    open("MANIFEST.sha256", "w").write("\n".join(lines) + "\n")
    print("Wrote MANIFEST.sha256")
    print("% ---- paste into repository README / Data Availability deposit ----")
    print(f"% manifest generated {datetime.date.today().isoformat()} from the archived extracts")
    print("\n".join(block))

if __name__ == "__main__":
    main(sys.argv[1:])
