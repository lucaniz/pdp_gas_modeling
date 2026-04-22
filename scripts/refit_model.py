#!/usr/bin/env python3
"""
FWSS gas model refit script.

Queries FOC Observer for all provePossession transactions since FWSS v1.2.0,
refits the logarithmic model gas = alpha + beta * log2(pieces),
and patches the model coefficients in calculator.html and capacity.html.

Run manually:   python scripts/refit_model.py
Run via CI:     triggered by .github/workflows/update_model.yml
"""

import json
import math
import re
import sys
import datetime
import requests
import numpy as np
from scipy.optimize import curve_fit

FOC_API        = "https://foc-observer.va.gg/sql"
BASELINE_BLOCK = 5864769
MIN_PROOFS     = 3
HTML_FILES     = ["calculator.html", "capacity.html"]


# ── model ──────────────────────────────────────────────────────────────────────
def log2_model(x, alpha, beta):
    return alpha + beta * np.log2(np.maximum(x, 1))


# ── fetch ──────────────────────────────────────────────────────────────────────
def foc_query(sql):
    resp = requests.post(
        FOC_API,
        headers={"Content-Type": "application/json"},
        json={"network": "mainnet", "sql": sql},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["rows"]


def fetch_proving_data():
    print("Querying FOC Observer for provePossession data...")
    rows = foc_query(f"""
        SELECT
          pp.set_id,
          AVG(pp.gas_used) AS avg_gas,
          COUNT(*) AS proofs,
          MAX(pc.total_pieces) AS pieces
        FROM pdp_possession_proven pp
        JOIN (
          SELECT data_set_id, COUNT(*) AS total_pieces
          FROM fwss_piece_added
          GROUP BY data_set_id
        ) pc ON pc.data_set_id = pp.set_id
        WHERE pp.block_number >= {BASELINE_BLOCK}
        GROUP BY pp.set_id
        HAVING COUNT(*) >= {MIN_PROOFS}
        ORDER BY pieces ASC
    """)
    print(f"  Got {len(rows)} datasets")
    return [(int(r["pieces"]), float(r["avg_gas"])) for r in rows]


def fetch_npp_data():
    print("Querying FOC Observer for nextProvingPeriod data...")
    rows = foc_query(f"""
        SELECT
          AVG(gas_used) AS avg_gas,
          COUNT(*) AS txns,
          STDDEV(gas_used) AS stddev_gas
        FROM pdp_next_proving_period
        WHERE block_number >= {BASELINE_BLOCK}
          AND gas_used < 500000000
    """)
    r = rows[0]
    avg = float(r["avg_gas"])
    txns = int(r["txns"])
    std = float(r["stddev_gas"] or 0)
    print(f"  nextProvingPeriod: {avg/1e6:.1f}M gas  (n={txns}, stddev={std/1e6:.1f}M)")
    return avg, txns


def fetch_daily_series():
    print("Querying 30-day proving trend...")
    rows = foc_query(f"""
        SELECT
          DATE_TRUNC('day', TO_TIMESTAMP(timestamp)) AS day,
          COUNT(DISTINCT set_id) AS active_datasets,
          COUNT(*) AS proofs,
          SUM(gas_used) AS total_gas
        FROM pdp_possession_proven
        WHERE block_number >= {BASELINE_BLOCK}
          AND timestamp >= EXTRACT(EPOCH FROM NOW()) - 86400*30
        GROUP BY 1
        ORDER BY 1 ASC
    """)
    print(f"  Got {len(rows)} days")
    return rows


# ── fit ────────────────────────────────────────────────────────────────────────
def fit_model(data):
    pieces = np.array([d[0] for d in data], dtype=float)
    gas    = np.array([d[1] for d in data], dtype=float)
    popt, _ = curve_fit(log2_model, pieces, gas, p0=[160e6, 8e6])
    alpha, beta = popt
    pred = log2_model(pieces, alpha, beta)
    ss_res = np.sum((gas - pred) ** 2)
    ss_tot = np.sum((gas - gas.mean()) ** 2)
    r2  = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(gas - pred))
    return alpha, beta, r2, mae, len(data)


# ── patch ──────────────────────────────────────────────────────────────────────
def patch_file(filepath, alpha, beta, npp, r2, mae, n, today):
    """
    Patches all known constant name variants in a given HTML file.
    Handles: MODEL_ALPHA/MODEL_BETA/NPP_CONSTANT (calculator)
             PP_ALPHA/PP_BETA/NPP_GAS (capacity)
    Uses scientific notation to stay compact and avoid floating point drift.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  SKIP: {filepath} not found")
        return False

    original = content

    # All constant name variants → replace with clean scientific notation
    # e.g. 158670000 → 1.587e+08  (but we write it as 158670000 for readability)
    a_int = round(alpha)
    b_int = round(beta)
    n_int = round(npp)

    replacements = [
        # calculator.html style
        (r"const MODEL_ALPHA\s*=\s*[^,;]+",  f"const MODEL_ALPHA = {a_int}"),
        (r"const MODEL_BETA\s*=\s*[^,;]+",   f"const MODEL_BETA = {b_int}"),
        (r"const NPP_CONSTANT\s*=\s*[^,;]+", f"const NPP_CONSTANT = {n_int}"),
        # capacity.html style
        (r"const PP_ALPHA\s*=\s*[^,;]+",     f"const PP_ALPHA = {a_int}"),
        (r"const PP_BETA\s*=\s*[^,;]+",      f"const PP_BETA = {b_int}"),
        (r"const NPP_GAS\s*=\s*[^,;]+",      f"const NPP_GAS = {n_int}"),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    # Update the human-readable formula in the model-box (both formats)
    alpha_m = alpha / 1e6
    beta_m  = beta  / 1e6
    content = re.sub(
        r"gas = [\d.]+M \+ [\d.]+M &times; log&#8322;\(pieces\)",
        f"gas = {alpha_m:.2f}M + {beta_m:.3f}M &times; log&#8322;(pieces)",
        content,
    )

    # Update the stats line (both "Fit from N real datasets" variants)
    content = re.sub(
        r"Fit from \d+ real(?: mainnet)? datasets[^<\n]*",
        f"Fit from {n} real mainnet datasets &middot; MAE = {mae/1e6:.1f}M gas &middot; R&sup2; = {r2:.4f} &middot; updated {today}",
        content,
    )

    changed = content != original
    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Patched {filepath}")
    else:
        print(f"  No changes in {filepath} (coefficients unchanged or pattern not found)")

    return changed


def patch_hist(filepath, hist_rows):
    """Replaces the hardcoded HIST array in capacity.html."""
    if not hist_rows:
        return False
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return False

    def fmt(iso):
        dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        # strftime %-d works on Linux (GitHub Actions). Windows uses %#d.
        try:
            return dt.strftime("%b %-d")
        except ValueError:
            return dt.strftime("%b %d").replace(" 0", " ")

    entries = [
        f"  {{day:'{fmt(r['day'])}',ds:{int(r['active_datasets'])},gas:{int(r['total_gas'])}}},"
        for r in hist_rows
    ]
    new_hist = "const HIST = [\n" + "\n".join(entries) + "\n];"

    patched = re.sub(r"const HIST = \[[\s\S]*?\];", new_hist, content)
    if patched == content:
        print(f"  WARNING: HIST array not found in {filepath}")
        return False

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(patched)
    print(f"  Patched HIST ({len(hist_rows)} days) in {filepath}")
    return True


def patch_readme(alpha, beta, npp, r2, mae, n, today):
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
        content = re.sub(
            r"\| Training datasets \|.*?\|",
            f"| Training datasets | {n} (updated {today}) |",
            content,
        )
        content = re.sub(r"\| R² \|.*?\|",  f"| R² | {r2:.4f} |", content)
        content = re.sub(r"\| MAE \|.*?\|", f"| MAE | {mae/1e6:.1f}M gas |", content)
        content = re.sub(
            r"gas_provePossession\(N\) = [\d.,M\s+×log₂()]+",
            f"gas_provePossession(N) = {alpha/1e6:.3f}M + {beta/1e6:.3f}M × log₂(N)",
            content,
        )
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(content)
        print("  Patched README.md")
    except FileNotFoundError:
        print("  README.md not found — skipping")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    today = datetime.date.today().strftime("%Y-%m-%d")
    print(f"=== FWSS model refit — {today} ===\n")

    pp_data  = fetch_proving_data()
    npp, _   = fetch_npp_data()
    hist     = fetch_daily_series()

    if len(pp_data) < 50:
        print(f"ERROR: only {len(pp_data)} datasets — too few to refit. Aborting.")
        sys.exit(1)

    print(f"\nFitting log2 model on {len(pp_data)} datasets...")
    alpha, beta, r2, mae, n = fit_model(pp_data)

    print(f"\ngas = {alpha/1e6:.2f}M + {beta/1e6:.3f}M × log₂(pieces)")
    print(f"R²={r2:.4f}  MAE={mae/1e6:.2f}M  npp={npp/1e6:.1f}M  n={n}")

    if r2 < 0.95:
        print(f"WARNING: R²={r2:.4f} is below 0.95 — check for protocol changes.")

    print(f"\n=== Patching files ===")
    changed = False
    for f in HTML_FILES:
        changed |= patch_file(f, alpha, beta, npp, r2, mae, n, today)

    changed |= patch_hist("capacity.html", hist)
    patch_readme(alpha, beta, npp, r2, mae, n, today)

    # Write summary for CI step
    summary = {
        "date": today, "alpha": alpha, "beta": beta, "npp": npp,
        "r2": r2, "mae": mae, "n_datasets": n, "files_changed": changed,
    }
    with open("/tmp/model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. files_changed={changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
