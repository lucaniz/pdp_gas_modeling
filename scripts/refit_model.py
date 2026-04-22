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
from scipy.stats import pearsonr

# ── configuration ──────────────────────────────────────────────────────────────
FOC_API       = "https://foc-observer.va.gg/sql"
BASELINE_BLOCK = 5864769   # FWSS v1.2.0 deployment block
MIN_PROOFS     = 3         # minimum proofs per dataset to include in fit
HTML_FILES     = ["calculator.html", "capacity.html"]

# ── model ──────────────────────────────────────────────────────────────────────
def log2_model(x, alpha, beta):
    return alpha + beta * np.log2(np.maximum(x, 1))

# ── fetch data ─────────────────────────────────────────────────────────────────
def fetch_proving_data():
    """
    Fetches per-dataset average gas and piece count from FOC Observer.
    Returns list of (pieces, avg_gas) tuples.
    Only includes datasets with >= MIN_PROOFS observations (more stable average).
    """
    print("Querying FOC Observer for provePossession data...")
    sql = f"""
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
    """
    resp = requests.post(
        FOC_API,
        headers={"Content-Type": "application/json"},
        json={"network": "mainnet", "sql": sql},
        timeout=60,
    )
    resp.raise_for_status()
    rows = resp.json()["rows"]
    print(f"  Got {len(rows)} datasets")
    return [(int(r["pieces"]), float(r["avg_gas"])) for r in rows]


def fetch_npp_data():
    """
    Fetches nextProvingPeriod average gas (flat constant, not model-dependent).
    Excludes outliers above 500M gas (anomalous transactions).
    """
    print("Querying FOC Observer for nextProvingPeriod data...")
    sql = f"""
        SELECT
          AVG(gas_used) AS avg_gas,
          MIN(gas_used) AS min_gas,
          MAX(gas_used) AS max_gas,
          COUNT(*) AS txns,
          STDDEV(gas_used) AS stddev_gas
        FROM pdp_next_proving_period
        WHERE block_number >= {BASELINE_BLOCK}
          AND gas_used < 500000000
    """
    resp = requests.post(
        FOC_API,
        headers={"Content-Type": "application/json"},
        json={"network": "mainnet", "sql": sql},
        timeout=30,
    )
    resp.raise_for_status()
    row = resp.json()["rows"][0]
    print(f"  nextProvingPeriod: avg={float(row['avg_gas'])/1e6:.1f}M gas"
          f"  (n={row['txns']}, stddev={float(row['stddev_gas'])/1e6:.1f}M)")
    return float(row["avg_gas"]), int(row["txns"])


def fetch_daily_proving_series():
    """
    Fetches the last 30 days of daily FWSS proving gas totals.
    Used to update the hardcoded HIST array in capacity.html.
    """
    print("Querying FOC Observer for 30-day proving trend...")
    sql = f"""
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
    """
    resp = requests.post(
        FOC_API,
        headers={"Content-Type": "application/json"},
        json={"network": "mainnet", "sql": sql},
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()["rows"]
    print(f"  Got {len(rows)} days of data")
    return rows


# ── fit ────────────────────────────────────────────────────────────────────────
def fit_model(data):
    """
    Fits gas = alpha + beta * log2(pieces) via OLS.
    Returns (alpha, beta, r2, mae, n_datasets).
    """
    pieces = np.array([d[0] for d in data], dtype=float)
    gas    = np.array([d[1] for d in data], dtype=float)

    popt, _ = curve_fit(log2_model, pieces, gas, p0=[160e6, 8e6])
    alpha, beta = popt

    gas_pred = log2_model(pieces, alpha, beta)
    residuals = gas - gas_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((gas - np.mean(gas)) ** 2)
    r2  = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(residuals))

    return alpha, beta, r2, mae, len(data)


# ── patch HTML ─────────────────────────────────────────────────────────────────
def patch_coefficients(filepath, alpha, beta, npp, r2, mae, n_datasets, today):
    """
    Replaces PP_ALPHA, PP_BETA, NPP_CONSTANT in the HTML file.
    Also updates the model stats text shown in the UI.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original = content

    # Patch JS constants
    content = re.sub(
        r"const MODEL_ALPHA\s*=\s*[\d.e+]+",
        f"const MODEL_ALPHA = {alpha:.0f}",
        content,
    )
    content = re.sub(
        r"const MODEL_BETA\s*=\s*[\d.e+]+",
        f"const MODEL_BETA = {beta:.0f}",
        content,
    )
    content = re.sub(
        r"const PP_ALPHA\s*=\s*[\d.e+]+",
        f"const PP_ALPHA = {alpha:.0f}",
        content,
    )
    content = re.sub(
        r"const PP_BETA\s*=\s*[\d.e+]+",
        f"const PP_BETA = {beta:.0f}",
        content,
    )
    content = re.sub(
        r"const NPP_CONSTANT\s*=\s*[\d.e+]+",
        f"const NPP_CONSTANT = {npp:.0f}",
        content,
    )
    content = re.sub(
        r"const NPP_GAS\s*=\s*[\d.e+]+",
        f"const NPP_GAS = {npp:.0f}",
        content,
    )

    # Patch human-readable formula display in the model-box
    alpha_m = alpha / 1e6
    beta_m  = beta  / 1e6
    npp_m   = npp   / 1e6
    content = re.sub(
        r"gas = [\d.]+M &times; log&#8322;\(pieces\)",
        f"gas = {alpha_m:.2f}M + {beta_m:.3f}M &times; log&#8322;(pieces)",
        content,
    )
    # Patch stats line
    content = re.sub(
        r"Fit from \d+ real mainnet datasets[^<]*",
        f"Fit from {n_datasets} real mainnet datasets · MAE = {mae/1e6:.1f}M gas · R² = {r2:.4f} · last updated {today}",
        content,
    )

    if content == original:
        print(f"  WARNING: no changes made to {filepath} — check regex patterns")
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Patched {filepath}")

    return content != original


def patch_hist_array(filepath, hist_rows):
    """
    Replaces the hardcoded HIST array in capacity.html with fresh data.
    """
    if not hist_rows:
        print("  No historical rows — skipping HIST patch")
        return False

    def fmt_day(iso):
        # "2026-04-21T00:00:00.000Z" → "Apr 21"
        dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%b %-d")  # Linux/Mac; Windows needs %#d

    js_entries = []
    for r in hist_rows:
        label = fmt_day(r["day"])
        ds    = int(r["active_datasets"])
        gas   = int(r["total_gas"])
        js_entries.append(f"  {{day:'{label}',ds:{ds},gas:{gas}}},")

    new_hist = "const HIST = [\n" + "\n".join(js_entries) + "\n];"

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    patched = re.sub(
        r"const HIST = \[[\s\S]*?\];",
        new_hist,
        content,
    )

    if patched == content:
        print(f"  WARNING: HIST array not found in {filepath}")
        return False

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(patched)
    print(f"  Patched HIST array in {filepath} ({len(hist_rows)} days)")
    return True


def patch_readme(alpha, beta, npp, r2, mae, n_datasets, today):
    """Updates model statistics table in README.md."""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()

        content = re.sub(
            r"\| Training datasets \| .* \|",
            f"| Training datasets | {n_datasets} (mainnet, updated {today}) |",
            content,
        )
        content = re.sub(
            r"\| R² \| .* \|",
            f"| R² | {r2:.4f} |",
            content,
        )
        content = re.sub(
            r"\| MAE \| .* \|",
            f"| MAE | {mae/1e6:.1f}M gas |",
            content,
        )
        # Update formula in README
        alpha_m = alpha / 1e6
        beta_m  = beta  / 1e6
        content = re.sub(
            r"gas_provePossession\(N\) = [\d.]+ \+ [\d.]+ × log₂\(N\)",
            f"gas_provePossession(N) = {alpha_m:.3f}M + {beta_m:.3f}M × log₂(N)",
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

    # 1. Fetch data
    pp_data = fetch_proving_data()
    npp_gas, npp_n = fetch_npp_data()
    hist_rows = fetch_daily_proving_series()

    if len(pp_data) < 50:
        print(f"ERROR: only {len(pp_data)} datasets — too few to refit safely. Aborting.")
        sys.exit(1)

    # 2. Fit model
    print(f"\nFitting log2 model on {len(pp_data)} datasets...")
    alpha, beta, r2, mae, n = fit_model(pp_data)
    npp = npp_gas  # flat constant

    print(f"\n=== Model results ===")
    print(f"  gas = {alpha/1e6:.2f}M + {beta/1e6:.3f}M × log₂(pieces)")
    print(f"  R² = {r2:.4f}")
    print(f"  MAE = {mae/1e6:.2f}M gas")
    print(f"  nextProvingPeriod = {npp/1e6:.1f}M gas (n={npp_n})")
    print(f"  n_datasets = {n}")

    if r2 < 0.95:
        print(f"\nWARNING: R²={r2:.4f} is below 0.95 — model quality degraded.")
        print("Check for new gas mechanism changes or data anomalies.")

    # 3. Validate: predictions at key points
    print(f"\n=== Predictions ===")
    for pieces in [1, 10, 100, 1000, 10000, 100000, 1000000]:
        pred = alpha + beta * math.log2(pieces)
        print(f"  {pieces:>10,} pieces → {pred/1e6:.1f}M gas")

    # 4. Patch HTML files
    print(f"\n=== Patching files ===")
    any_changed = False
    for filepath in HTML_FILES:
        try:
            changed = patch_coefficients(filepath, alpha, beta, npp, r2, mae, n, today)
            any_changed = any_changed or changed
        except FileNotFoundError:
            print(f"  {filepath} not found — skipping")

    # 5. Patch HIST array in capacity.html
    try:
        changed = patch_hist_array("capacity.html", hist_rows)
        any_changed = any_changed or changed
    except FileNotFoundError:
        print("  capacity.html not found — skipping HIST patch")

    # 6. Patch README
    patch_readme(alpha, beta, npp, r2, mae, n, today)

    # 7. Output summary for CI
    print(f"\n=== Summary ===")
    print(f"  alpha = {alpha:.0f}  ({alpha/1e6:.2f}M)")
    print(f"  beta  = {beta:.0f}  ({beta/1e6:.3f}M)")
    print(f"  npp   = {npp:.0f}  ({npp/1e6:.1f}M)")
    print(f"  r2    = {r2:.4f}")
    print(f"  mae   = {mae/1e6:.2f}M gas")
    print(f"  n     = {n}")
    print(f"  files_changed = {any_changed}")

    # Write a machine-readable summary for the CI commit message
    with open("/tmp/model_summary.json", "w") as f:
        json.dump({
            "date": today,
            "alpha": alpha,
            "beta": beta,
            "npp": npp,
            "r2": r2,
            "mae": mae,
            "n_datasets": n,
            "files_changed": any_changed,
        }, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
