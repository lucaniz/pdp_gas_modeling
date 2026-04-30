# FWSS Gas Modeling

Tools for understanding, estimating, and forecasting the on-chain gas costs of FWSS (Filecoin Web3 Storage Service) operations on Filecoin mainnet.

**Live site → [lucaniz.github.io/pdp_gas_modeling](https://lucaniz.github.io/pdp_gas_modeling/)**

---

## Tools

### [Gas & Cost Calculator](https://lucaniz.github.io/pdp_gas_modeling/calculator.html)

Estimates the real gas cost of every FWSS operation for any dataset configuration:

- `createDataSet`, `addPieces` (with batching), `terminateService` — live empirical averages from FOC Observer
- `provePossession` — **logarithmic model** fitted to 630 real mainnet datasets (R²=0.99)
- `nextProvingPeriod` — empirical constant (~124M gas, independent of piece count)
- Live base fee from Glif, live FIL/USD from CoinGecko
- Floor price calculator: minimum a client must pay per dataset per 30 days to cover SP proving costs

### [Network Capacity & Forecast](https://lucaniz.github.io/pdp_gas_modeling/capacity.html)

Models how much of Filecoin's gas budget FWSS uses, how it grows, and when it becomes relevant to network congestion:

- Live network gas headroom: FWSS vs total gas target (23.5B gas/tipset)
- 30-day historical trend of FWSS proving gas (real on-chain data)
- Dataset piece distribution analysis (why using the mean is misleading)
- Growth forecast with configurable scenarios and churn rate
- Sensitivity analysis: dataset count vs piece count as growth drivers
- Scale thresholds: how many datasets to reach 1%, 5%, 10% of gas target

---

## Gas Model

The central model is a **single-predictor logarithmic regression** for `provePossession` gas:

```
gas_provePossession(N) = 157.491M + 7.987M × log₂(N)N)N)N)N)N)N)N)N)N)N)
```

where `N` is the number of pieces in the dataset.

**Why logarithmic?** The PDP protocol samples exactly 5 random pieces per proof (hardcoded). Each challenge verifies a Merkle inclusion proof, traversing a tree of depth `log₂(N)`. Gas therefore scales with `log₂(N)`, not `N`. Each doubling of pieces adds ~8.5M gas — not 2× the gas.

**Why not multilinear?** [FIPs discussion #761](https://github.com/filecoin-project/FIPs/discussions/761) proposes a multilinear regression for Filecoin gas estimation, appropriate for the miner actor cron job where several structurally distinct operations contribute independently. For `provePossession`, piece count is the single dominant predictor. All other candidate variables are either constant (challenge count = 5, always) or collinear with piece count. OLS on the log-transformed predictor achieves R²=0.99 — adding further terms does not improve fit materially. Weighted regression was considered but not used for the same reason.

### Model statistics

| Metric | Value |
|--------|-------|
| Training datasets | 721 (updated 2026-04-30) |
| Piece count range | 1 → 868,515 |
| R² | 0.9267 |
| MAE | 5.1M gas |
| `nextProvingPeriod` model | constant ~124M gas (flat across all piece counts) |

---

## Data Sources

| Data | Source | How queried |
|------|--------|-------------|
| Gas unit averages (createDataSet, addPieces, nextProvingPeriod, terminateService) | [FOC Observer](https://foc-observer.va.gg) REST API | Live on page load (`POST /sql`) |
| provePossession model (α, β) | OLS on FOC Observer data | Hardcoded — refit periodically |
| Base fee (current) | [Glif](https://api.node.glif.io) public Lotus node | Live (`Filecoin.ChainHead`) |
| Base fee (historical) | FOC Observer (`pdp_pieces_added.effective_gas_price`) | Aggregated live |
| FIL/USD price | [CoinGecko](https://coingecko.com) public API | Live on page load |
| Network gas utilisation | [Glif](https://api.node.glif.io) (`eth_feeHistory`) | Live on page load |
| Average blocks/tipset | [Filfox](https://filfox.info) (`/api/v1/stats/base-fee`) | Live on page load |
| 30-day FWSS proving trend | FOC Observer | Hardcoded from last query — update monthly |

---

## Filecoin Gas Mechanics

Filecoin's base fee adjustment mechanism (introduced in [FIP-0001](https://github.com/filecoin-project/FIPs/blob/master/FIPS/fip-0001.md), inspired by but distinct from EIP-1559) works as follows:

- Each block has a **gas target of 5 billion gas units**
- With ~4.7 blocks per tipset, the effective target is **~23.5B gas per tipset**
- When a tipset exceeds this target, `BaseFee` rises by up to +12.5%
- When below, it falls by up to −12.5%
- Sustained overshoot causes **exponential base fee growth** — a 10% sustained overshoot doubles the base fee in ~6 tipsets (~3 minutes)
- The `BaseFee` is **burned entirely** (not paid to miners); miners receive `GasPremium × GasLimit`

Key differences from EIP-1559: Filecoin has `GasPremium` applied on `GasLimit` (not `GasUsed`), an `OverEstimationBurn` mechanism for over-estimated gas limits, and full BaseFee burn rather than partial burn.

---

## Files

```
index.html        Landing page
calculator.html   Gas & cost calculator
capacity.html     Network capacity & forecast
README.md         This file
```

---

## Updating the Model

The model coefficients (α=158.67M, β=8.485M) are hardcoded. To refit:

1. Query FOC Observer for all `provePossession` transactions since the last fit:
   ```sql
   SELECT pp.set_id, AVG(pp.gas_used) AS avg_gas, MAX(pc.total_pieces) AS pieces
   FROM pdp_possession_proven pp
   JOIN (SELECT data_set_id, COUNT(*) AS total_pieces FROM fwss_piece_added GROUP BY data_set_id) pc
     ON pc.data_set_id = pp.set_id
   WHERE pp.block_number >= 5864769
   GROUP BY pp.set_id
   ```
2. Fit OLS: `gas = α + β × log₂(pieces)`
3. Update `PP_ALPHA` and `PP_BETA` in both `calculator.html` and `capacity.html`
4. Update model statistics in the methodology sections

---

Built by Luca — data from [FOC Observer](https://foc-observer.va.gg), [Glif](https://api.node.glif.io), [Filfox](https://filfox.info), [CoinGecko](https://coingecko.com).
