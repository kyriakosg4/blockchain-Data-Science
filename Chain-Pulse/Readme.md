<h1>Chain Pulse — Data Collector • Metrics Processor • Daily Reporter</h1>

End-to-end pipeline for collecting blockchain metrics, structuring them into CSVs, generating visualizations, and producing a daily Markdown report. Runs locally or on AWS with the same code paths (toggled by an env flag).

---

# Contents

- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Dual-mode (Local vs AWS)](#dual-mode-local-vs-aws)
- [Environment variables](#environment-variables)
- [Install & run (local)](#install--run-local)
- [Deploy on AWS](#deploy-on-aws)
- [Data flow & folder structure](#data-flow--folder-structure)
- [Generated outputs](#generated-outputs)
- [Scheduling](#scheduling)
- [IAM permissions](#iam-permissions)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Architecture

```text
┌───────────────┐        JSON        ┌──────────────────┐        CSV        ┌──────────────────┐
│ Data Collector│ ────────────────▶ │ Metrics Processor │ ────────────────▶ │  Daily Reporter  │
│ (APIs → JSON) │                   │ (JSON → CSV)      │                   │ (Charts+Markdown)│
└───────────────┘                   └──────────────────┘                   └──────────────────┘

S3: raw-data-all-types   →   S3: processed-data-all-types   →   S3: reports

```

- **Collector**: pulls data from CoinGecko, DeFiLlama, StakingRewards, Etherscan/BscScan, and RPC nodes,  
  and stores **normalized JSON** per run.  

- **Processor**: reads daily JSONs, maps/normalizes into **economic**, **network**, and **staking** CSVs;  
  also writes a per-day processed log and optional weekly aggregates.  

- **Reporter**: loads processed CSVs, builds enhanced CSVs (with “Average” rows and change %),  
  generates **Matplotlib** visualizations, and writes a final **Markdown** report.

  ---

# Repository Layout

```text
├─ chain-data-collector.py
├─ chain-metrics-processor.py
├─ chain-daily-reporter.py
├─ requirements.txt
├─ .env              # not committed; use .env.example
└─ s3/               # (local mode) S3-like folder structure on disk
   └─ buckets/chain-pulse-metrics/
      ├─ raw-data-all-types/
      ├─ processed-data-all-types/
      └─ reports/

```
In AWS mode, the same paths under s3://chain-pulse-metrics/... are used. In local mode, identical S3-style subpaths are mirrored under your local s3/buckets/chain-pulse-metrics/… root so you can inspect results easily.


## Dual-mode (Local vs AWS)

All three scripts share the same “dual-mode” helpers:

- Set `LOCAL_MODE=true` in `.env` → read/write **local folders**.  
- Set `LOCAL_MODE=false` (or omit) → read/write **S3** with `boto3`.  

This keeps identical behavior and folder structure across environments,  
changing only the IO backend.

---

## Enviroment Variables
Create a .env in the project root:

**switch: local = true, aws = false**

LOCAL_MODE=true

## AWS mode only
S3_BUCKET=chain-pulse-metrics
AWS_REGION=eu-central-1

**Local folders (when LOCAL_MODE=true)**

**Use absolute paths. Example below matches the Windows layout you’ve been using.**


PROCESSED_DIR=C:/Users/User/Desktop/Chain-Pulse-AWS-lambda/s3/buckets/chain-pulse-metrics/processed-data-all-types
REPORTS_DIR=C:/Users/User/Desktop/Chain-Pulse-AWS-lambda/s3/buckets/chain-pulse-metrics/reports
TEMPLATES_DIR=C:/Users/User/Desktop/Chain-Pulse-AWS-lambda/s3/buckets/chain-pulse-metrics/reports/templates

---

## Install & run (local)

**Requirements:**
- Python 3.11+ (your Lambda uses 3.11; local 3.12 is fine as long as you run locally)
- pip

```bash
# from repo root
pip install -r requirements.txt
```

---
### 1) Run the Collector (optional in local dev)

This fetches live data from APIs and writes JSON under `raw-data-all-types/YYYY-MM/YYYY-MM-DD/`.

If you already have sample JSON from AWS, you can skip this.

```bash
python chain-data-collector.py
```

---

### 2) Run the Processor

Reads raw JSON, writes **economic/network/staking** CSVs into  
`processed-data-all-types/YYYY-MM/YYYY-MM-DD/` with one row per timeslot, plus weekly files.

```bash
python chain-metrics-processor.py
```

### 3) Run the Reporter
Auto-selects today & yesterday; if either day is missing, it falls back to the latest two available days that each have all three CSVs.
Outputs enhanced CSVs, charts, and a Markdown report under reports/YYYY-MM/YYYY-MM-DD/....
```bash
python chain-daily.reporter.py
```

---

## Deploy on AWS

- **Collector**: EventBridge schedule **every 6 hours** (00:00, 06:00, 12:00, 18:00 UTC), writes raw JSON to `raw-data-all-types`.

- **Processor**: EventBridge schedule **00:15, 06:15, 12:15, 18:15 UTC**, reads raw JSON and writes processed CSVs to `processed-data-all-types` and weekly aggregates; keeps a per-day processed log.

- **Reporter**: EventBridge schedule **daily 19:00 UTC**, reads processed CSVs, generates **charts** and the daily **Markdown** report under `reports`.

The Reporter switched to **Matplotlib** to fit Lambda layer size limits; figures are closed explicitly to free memory.

---

## Data flow & folder structure

### Collector → S3

```text
raw-data-all-types/
└─ YYYY-MM/
   └─ YYYY-MM-DD/
      ├─ coingecko.json
      ├─ stakingrewards.json
      ├─ defillama.json
      ├─ fees_gas.json
      └─ tps.json

- Normalized JSON structures for each source (price, market cap, 24h vol;
TVL & DEX vol; staking metrics; gas/fees; TPS).
```

### Processor → S3
```text
processed-data-all-types/
└─ YYYY-MM/
   └─ YYYY-MM-DD/
      ├─ economic_metrics.csv
      ├─ network_metrics.csv
      └─ staking_metrics.csv

weekly-data/
└─ YYYY-MM-weekX/
   ├─ economic_metrics.csv
   ├─ network_metrics.csv
   └─ staking_metrics.csv

processed-files-log/
└─ processed_files.json
```

### Reporter → S3
```text
reports/
├─ templates/
│  └─ daily_report_template.md
└─ YYYY-MM/
   └─ YYYY-MM-DD/
      ├─ data/
      │  ├─ economic_metrics.csv     # enriched, includes {date1}/{date2} columns + Average row
      │  ├─ network_metrics.csv
      │  └─ staking_metrics.csv
      ├─ visualizations/
      │  ├─ economic_charts/
      │  │  ├─ market_trends.png
      │  │  ├─ supply_comparison.png
      │  │  └─ trading_volume.png
      │  ├─ network_charts/
      │  │  ├─ volume_comparison.png
      │  │  ├─ gas_and_fee_trends.png
      │  │  └─ tps_comparison.png
      │  └─ staking_charts/
      │     └─ staking_comparison.png
      └─ generated/
         └─ daily_report.md
```
- Reporter reads processed data, computes Average rows (averaging absolutes and computing % change based on those averages), then renders charts and a Markdown report.

---

## Generated outputs

### Enhanced CSVs (for the report)

Reporter rewrites per-day CSVs into `reports/.../data/` with:

- Two columns per metric: (... date2) and (... date1)  
- A `{metric} Change (%)` column  
- An extra **“Average”** row per blockchain (averages absolutes; computes the change % from those averages rather than averaging per-row percentages).

---

### Visualizations

Saved under `reports/.../visualizations/`:

- **Network**: transaction volume trends, gas & fees, TPS.  
- **Economic**: price + market cap trends, supply comparison, trading volume.  
- **Staking**: staking ratio comparison.  

---

### Markdown report

`reports/.../generated/daily_report.md` embeds the above charts and table values;  
placeholders are replaced from the enriched CSVs (not from raw data).

**Minimal template example** (`reports/templates/daily_report_template.md`)

# 📊 Chain Pulse Daily Report — {date}

Compared to: {previous_date}

## Network
![Transaction Volume](volume_comparison.png)
![Gas & Fees](gas_and_fee_trends.png)
![TPS](tps_comparison.png)

## Economics
![Price & Market Cap](market_trends.png)
![Supply](supply_comparison.png)
![Trading Volume](trading_volume.png)

## Staking
![Staking Ratio](staking_comparison.png)

---

## Scheduling

- **Collector** — every 6h: `00:00, 06:00, 12:00, 18:00 UTC`.
- **Processor** — every 6h + 15m: `00:15, 06:15, 12:15, 18:15 UTC`.
- **Reporter** — daily **19:00 UTC** to ensure all data is present.

---

## IAM permissions

**Minimum:**

- **S3**: read/write to `raw-data-all-types/`, `processed-data-all-types/`, `reports/`.
- **EventBridge**: rule to trigger schedules.
- **Lambda** execution role.

---

## Troubleshooting

- **“Missing one or both CSV files”** when running the reporter  
  Ensure the **Processor** has produced all three CSVs for the target date. The reporter will **fallback** to the latest two dates that each have all three CSVs if today/yesterday aren’t present (built into `select_dates_for_report()` in `chain-daily-reporter.py`).

- **KeyError** (e.g., `Price Change (%)`)  
  This usually means the reporter is reading the **pre-enrichment** CSVs. Always let the reporter generate its report-ready CSVs in `reports/.../data/` and then read those for charting/report text (the script already does this flow).

- **Windows paths**  
  Use **absolute** Windows paths in `.env` for `PROCESSED_DIR`, `REPORTS_DIR`, `TEMPLATES_DIR`.  
  The reporter/processor/collector will create missing folders as needed.

- **Matplotlib memory / blank figures**  
  The reporter explicitly calls `plt.close(fig)` after saving. If you add custom plots, do the same.





