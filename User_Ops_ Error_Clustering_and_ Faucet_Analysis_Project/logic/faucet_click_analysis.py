# faucet_click_analysis.py

import os
from collections import Counter
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from eth_abi import decode
from tqdm import tqdm
from web3 import Web3, HTTPProvider
from web3.middleware import ExtraDataToPOAMiddleware


# -------------------------
# Web3 init (sync, v7-ready)
# -------------------------
def init_web3(rpc_url: str) -> Web3:
    """
    Initialize a synchronous Web3 client and inject POA middleware (v7+).
    """
    w3 = Web3(HTTPProvider(rpc_url))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w3


# -------------------------------------
# Helpers for safe hex/method-id checks
# -------------------------------------
def _normalize_hex(s) -> str:
    """
    Return lowercase hex string without '0x' prefix.
    Accepts str, HexBytes, bytes, bytearray, or None.
    """
    if s is None:
        return ""
    if hasattr(s, "hex") and not isinstance(s, str):
        s = s.hex()
    elif isinstance(s, (bytes, bytearray)):
        s = s.hex()
    else:
        s = str(s)
    s = s.lower()
    return s[2:] if s.startswith("0x") else s


def _tx_matches(tx: dict, faucet_address: str, method_id: str) -> bool:
    """
    Check if the transaction targets faucet_address and starts with method_id.
    """
    to_addr = (tx.get("to") or "").lower()
    if not to_addr or to_addr != faucet_address.lower():
        return False

    hex_input = _normalize_hex(tx.get("input"))
    selector = _normalize_hex(method_id)

    return hex_input.startswith(selector)


# ---------------------------------------
# Main extraction (writes CSVs with heads)
# ---------------------------------------
def extract_faucet_clicks(
    w3: Web3,
    faucet_address: str,
    method_id: str,
    start_block: int,
    end_block: int,
    log_path: str = "data/faucet_click_log.csv",
    summary_path: str = "data/faucet_clicks_by_address.csv",
    save_interval: int = 500,
) -> Tuple[str, str]:
    """
    Scan blocks and collect faucet click senders. Writes progress periodically.
    Always writes CSVs with headers (even when empty).
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    click_counter: Counter[str] = Counter()
    click_rows: list[dict] = []

    # Ensure files exist with headers
    pd.DataFrame(columns=["sender", "block_number"]).to_csv(log_path, index=False)
    pd.DataFrame(columns=["sender", "faucet_clicks"]).to_csv(summary_path, index=False)

    for idx, bn in enumerate(tqdm(range(start_block, end_block + 1), desc="Scanning blocks")):
        try:
            block = w3.eth.get_block(bn, full_transactions=True)
            txs = getattr(block, "transactions", None)
            if txs is None:
                txs = block.get("transactions", [])

            for tx in txs:
                tx_dict = dict(tx)
                if not _tx_matches(tx_dict, faucet_address, method_id):
                    continue

                # Decode calldata after selector
                hex_input = _normalize_hex(tx_dict.get("input"))
                selector = _normalize_hex(method_id)
                payload_hex = hex_input[len(selector):]

                try:
                    payload_bytes = bytes.fromhex(payload_hex)
                    recipients, _, _ = decode(["address[]", "uint256[]", "bool"], payload_bytes)
                except Exception as e:
                    print(f"⚠️ Decode failed on tx {tx_dict.get('hash')}: {e}")
                    continue

                for recipient in recipients:
                    addr = recipient.lower()
                    click_counter[addr] += 1
                    click_rows.append({"sender": addr, "block_number": bn})

        except Exception as e:
            print(f"❌ Block {bn} fetch failed: {e}")

        if (idx + 1) % save_interval == 0 or bn == end_block:
            df_clicks = pd.DataFrame(click_rows, columns=["sender", "block_number"])
            df_clicks.to_csv(log_path, index=False)

            if click_counter:
                df_summary = pd.DataFrame(click_counter.items(), columns=["sender", "faucet_clicks"])
                df_summary.sort_values(by="faucet_clicks", ascending=False, inplace=True)
            else:
                df_summary = pd.DataFrame(columns=["sender", "faucet_clicks"])

            df_summary.to_csv(summary_path, index=False)
            print(f"💾 Saved progress at block {bn} (rows: {len(df_clicks)})")

    return log_path, summary_path


# ----------------------------------------
# Analysis (robust to empty / no-data case)
# ----------------------------------------
def analyze_click_behavior(
    log_path: str,
    summary_path: str,
    output_file: str = "data/faucet_clicks_fulldata_sorted.csv",
    block_time_seconds: int = 5,
) -> Optional[str]:
    """
    Merge summary with timing features; skip gracefully if no data.
    """
    if not (os.path.exists(summary_path) and os.path.getsize(summary_path) > 0):
        print("ℹ️ No summary data found; skipping analysis.")
        return None
    if not (os.path.exists(log_path) and os.path.getsize(log_path) > 0):
        print("ℹ️ No click log found; skipping analysis.")
        return None

    df_summary = pd.read_csv(summary_path)
    df_clicks = pd.read_csv(log_path)

    if df_clicks.empty or df_summary.empty:
        print("ℹ️ No rows to analyze; skipping.")
        return None

    df_clicks = df_clicks.sort_values(by=["sender", "block_number"])
    grouped = df_clicks.groupby("sender")["block_number"].agg(list).reset_index()

    def compute_gap_hours(blocks: list[int]) -> pd.Series:
        if len(blocks) < 2:
            return pd.Series([np.nan, np.nan, np.nan])
        gaps = np.diff(blocks)
        gaps_in_hours = (gaps * block_time_seconds) / 3600.0
        return pd.Series([gaps_in_hours.mean(), gaps_in_hours.min(), gaps_in_hours.max()])

    grouped[
        ["avg_time_between_clicks_hr", "min_time_between_clicks_hr", "max_time_between_clicks_hr"]
    ] = grouped["block_number"].apply(compute_gap_hours)

    merged = df_summary.merge(grouped.drop(columns=["block_number"]), on="sender", how="left")

    # === FIXED avg_clicks_per_day ===
    # calculate based on sender's first/last block
    first_last = (
        df_clicks.groupby("sender")["block_number"]
        .agg(first_block="min", last_block="max")
        .reset_index()
    )
    merged = merged.merge(first_last, on="sender", how="left")

    merged["active_seconds"] = (
        (merged["last_block"] - merged["first_block"]).clip(lower=0) * block_time_seconds
    )
    merged["active_days"] = (merged["active_seconds"] / 86400.0).clip(lower=1.0).fillna(1.0)
    merged["avg_clicks_per_day"] = merged["faucet_clicks"] / merged["active_days"]

    # drop helpers if not needed
    merged = merged.drop(columns=["first_block", "last_block", "active_seconds", "active_days"])

    # Percentiles
    clicks_95 = merged["faucet_clicks"].quantile(0.95)
    clicks_98 = merged["faucet_clicks"].quantile(0.98)
    merged["is_top_5_percent"] = merged["faucet_clicks"] >= clicks_95
    merged["is_top_2_percent"] = merged["faucet_clicks"] >= clicks_98

    merged = merged.sort_values(by="faucet_clicks", ascending=False)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    merged.to_csv(output_file, index=False)
    print(f"✅ Final summary saved as: {output_file}")
    return output_file


# ------------------------------
# Orchestrator / public function
# ------------------------------
def generate_faucet_summary(
    rpc_url: str,
    faucet_address: str,
    method_id: str,
    start_block: int,
    end_block: int,
    log_path: str = "data/faucet_click_log.csv",
    summary_path: str = "data/faucet_clicks_by_address.csv",
    output_path: str = "data/faucet_clicks_fulldata_sorted.csv",
) -> Optional[str]:
    """
    End-to-end: extract, then analyze. Returns output CSV path (or None).
    """
    os.makedirs("data", exist_ok=True)
    w3 = init_web3(rpc_url)

    print("🚰 Extracting faucet clicks...")
    log_file, summary_file = extract_faucet_clicks(
        w3=w3,
        faucet_address=faucet_address,
        method_id=method_id,
        start_block=start_block,
        end_block=end_block,
        log_path=log_path,
        summary_path=summary_path,
    )

    print("📊 Analyzing faucet click patterns...")
    return analyze_click_behavior(log_path=log_file, summary_path=summary_file, output_file=output_path)
