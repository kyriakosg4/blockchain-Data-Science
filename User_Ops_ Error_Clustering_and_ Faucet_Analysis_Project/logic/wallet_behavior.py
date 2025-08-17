# src/wallet_behavior.py

import pandas as pd
import glob
import re
import os


os.makedirs("data", exist_ok=True)


def generate_wallet_behavior_summary(
    user_ops_pattern="uops/user_operations_chunk_*.csv",
    block_file_pattern="blocks/blocks_chunk_{}.csv",
    output_file="data/wallet_behavior_summary_pre_faucet.csv"
):
    enriched_files = []
    for user_ops_file in glob.glob(user_ops_pattern):
        match = re.search(r"user_operations_chunk_(\d+).csv", user_ops_file)
        if not match:
            continue

        chunk_num = match.group(1)
        blocks_file = block_file_pattern.format(chunk_num)

        try:
            uops_df = pd.read_csv(user_ops_file, dtype={"tx_hash": "string"})
            uops_df["tx_hash"] = uops_df["tx_hash"].str.replace("0x", "", case=False)

            blocks_df = pd.read_csv(blocks_file, dtype={"hash": "string", "timestamp": "string"})
            blocks_df = blocks_df.rename(columns={"hash": "tx_hash"})
            blocks_df["timestamp"] = pd.to_datetime(blocks_df["timestamp"], errors="coerce")

            merged_df = uops_df.merge(blocks_df[["tx_hash", "timestamp"]], on="tx_hash", how="left")
            merged_df["date"] = merged_df["timestamp"].dt.date

            if "native_value" not in merged_df.columns or "success" not in merged_df.columns:
                print(f"⚠️ Skipping chunk {chunk_num}: missing required columns")
                continue

            enriched_files.append(merged_df)
            print(f"✅ Merged chunk {chunk_num}")

        except Exception as e:
            print(f"❌ Error merging chunk {chunk_num}: {e}")

    if not enriched_files:
        raise ValueError("❌ No valid user_operations files merged.")

    user_ops = pd.concat(enriched_files, ignore_index=True)

    wallet_stats = user_ops.groupby("sender").agg(
        tx_count=('tx_hash', 'count'),
        total_gas_cost=('actualGasCost', 'sum'),
        total_gas_used=('actualGasUsed', 'sum'),
        failed_tx=('success', lambda x: (~x).sum()),
        active_days=('date', pd.Series.nunique)
    ).reset_index()

    wallet_stats["avg_tx_per_day"] = wallet_stats["tx_count"] / wallet_stats["active_days"].replace(0, 1)
    wallet_stats["fail_rate"] = wallet_stats["failed_tx"] / wallet_stats["tx_count"].replace(0, 1)

    successful_ops = user_ops[user_ops["success"] == True]
    volume_stats = successful_ops.groupby("sender").agg(
        transaction_volume=("native_value", "sum"),
        avg_transaction_volume=("native_value", "mean"),
        std_transaction_volume=("native_value", "std")
    ).reset_index()

    wallet_stats = wallet_stats.merge(volume_stats, on="sender", how="left")
    wallet_stats = wallet_stats.sort_values(by="tx_count", ascending=False).reset_index(drop=True)
    wallet_stats["tx_rank"] = wallet_stats.index + 1
    wallet_stats["tx_rank_pct"] = wallet_stats["tx_count"].rank(pct=True)
    wallet_stats["is_top_2_percent_tx_count"] = wallet_stats["tx_rank_pct"] > 0.98
    wallet_stats["is_top_5_percent_tx_count"] = wallet_stats["tx_rank_pct"] > 0.95

    # Reorder top flags
    top_5 = wallet_stats.pop("is_top_5_percent_tx_count")
    top_2 = wallet_stats.pop("is_top_2_percent_tx_count")
    wallet_stats["is_top_5_percent_tx_count"] = top_5
    wallet_stats["is_top_2_percent_tx_count"] = top_2

    wallet_stats.to_csv(output_file, index=False)
    print(f"✅ Final wallet behavior summary saved to: {output_file}")
