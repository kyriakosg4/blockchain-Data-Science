# src/cluster_errors.py

import os
import glob
import pandas as pd

os.makedirs("blocks", exist_ok=True)
os.makedirs("uops", exist_ok=True)
os.makedirs("data", exist_ok=True)

def extract_all_error_transactions(input_dir: str = ".", 
                                   pattern: str = "blocks/blocks_chunk_*.csv", 
                                   output_csv: str = "data/all_error_transactions.csv") -> None:
    """
    Extracts all transactions with status == 0 from block chunk CSVs.
    """
    file_pattern = os.path.join(input_dir, pattern)
    chunk_files = glob.glob(file_pattern)

    print(f"📁 Found {len(chunk_files)} chunk files.")
    error_chunks = []

    for file in chunk_files:
        df = pd.read_csv(file)
        error_df = df[df["status"] == 0]
        if not error_df.empty:
            error_chunks.append(error_df)
            print(f"✅ {file}: {len(error_df)} error rows found.")
        else:
            print(f"⚪ {file}: no errors.")

    if error_chunks:
        final_df = pd.concat(error_chunks, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"🎉 Saved {len(final_df)} error transactions to '{output_csv}'")
    else:
        print("⚠️ No error transactions found in any chunk.")

def cluster_from_block_errors(file_path: str = "data/all_error_transactions.csv", output_dir: str = "data") -> None:
    """
    Clusters errors by sender and block group using raw block data.
    """
    df_errors = pd.read_csv(file_path)
    total_errors = len(df_errors)

    # ----------------- By sender -----------------
    by_sender = df_errors.groupby("from").size().sort_values(ascending=False)
    df_sender = by_sender.reset_index()
    df_sender.columns = ["from", "total"]
    df_sender["percent_of_total"] = (df_sender["total"] / total_errors * 100).round(2)
    df_sender["rank"] = df_sender["total"].rank(method="first", ascending=False).astype(int)
    sender_path = os.path.join(output_dir, "error_cluster_by_sender.csv")
    df_sender.to_csv(sender_path, index=False)
    print(f"✅ Saved clustering by sender to '{sender_path}'.")

    # ----------------- By block group -----------------
    df_errors["block_group"] = df_errors["blockNumber"] // 1000 * 1000
    by_block = df_errors.groupby("block_group").size().sort_values(ascending=False)
    df_block = by_block.reset_index()
    df_block.columns = ["block_group (1000 blocks)", "total"]
    df_block["percent_of_total"] = (df_block["total"] / total_errors * 100).round(2)
    df_block["rank"] = df_block["total"].rank(method="first", ascending=False).astype(int)
    block_path = os.path.join(output_dir, "error_cluster_by_block.csv")
    df_block.to_csv(block_path, index=False)
    print(f"✅ Saved clustering by block group to '{block_path}'.")

def cluster_from_userops(directory: str = "uops", 
                         pattern: str = "user_operations_chunk_*.csv", 
                         output_dir: str = "data") -> None:

    """
    Summarizes failures in user operations by sender, bundler, and paymaster.
    """
    files = glob.glob(os.path.join(directory, pattern))
    all_dfs = [pd.read_csv(file) for file in files]
    merged_df = pd.concat(all_dfs, ignore_index=True)
    failed_df = merged_df[merged_df["success"] == False]

    # ============== Sender Summary ==============
    if "sender" in merged_df.columns:
        total_ops_sender = merged_df.groupby("sender").size().reset_index(name="total_ops")
        fail_sender = failed_df.groupby("sender").size().reset_index(name="fail_count")
        sender_summary = pd.merge(total_ops_sender, fail_sender, on="sender", how="left").fillna(0)
        sender_summary["fail_count"] = sender_summary["fail_count"].astype(int)
        sender_summary = sender_summary[sender_summary["fail_count"] > 0]
        sender_summary["failure_rate_percent"] = (sender_summary["fail_count"] / sender_summary["total_ops"] * 100).round(2)
        total_failures = sender_summary["fail_count"].sum()
        sender_summary["percent_of_total_failures"] = (sender_summary["fail_count"] / total_failures * 100).round(2)
        sender_summary["rank"] = sender_summary["fail_count"].rank(method="dense", ascending=False).astype(int)
        sender_summary = sender_summary.sort_values(by="fail_count", ascending=False).reset_index(drop=True)
        sender_summary.to_csv(os.path.join(output_dir, "summary_failures_per_sender.csv"), index=False)
        print("💾 Saved: summary_failures_per_sender.csv")

    # ============== Bundler Summary ==============
    if "bundler" in merged_df.columns:
        total_ops_bundler = merged_df.groupby("bundler").size().reset_index(name="total_ops")
        fail_bundler = failed_df.groupby("bundler").size().reset_index(name="fail_count")
        bundler_summary = pd.merge(total_ops_bundler, fail_bundler, on="bundler", how="left").fillna(0)
        bundler_summary["fail_count"] = bundler_summary["fail_count"].astype(int)
        bundler_summary = bundler_summary[bundler_summary["fail_count"] > 0]
        bundler_summary["failure_rate_percent"] = (bundler_summary["fail_count"] / bundler_summary["total_ops"] * 100).round(2)
        total_failures = bundler_summary["fail_count"].sum()
        bundler_summary["percent_of_total_failures"] = (bundler_summary["fail_count"] / total_failures * 100).round(2)
        bundler_summary["rank"] = bundler_summary["fail_count"].rank(method="dense", ascending=False).astype(int)
        bundler_summary = bundler_summary.sort_values(by="fail_count", ascending=False).reset_index(drop=True)
        bundler_summary.to_csv(os.path.join(output_dir, "summary_failures_per_bundler.csv"), index=False)
        print("💾 Saved: summary_failures_per_bundler.csv")

    # ============== Paymaster Summary ==============
    if "paymaster" in merged_df.columns:
        total_ops_paymaster = merged_df.groupby("paymaster").size().reset_index(name="total_ops")
        fail_paymaster = failed_df.groupby("paymaster").size().reset_index(name="fail_count")
        paymaster_summary = pd.merge(total_ops_paymaster, fail_paymaster, on="paymaster", how="left").fillna(0)
        paymaster_summary["fail_count"] = paymaster_summary["fail_count"].astype(int)
        paymaster_summary = paymaster_summary[paymaster_summary["fail_count"] > 0]
        paymaster_summary["failure_rate_percent"] = (paymaster_summary["fail_count"] / paymaster_summary["total_ops"] * 100).round(2)
        total_failures = paymaster_summary["fail_count"].sum()
        paymaster_summary["percent_of_total_failures"] = (paymaster_summary["fail_count"] / total_failures * 100).round(2)
        paymaster_summary["rank"] = paymaster_summary["fail_count"].rank(method="dense", ascending=False).astype(int)
        paymaster_summary = paymaster_summary.sort_values(by="fail_count", ascending=False).reset_index(drop=True)
        paymaster_summary.to_csv(os.path.join(output_dir, "summary_failures_per_paymaster.csv"), index=False)
        print("💾 Saved: summary_failures_per_paymaster.csv")

    print("✅ All summary CSVs generated successfully.")



