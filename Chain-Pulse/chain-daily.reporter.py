import boto3
import pandas as pd
import os
import io
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone   
from pathlib import Path                              
import re    

from dotenv import load_dotenv
load_dotenv()

LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"

# S3 (used when LOCAL_MODE = false)
BUCKET_NAME = os.getenv("S3_BUCKET", "chain-pulse-metrics")

# Local folders (used when LOCAL_MODE = true)
LOCAL_PROCESSED_DIR = os.getenv("PROCESSED_DIR", r"C:/.../processed-data-all-types")
LOCAL_REPORTS_DIR   = os.getenv("REPORTS_DIR",   r"C:/.../reports")
LOCAL_TEMPLATES_DIR = os.getenv("TEMPLATES_DIR", r"C:/.../reports/templates")

# Prefixes (same structure as S3)
PROCESSED_DATA_PATH = "processed-data-all-types/"
REPORTS_PATH        = "reports/"

# S3 client only if needed
s3 = None if LOCAL_MODE else boto3.client("s3")


# ---------- dual-mode helpers ----------

def _local_read_csv(path_abs: str):
    return pd.read_csv(path_abs) if os.path.exists(path_abs) else None

def _local_write_csv(df: pd.DataFrame, path_abs: str):
    os.makedirs(os.path.dirname(path_abs), exist_ok=True)
    df.to_csv(path_abs, index=False)
    print(f"✅ Saved locally: {path_abs}")

def _local_write_bytes(data: bytes, path_abs: str):
    os.makedirs(os.path.dirname(path_abs), exist_ok=True)
    with open(path_abs, "wb") as f:
        f.write(data)
    print(f"✅ Saved locally: {path_abs}")

def _rel_to_local_processed(path_rel: str) -> str:
    # path_rel starts with PROCESSED_DATA_PATH
    return os.path.join(LOCAL_PROCESSED_DIR, os.path.relpath(path_rel, PROCESSED_DATA_PATH))

def _rel_to_local_reports(path_rel: str) -> str:
    # path_rel starts with REPORTS_PATH
    return os.path.join(LOCAL_REPORTS_DIR, os.path.relpath(path_rel, REPORTS_PATH))

def read_csv(path_rel: str):
    """Read a CSV by S3-style relative path."""
    if LOCAL_MODE:
        if path_rel.startswith(PROCESSED_DATA_PATH):
            return _local_read_csv(_rel_to_local_processed(path_rel))
        if path_rel.startswith(REPORTS_PATH):
            return _local_read_csv(_rel_to_local_reports(path_rel))
        # fallback (processed)
        return _local_read_csv(_rel_to_local_processed(path_rel))
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=path_rel)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        print(f"❌ read_csv failed for {path_rel}: {e}")
        return None


def write_csv(df: pd.DataFrame, path_rel: str):
    """Write a CSV by S3-style relative path under reports/."""
    if LOCAL_MODE:
        _local_write_csv(df, _rel_to_local_reports(path_rel))
        return
    csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=path_rel, Body=csv_buf.getvalue())
    print(f"✅ Uploaded CSV: {path_rel}")

def save_fig(fig, path_rel: str):
    """Save a Matplotlib figure to PNG."""
    img = io.BytesIO(); fig.savefig(img, format="png", bbox_inches="tight"); img.seek(0)
    if LOCAL_MODE:
        _local_write_bytes(img.getvalue(), _rel_to_local_reports(path_rel))
    else:
        s3.put_object(Bucket=BUCKET_NAME, Key=path_rel, Body=img.getvalue(), ContentType="image/png")
        print(f"✅ Uploaded plot: {path_rel}")

def read_template(path_rel: str) -> str:
    """Read the markdown template."""
    if LOCAL_MODE:
        full = os.path.join(LOCAL_TEMPLATES_DIR, os.path.basename(path_rel))
        with open(full, "r", encoding="utf-8") as f:
            return f.read()
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=path_rel)
    return obj["Body"].read().decode("utf-8")

def write_markdown(text: str, path_rel: str):
    """Write the final markdown report."""
    if LOCAL_MODE:
        full = _rel_to_local_reports(path_rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Report saved locally: {full}")
    else:
        s3.put_object(Bucket=BUCKET_NAME, Key=path_rel, Body=text.encode("utf-8"),
                      ContentType="text/markdown")
        print(f"✅ Report uploaded: {path_rel}")


def calculate_percentage_change(current, previous):
    """Calculates the percentage change from previous to current."""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return None  # Avoid division errors
    return round(((current - previous) / previous) * 100, 2)



# Function to dynamically rename and clean data
def rename_and_clean_data(df, dataset_type, date1, date2):
    """
    Rename columns and clean up data based on the dataset type dynamically.
    
    Parameters:
    - df: The DataFrame to process.
    - dataset_type: The type of dataset ('network', 'economic', 'staking').
    - date1: Previous date string.
    - date2: Current date string.
    """

    # Define column mappings dynamically
    column_mappings = {
        "network": {
            "timestamp": "Timestamp",
            "chain": "Blockchain",
            f"volume ({date2})": f"Transaction Volume ({date2})",
            f"volume ({date1})": f"Transaction Volume ({date1})",
            "volume Change (%)": "Transaction Volume Change (%)",
            f"active_validators ({date2})": f"Active Validators ({date2})",
            f"active_validators ({date1})": f"Active Validators ({date1})",
            "active_validators Change (%)": "Active Validators Change (%)",
            f"gas_price_gwei ({date2})": f"Gas Price ({date2})",
            f"gas_price_gwei ({date1})": f"Gas Price ({date1})",
            "gas_price_gwei Change (%)": "Gas Price Change (%)",
            f"avg_cu_per_tx ({date2})": f"Avg CU per Tx ({date2})",
            f"avg_cu_per_tx ({date1})": f"Avg CU per Tx ({date1})",
            "avg_cu_per_tx Change (%)": "Avg CU per Tx Change (%)",
            f"cost_per_cu_lamports ({date2})": f"Cost per CU (Lamports) ({date2})",
            f"cost_per_cu_lamports ({date1})": f"Cost per CU (Lamports) ({date1})",
            "cost_per_cu_lamports Change (%)": "Cost per CU (Lamports) Change (%)",
            f"avg_gas_used ({date2})": f"Avg Gas Used ({date2})",
            f"avg_gas_used ({date1})": f"Avg Gas Used ({date1})",
            "avg_gas_used Change (%)": "Avg Gas Used Change (%)",
            f"transaction_fee ({date2})": f"Transaction Fee ({date2})",
            f"transaction_fee ({date1})": f"Transaction Fee ({date1})",
            "transaction_fee Change (%)": "Transaction Fee Change (%)",
            f"tps ({date2})": f"TPS ({date2})",
            f"tps ({date1})": f"TPS ({date1})",
            "tps Change (%)": "TPS Change (%)",
        },
        "economic": {
            "timestamp": "Timestamp (UTC)",
            "chain": "Blockchain",
            f"price ({date2})": f"Price ({date2})",
            f"price ({date1})": f"Price ({date1})",
            "price Change (%)": "Price Change (%)",
            f"usd_market_cap ({date2})": f"Market Cap ({date2})",
            f"usd_market_cap ({date1})": f"Market Cap ({date1})",
            "usd_market_cap Change (%)": "Market Cap Change (%)",
            f"usd_24h_vol ({date2})": f"Trading Volume ({date2})",
            f"usd_24h_vol ({date1})": f"Trading Volume ({date1})",
            "usd_24h_vol Change (%)": "Trading Volume Change (%)",
            f"circulating_supply ({date2})": f"Circulating Supply ({date2})",
            f"circulating_supply ({date1})": f"Circulating Supply ({date1})",
            "circulating_supply Change (%)": "Circulating Supply Change (%)",
            f"inflation_rate ({date2})": f"Inflation Rate ({date2})",
            f"inflation_rate ({date1})": f"Inflation Rate ({date1})",
            "inflation_rate Change (%)": "Inflation Rate Change (%)",
            f"total_supply ({date2})": f"Total Supply ({date2})",
            f"total_supply ({date1})": f"Total Supply ({date1})",
            "total_supply Change (%)": "Total Supply Change (%)"
        },
        "staking": {
            "timestamp": "Timestamp",
            "chain": "Blockchain",
            f"reward_rate ({date2})": f"Reward Rate ({date2})",
            f"reward_rate ({date1})": f"Reward Rate ({date1})",
            "reward_rate Change (%)": "Reward Rate Change (%)",
            f"staked_tokens ({date2})": f"Staked Tokens ({date2})",
            f"staked_tokens ({date1})": f"Staked Tokens ({date1})",
            "staked_tokens Change (%)": "Staked Tokens Change (%)",
            f"staking_ratio ({date2})": f"Staking Ratio ({date2})",
            f"staking_ratio ({date1})": f"Staking Ratio ({date1})",
            "staking_ratio Change (%)": "Staking Ratio Change (%)",
            f"tvl ({date2})": f"TVL ({date2})",
            f"tvl ({date1})": f"TVL ({date1})",
            "tvl Change (%)": "TVL Change (%)"
        }
    }
    

    # Apply the renaming logic if dataset_type exists in the mappings
    if dataset_type in column_mappings:
        df.columns = df.columns.str.strip()
        df.rename(columns=column_mappings[dataset_type], inplace=True)

    # Define columns to remove based on dataset type
    columns_to_remove = []
    if dataset_type == "economic":
        columns_to_remove = [
            f"usd_24h_change ({date2})",
            f"usd_24h_change ({date1})",
            "usd_24h_change Change (%)"
        ]

    # Drop unwanted columns
    df.drop(columns=columns_to_remove, inplace=True, errors='ignore')

    # Convert change columns to numeric
    change_columns = [col for col in df.columns if 'Change (%)' in col]
    for col in change_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')



    
    return df


def format_timestamps(df):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    elif 'Timestamp (UTC)' in df.columns:
        df['Timestamp (UTC)'] = pd.to_datetime(df['Timestamp (UTC)']).dt.strftime('%Y-%m-%d %H:%M')
    return df



# Function to apply consistent formatting
def apply_formatting(df, for_table=False):
    def format_large_values(value):
        if pd.isnull(value) or not isinstance(value, (int, float)):
            return value  # Skip formatting for NaN or non-numeric values
        if value >= 1_000_000_000:
            return f"${value / 1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        elif value >= 1_000:
            return f"${value / 1_000:.2f}K"
        else:
            return f"${value:,.2f}"

    def format_percentage(value):
        if pd.isnull(value) or not isinstance(value, (int, float)):
            return value  # Skip formatting for NaN or non-numeric values
        return f"{value:.2f}%"

    # Format columns for large values or percentages
    for col in df.columns:
        if '_price' in col or '_volume' in col or '_supply' in col or 'gas_price' in col:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Ensure numeric values
            df[col] = df[col].apply(format_large_values)
        elif '_change' in col or 'Change (%)' in col:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Ensure numeric values
            df[col] = df[col].apply(format_percentage)

    return df

BLOCKCHAIN_COLORS = {
    "Avalanche": "red",
    "Ethereum": "green",
    "Solana": "blue",
    "Binance": "orange"
}

def generate_transaction_volume_chart(network_data, date):
    """
    Generates a multi-line chart for transaction volume trends and a grouped bar chart for percentage changes.
    Parameters:
    - network_data: DataFrame containing transaction volume and change data.
    - date: Date to dynamically select the transaction volume column.
    """
    volume_column = f"Transaction Volume ({date})"
    change_column = f"Transaction Volume Change (%)"

    if volume_column not in network_data.columns or change_column not in network_data.columns:
        print(f"❌ Missing required columns for date {date}. Skipping chart generation.")
        return

    network_data[change_column] = network_data[change_column].astype(str).str.replace('%', '', regex=False)
    network_data[change_column] = pd.to_numeric(network_data[change_column], errors='coerce')

    network_data = network_data[~network_data['Blockchain'].str.contains("Average", na=False)]
    network_data = network_data[~network_data['Timestamp'].str.contains("Average", na=False)]

    blockchains = network_data['Blockchain'].unique()
    timestamps = [t for t in network_data['Timestamp'].unique() if "Average" not in t]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 14), gridspec_kw={'height_ratios': [2, 2]})
    plt.subplots_adjust(hspace=0.6, bottom=0.12, top=0.95)

    for blockchain in blockchains:
        blockchain_data = network_data[network_data['Blockchain'] == blockchain]
        blockchain_data = blockchain_data[~blockchain_data['Timestamp'].str.contains("Average", na=False)]
        color = BLOCKCHAIN_COLORS.get(blockchain, "gray")
        axes[0].plot(blockchain_data['Timestamp'], blockchain_data[volume_column], marker='o', linestyle='-', label=blockchain, color=color)

    axes[0].set_title("Transaction Volume Trends Over Time")
    axes[0].set_xlabel("Timestamp")
    axes[0].set_ylabel("Transaction Volume")
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

    bar_width = 0.2
    gap = 0.8

    x_labels = []
    x_positions = []
    current_position = 0

    for blockchain in blockchains:
        x_labels.append(blockchain)
        x_positions.append(current_position + (bar_width * len(timestamps) / 2))
        current_position += (bar_width * len(timestamps)) + gap

    for i, timestamp in enumerate(timestamps):
        timestamp_data = network_data[network_data['Timestamp'] == timestamp]
        y_values = timestamp_data[change_column].fillna(0).values
        colors = ['green' if v > 0 else 'red' for v in y_values]
        bars = axes[1].bar(np.array(x_positions) + i * bar_width - (bar_width * len(timestamps) / 2), y_values, width=bar_width, color=colors)

        for bar, v in zip(bars, y_values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}%", ha='center', va='bottom' if v > 0 else 'top', fontsize=10, fontweight='bold', rotation=90)

    axes[1].set_title("Percentage Changes by Blockchain (Each Timestamp)")
    axes[1].set_xlabel("Blockchains")
    axes[1].set_ylabel("Change (%)")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(x_labels, rotation=90)

    y_min, y_max = axes[1].get_ylim()
    padding = (y_max - y_min) * 0.2
    axes[1].set_ylim(y_min - padding, y_max + padding)

    month_folder = date[:7]
    chart_s3_path = f"{REPORTS_PATH}{month_folder}/{date}/visualizations/network_charts/volume_comparison.png"
    save_fig(fig, chart_s3_path)
    plt.close()


def generate_market_trends_chart(economic_data, date):
    """
    Generates a dual-subplot chart for Price and Market Cap trends over time with daily percentage changes.
    """
    change_columns = ["Price Change (%)", "Market Cap Change (%)"]

    # ✅ Convert to numeric and remove '%'
    for col in change_columns:
        economic_data[col] = economic_data[col].astype(str).str.replace('%', '', regex=False)
        economic_data[col] = pd.to_numeric(economic_data[col], errors='coerce')

    # ✅ Filter out "Average" rows to prevent shape mismatch
    economic_data = economic_data[~economic_data['Timestamp (UTC)'].str.contains("Average", na=False)]

    blockchains = economic_data['Blockchain'].unique()
    timestamps = economic_data['Timestamp (UTC)'].unique()
    index = np.arange(len(timestamps))  # ✅ Ensure index length matches timestamps

    price_column = f"Price ({date})"
    market_cap_column = f"Market Cap ({date})"
    price_change_column = "Price Change (%)"
    market_cap_change_column = "Market Cap Change (%)"

    # ✅ Adjust figure size for better spacing
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=100)
    bar_width = 0.25

    # ✅ Compute dynamic y-axis ranges
    price_change_range = [economic_data[price_change_column].dropna().min() - 1, 
                          economic_data[price_change_column].dropna().max() + 1]
    market_cap_change_range = [economic_data[market_cap_change_column].dropna().min() - 1, 
                               economic_data[market_cap_change_column].dropna().max() + 1]

    # --- Price Trends (Left Subplot) ---
    ax1 = axes[0]
    ax2 = ax1.twinx()

    for i, blockchain in enumerate(blockchains):
        data = economic_data[economic_data['Blockchain'] == blockchain]
        if len(data) == len(index):  # ✅ Prevent broadcasting errors
            ax2.bar(index + i * bar_width, data[price_change_column].values, width=bar_width,
                    label=f"{blockchain} Change (%)", alpha=0.7)

    for blockchain in blockchains:
        data = economic_data[economic_data['Blockchain'] == blockchain]
        color = BLOCKCHAIN_COLORS.get(blockchain, "gray")  # ✅ Apply stable colors
        ax1.plot(data['Timestamp (UTC)'].values, data[price_column].values, marker='o', markersize=6,
                 label=blockchain, linewidth=2, color=color)

    ax1.set_xlabel("Timestamp (UTC)", fontsize=12)
    ax1.set_ylabel("Price", fontsize=12)
    ax2.set_ylabel("Price Change (%)", fontsize=12)
    ax1.set_title("Price Trends with Daily Changes", fontsize=14, fontweight="bold")
    ax1.set_xticks(index)
    ax1.set_xticklabels(timestamps, rotation=45, fontsize=10)
    ax2.set_ylim(price_change_range)

    # ❌ Remove legends from the left subplot
    ax1.legend().remove()
    ax2.legend().remove()

    # --- Market Cap Trends (Right Subplot) ---
    ax3 = axes[1]
    ax4 = ax3.twinx()

    for i, blockchain in enumerate(blockchains):
        data = economic_data[economic_data['Blockchain'] == blockchain]
        if len(data) == len(index):
            ax4.bar(index + i * bar_width, data[market_cap_change_column].values, width=bar_width,
                    label=f"{blockchain} Change (%)", alpha=0.7)

    for blockchain in blockchains:
        data = economic_data[economic_data['Blockchain'] == blockchain]
        color = BLOCKCHAIN_COLORS.get(blockchain, "gray")  # ✅ Apply stable colors
        ax3.plot(data['Timestamp (UTC)'].values, data[market_cap_column].values, marker='o', markersize=6,
                 label=blockchain, linewidth=2, color=color)

    ax3.set_xlabel("Timestamp (UTC)", fontsize=12)
    ax3.set_ylabel("Market Cap", fontsize=12)
    ax4.set_ylabel("Market Cap Change (%)", fontsize=12)
    ax3.set_title("Market Cap Trends with Daily Changes", fontsize=14, fontweight="bold")
    ax3.set_xticks(index)
    ax3.set_xticklabels(timestamps, rotation=45, fontsize=10)
    ax4.set_ylim(market_cap_change_range)

    # ✅ Fix Legend Overlap: Move Lower, Use Two Columns, Reduce Spacing
    ax3.legend(loc="upper left", bbox_to_anchor=(1.2, 0.6), fontsize=10, 
               frameon=False, ncol=2, handletextpad=0.5, columnspacing=1.5)
    ax4.legend(loc="upper left", bbox_to_anchor=(1.2, 0.45), fontsize=10, 
               frameon=False, ncol=2, handletextpad=0.5, columnspacing=1.5)

    # ✅ Adjust layout for better spacing
    plt.subplots_adjust(wspace=0.4, right=0.75)  # More spacing between subplots

    # ✅ Save optimized chart
    month_folder = date[:7]
    chart_s3_path = f"{REPORTS_PATH}{month_folder}/{date}/visualizations/economic_charts/market_trends.png"
    save_fig(fig, chart_s3_path)

    # ✅ Explicitly free memory
    plt.close(fig)



def generate_trading_volume_chart(economic_data, date):
    """
    Generates a multi-line chart for trading volume trends and a grouped bar chart for percentage changes.
    """
    volume_column = f"Trading Volume ({date})"
    change_column = "Trading Volume Change (%)"
    
    # Ensure columns exist
    if volume_column not in economic_data.columns or change_column not in economic_data.columns:
        print(f"❌ Missing required columns for date {date}. Skipping chart generation.")
        return

    economic_data[change_column] = economic_data[change_column].astype(str).str.replace('%', '', regex=False)
    economic_data[change_column] = pd.to_numeric(economic_data[change_column], errors='coerce')
    
    # Remove "Average" rows
    economic_data = economic_data[~economic_data['Blockchain'].str.contains("Average", na=False)]
    economic_data = economic_data[~economic_data['Timestamp (UTC)'].str.contains("Average", na=False)]

    blockchains = economic_data['Blockchain'].unique()
    timestamps = [t for t in economic_data['Timestamp (UTC)'].unique() if "Average" not in t]
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 14), gridspec_kw={'height_ratios': [2, 2]})
    plt.subplots_adjust(hspace=0.6, bottom=0.12, top=0.95)
    
    # First Subplot: Multi-line chart for trading volume trends
    for blockchain in blockchains:
        blockchain_data = economic_data[economic_data['Blockchain'] == blockchain]
        color = BLOCKCHAIN_COLORS.get(blockchain, "gray")  # ✅ Apply stable colors
        axes[0].plot(blockchain_data['Timestamp (UTC)'], blockchain_data[volume_column], 
                     marker='o', linestyle='-', label=blockchain, color=color)
    
    axes[0].set_title("Trading Volume Trends Over Time")
    axes[0].set_xlabel("Timestamp (UTC)")
    axes[0].set_ylabel("Trading Volume")
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Second Subplot: Grouped Bar Chart for Trading Volume Change (%)
    bar_width = 0.2
    gap = 0.8

    x_labels = []
    x_positions = []
    current_position = 0

    for blockchain in blockchains:
        x_labels.append(blockchain)
        x_positions.append(current_position + (bar_width * len(timestamps) / 2))  
        current_position += (bar_width * len(timestamps)) + gap

    for i, timestamp in enumerate(timestamps):
        timestamp_data = economic_data[economic_data['Timestamp (UTC)'] == timestamp]
        y_values = timestamp_data[change_column].fillna(0).values

        colors = ['green' if v > 0 else 'red' for v in y_values]

        bars = axes[1].bar(np.array(x_positions) + i * bar_width - (bar_width * len(timestamps) / 2), 
                           y_values, width=bar_width, color=colors)
        for bar, v in zip(bars, y_values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}%", 
                         ha='center', va='bottom' if v > 0 else 'top', 
                         fontsize=10, fontweight='bold', rotation=90)

    axes[1].set_title("Trading Volume Percentage Changes by Blockchain")
    axes[1].set_xlabel("Blockchains")
    axes[1].set_ylabel("Change (%)")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(x_labels, rotation=90)

    y_min, y_max = axes[1].get_ylim()
    padding = (y_max - y_min) * 0.2
    axes[1].set_ylim(y_min - padding, y_max + padding)

    # Save chart to S3
    month_folder = date[:7]
    chart_s3_path = f"{REPORTS_PATH}{month_folder}/{date}/visualizations/economic_charts/trading_volume.png"
    save_fig(fig, chart_s3_path)
    plt.close()




def generate_supply_comparison_chart(economic_data, date):
    """
    Generates grouped bar charts for Circulating and Total Supply comparison with percentage changes.
    """
    # Define dynamic column names
    circulating_supply_column = f"Circulating Supply ({date})"
    total_supply_column = f"Total Supply ({date})"
    circulating_change_column = "Circulating Supply Change (%)"
    total_change_column = "Total Supply Change (%)"

    # Ensure numeric conversion for supply values
    for column in [circulating_supply_column, total_supply_column]:
        if column in economic_data.columns:
            economic_data[column] = pd.to_numeric(economic_data[column], errors='coerce')

    # Ensure numeric conversion for percentage changes (removing % signs if necessary)
    for column in [circulating_change_column, total_change_column]:
        if column in economic_data.columns:
            economic_data[column] = economic_data[column].astype(str).str.replace('%', '', regex=False)
            economic_data[column] = pd.to_numeric(economic_data[column], errors='coerce')

    # Remove "Average" rows
    economic_data = economic_data[~economic_data['Blockchain'].str.contains("Average", na=False)]
    economic_data = economic_data[~economic_data['Timestamp (UTC)'].str.contains("Average", na=False)]

    if economic_data.empty:
        print(f"❌ No valid data for Supply Comparison on {date}. Skipping chart generation.")
        return

    # Extract blockchains and timestamps
    blockchains = economic_data['Blockchain'].unique()
    timestamps = [t for t in economic_data['Timestamp (UTC)'].unique() if "Average" not in t]

    # Create figure with two subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 12))
    bar_width = 0.2
    gap = 0.8

    x_labels = []
    x_positions = []
    current_position = 0

    for timestamp in timestamps:
        x_labels.append(timestamp)
        x_positions.append(current_position + (bar_width * len(blockchains) / 2))
        current_position += (bar_width * len(blockchains)) + gap

    # Circulating Supply Subplot
    for i, blockchain in enumerate(blockchains):
        blockchain_data = economic_data[economic_data['Blockchain'] == blockchain]
        y_values = blockchain_data[circulating_supply_column].values
        change_values = blockchain_data[circulating_change_column].fillna(0).values
        color = BLOCKCHAIN_COLORS.get(blockchain, "gray")  # ✅ Apply stable colors

        bars = axes[0].bar(
            np.array(x_positions) + i * bar_width - (bar_width * len(blockchains) / 2),
            y_values,
            width=bar_width,
            color=color,
            label=blockchain,
            alpha=0.8
        )

        # Add percentage change as text labels
        for bar, change in zip(bars, change_values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{change:.2f}%",
                ha='center',
                va='bottom' if change > 0 else 'top',
                fontsize=10,
                fontweight='bold',
                color='black' if change > 0 else 'red',
                rotation=90
            )

    axes[0].set_title("Circulating Supply Comparison", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Timestamps", fontsize=12)
    axes[0].set_ylabel("Circulating Supply", fontsize=12)
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[0].legend(title="Blockchains", loc="upper left", bbox_to_anchor=(1, 1))

    # Adjust y-limits dynamically to prevent labels from exceeding the plot
    y_min, y_max = axes[0].get_ylim()
    padding = (y_max - y_min) * 0.25
    axes[0].set_ylim(y_min, y_max + padding)

    # Total Supply Subplot
    for i, blockchain in enumerate(blockchains):
        blockchain_data = economic_data[economic_data['Blockchain'] == blockchain]
        y_values = blockchain_data[total_supply_column].values
        change_values = blockchain_data[total_change_column].fillna(0).values
        color = BLOCKCHAIN_COLORS.get(blockchain, "gray")  # ✅ Apply stable colors

        bars = axes[1].bar(
            np.array(x_positions) + i * bar_width - (bar_width * len(blockchains) / 2),
            y_values,
            width=bar_width,
            color=color,
            label=blockchain,
            alpha=0.8
        )

        # Add percentage change as text labels
        for bar, change in zip(bars, change_values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{change:.2f}%",
                ha='center',
                va='bottom' if change > 0 else 'top',
                fontsize=10,
                fontweight='bold',
                color='black' if change > 0 else 'red',
                rotation=90
            )

    axes[1].set_title("Total Supply Comparison", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Timestamps", fontsize=12)
    axes[1].set_ylabel("Total Supply", fontsize=12)
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[1].legend(title="Blockchains", loc="upper left", bbox_to_anchor=(1, 1))

    # Adjust y-limits dynamically to prevent labels from exceeding the plot
    y_min, y_max = axes[1].get_ylim()
    padding = (y_max - y_min) * 0.25
    axes[1].set_ylim(y_min, y_max + padding)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(hspace=0.5)

    # Save chart to S3
    month_folder = date[:7]
    chart_s3_path = f"{REPORTS_PATH}{month_folder}/{date}/visualizations/economic_charts/supply_comparison.png"
    save_fig(fig, chart_s3_path)
    plt.close()




def generate_gas_fees_trends_chart(network_data, date):
    """
    Generates a grouped bar chart with two subplots:
    1. Gas Price Trends (Avalanche, Binance, Ethereum)
    2. Transaction Fee Trends (All blockchains including Solana)
    Each subplot has its own legend and data labels.
    """
    # Define dynamic column names
    gas_price_column = f"Gas Price ({date})"
    gas_change_column = "Gas Price Change (%)"
    transaction_fee_column = f"Transaction Fee ({date})"
    transaction_fee_change_column = "Transaction Fee Change (%)"

    # Ensure numeric conversion for Gas Prices and Transaction Fees
    for col in [gas_price_column, gas_change_column, transaction_fee_column, transaction_fee_change_column]:
        if col in network_data.columns:
            network_data[col] = pd.to_numeric(network_data[col], errors='coerce')

    # Filter data for the first subplot (Gas Price Trends) - Only Avalanche, Binance, Ethereum
    gas_data = network_data[
        (network_data['Blockchain'].isin(["Avalanche", "Binance", "Ethereum"])) & 
        (~network_data['Blockchain'].str.contains("Average", na=False)) & 
        (~network_data['Timestamp'].str.contains("Average", na=False))
    ].copy()

    # Filter data for the second subplot (Transaction Fee Trends) - Includes Solana
    fee_data = network_data[
        (~network_data['Blockchain'].str.contains("Average", na=False)) & 
        (~network_data['Timestamp'].str.contains("Average", na=False))
    ].copy()

    if gas_data.empty or fee_data.empty:
        print(f"❌ No valid data for Gas Price or Transaction Fee trends on {date}. Skipping chart generation.")
        return

    # Extract timestamps that exist in both datasets
    valid_timestamps = sorted(set(gas_data["Timestamp"]).intersection(set(fee_data["Timestamp"])))
    gas_data = gas_data[gas_data["Timestamp"].isin(valid_timestamps)]
    fee_data = fee_data[fee_data["Timestamp"].isin(valid_timestamps)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 16))  # Two subplots
    bar_width = 0.2  # Width of bars

    def create_bar_chart(ax, data, value_col, change_col, title, legend_title):
        """Helper function to create grouped bar charts with labels."""
        blockchains = data['Blockchain'].unique()
        x_positions = np.arange(len(valid_timestamps)) * (len(blockchains) * bar_width + 0.2)

        for i, blockchain in enumerate(blockchains):
            blockchain_data = data[data['Blockchain'] == blockchain]

            # Ensure timestamps match, filling missing ones
            blockchain_data = blockchain_data.set_index("Timestamp").reindex(valid_timestamps).fillna(0).reset_index()

            y_values = blockchain_data[value_col].fillna(0).values
            change_values = blockchain_data[change_col].fillna(0).values  # Replace NaN% with 0%
            color = BLOCKCHAIN_COLORS.get(blockchain, "gray")  # ✅ Apply stable colors

            bars = ax.bar(
                x_positions + i * bar_width - (bar_width * len(blockchains) / 2),
                y_values,
                width=bar_width,
                color=color,
                label=blockchain
            )

            # Add percentage change as text labels
            for bar, change in zip(bars, change_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.01 * max(y_values)),  # Adjust position
                    f"{change:.2f}%",
                    ha='center',
                    va='bottom' if change >= 0 else 'top',
                    fontsize=10,
                    fontweight='bold',
                    color='black' if change >= 0 else 'red',
                    rotation=90
                )

        # Customize chart labels and title
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Timestamps", fontsize=12)
        ax.set_ylabel(value_col, fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(valid_timestamps, rotation=45, ha='right')

        # Move legend outside but within figure bounds
        ax.legend(title=legend_title, loc="center left", bbox_to_anchor=(1, 0.5))

        # Adjust y-limits dynamically to prevent labels from exceeding the plot
        y_min, y_max = ax.get_ylim()
        padding = (y_max - y_min) * 0.25  # Increase padding to ensure labels fit
        ax.set_ylim(y_min, y_max + padding)

    # Create first subplot (Gas Prices)
    create_bar_chart(
        axes[0], gas_data, gas_price_column, gas_change_column, 
        "Gas Price Trends with Percentage Changes", "Gas Price - Blockchains"
    )

    # Create second subplot (Transaction Fees)
    create_bar_chart(
        axes[1], fee_data, transaction_fee_column, transaction_fee_change_column, 
        "Transaction Fee Trends with Percentage Changes", "Transaction Fee - Blockchains"
    )

    # Adjust layout so everything is visible
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save chart to S3
    month_folder = date[:7]
    chart_s3_path = f"{REPORTS_PATH}{month_folder}/{date}/visualizations/network_charts/gas_and_fee_trends.png"
    save_fig(fig, chart_s3_path)
    plt.close()




def generate_staking_comparison_chart(staking_data, date):
    """
    Generates a grouped bar chart for staking ratio comparison with percentage changes.
    """
    # Define dynamic column names
    staking_column = f"Staking Ratio ({date})"
    change_column = "Staking Ratio Change (%)"

    # Ensure numeric conversion for Staking Ratio and Change Percentage
    if staking_column in staking_data.columns and change_column in staking_data.columns:
        staking_data[staking_column] = pd.to_numeric(staking_data[staking_column], errors='coerce')
        staking_data[change_column] = (
            staking_data[change_column]
            .astype(str)
            .str.replace('%', '', regex=False)
            .replace('', '0')
            .astype(float)
        )

    # Remove "Average" rows from both blockchains and timestamps
    staking_data = staking_data[~staking_data['Blockchain'].str.contains("Average", na=False)]
    staking_data = staking_data[~staking_data['Timestamp'].str.contains("Average", na=False)]  # Extra safeguard

    if staking_data.empty:
        print(f"❌ No valid data for Staking Ratio on {date}. Skipping chart generation.")
        return

    # Extract blockchains and timestamps (excluding "Average" timestamps)
    blockchains = staking_data['Blockchain'].unique()
    timestamps = [t for t in staking_data['Timestamp'].unique() if "Average" not in t]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 9))  # Increased height for better spacing
    bar_width = 0.2  # Width of bars
    gap = 0.8  # Space between groups

    x_labels = []
    x_positions = []
    current_position = 0

    for timestamp in timestamps:
        x_labels.append(timestamp)
        x_positions.append(current_position + (bar_width * len(blockchains) / 2))
        current_position += (bar_width * len(blockchains)) + gap

    for i, blockchain in enumerate(blockchains):
        blockchain_data = staking_data[staking_data['Blockchain'] == blockchain]
        blockchain_data = blockchain_data[~blockchain_data['Timestamp'].str.contains("Average", na=False)]  # Extra filter
        y_values = blockchain_data[staking_column].values
        change_values = blockchain_data[change_column].fillna(0).values  # Replace NaN with 0%
        color = BLOCKCHAIN_COLORS.get(blockchain, 'gray')  # ✅ Apply stable colors

        bars = ax.bar(
            np.array(x_positions) + i * bar_width - (bar_width * len(blockchains) / 2),
            y_values,
            width=bar_width,
            color=color,
            label=blockchain,
            alpha=0.8
        )

        # Add percentage change as text labels
        for bar, change in zip(bars, change_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{change:.2f}%",
                ha='center',
                va='bottom' if change > 0 else 'top',
                fontsize=10,
                fontweight='bold',
                color='black' if change > 0 else 'red',
                rotation=90
            )

    # Customize chart labels and title
    ax.set_title("Staking Ratio Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Timestamps", fontsize=12)
    ax.set_ylabel("Staking Ratio (%)", fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend(title="Blockchains", loc="upper left", bbox_to_anchor=(1, 1))

    # Adjust y-limits dynamically to prevent labels from exceeding the plot
    y_min, y_max = ax.get_ylim()
    padding = (y_max - y_min) * 0.25  # Increase padding to ensure labels fit
    ax.set_ylim(y_min, y_max + padding)

    # Save chart to S3
    month_folder = date[:7]
    chart_s3_path = f"{REPORTS_PATH}{month_folder}/{date}/visualizations/staking_charts/staking_comparison.png"
    save_fig(fig, chart_s3_path)
    plt.close()



def generate_tps_comparison_chart(network_data, date):
    """Generates a multi-line TPS trends chart and a grouped bar chart for TPS changes."""
    
    tps_column = f"TPS ({date})"
    change_column = "TPS Change (%)"

    if tps_column not in network_data.columns or change_column not in network_data.columns:
        print(f"❌ Missing required TPS columns: {tps_column}, {change_column}")
        return

    network_data[change_column] = network_data[change_column].astype(str).str.replace('%', '', regex=False)
    network_data[change_column] = pd.to_numeric(network_data[change_column], errors='coerce')

    network_data = network_data[~network_data['Blockchain'].str.contains("Average", na=False)]
    network_data = network_data[~network_data['Timestamp'].str.contains("Average", na=False)]

    blockchains = network_data['Blockchain'].unique()
    timestamps = [t for t in network_data['Timestamp'].unique() if "Average" not in t]

    print(f"🟢 Blockchains found: {blockchains}")
    print(f"🟢 Timestamps found: {timestamps}")

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 14), gridspec_kw={'height_ratios': [2, 2]})
    plt.subplots_adjust(hspace=0.6, bottom=0.12, top=0.95)

    for blockchain in blockchains:
        blockchain_data = network_data[network_data['Blockchain'] == blockchain]
        axes[0].plot(blockchain_data['Timestamp'], blockchain_data[tps_column],
                     marker='o', linestyle='-', label=blockchain)

    axes[0].set_title("TPS Trends Over Time")
    axes[0].set_xlabel("Timestamp")
    axes[0].set_ylabel("Transactions Per Second (TPS)")
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

    bar_width = 0.2
    gap = 0.8

    x_labels = []
    x_positions = []
    current_position = 0

    for blockchain in blockchains:
        x_labels.append(blockchain)
        x_positions.append(current_position + (bar_width * len(timestamps) / 2))
        current_position += (bar_width * len(timestamps)) + gap

    for i, timestamp in enumerate(timestamps):
        timestamp_data = network_data[network_data['Timestamp'] == timestamp]
        y_values = timestamp_data[change_column].fillna(0).values
        colors = ['green' if v > 0 else 'red' for v in y_values]

        bars = axes[1].bar(np.array(x_positions) + i * bar_width - (bar_width * len(timestamps) / 2),
                           y_values, width=bar_width, color=colors)
        for bar, v in zip(bars, y_values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}%",
                         ha='center', va='bottom' if v > 0 else 'top',
                         fontsize=10, fontweight='bold', rotation=90)

    axes[1].set_title("TPS Percentage Changes by Blockchain")
    axes[1].set_xlabel("Blockchains")
    axes[1].set_ylabel("Change (%)")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(x_labels, rotation=90)

    y_min, y_max = axes[1].get_ylim()
    padding = (y_max - y_min) * 0.2
    axes[1].set_ylim(y_min - padding, y_max + padding)

    # ✅ Debug print before saving chart
    month_folder = date[:7]
    chart_s3_path = f"{REPORTS_PATH}{month_folder}/{date}/visualizations/network_charts/tps_comparison.png"

    print(f"📊 Saving TPS Comparison Chart to {chart_s3_path}")  # Debug print

    save_fig(fig, chart_s3_path)

    print(f"✅ TPS Comparison Chart uploaded successfully!")  # Confirm upload
    plt.close()




# Function to generate network metrics CSV
def generate_network_metrics_csv(date1_str, date2_str):
    """
    Generates a network metrics CSV with daily comparisons and saves it to S3.
    Ensures that the "Change (%)" values in the average row are computed based on
    the averaged metric values instead of averaging individual percentage changes.
    """

    month_folder_today = date2_str[:7]
    month_folder_yesterday = date1_str[:7]
    input_path_today = f"{PROCESSED_DATA_PATH}{month_folder_today}/{date2_str}/network_metrics.csv"
    input_path_yesterday = f"{PROCESSED_DATA_PATH}{month_folder_yesterday}/{date1_str}/network_metrics.csv"
    output_path = f"{REPORTS_PATH}{month_folder_today}/{date2_str}/data/network_metrics.csv"

    # ✅ Download files from S3
    metrics_today = read_csv(input_path_today)
    metrics_yesterday = read_csv(input_path_yesterday)

    if metrics_today is None or metrics_yesterday is None:
        print("❌ Missing one or both CSV files. Exiting.")
        return

    try:
        # ✅ Rename Solana's transaction cost to transaction fee
        metrics_today.rename(columns={"solana_transaction_cost": "solana_transaction_fee"}, inplace=True)
        metrics_yesterday.rename(columns={"solana_transaction_cost": "solana_transaction_fee"}, inplace=True)

        # ✅ Remove unnecessary transaction cost columns
        cost_columns_to_remove = [col for col in metrics_today.columns if "transaction_cost" in col]
        metrics_today.drop(columns=cost_columns_to_remove, inplace=True, errors='ignore')
        metrics_yesterday.drop(columns=cost_columns_to_remove, inplace=True, errors='ignore')

        # ✅ Format timestamps
        metrics_today["timestamp"] = pd.to_datetime(metrics_today["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        metrics_yesterday["timestamp"] = pd.to_datetime(metrics_yesterday["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

    except Exception as e:
        print(f"❌ Error processing network metrics: {e}")
        return

    output_data = []

    # ✅ Identify chain names dynamically
    chain_names = list(set(col.split("_")[0] for col in metrics_today.columns if "_" in col))

    for chain in chain_names:
        # ✅ Extract all relevant columns dynamically
        chain_columns = [col for col in metrics_today.columns if col.startswith(chain) and "block_time" not in col]

        # ✅ Prepare dataframes for today and yesterday
        chain_data_today = metrics_today[["timestamp"] + chain_columns].copy()
        chain_data_yesterday = metrics_yesterday[["timestamp"] + chain_columns].copy()

        chain_data_today = chain_data_today.reset_index(drop=True)
        chain_data_yesterday = chain_data_yesterday.reset_index(drop=True)

        chain_output_data = []

        for idx, row_today in chain_data_today.iterrows():
            row_yesterday = chain_data_yesterday.iloc[idx] if idx < len(chain_data_yesterday) else {}

            row_data = {"timestamp": row_today["timestamp"], "chain": chain.capitalize()}

            # ✅ Process all metrics dynamically
            for col in chain_columns:
                metric = col.split("_", 1)[1]  # Extract metric name (e.g., volume, gas_price)
                today_value = row_today.get(col)
                yesterday_value = row_yesterday.get(col)
                change_24h = calculate_percentage_change(today_value, yesterday_value)

                # ✅ Store values dynamically
                row_data[f"{metric} ({date2_str})"] = round(today_value, 8) if pd.notnull(today_value) else None
                row_data[f"{metric} ({date1_str})"] = round(yesterday_value, 8) if pd.notnull(yesterday_value) else None
                row_data[f"{metric} Change (%)"] = round(change_24h, 2) if pd.notnull(change_24h) else None

            chain_output_data.append(row_data)

        # ✅ Convert to DataFrame
        chain_df = pd.DataFrame(chain_output_data)

        # ✅ Compute averages correctly (DO NOT AVERAGE % CHANGES)
        numeric_columns = chain_df.select_dtypes(include=["float", "int"]).columns
        averages = {col: chain_df[col].mean() for col in numeric_columns}

        # ✅ Calculate "Change (%)" from average values, NOT averaging Change (%)
        for metric in [col.split(" (")[0] for col in numeric_columns if col.endswith(f"({date2_str})")]:
            avg_today = averages.get(f"{metric} ({date2_str})", None)
            avg_yesterday = averages.get(f"{metric} ({date1_str})", None)
            averages[f"{metric} Change (%)"] = calculate_percentage_change(avg_today, avg_yesterday)

        averages["timestamp"] = "Average"
        averages["chain"] = chain.capitalize()

        # ✅ Append the average row
        chain_output_data.append(averages)

        output_data.extend(chain_output_data)

    output_df = pd.DataFrame(output_data)

    # ✅ Ensure column order remains consistent
    fixed_column_order = ["timestamp", "chain"] + sorted(
        [col for col in output_df.columns if col not in ["timestamp", "chain"]]
    )

    # ✅ Apply the consistent column order
    output_df = output_df[fixed_column_order]

    # ✅ Upload cleaned CSV to S3
    write_csv(output_df, output_path)
    print(f"✅ Network metrics saved to {output_path}")



# Function to generate market metrics CSV
def generate_market_metrics_csv(date1_str, date2_str):
    """
    Generates an economic metrics CSV with daily comparisons and saves it to S3.
    Ensures that the "Change (%)" values in the average row are computed based on
    the averaged metric values instead of averaging individual percentage changes.
    """
    month_folder_today = date2_str[:7]
    month_folder_yesterday = date1_str[:7]
    input_path_today = f"{PROCESSED_DATA_PATH}{month_folder_today}/{date2_str}/economic_metrics.csv"
    input_path_yesterday = f"{PROCESSED_DATA_PATH}{month_folder_yesterday}/{date1_str}/economic_metrics.csv"
    output_path = f"{REPORTS_PATH}{month_folder_today}/{date2_str}/data/economic_metrics.csv"

    # Download files from S3
    metrics_today = read_csv(input_path_today)
    metrics_yesterday = read_csv(input_path_yesterday)

    if metrics_today is None or metrics_yesterday is None:
        print("❌ Missing one or both CSV files. Exiting.")
        return

    # Format timestamps
    metrics_today["timestamp"] = pd.to_datetime(metrics_today["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    metrics_yesterday["timestamp"] = pd.to_datetime(metrics_yesterday["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

    output_data = []

    # Identify chain names dynamically
    chain_names = [col.split("_")[0] for col in metrics_today.columns if col.endswith("_price")]

    for chain in chain_names:
        chain_columns = [col for col in metrics_today.columns if col.startswith(chain)]
        chain_data_today = metrics_today[["timestamp"] + chain_columns].copy()
        chain_data_yesterday = metrics_yesterday[["timestamp"] + chain_columns].copy()

        chain_data_today = chain_data_today.reset_index(drop=True)
        chain_data_yesterday = chain_data_yesterday.reset_index(drop=True)

        chain_output_data = []

        for idx, row_today in chain_data_today.iterrows():
            row_yesterday = chain_data_yesterday.iloc[idx]
            row_data = {"timestamp": row_today["timestamp"], "chain": chain.capitalize()}

            for col in chain_columns:
                metric = col.split("_", 1)[1]  # Extract metric name (e.g., price, volume)
                today_value = row_today.get(col)
                yesterday_value = row_yesterday.get(col)
                change_24h = calculate_percentage_change(today_value, yesterday_value)

                # Store today's and yesterday's values
                row_data[f"{metric} ({date2_str})"] = round(today_value, 2) if pd.notnull(today_value) else None
                row_data[f"{metric} ({date1_str})"] = round(yesterday_value, 2) if pd.notnull(yesterday_value) else None

                # Store calculated percentage change
                row_data[f"{metric} Change (%)"] = round(change_24h, 2) if pd.notnull(change_24h) else None

            chain_output_data.append(row_data)

        # Convert to DataFrame
        chain_df = pd.DataFrame(chain_output_data)

        # Convert required columns to numeric
        numeric_columns = chain_df.select_dtypes(include=["number"]).columns

        # Calculate averages for numeric columns
        averages = {col: chain_df[col].mean() for col in numeric_columns}

        # Correctly compute "Change (%)" for the "Average" row
        for metric in [col.split(" (")[0] for col in numeric_columns if col.endswith(f"({date2_str})")]:
            avg_today = averages.get(f"{metric} ({date2_str})", None)
            avg_yesterday = averages.get(f"{metric} ({date1_str})", None)
            if avg_today is not None and avg_yesterday is not None:
                averages[f"{metric} Change (%)"] = round(calculate_percentage_change(avg_today, avg_yesterday), 2)
            else:
                averages[f"{metric} Change (%)"] = None  # Handle missing data

        # Add timestamp and chain name for the "Average" row
        averages["timestamp"] = "Average"
        averages["chain"] = chain.capitalize()

        # Append the average row
        chain_output_data.append(averages)

        output_data.extend(chain_output_data)

    output_df = pd.DataFrame(output_data)

    # Ensure numeric rounding
    numeric_columns = output_df.select_dtypes(include=["float", "int"]).columns
    output_df[numeric_columns] = output_df[numeric_columns].applymap(lambda x: round(x, 2) if pd.notnull(x) else x)

    print("✅ Generated economic metrics:")
    print(output_df.head())

    # Upload updated CSV to S3
    write_csv(output_df, output_path)



def generate_staking_metrics_csv(date1_str, date2_str):
    """
    Generates a staking metrics CSV with daily comparisons and saves it to S3.
    Ensures that the "Change (%)" values in the average row are computed based on
    the averaged metric values instead of averaging individual percentage changes.
    """
    month_folder_today = date2_str[:7]  # Extract YYYY-MM from today's date
    month_folder_yesterday = date1_str[:7]  # Extract YYYY-MM from yesterday's date
    input_path_today = f"{PROCESSED_DATA_PATH}{month_folder_today}/{date2_str}/staking_metrics.csv"
    input_path_yesterday = f"{PROCESSED_DATA_PATH}{month_folder_yesterday}/{date1_str}/staking_metrics.csv"
    output_path = f"{REPORTS_PATH}{month_folder_today}/{date2_str}/data/staking_metrics.csv"

    # Download files from S3
    metrics_today = read_csv(input_path_today)
    metrics_yesterday = read_csv(input_path_yesterday)

    if metrics_today is None or metrics_yesterday is None:
        print("❌ Missing one or both CSV files. Exiting.")
        return

    # Format timestamps
    metrics_today["timestamp"] = pd.to_datetime(metrics_today["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    metrics_yesterday["timestamp"] = pd.to_datetime(metrics_yesterday["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

    output_data = []

    # Identify chain names dynamically
    chain_names = [col.split("_")[0] for col in metrics_today.columns if col.endswith("_staking_ratio")]

    for chain in chain_names:
        chain_columns = [col for col in metrics_today.columns if col.startswith(chain)]
        chain_data_today = metrics_today[["timestamp"] + chain_columns].copy()
        chain_data_yesterday = metrics_yesterday[["timestamp"] + chain_columns].copy()

        chain_data_today = chain_data_today.reset_index(drop=True)
        chain_data_yesterday = chain_data_yesterday.reset_index(drop=True)

        chain_output_data = []

        for idx, row_today in chain_data_today.iterrows():
            row_yesterday = chain_data_yesterday.iloc[idx]
            row_data = {"timestamp": row_today["timestamp"], "chain": chain.capitalize()}

            for col in chain_columns:
                metric = col.split("_", 1)[1]  # Extract metric name (e.g., staking_ratio, reward_rate)
                today_value = row_today.get(col)
                yesterday_value = row_yesterday.get(col)
                change_24h = calculate_percentage_change(today_value, yesterday_value)

                row_data[f"{metric} ({date2_str})"] = round(today_value, 2) if pd.notnull(today_value) else None
                row_data[f"{metric} ({date1_str})"] = round(yesterday_value, 2) if pd.notnull(yesterday_value) else None
                row_data[f"{metric} Change (%)"] = round(change_24h, 2) if pd.notnull(change_24h) else None

            chain_output_data.append(row_data)

        # Convert to DataFrame
        chain_df = pd.DataFrame(chain_output_data)

        # Compute averages correctly
        numeric_columns = chain_df.select_dtypes(include=["float", "int"]).columns
        averages = {col: chain_df[col].mean() for col in numeric_columns}

        # Calculate "Change (%)" from average values, NOT averaging Change (%)
        for metric in [col.split(" (")[0] for col in numeric_columns if col.endswith(f"({date2_str})")]:
            avg_today = averages.get(f"{metric} ({date2_str})", None)
            avg_yesterday = averages.get(f"{metric} ({date1_str})", None)
            averages[f"{metric} Change (%)"] = calculate_percentage_change(avg_today, avg_yesterday)

        averages["timestamp"] = "Average"
        averages["chain"] = chain.capitalize()

        # Append the average row
        chain_output_data.append(averages)

        output_data.extend(chain_output_data)

    output_df = pd.DataFrame(output_data)
    write_csv(output_df, output_path)





# def generate_template_markdown(template_path):
#     """
#     Generates a markdown template file for the daily report with placeholders for dynamic data.

#     Parameters:
#     - template_path: The path where the template markdown file will be saved.
#     """
#     # Markdown content with placeholders using flat keys
#     template_content = """
# # 📊 Chain Pulse Daily Report - {date}

# ## Overview
# This report highlights blockchain metrics for {date}, compared to {previous_date}. 
# It includes data on transaction volumes, gas prices, staking metrics, and economic indicators.

# ---

# ## 1️⃣ Network Performance


# ### Transaction Volume
# | Blockchain  | Volume ({previous_date}) (USD) | Volume ({date}) (USD) | Change (%) |
# |-------------|--------------------------|-----------------|------------|
# | Ethereum    | {eth_volume_prev}        | {eth_volume}    | {eth_volume_change}% |
# | Binance     | {bnb_volume_prev}        | {bnb_volume}    | {bnb_volume_change}% |
# | Solana      | {sol_volume_prev}        | {sol_volume}    | {sol_volume_change}% |
# | Avalanche   | {avax_volume_prev}       | {avax_volume}   | {avax_volume_change}% |

# ### Gas Prices
# | Blockchain  | Gas Price ({previous_date}) (Gwei) | Gas Price ({date}) (Gwei) | Change (%) |
# |-------------|-----------------------------|--------------------|------------|
# | Ethereum    | {eth_gas_prev}             | {eth_gas}          | {eth_gas_change}% |
# | Binance     | {bnb_gas_prev}             | {bnb_gas}          | {bnb_gas_change}% |
# | Avalanche   | {avax_gas_prev}            | {avax_gas}         | {avax_gas_change}% |

# ### Transaction Fees
# | Blockchain | Fee ({previous_date}) (Token) | Fee ({date}) (Token) | Change (%) |
# |------------|----------------------|-------------|------------|
# | Ethereum   | {eth_fee_prev}        | {eth_fee}   | {eth_fee_change}% |
# | Binance    | {bnb_fee_prev}        | {bnb_fee}   | {bnb_fee_change}% |
# | Solana     | {sol_fee_prev}        | {sol_fee}   | {sol_fee_change}% |
# | Avalanche  | {avax_fee_prev}       | {avax_fee}  | {avax_fee_change}% |

# ---


# ![Transaction Volume Trends](volume_comparison.png)

# ![Gas Price & Fees Trends](gas_and_fee_trends.png)


# ---

# ## 2️⃣ Economic Indicators

# ### Price and Market Cap
# | Blockchain  | Price ({previous_date}) (USD) | Price ({date}) (USD) | Change (%) | Market Cap ({previous_date}) (USD) | Market Cap ({date}) (USD) | Change (%) |
# |-------------|-------------------------|----------------|------------|-----------------------------|---------------------|------------|
# | Ethereum    | {eth_price_prev}        | {eth_price}    | {eth_price_change}% | {eth_market_cap_prev} | {eth_market_cap} | {eth_market_cap_change}% |
# | Binance     | {bnb_price_prev}        | {bnb_price}    | {bnb_price_change}% | {bnb_market_cap_prev} | {bnb_market_cap} | {bnb_market_cap_change}% |
# | Solana      | {sol_price_prev}        | {sol_price}    | {sol_price_change}% | {sol_market_cap_prev} | {sol_market_cap} | {sol_market_cap_change}% |
# | Avalanche   | {avax_price_prev}       | {avax_price}   | {avax_price_change}% | {avax_market_cap_prev} | {avax_market_cap} | {avax_market_cap_change}% |

# ### Circulating and Total Supply
# | Blockchain  | Circulating Supply ({previous_date}) (Token) | Circulating Supply ({date}) (Token) | Change (%) | Total Supply ({previous_date}) (Token) | Total Supply ({date}) (Token) | Change (%) |
# |-------------|--------------------------------------|-----------------------------|------------|-------------------------------|------------------------|------------|
# | Ethereum    | {eth_circulating_supply_prev}        | {eth_circulating_supply}    | {eth_circulating_supply_change}% | {eth_total_supply_prev} | {eth_total_supply} | {eth_total_supply_change}% |
# | Binance     | {bnb_circulating_supply_prev}        | {bnb_circulating_supply}    | {bnb_circulating_supply_change}% | {bnb_total_supply_prev} | {bnb_total_supply} | {bnb_total_supply_change}% |
# | Solana      | {sol_circulating_supply_prev}        | {sol_circulating_supply}    | {sol_circulating_supply_change}% | {sol_total_supply_prev} | {sol_total_supply} | {sol_total_supply_change}% |
# | Avalanche   | {avax_circulating_supply_prev}       | {avax_circulating_supply}   | {avax_circulating_supply_change}% | {avax_total_supply_prev} | {avax_total_supply} | {avax_total_supply_change}% |

# ### Trading Volume
# | Blockchain  | Trading Volume ({previous_date}) (USD) | Trading Volume ({date}) (USD) | Change (%) |
# |-------------|---------------------------------|--------------------------|------------|
# | Ethereum    | {eth_trading_volume_prev}       | {eth_trading_volume}    | {eth_trading_volume_change}% |
# | Binance     | {bnb_trading_volume_prev}       | {bnb_trading_volume}    | {bnb_trading_volume_change}% |
# | Solana      | {sol_trading_volume_prev}       | {sol_trading_volume}    | {sol_trading_volume_change}% |
# | Avalanche   | {avax_trading_volume_prev}      | {avax_trading_volume}   | {avax_trading_volume_change}% |

# ---


# ![Price and Market Cap Trends](market_trends.png)


# ![Circulating and Total Supply Trends](supply_comparison.png)


# ![Trading Volume Trends](trading_volume.png)


# ---

# ## 3️⃣ Staking Metrics

# ### Staking Ratio
# | Blockchain  | Staking Ratio ({previous_date}) (%) | Staking Ratio ({date}) (%) | Change (%) |
# |-------------|--------------------------------|-------------------------|------------|
# | Ethereum    | {eth_staking_ratio_prev}%      | {eth_staking_ratio}%    | {eth_staking_ratio_change}% |
# | Binance     | {bnb_staking_ratio_prev}%      | {bnb_staking_ratio}%    | {bnb_staking_ratio_change}% |
# | Solana      | {sol_staking_ratio_prev}%      | {sol_staking_ratio}%    | {sol_staking_ratio_change}% |
# | Avalanche   | {avax_staking_ratio_prev}%     | {avax_staking_ratio}%   | {avax_staking_ratio_change}% |

# ---


# ![Staking Ratio Trends](staking_comparison.png)


# **Generated automatically by Chain Pulse Report Generator.**
# """

#     # Create the directory if it doesn't exist
#     os.makedirs(os.path.dirname(template_path), exist_ok=True)

#     # Write template to file
#     with open(template_path, "w", encoding="utf-8") as file:
#         file.write(template_content)

#     print(f"Template generated and saved to {template_path}")


def generate_actual_report(template_s3_path, output_s3_path, network_data, economic_data, staking_data, visualizations_s3_folder, date, previous_date):
    """
    Generate a daily report with actual data and visualizations based on the template, working with S3 paths.

    Parameters:
    - template_s3_path: S3 path to the markdown template.
    - output_s3_path: S3 path where the generated markdown report will be saved.
    - network_s3_path: S3 path to the CSV file containing network data.
    - economic_s3_path: S3 path to the CSV file containing economic data.
    - staking_s3_path: S3 path to the CSV file containing staking data.
    - visualizations_s3_folder: S3 folder path containing visualization files.
    - date: Current report date.
    - previous_date: Previous report date.
    """
    print("🔍 Debug: Average row for Gas Price and Fees")
    print(network_data[network_data["Timestamp"] == "Average"])

    # Read the markdown template (works local or S3 depending on LOCAL_MODE)
    try:
        template_content = read_template(template_s3_path)
    except Exception as e:
        print(f"❌ Error reading template: {e}")
        return

    def format_large_values(value):
        """Formats large numeric values using K, M, B notation for the report."""
        if pd.isnull(value) or not isinstance(value, (int, float)):
            return value  # Skip formatting for NaN or non-numeric values
        if abs(value) >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        elif abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        elif abs(value) >= 1_000:
            return f"{value / 1_000:.2f}K"
        else:
            return f"{value:,.2f}"  # Keep small numbers formatted with commas

    def get_metric(df, blockchain, column, apply_kmb=False, use_scientific=False):
        """Fetches the required metric and applies formatting only for the report."""
        try:
            timestamp_column = "Timestamp (UTC)" if "Timestamp (UTC)" in df.columns else "Timestamp"
            avg_row = df[(df['Blockchain'] == blockchain) & (df[timestamp_column] == "Average")]

            print(f"🔍 Debugging `get_metric()`: Available columns = {df.columns.tolist()}")

            # print(f"🔍 Debugging: Fetching `{column}` for `{blockchain}`")
            # print(avg_row[[column]])  # Print actual values before returning

            value = avg_row[column].values
            if len(value) > 0:
                formatted_value = value[0]

                if "Staking Ratio" in column:
                    return round(formatted_value, 2) if pd.notnull(formatted_value) else "N/A"

                # Apply scientific notation for fees
                if use_scientific:
                    return f"{formatted_value:.2e}"  # Example: 3.00e-4 for 0.0003

                # Apply K, M, B formatting if requested
                if apply_kmb:
                    return format_large_values(formatted_value)

                return formatted_value  # Keep other values unchanged
            else:
                return "N/A"
        except KeyError:
            return "N/A"

    #print("🔍 Debug: Checking Gas Price & Transaction Fee Columns in `network_data`")
    #print(network_data[["Blockchain", "Gas Price Change (%)", "Transaction Fee Change (%)"]])


    
    # Prepare the actual data to replace placeholders
    data = {
        "date": date,
        "previous_date": previous_date,

        # Network Performance - Transaction Volume and Gas Prices
        "eth_volume_prev": get_metric(network_data, "Ethereum", f"Transaction Volume ({previous_date})",apply_kmb=True),
        "eth_volume": get_metric(network_data, "Ethereum", f"Transaction Volume ({date})",apply_kmb=True),
        "eth_volume_change": get_metric(network_data, "Ethereum", "Transaction Volume Change (%)"),
        "eth_gas_prev": get_metric(network_data, "Ethereum", f"Gas Price ({previous_date})",apply_kmb=True),
        "eth_gas": get_metric(network_data, "Ethereum", f"Gas Price ({date})",apply_kmb=True),
        "eth_gas_change": get_metric(network_data, "Ethereum", "Gas Price Change (%)"),
        "eth_fee_prev": get_metric(network_data, "Ethereum", f"Transaction Fee ({previous_date})", use_scientific=True),
        "eth_fee": get_metric(network_data, "Ethereum", f"Transaction Fee ({date})", use_scientific=True),
        "eth_fee_change": get_metric(network_data, "Ethereum", "Transaction Fee Change (%)"),
        "eth_tps_prev": get_metric(network_data, "Ethereum", f"TPS ({previous_date})",apply_kmb=True),
        "eth_tps": get_metric(network_data, "Ethereum", f"TPS ({date})",apply_kmb=True),
        "eth_tps_change": get_metric(network_data, "Ethereum", "TPS Change (%)"),


        "bnb_volume_prev": get_metric(network_data, "Binance", f"Transaction Volume ({previous_date})",apply_kmb=True),
        "bnb_volume": get_metric(network_data, "Binance", f"Transaction Volume ({date})",apply_kmb=True),
        "bnb_volume_change": get_metric(network_data, "Binance", "Transaction Volume Change (%)"),
        "bnb_gas_prev": get_metric(network_data, "Binance", f"Gas Price ({previous_date})",apply_kmb=True),
        "bnb_gas": get_metric(network_data, "Binance", f"Gas Price ({date})",apply_kmb=True),
        "bnb_gas_change": get_metric(network_data, "Binance", "Gas Price Change (%)"),
        "bnb_fee_prev": get_metric(network_data, "Binance", f"Transaction Fee ({previous_date})", use_scientific=True),
        "bnb_fee": get_metric(network_data, "Binance", f"Transaction Fee ({date})", use_scientific=True),
        "bnb_fee_change": get_metric(network_data, "Binance", "Transaction Fee Change (%)"),
        "bnb_tps_prev": get_metric(network_data, "Binance", f"TPS ({previous_date})",apply_kmb=True),
        "bnb_tps": get_metric(network_data, "Binance", f"TPS ({date})",apply_kmb=True),
        "bnb_tps_change": get_metric(network_data, "Binance", "TPS Change (%)"),



        "sol_volume_prev": get_metric(network_data, "Solana", f"Transaction Volume ({previous_date})",apply_kmb=True),
        "sol_volume": get_metric(network_data, "Solana", f"Transaction Volume ({date})",apply_kmb=True),
        "sol_volume_change": get_metric(network_data, "Solana", "Transaction Volume Change (%)"),
        "sol_gas_prev": get_metric(network_data, "Solana", f"Gas Price ({previous_date})",apply_kmb=True),
        "sol_gas": get_metric(network_data, "Solana", f"Gas Price ({date})",apply_kmb=True),
        "sol_gas_change": get_metric(network_data, "Solana", "Gas Price Change (%)"),
        "sol_fee_prev": get_metric(network_data, "Solana", f"Transaction Fee ({previous_date})", use_scientific=True),
        "sol_fee": get_metric(network_data, "Solana", f"Transaction Fee ({date})", use_scientific=True),
        "sol_fee_change": get_metric(network_data, "Solana", "Transaction Fee Change (%)"),
        "sol_tps_prev": get_metric(network_data, "Solana", f"TPS ({previous_date})",apply_kmb=True),
        "sol_tps": get_metric(network_data, "Solana", f"TPS ({date})",apply_kmb=True),
        "sol_tps_change": get_metric(network_data, "Solana", "TPS Change (%)"),


        "avax_volume_prev": get_metric(network_data, "Avalanche", f"Transaction Volume ({previous_date})",apply_kmb=True),
        "avax_volume": get_metric(network_data, "Avalanche", f"Transaction Volume ({date})",apply_kmb=True),
        "avax_volume_change": get_metric(network_data, "Avalanche", "Transaction Volume Change (%)"),
        "avax_gas_prev": get_metric(network_data, "Avalanche", f"Gas Price ({previous_date})",apply_kmb=True),
        "avax_gas": get_metric(network_data, "Avalanche", f"Gas Price ({date})",apply_kmb=True),
        "avax_gas_change": get_metric(network_data, "Avalanche", "Gas Price Change (%)"),
        "avax_fee_prev": get_metric(network_data, "Avalanche", f"Transaction Fee ({previous_date})", use_scientific=True),
        "avax_fee": get_metric(network_data, "Avalanche", f"Transaction Fee ({date})", use_scientific=True),
        "avax_fee_change": get_metric(network_data, "Avalanche", "Transaction Fee Change (%)"),
        "avax_tps_prev": get_metric(network_data, "Avalanche", f"TPS ({previous_date})",apply_kmb=True),
        "avax_tps": get_metric(network_data, "Avalanche", f"TPS ({date})",apply_kmb=True),
        "avax_tps_change": get_metric(network_data, "Avalanche", "TPS Change (%)"),



        # Economic Indicators - Trading Volume, Price, Market Cap, Circulating and Total Supply
        "eth_trading_volume_prev": get_metric(economic_data, "Ethereum", f"Trading Volume ({previous_date})",apply_kmb=True),
        "eth_trading_volume": get_metric(economic_data, "Ethereum", f"Trading Volume ({date})",apply_kmb=True),
        "eth_trading_volume_change": get_metric(economic_data, "Ethereum", "Trading Volume Change (%)"),
        "eth_price_prev": get_metric(economic_data, "Ethereum", f"Price ({previous_date})",apply_kmb=True),
        "eth_price": get_metric(economic_data, "Ethereum", f"Price ({date})",apply_kmb=True),
        "eth_price_change": get_metric(economic_data, "Ethereum", "Price Change (%)"),
        "eth_market_cap_prev": get_metric(economic_data, "Ethereum", f"Market Cap ({previous_date})",apply_kmb=True),
        "eth_market_cap": get_metric(economic_data, "Ethereum", f"Market Cap ({date})",apply_kmb=True),
        "eth_market_cap_change": get_metric(economic_data, "Ethereum", "Market Cap Change (%)"),
        "eth_circulating_supply_prev": get_metric(economic_data, "Ethereum", f"Circulating Supply ({previous_date})",apply_kmb=True),
        "eth_circulating_supply": get_metric(economic_data, "Ethereum", f"Circulating Supply ({date})",apply_kmb=True),
        "eth_circulating_supply_change": get_metric(economic_data, "Ethereum", "Circulating Supply Change (%)"),
        "eth_total_supply_prev": get_metric(economic_data, "Ethereum", f"Total Supply ({previous_date})",apply_kmb=True),
        "eth_total_supply": get_metric(economic_data, "Ethereum", f"Total Supply ({date})",apply_kmb=True),
        "eth_total_supply_change": get_metric(economic_data, "Ethereum", "Total Supply Change (%)"),

        "bnb_trading_volume_prev": get_metric(economic_data, "Binance", f"Trading Volume ({previous_date})",apply_kmb=True),
        "bnb_trading_volume": get_metric(economic_data, "Binance", f"Trading Volume ({date})",apply_kmb=True),
        "bnb_trading_volume_change": get_metric(economic_data, "Binance", "Trading Volume Change (%)"),
        "bnb_price_prev": get_metric(economic_data, "Binance", f"Price ({previous_date})",apply_kmb=True),
        "bnb_price": get_metric(economic_data, "Binance", f"Price ({date})",apply_kmb=True),
        "bnb_price_change": get_metric(economic_data, "Binance", "Price Change (%)"),
        "bnb_market_cap_prev": get_metric(economic_data, "Binance", f"Market Cap ({previous_date})",apply_kmb=True),
        "bnb_market_cap": get_metric(economic_data, "Binance", f"Market Cap ({date})",apply_kmb=True),
        "bnb_market_cap_change": get_metric(economic_data, "Binance", "Market Cap Change (%)"),
        "bnb_circulating_supply_prev": get_metric(economic_data, "Binance", f"Circulating Supply ({previous_date})",apply_kmb=True),
        "bnb_circulating_supply": get_metric(economic_data, "Binance", f"Circulating Supply ({date})",apply_kmb=True),
        "bnb_circulating_supply_change": get_metric(economic_data, "Binance", "Circulating Supply Change (%)"),
        "bnb_total_supply_prev": get_metric(economic_data, "Binance", f"Total Supply ({previous_date})",apply_kmb=True),
        "bnb_total_supply": get_metric(economic_data, "Binance", f"Total Supply ({date})",apply_kmb=True),
        "bnb_total_supply_change": get_metric(economic_data, "Binance", "Total Supply Change (%)"),


        "sol_trading_volume_prev": get_metric(economic_data, "Solana", f"Trading Volume ({previous_date})",apply_kmb=True),
        "sol_trading_volume": get_metric(economic_data, "Solana", f"Trading Volume ({date})",apply_kmb=True),
        "sol_trading_volume_change": get_metric(economic_data, "Solana", "Trading Volume Change (%)"),
        "sol_price_prev": get_metric(economic_data, "Solana", f"Price ({previous_date})",apply_kmb=True),
        "sol_price": get_metric(economic_data, "Solana", f"Price ({date})",apply_kmb=True),
        "sol_price_change": get_metric(economic_data, "Solana", "Price Change (%)"),
        "sol_market_cap_prev": get_metric(economic_data, "Solana", f"Market Cap ({previous_date})",apply_kmb=True),
        "sol_market_cap": get_metric(economic_data, "Solana", f"Market Cap ({date})",apply_kmb=True),
        "sol_market_cap_change": get_metric(economic_data, "Solana", "Market Cap Change (%)"),
        "sol_circulating_supply_prev": get_metric(economic_data, "Solana", f"Circulating Supply ({previous_date})",apply_kmb=True),
        "sol_circulating_supply": get_metric(economic_data, "Solana", f"Circulating Supply ({date})",apply_kmb=True),
        "sol_circulating_supply_change": get_metric(economic_data, "Solana", "Circulating Supply Change (%)"),
        "sol_total_supply_prev": get_metric(economic_data, "Solana", f"Total Supply ({previous_date})",apply_kmb=True),
        "sol_total_supply": get_metric(economic_data, "Solana", f"Total Supply ({date})",apply_kmb=True),
        "sol_total_supply_change": get_metric(economic_data, "Solana", "Total Supply Change (%)"),


        "avax_trading_volume_prev": get_metric(economic_data, "Avalanche", f"Trading Volume ({previous_date})",apply_kmb=True),
        "avax_trading_volume": get_metric(economic_data, "Avalanche", f"Trading Volume ({date})",apply_kmb=True),
        "avax_trading_volume_change": get_metric(economic_data, "Avalanche", "Trading Volume Change (%)"),
        "avax_price_prev": get_metric(economic_data, "Avalanche", f"Price ({previous_date})",apply_kmb=True),
        "avax_price": get_metric(economic_data, "Avalanche", f"Price ({date})",apply_kmb=True),
        "avax_price_change": get_metric(economic_data, "Avalanche", "Price Change (%)"),
        "avax_market_cap_prev": get_metric(economic_data, "Avalanche", f"Market Cap ({previous_date})",apply_kmb=True),
        "avax_market_cap": get_metric(economic_data, "Avalanche", f"Market Cap ({date})",apply_kmb=True),
        "avax_market_cap_change": get_metric(economic_data, "Avalanche", "Market Cap Change (%)"),
        "avax_circulating_supply_prev": get_metric(economic_data, "Avalanche", f"Circulating Supply ({previous_date})",apply_kmb=True),
        "avax_circulating_supply": get_metric(economic_data, "Avalanche", f"Circulating Supply ({date})",apply_kmb=True),
        "avax_circulating_supply_change": get_metric(economic_data, "Avalanche", "Circulating Supply Change (%)"),
        "avax_total_supply_prev": get_metric(economic_data, "Avalanche", f"Total Supply ({previous_date})",apply_kmb=True),
        "avax_total_supply": get_metric(economic_data, "Avalanche", f"Total Supply ({date})",apply_kmb=True),
        "avax_total_supply_change": get_metric(economic_data, "Avalanche", "Total Supply Change (%)"),


        # Staking Metrics
        "eth_staking_ratio_prev": get_metric(staking_data, "Ethereum", f"Staking Ratio ({previous_date})"),
        "eth_staking_ratio": get_metric(staking_data, "Ethereum", f"Staking Ratio ({date})"),
        "eth_staking_ratio_change": get_metric(staking_data, "Ethereum", "Staking Ratio Change (%)"),
        "bnb_staking_ratio_prev": get_metric(staking_data, "Binance", f"Staking Ratio ({previous_date})"),
        "bnb_staking_ratio": get_metric(staking_data, "Binance", f"Staking Ratio ({date})"),
        "bnb_staking_ratio_change": get_metric(staking_data, "Binance", "Staking Ratio Change (%)"),
        "sol_staking_ratio_prev": get_metric(staking_data, "Solana", f"Staking Ratio ({previous_date})"),
        "sol_staking_ratio": get_metric(staking_data, "Solana", f"Staking Ratio ({date})"),
        "sol_staking_ratio_change": get_metric(staking_data, "Solana", "Staking Ratio Change (%)"),
        "avax_staking_ratio_prev": get_metric(staking_data, "Avalanche", f"Staking Ratio ({previous_date})"),
        "avax_staking_ratio": get_metric(staking_data, "Avalanche", f"Staking Ratio ({date})"),
        "avax_staking_ratio_change": get_metric(staking_data, "Avalanche", "Staking Ratio Change (%)"),
    }

    # Generate visualization S3 paths
    if LOCAL_MODE:
        base = visualizations_s3_folder              # e.g. "reports/2025-08/2025-08-18/visualizations"
        visualizations = {
            "volume_comparison.png": f"{base}/network_charts/volume_comparison.png",
            "gas_and_fee_trends.png": f"{base}/network_charts/gas_and_fee_trends.png",
            "market_trends.png": f"{base}/economic_charts/market_trends.png",
            "supply_comparison.png": f"{base}/economic_charts/supply_comparison.png",
            "trading_volume.png": f"{base}/economic_charts/trading_volume.png",
            "staking_comparison.png": f"{base}/staking_charts/staking_comparison.png",
        }
    else:
        visualizations = {
            "volume_comparison.png": f"s3://{BUCKET_NAME}/{visualizations_s3_folder}/network_charts/volume_comparison.png",
            "gas_and_fee_trends.png": f"s3://{BUCKET_NAME}/{visualizations_s3_folder}/network_charts/gas_and_fee_trends.png",
            "market_trends.png": f"s3://{BUCKET_NAME}/{visualizations_s3_folder}/economic_charts/market_trends.png",
            "supply_comparison.png": f"s3://{BUCKET_NAME}/{visualizations_s3_folder}/economic_charts/supply_comparison.png",
            "trading_volume.png": f"s3://{BUCKET_NAME}/{visualizations_s3_folder}/economic_charts/trading_volume.png",
            "staking_comparison.png": f"s3://{BUCKET_NAME}/{visualizations_s3_folder}/staking_charts/staking_comparison.png",
        }


    # ✅ Debugging: Check what get_metric() returns
    #print("🔍 Debug: get_metric() output for Ethereum Gas Price Change:", get_metric(network_data, "Ethereum", "Gas Price Change (%)"))
    #print("🔍 Debug: get_metric() output for Ethereum Transaction Fee Change:", get_metric(network_data, "Ethereum", "Transaction Fee Change (%)"))

    # Replace placeholders in the template
    for placeholder, s3_path in visualizations.items():
        template_content = template_content.replace(f"({placeholder})", f"({s3_path})")

    # Replace other placeholders dynamically
    try:
        report_content = template_content.format(**data)
    except KeyError as e:
        raise KeyError(f"❌ Template placeholder not found in data: {e}")

    # Save final markdown (works local or S3 depending on LOCAL_MODE)
    try:
        write_markdown(report_content, output_s3_path)
        print(f"✅ Report successfully saved: {output_s3_path}")
    except Exception as e:
        print(f"❌ Error saving report: {e}")




def lambda_handler(event, context):
    """
    AWS Lambda Handler to generate daily blockchain reports based on event input.
    Expects `today_date` and `yesterday_date` in the event JSON.
    """

    try:
        #Get today's date (UTC) and yesterday's date
        today_date = datetime.now(timezone.utc).date()
        yesterday_date = today_date - timedelta(days=1)

        # Format dates as strings (YYYY-MM-DD)
        date2_str = today_date.strftime("%Y-%m-%d")
        date1_str = yesterday_date.strftime("%Y-%m-%d")

        print(f"🕒 Auto-detected dates: Today = {date2_str}, Yesterday = {date1_str}")

        # # Check if test dates are provided; otherwise, use the default dynamic calculation
        # if "today_date" in event and "yesterday_date" in event:
        #     date2_str = event["today_date"]
        #     date1_str = event["yesterday_date"]
        # else:
        #     today_date = datetime.utcnow().date()
        #     yesterday_date = today_date - timedelta(days=1)
        #     date2_str = today_date.strftime("%Y-%m-%d")
        #     date1_str = yesterday_date.strftime("%Y-%m-%d")


        month_folder_today = date2_str[:7]  # Extract YYYY-MM
        month_folder_yesterday = date1_str[:7]
        base_folder = f"{REPORTS_PATH}{date2_str}/"


        # Generate metrics CSVs
        generate_network_metrics_csv(date1_str, date2_str)
        generate_market_metrics_csv(date1_str, date2_str)
        generate_staking_metrics_csv(date1_str, date2_str)

        # Define S3 paths for files
        network_s3_path = f"{REPORTS_PATH}{month_folder_today}/{date2_str}/data/network_metrics.csv"
        economic_s3_path = f"{REPORTS_PATH}{month_folder_today}/{date2_str}/data/economic_metrics.csv"
        staking_s3_path = f"{REPORTS_PATH}{month_folder_today}/{date2_str}/data/staking_metrics.csv"
        template_s3_path = f"{REPORTS_PATH}templates/daily_report_template.md"
        output_s3_path = f"{REPORTS_PATH}{month_folder_today}/{date2_str}/generated/daily_report.md"
        visualizations_s3_folder = f"{REPORTS_PATH}{month_folder_today}/{date2_str}/visualizations"


        # Load data from S3
        network_data = read_csv(network_s3_path)
        economic_data = read_csv(economic_s3_path)
        staking_data = read_csv(staking_s3_path)

        if network_data is None or economic_data is None or staking_data is None:
            print("❌ Error: One or more required CSV files could not be loaded. Exiting function.")
            return {
                'statusCode': 500,
                'body': json.dumps('Error: Missing data files from S3.')
            }

        # Rename and clean headers for each dataset
        network_data = rename_and_clean_data(network_data, "network", date1_str, date2_str)
        economic_data = rename_and_clean_data(economic_data, "economic", date1_str, date2_str)
        staking_data = rename_and_clean_data(staking_data, "staking", date1_str, date2_str)

        # 🔍 Debugging print: Verify economic data after renaming and cleaning
        #print("🔍 Economic Data After Cleaning:")
        #print(economic_data.columns)
        #print(economic_data.head())

        # # Apply formatting to loaded data for visualization
        # network_data = apply_formatting(network_data)
        # economic_data = apply_formatting(economic_data)
        # staking_data = apply_formatting(staking_data)


        # Upload cleaned and formatted data back to S3
        write_csv(network_data, network_s3_path)
        write_csv(economic_data, economic_s3_path)
        write_csv(staking_data, staking_s3_path)

        #print("🔍 Economic Data Before Charting:")
        #print(economic_data[['Blockchain', 'Timestamp (UTC)', 'Circulating Supply Change (%)', 'Total Supply Change (%)']])


        # Generate visualizations
        generate_transaction_volume_chart(network_data, date2_str)
        generate_market_trends_chart(economic_data, date2_str)
        generate_trading_volume_chart(economic_data, date2_str)
        generate_supply_comparison_chart(economic_data, date2_str)
        generate_gas_fees_trends_chart(network_data, date2_str)
        generate_staking_comparison_chart(staking_data, date2_str)
        print("🔄 Calling generate_tps_comparison_chart()...")
        generate_tps_comparison_chart(network_data, date2_str)
        print("✅ Finished calling generate_tps_comparison_chart()")



        # Generate the report
        generate_actual_report(
        template_s3_path=template_s3_path,
        output_s3_path=output_s3_path,
        network_data=network_data,  # ✅ Pass actual data instead of paths
        economic_data=economic_data,
        staking_data=staking_data,
        visualizations_s3_folder=visualizations_s3_folder,
        date=date2_str,
        previous_date=date1_str
    )


        print(f"✅ Report generation completed successfully for {date2_str}.")
        return {
            'statusCode': 200,
            'body': json.dumps(f"Report generated successfully for {date2_str}.")
        }

    except Exception as e:
        print(f"❌ Lambda execution error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Lambda execution failed: {str(e)}")
        }

# --- date selection helpers ---

REQUIRED_FILES = {"network_metrics.csv", "economic_metrics.csv", "staking_metrics.csv"}

def date_has_all_three(date_str: str) -> bool:
    """True if all three processed CSVs exist for date_str."""
    year_month = date_str[:7]
    if LOCAL_MODE:
        dpath = Path(LOCAL_PROCESSED_DIR) / year_month / date_str
        return dpath.is_dir() and all((dpath / f).exists() for f in REQUIRED_FILES)
    # S3 mode
    prefix = f"{PROCESSED_DATA_PATH}{year_month}/{date_str}/"
    try:
        resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        names = {obj["Key"].split("/")[-1] for obj in resp.get("Contents", [])}
        return REQUIRED_FILES.issubset(names)
    except Exception:
        return False

def find_latest_two_dates_available():
    """Scan processed folders (local or S3) and return (prev_date, latest_date)."""
    dates = set()
    if LOCAL_MODE:
        root = Path(LOCAL_PROCESSED_DIR)
        if not root.exists():
            return None, None
        for month_dir in root.iterdir():
            if not (month_dir.is_dir() and re.fullmatch(r"\d{4}-\d{2}", month_dir.name)):
                continue
            for day_dir in month_dir.iterdir():
                if not (day_dir.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", day_dir.name)):
                    continue
                files = {p.name for p in day_dir.glob("*.csv")}
                if REQUIRED_FILES.issubset(files):
                    dates.add(day_dir.name)
    else:
        paginator = s3.get_paginator("list_objects_v2")
        seen = {}
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=PROCESSED_DATA_PATH):
            for obj in page.get("Contents", []):
                m = re.search(r"processed-data-all-types/\d{4}-\d{2}/(\d{4}-\d{2}-\d{2})/([^/]+)$", obj["Key"])
                if m:
                    d, fname = m.group(1), m.group(2)
                    seen.setdefault(d, set()).add(fname)
        for d, names in seen.items():
            if REQUIRED_FILES.issubset(names):
                dates.add(d)

    if not dates:
        return None, None
    ds = sorted(dates)
    return (ds[-2], ds[-1]) if len(ds) >= 2 else (None, ds[-1])

def select_dates_for_report():
    """Prefer (yesterday, today). If missing, fallback to latest two available."""
    today = datetime.now(timezone.utc).date()
    d2 = today.strftime("%Y-%m-%d")
    d1 = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    if date_has_all_three(d1) and date_has_all_three(d2):
        return d1, d2
    return find_latest_two_dates_available()


if __name__ == "__main__":
    d1, d2 = select_dates_for_report()
    if not d2:
        print("❌ No processed days found with all required CSVs. Run the processor first.")
        raise SystemExit(1)
    if not d1:
        print(f"⚠️ Only one suitable day found ({d2}). Need two days to build comparisons. Exiting.")
        raise SystemExit(1)

    ym = d2[:7]

    # 1) Build report input CSVs under reports/<ym>/<d2>/data/
    generate_network_metrics_csv(d1, d2)
    generate_market_metrics_csv(d1, d2)
    generate_staking_metrics_csv(d1, d2)

    # 2) Load them back for charts + report
    net_rel = f"{REPORTS_PATH}{ym}/{d2}/data/network_metrics.csv"
    eco_rel = f"{REPORTS_PATH}{ym}/{d2}/data/economic_metrics.csv"
    stk_rel = f"{REPORTS_PATH}{ym}/{d2}/data/staking_metrics.csv"

    net_df = read_csv(net_rel)
    eco_df = read_csv(eco_rel)
    stk_df = read_csv(stk_rel)

    if any(x is None for x in (net_df, eco_df, stk_df)):
        print("❌ Report input CSVs were not created; aborting charts/report.")
        raise SystemExit(1)
    
    # ✅ Normalize headers to the names your charts expect
    net_df = rename_and_clean_data(net_df, "network", d1, d2)
    eco_df = rename_and_clean_data(eco_df, "economic", d1, d2)
    stk_df = rename_and_clean_data(stk_df, "staking", d1, d2)

    # 3) Charts
    generate_transaction_volume_chart(net_df, d2)
    generate_market_trends_chart(eco_df, d2)
    generate_trading_volume_chart(eco_df, d2)
    generate_supply_comparison_chart(eco_df, d2)
    generate_gas_fees_trends_chart(net_df, d2)
    generate_staking_comparison_chart(stk_df, d2)
    generate_tps_comparison_chart(net_df, d2)

    # 4) Markdown report
    template_rel = f"{REPORTS_PATH}templates/daily_report_template.md"
    report_rel   = f"{REPORTS_PATH}{ym}/{d2}/generated/daily_report.md"
    visuals_rel_folder = f"{REPORTS_PATH}{ym}/{d2}/visualizations"
    generate_actual_report(template_rel, report_rel, net_df, eco_df, stk_df, visuals_rel_folder, d2, d1)

    print(f"🎉 Daily report generated for {d1} → {d2}")




