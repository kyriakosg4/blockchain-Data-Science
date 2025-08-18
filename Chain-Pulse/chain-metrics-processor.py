import os
import json
import csv
import boto3
from datetime import datetime, timedelta, timezone
from statistics import mean
import shutil

from dotenv import load_dotenv
load_dotenv()

LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"

# Local folders (USED ONLY WHEN LOCAL_MODE=true)
RAW_DIR        = os.getenv("RAW_DIR",  r"C:/Users/User/Desktop/Chain-Pulse-AWS-lambda/s3/buckets/chain-pulse-metrics/raw-data-all-types")
PROCESSED_DIR  = os.getenv("PROCESSED_DIR", r"C:/Users/User/Desktop/Chain-Pulse-AWS-lambda/s3/buckets/chain-pulse-metrics/processed-data-all-types")
LOG_DIR        = os.getenv("LOG_DIR",  r"C:/Users/User/Desktop/Chain-Pulse-AWS-lambda/s3/buckets/chain-pulse-metrics/processed-files-log")

# AWS (USED ONLY WHEN LOCAL_MODE=false)
S3_BUCKET = os.getenv("S3_BUCKET", "chain-pulse-metrics")
S3_RAW_PREFIX = "raw-data-all-types"
S3_PROCESSED_PREFIX = "processed-data-all-types"
S3_LOG_PREFIX = "processed-files-log"

s3_client = None if LOCAL_MODE else boto3.client("s3")
PROCESSED_LOG_FILE = os.path.join("/tmp" if not LOCAL_MODE else LOG_DIR, "processed_files.json")


def get_current_month_prefix():
    """Generate the S3 prefix for the current month's folder."""
    current_month = datetime.now().strftime("%Y-%m")
    return f"raw-data-all-types/{current_month}/"



def get_week_of_month(date):
    """Return week number within the current month (1..4/5), in 7-day buckets."""
    # Keep timezone info if present
    if date.tzinfo is not None:
        start_of_month = date.replace(day=1)
    else:
        # make it explicit that both are naive
        start_of_month = date.replace(day=1)

    # Day 1-7 -> week 1, 8-14 -> week 2, etc.
    return ((date.day - 1) // 7) + 1





def get_log_file_key():
    # S3 key for the daily log (AWS mode)
    return f"{S3_LOG_PREFIX}/processed_files.json"



def get_last_processed_date():
    """Find the last processed date from the daily log or start fresh if missing."""
    processed_files = load_processed_log()

    today_str = datetime.now().strftime("%Y-%m-%d")

    if not processed_files["processed_files"]:
        # No files have been processed yet today
        print(f"🆕 No previous log data. Starting fresh from {today_str}")
        return today_str  # Start fresh from today

    # Extract dates from processed files
    dates = sorted(set(key.split('/')[2] for key in processed_files["processed_files"].keys() if key.startswith("raw-data-all-types/")))

    last_date = dates[-1] if dates else today_str  # Default to today if no dates found
    print(f"🔄 Resuming from last processed date: {last_date}")
    return last_date



def load_processed_log():
    """Load today's log; if missing/corrupted, start fresh."""
    today_str = datetime.now().strftime("%Y-%m-%d")

    if LOCAL_MODE:
        os.makedirs(LOG_DIR, exist_ok=True)
        if os.path.exists(PROCESSED_LOG_FILE):
            try:
                with open(PROCESSED_LOG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("date") != today_str:
                    data = {"date": today_str, "processed_files": {}}
                    save_processed_log(data)
                return data
            except Exception:
                pass
        # create fresh
        data = {"date": today_str, "processed_files": {}}
        save_processed_log(data)
        return data

    # AWS mode (S3)
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=get_log_file_key())
        s3_client.download_file(S3_BUCKET, get_log_file_key(), PROCESSED_LOG_FILE)
        with open(PROCESSED_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") != today_str:
            data = {"date": today_str, "processed_files": {}}
            save_processed_log(data)
        return data
    except Exception:
        data = {"date": today_str, "processed_files": {}}
        save_processed_log(data)
        return data


def save_processed_log(processed_files):
    """Persist the daily log (local file or S3)."""
    processed_files["date"] = datetime.now().strftime("%Y-%m-%d")

    # Always write locally (so we have a copy)
    os.makedirs(os.path.dirname(PROCESSED_LOG_FILE), exist_ok=True)
    with open(PROCESSED_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_files, f, indent=4)

    if not LOCAL_MODE:
        s3_client.upload_file(PROCESSED_LOG_FILE, S3_BUCKET, get_log_file_key())
        print(f"✅ Log saved to S3: {get_log_file_key()}")
    else:
        print(f"✅ Log saved locally: {PROCESSED_LOG_FILE}")



def list_all_objects_for_date(year_month: str, day: str):
    """Return a list of file identifiers for the given date folder.
       - LOCAL_MODE: absolute file paths
       - AWS mode:   S3 object dicts like {'Key': '...'}
    """
    if LOCAL_MODE:
        day_dir = os.path.join(RAW_DIR, year_month, day)
        if not os.path.isdir(day_dir):
            return []
        files = []
        for root, _, filenames in os.walk(day_dir):
            for name in filenames:
                if name.endswith(".json"):
                    files.append(os.path.join(root, name))
        return files

    # AWS mode
    prefix = f"{S3_RAW_PREFIX}/{year_month}/{day}/"
    objects, token = [], None
    while True:
        kwargs = dict(Bucket=S3_BUCKET, Prefix=prefix)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3_client.list_objects_v2(**kwargs)
        objects.extend(resp.get("Contents", []))
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return objects



def lambda_handler(event, context):
    processed_bucket = S3_BUCKET
    processed_folder = S3_PROCESSED_PREFIX

    last_processed_date = get_last_processed_date()
    print(f"🔄 Resuming from {last_processed_date}")

    today = datetime.now().strftime("%Y-%m-%d")
    if last_processed_date < today:
        last_processed_date = today

    year_month = last_processed_date[:7]
    print(f"📂 Date folder: {year_month}/{last_processed_date}")

    processed_log = load_processed_log()
    print(f"✅ Loaded log with {len(processed_log['processed_files'])} entries.")

    network_data, economic_data, staking_data = {}, {}, {}
    timestamps = []
    files_found = files_processed = 0

    objects = list_all_objects_for_date(year_month, last_processed_date)
    if not objects:
        print("⚠️ No files found for the date.")
    else:
        for obj in objects:
            file_key = obj if LOCAL_MODE else obj["Key"]
            files_found += 1

            # ✅ bug fix: check the right dict
            if file_key in processed_log["processed_files"]:
                print(f"⏩ Skip (already processed): {file_key}")
                continue

            try:
                process_file(obj, S3_BUCKET, network_data, economic_data, staking_data, timestamps, processed_log)
                files_processed += 1
            except Exception as e:
                print(f"❌ Error processing {file_key}: {e}")

    print(f"📊 Files found: {files_found}, processed: {files_processed}, skipped: {files_found - files_processed}")

    if timestamps:
        latest_timestamp = max(timestamps)
        latest_dt = datetime.fromtimestamp(latest_timestamp, tz=timezone.utc)
        final_ts = latest_dt.isoformat()
        print(f"📌 Final timestamp used in CSV: {final_ts}")

        # Save daily + weekly CSVs (function already dispatches local/S3)
        save_metric_csv_by_date(network_data, economic_data, staking_data, today, processed_bucket, processed_folder, final_ts)

    save_processed_log(processed_log)
    print("🎯 All files processed successfully.")
    return {'statusCode': 200, 'body': 'All files processed successfully.'}





def process_file(file_id, raw_bucket, network_data, economic_data, staking_data, timestamps, processed_files):
    """file_id is a path (local) or an S3 object dict (AWS)."""
    try:
        if LOCAL_MODE:
            file_key = file_id  # path
            local_file_path = file_key
        else:
            file_key = file_id["Key"]
            local_file_path = f"/tmp/{os.path.basename(file_key)}"
            try:
                s3_client.download_file(raw_bucket, file_key, local_file_path)
            except Exception as e:
                print(f"❌ Error downloading {file_key}: {e}")
                return

        # Read JSON
        try:
            with open(local_file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"❌ Error decoding JSON from {file_key}: {e}")
            return

        # Gas fees special case
        if "gas_fees&prices" in os.path.basename(file_key):
            process_gas_fees_file(json_data, network_data)
            processed_files["processed_files"][file_key] = datetime.now().isoformat()
            return

        source = identify_source(file_key)
        if source == "coingecko":
            entries = process_coingecko(json_data)
        elif source == "stakingrewards":
            entries = process_stakingrewards(json_data)
        elif source == "defillama":
            entries = process_defillama(json_data)
        elif source == "tps_values":
            entries = process_tps(json_data)
        else:
            print(f"⚠️ Unknown source: {file_key}")
            return

        for entry in entries:
            metric = entry.get("metric", "unknown")
            blockchain = normalize_blockchain_name(entry.get("blockchain", "unknown"))
            value = entry.get("value", "null")
            ts = normalize_timestamp(entry.get("timestamp"))
            if ts:
                try:
                    timestamps.append(datetime.fromisoformat(ts).timestamp())
                except Exception:
                    pass

            if metric in ["volume", "active_validators", "gas_price", "block_time", "tps"]:
                update_metric_data(network_data, blockchain, metric, value)
            elif metric in ["price", "usd_market_cap", "usd_24h_vol", "usd_24h_change", "circulating_supply", "inflation_rate", "total_supply"]:
                update_metric_data(economic_data, blockchain, metric, value)
            elif metric in ["reward_rate", "staked_tokens", "staking_ratio", "tvl"]:
                update_metric_data(staking_data, blockchain, metric, value)

        processed_files["processed_files"][file_key] = datetime.now().isoformat()
        print(f"✅ Processed: {file_key}")

    except Exception as e:
        print(f"❌ Unhandled error while processing {file_key}: {e}")



def save_csv_to_s3(data, bucket, file_key, timestamp, metric_type):
    """Save metrics with strict column order. Writes to local when LOCAL_MODE=true."""
    if metric_type == "network":
        columns = [
            "timestamp", "solana_volume", "avalanche_volume", "ethereum_volume", "binance_volume",
            "solana_active_validators", "avalanche_active_validators", "ethereum_active_validators", "binance_active_validators",
            "solana_block_time", "avalanche_block_time", "ethereum_block_time", "binance_block_time",
            "solana_avg_cu_per_tx", "avalanche_gas_price_gwei", "ethereum_gas_price_gwei", "binance_gas_price_gwei",
            "solana_cost_per_cu_lamports", "avalanche_avg_gas_used", "ethereum_avg_gas_used", "binance_avg_gas_used",
            "solana_transaction_cost", "avalanche_transaction_fee", "ethereum_transaction_fee", "binance_transaction_fee",
            "solana_tps", "avalanche_tps", "ethereum_tps", "binance_tps"
        ]
    elif metric_type == "economic":
        columns = [
            "timestamp", "solana_price", "avalanche_price", "ethereum_price", "binance_price",
            "solana_usd_market_cap", "avalanche_usd_market_cap", "ethereum_usd_market_cap", "binance_usd_market_cap",
            "solana_usd_24h_vol", "avalanche_usd_24h_vol", "ethereum_usd_24h_vol", "binance_usd_24h_vol",
            "solana_usd_24h_change", "avalanche_usd_24h_change", "ethereum_usd_24h_change", "binance_usd_24h_change",
            "solana_circulating_supply", "avalanche_circulating_supply", "ethereum_circulating_supply", "binance_circulating_supply",
            "solana_inflation_rate", "avalanche_inflation_rate", "ethereum_inflation_rate", "binance_inflation_rate",
            "solana_total_supply", "avalanche_total_supply", "ethereum_total_supply", "binance_total_supply"
        ]
    elif metric_type == "staking":
        columns = [
            "timestamp", "solana_reward_rate", "avalanche_reward_rate", "ethereum_reward_rate", "binance_reward_rate",
            "solana_staked_tokens", "avalanche_staked_tokens", "ethereum_staked_tokens", "binance_staked_tokens",
            "solana_staking_ratio", "avalanche_staking_ratio", "ethereum_staking_ratio", "binance_staking_ratio",
            "solana_tvl", "avalanche_tvl", "ethereum_tvl", "binance_tvl"
        ]
    else:
        print(f"❌ Invalid metric type: {metric_type}")
        return
    
    # derive a local path mirroring file_key
    if LOCAL_MODE:
        rel_key = file_key
        prefix = f"{S3_PROCESSED_PREFIX}/"
        if rel_key.startswith(prefix):
            rel_key = rel_key[len(prefix):]  # remove leading 'processed-data-all-types/'
        local_file_path = os.path.join(PROCESSED_DIR, rel_key)
    else:
        local_file_path = f"/tmp/{os.path.basename(file_key)}"


    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

    # Try to append previous rows (local or S3)
    existing_rows = []
    if LOCAL_MODE:
        if os.path.exists(local_file_path):
            with open(local_file_path, "r", newline='', encoding="utf-8") as csvfile:
                existing_rows = list(csv.DictReader(csvfile))
    else:
        try:
            s3_client.download_file(bucket, file_key, local_file_path)
            with open(local_file_path, "r", newline='', encoding="utf-8") as csvfile:
                existing_rows = list(csv.DictReader(csvfile))
            print(f"📂 Existing file found: {file_key}. Appending data.")
        except s3_client.exceptions.ClientError:
            print(f"📂 File not found in S3: {file_key}. Creating new.")

    cleaned_data = {"timestamp": timestamp.replace("+00:00", "")}
    for col in columns[1:]:
        cleaned_data[col] = data.get(col, "null")
    existing_rows.append(cleaned_data)

    with open(local_file_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(existing_rows)

    if LOCAL_MODE:
        print(f"✅ Saved locally: {local_file_path}")
    else:
        s3_client.upload_file(local_file_path, bucket, file_key)
        print(f"✅ Saved/Updated CSV to S3: {file_key}")




def process_coingecko(data):
    try:
        timestamp = data.get('timestamp', 'null')
        blockchain_data = data.get('data', {})  # Extract the "data" key containing blockchain info

        results = []
        for primary_chain, nested_data in blockchain_data.items():
            for blockchain, metrics in nested_data.items():  # ✅ Ensure both levels are iterated
                blockchain = normalize_blockchain_name(blockchain)
                results.append({'timestamp': timestamp, 'blockchain': blockchain, 'metric': 'price', 'value': metrics.get('usd', 'null')})
                results.append({'timestamp': timestamp, 'blockchain': blockchain, 'metric': 'usd_market_cap', 'value': metrics.get('usd_market_cap', 'null')})
                results.append({'timestamp': timestamp, 'blockchain': blockchain, 'metric': 'usd_24h_vol', 'value': metrics.get('usd_24h_vol', 'null')})
                results.append({'timestamp': timestamp, 'blockchain': blockchain, 'metric': 'usd_24h_change', 'value': metrics.get('usd_24h_change', 'null')})

        return results
    except Exception as e:
        print(f"❌ Error processing CoinGecko data: {str(e)}")
        return []



def process_stakingrewards(data):
    try:
        timestamp = data.get('timestamp', 'null')
        staking_data = data.get('staking_data', {})  # Extract the "staking_data" key containing metrics

        results = []
        for blockchain, metrics in staking_data.items():
            blockchain = normalize_blockchain_name(blockchain)
            for metric, value in metrics.items():
                results.append({'timestamp': timestamp, 'blockchain': blockchain, 'metric': metric, 'value': value})

        return results
    except Exception as e:
        print(f"❌ Error processing StakingRewards data: {str(e)}")
        return []



def process_defillama(data):
    try:
        timestamp = data.get('timestamp', 'null')
        transaction_volumes = data.get('transaction_volume', {})  # Extract transaction volume data
        tvl_data = data.get('tvl', [])  # Extract TVL data

        results = []
        # Process transaction volumes
        for blockchain, volume_info in transaction_volumes.items():
            blockchain = normalize_blockchain_name(blockchain)
            results.append({
                'timestamp': timestamp,
                'blockchain': blockchain,
                'metric': 'volume',
                'value': volume_info.get('volume', 'null')
            })

        # Process TVL data
        for chain_entry in tvl_data:
            blockchain = normalize_blockchain_name(chain_entry.get('chain'))
            results.append({
                'timestamp': timestamp,
                'blockchain': blockchain,
                'metric': 'tvl',
                'value': chain_entry.get('tvl', 'null')
            })

        return results
    except Exception as e:
        print(f"❌ Error processing DeFiLlama data: {str(e)}")
        return []



def process_gas_fees_file(json_data, network_data):
    """
    Extracts gas fees and transaction costs from JSON and updates network_data.
    """
    try:
        # Define expected fields and initialize them with None
        gas_fee_metrics = {
            "solana_avg_cu_per_tx": None,
            "solana_cost_per_cu_lamports": None,
            "solana_transaction_cost": None,
            "avalanche_gas_price_gwei": None,
            "avalanche_avg_gas_used": None,
            "avalanche_transaction_fee": None,
            "ethereum_gas_price_gwei": None,
            "ethereum_avg_gas_used": None,
            "ethereum_transaction_fee": None,
            "binance_gas_price_gwei": None,
            "binance_avg_gas_used": None,
            "binance_transaction_fee": None
        }

        # Extract fees data from JSON
        for fee_entry in json_data.get("fees", []):
            chain = fee_entry.get("chain", "").lower()  # Ensure lowercase for consistency

            if chain == "solana":
                gas_fee_metrics["solana_avg_cu_per_tx"] = fee_entry.get("avg_cu_per_tx", None)
                gas_fee_metrics["solana_cost_per_cu_lamports"] = fee_entry.get("avg_cost_per_cu_lamports", None)
                gas_fee_metrics["solana_transaction_cost"] = fee_entry.get("total_transaction_cost", None)

            elif chain == "avalanche":
                gas_fee_metrics["avalanche_gas_price_gwei"] = fee_entry.get("avg_gas_price_gwei", None)
                gas_fee_metrics["avalanche_avg_gas_used"] = fee_entry.get("avg_gas_used", None)
                gas_fee_metrics["avalanche_transaction_fee"] = fee_entry.get("transaction_fee", None)

            elif chain == "ethereum":
                gas_fee_metrics["ethereum_gas_price_gwei"] = fee_entry.get("avg_gas_price_gwei", None)
                gas_fee_metrics["ethereum_avg_gas_used"] = fee_entry.get("avg_gas_used", None)
                gas_fee_metrics["ethereum_transaction_fee"] = fee_entry.get("transaction_fee", None)

            elif chain == "bsc":
                gas_fee_metrics["binance_gas_price_gwei"] = fee_entry.get("avg_gas_price_gwei", None)
                gas_fee_metrics["binance_avg_gas_used"] = fee_entry.get("avg_gas_used", None)
                gas_fee_metrics["binance_transaction_fee"] = fee_entry.get("transaction_fee", None)

        # Print extracted values for debugging
        print("✅ Extracted Gas Fees Data:", json.dumps(gas_fee_metrics, indent=4))

        # Store extracted data in network_data
        network_data.update(gas_fee_metrics)

        print("✅ Gas fees data extracted and stored in network_data.")

    except Exception as e:
        print(f"❌ Error processing gas fees file: {str(e)}")


def process_tps(data):
    try:
        timestamp = data.get('timestamp', 'null')
        tps_data = data.get('tps', {})

        results = []
        for blockchain, tps_value in tps_data.items():
            blockchain = normalize_blockchain_name(blockchain)  # ✅ Normalize names
            results.append({
                'timestamp': timestamp,
                'blockchain': blockchain,
                'metric': 'tps',
                'value': tps_value
            })

        return results
    except Exception as e:
        print(f"❌ Error processing TPS data: {str(e)}")
        return []






def update_metric_data(data, blockchain, metric, value):
    key = f"{blockchain}_{metric}"
    data[key] = value

def normalize_blockchain_name(blockchain):
    blockchain_mapping = {
        "eth": "ethereum",
        "ethereum": "ethereum",
        "ethereum-2-0": "ethereum",
        "bsc": "binance",  
        "binance-smart-chain": "binance",  
        "bnb": "binance",
        "binancecoin": "binance",
        "solana": "solana",
        "avalanche": "avalanche",
        "avalanche-2": "avalanche",
        "avax": "avalanche",
    }
    return blockchain_mapping.get(blockchain.lower(), blockchain.lower())


def identify_source(file_path):
    """Identify the source of the data based on the file name."""
    file_name = os.path.basename(file_path).lower()
    if "stakingrewards" in file_name:
        return "stakingrewards"
    elif "defillama" in file_name:
        return "defillama"
    elif "coingecko" in file_name:
        return "coingecko"
    elif "rpc_avalanche" in file_name:
        return "rpc_avalanche"
    elif "etherscan" in file_name:
        return "etherscan"
    elif "tps_values" in file_name:  
        return "tps_values"
    else:
        return "unknown"



def normalize_timestamp(timestamp):
    try:
        print(f"🕒 Before normalization: {timestamp}")

        if isinstance(timestamp, (int, float)):
            ts = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        elif isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = dt.astimezone(timezone.utc)
        else:
            raise ValueError("Unsupported timestamp format")

        # ✅ Ensure it is fully naive (removing timezone) and formatted correctly
        formatted_ts = ts.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S.%f")  # microseconds kept

        print(f"✅ After normalization: {formatted_ts}")
        return formatted_ts

    except Exception as e:
        print(f"❌ Error normalizing timestamp {timestamp}: {str(e)}")
        return None


def save_metric_csv_by_date(network_data, economic_data, staking_data, folder_date, bucket, folder, timestamp):
    """Ensure correct column order and save CSVs for network, economic, and staking metrics."""
    year_month = folder_date[:7]  # e.g., "2025-02"

    timestamp_dt = datetime.fromisoformat(timestamp)
    if timestamp_dt.tzinfo is None:  # If it's naive, set it to UTC
        timestamp_dt = timestamp_dt.replace(tzinfo=timezone.utc)

    week_number = get_week_of_month(timestamp_dt)  # Now always timezone-aware

    # Define file paths
    network_metrics_file = f"{folder}/{year_month}/{folder_date}/network_metrics.csv"
    economic_metrics_file = f"{folder}/{year_month}/{folder_date}/economic_metrics.csv"
    staking_metrics_file = f"{folder}/{year_month}/{folder_date}/staking_metrics.csv"

    weekly_network_file = f"{folder}/weekly-data/{year_month}-week{week_number}/network_metrics.csv"
    weekly_economic_file = f"{folder}/weekly-data/{year_month}-week{week_number}/economic_metrics.csv"
    weekly_staking_file = f"{folder}/weekly-data/{year_month}-week{week_number}/staking_metrics.csv"

    # Save CSVs with the correct metric type
    save_csv_to_s3(network_data, bucket, network_metrics_file, timestamp, "network")
    save_csv_to_s3(economic_data, bucket, economic_metrics_file, timestamp, "economic")
    save_csv_to_s3(staking_data, bucket, staking_metrics_file, timestamp, "staking")

    # Save weekly data
    save_csv_to_s3(network_data, bucket, weekly_network_file, timestamp, "network")
    save_csv_to_s3(economic_data, bucket, weekly_economic_file, timestamp, "economic")
    save_csv_to_s3(staking_data, bucket, weekly_staking_file, timestamp, "staking")



if __name__ == "__main__":
    # helpful debug prints
    print(f"LOCAL_MODE={LOCAL_MODE}")
    if LOCAL_MODE:
        print(f"RAW_DIR={RAW_DIR}")
        print(f"PROCESSED_DIR={PROCESSED_DIR}")
        print(f"LOG_DIR={LOG_DIR}")
    else:
        print(f"S3_BUCKET={S3_BUCKET}")
        print(f"S3_RAW_PREFIX={S3_RAW_PREFIX}")
        print(f"S3_PROCESSED_PREFIX={S3_PROCESSED_PREFIX}")
        print(f"S3_LOG_PREFIX={S3_LOG_PREFIX}")

    # run the same flow Lambda would run
    result = lambda_handler({}, None)
    print(result)



