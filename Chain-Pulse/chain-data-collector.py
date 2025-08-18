import boto3
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import requests
import os
from concurrent.futures import ThreadPoolExecutor



# (optional but convenient)
try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env when present
except Exception:
    pass


LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")  # where local JSONs go


# 🔑 Load configuration from environment variables
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
BSCSCAN_API_KEY = os.getenv("BSCSCAN_API_KEY", "")
STAKINGREWARDS_API_KEY = os.getenv("STAKINGREWARDS_API_KEY", "")

S3_BUCKET = os.getenv("S3_BUCKET", "chain-pulse-metrics")
S3_PATH = os.getenv("S3_PATH", "raw-data-all-types")


class ChainPulseError(Exception):
    """Base exception for ChainPulse errors."""
    pass


class BaseCollector(ABC):
    """Abstract base collector with common functionality."""

    def __init__(self, s3_bucket: Optional[str], s3_path: Optional[str], *, local_mode: Optional[bool] = None, output_dir: str = OUTPUT_DIR):
        self.local_mode = LOCAL_MODE if local_mode is None else local_mode
        self.output_dir = output_dir

        self.s3_bucket = s3_bucket
        self.s3_path = s3_path
        self.s3_client = None if self.local_mode else boto3.client("s3")

    def fetch_with_retry(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
        """Fetch data from API with retry and exponential backoff."""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, headers=headers)
                if response.status_code == 429:  # Rate limiting
                    retry_after = 2 ** attempt
                    print(f"Rate limit hit. Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"Error fetching data from {url} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise ChainPulseError(f"Failed to fetch data after {max_retries} attempts.")
        return None

    def _build_paths(self, source: str, chain: Optional[str] = None):
        timestamp = datetime.now(timezone.utc)
        year_month = timestamp.strftime('%Y-%m')  # e.g., "2025-02"
        date = timestamp.strftime('%Y-%m-%d')     # e.g., "2025-02-09"
        filename = f"{source}_{chain}_{timestamp.strftime('%H%M%S')}.json" if chain else f"{source}_{timestamp.strftime('%H%M%S')}.json"
        return year_month, date, filename

    def save_output(self, data: Dict, source: str, chain: Optional[str] = None) -> None:
        """
        Save JSON either to S3 (prod) or to local filesystem (dev), based on LOCAL_MODE/env.
        """
        year_month, date, filename = self._build_paths(source, chain)

        if self.local_mode:
            # Save locally (dev/testing)
            folder = os.path.join(self.output_dir, year_month, date)
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"✅ Saved locally: {filepath}")
            return

        # Save to S3 (production)
        if not self.s3_bucket or not self.s3_path:
            raise ChainPulseError("S3 configuration missing (s3_bucket/s3_path).")

        s3_key = f"{self.s3_path}/{year_month}/{date}/{filename}"
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=json.dumps(data, indent=2),
                ContentType="application/json"
            )
            print(f"✅ Saved to S3: {s3_key}")
        except Exception as e:
            print(f"Error saving data to S3: {e}")
            raise

    @abstractmethod
    def collect(self):
        pass



class CoinGeckoCollector(BaseCollector):
    """Collector for CoinGecko API data."""

    def __init__(self, s3_bucket: str, s3_path: str):
        super().__init__(s3_bucket, s3_path)
        self.coins = {
            "avax": "avalanche-2",
            "eth": "ethereum",
            "bnb": "binancecoin",
            "sol": "solana"
        }
        self.base_url = "https://api.coingecko.com/api/v3"

    def collect_single_coin(self, coin_id: str) -> Dict:
        """Fetch data for a single coin from CoinGecko API."""
        url = f"{self.base_url}/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true"
        }
        return self.fetch_with_retry(url, params=params)

    def collect(self) -> None:
        """Collect data for all coins and store in a single JSON file in S3."""
        all_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {}
        }

        for short_name, coin_id in self.coins.items():
            try:
                data = self.collect_single_coin(coin_id)
                if data:
                    all_data["data"][short_name] = data  # Store under respective blockchain key
            except Exception as e:
                print(f"Error collecting data for {short_name}: {e}")

        # Save the combined JSON file to S3
        self.save_output(all_data, "coingecko")




class DefiLlamaCollector(BaseCollector):
    """Collector for fetching both DefiLlama Transaction Volume and TVL in a single file."""

    def __init__(self, s3_bucket: str, s3_path: str):
        super().__init__(s3_bucket, s3_path)
        self.volume_chains = ["ethereum", "avalanche", "solana", "bsc"]
        self.base_url = "https://api.llama.fi"
        self.tvl_url = "https://api.llama.fi/v2/chains"
        self.target_tvl_chains = ["solana", "avalanche", "bsc", "ethereum"]

    def fetch_transaction_volume(self, blockchain: str) -> Optional[Dict]:
        """Fetch the latest DEX volume for a given blockchain."""
        url = f"{self.base_url}/overview/dexs/{blockchain}"
        try:
            data = self.fetch_with_retry(url)
            if data and "totalDataChart" in data and data["totalDataChart"]:
                latest_data = data["totalDataChart"][-1]
                timestamp, volume = latest_data
                date = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')
                return {"blockchain": blockchain, "date": date, "volume": volume}
        except Exception as e:
            print(f"Error fetching volume for {blockchain}: {e}")
        return None

    def fetch_tvl_data(self) -> Optional[List[Dict]]:
        """Fetch TVL data for selected blockchains."""
        try:
            data = self.fetch_with_retry(self.tvl_url)
            if data:
                return [
                    {
                        "chain": chain.get("name"),
                        "tvl": chain.get("tvl"),
                    }
                    for chain in data if chain.get("name", "").lower() in self.target_tvl_chains
                ]
        except Exception as e:
            print(f"Error fetching TVL data: {e}")
        return None

    def collect(self) -> None:
        """Collect both transaction volume and TVL data and save in a single JSON file."""
        combined_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transaction_volume": {},
            "tvl": []
        }

        # Collect transaction volume for each chain
        for chain in self.volume_chains:
            try:
                volume_data = self.fetch_transaction_volume(chain)
                if volume_data:
                    combined_data["transaction_volume"][chain] = volume_data
            except Exception as e:
                print(f"Error collecting volume data for {chain}: {e}")

        # Collect TVL data
        try:
            tvl_data = self.fetch_tvl_data()
            if tvl_data:
                combined_data["tvl"] = tvl_data
        except Exception as e:
            print(f"Error collecting TVL data: {e}")

        # Save the combined JSON file to S3
        self.save_output(combined_data, "defillama")




class StakingRewardsCollector(BaseCollector):
    """Collector for StakingRewards API data, storing all data in one JSON file."""

    def __init__(self, api_key: str, s3_bucket: str, s3_path: str):
        super().__init__(s3_bucket, s3_path)
        self.api_key = api_key
        self.chains = ["ethereum-2-0", "solana", "avalanche", "binance-smart-chain"]
        self.base_url = "https://api.stakingrewards.com/public/query"

    def fetch_staking_data(self) -> Optional[Dict]:
        """Fetch staking data for all blockchains from StakingRewards API."""
        query = """
        query HistoricalStakingRatios($chains: [String!]!, $limit: Int!) {
            assets(where: { slugs: $chains }, limit: $limit) {
                slug
                metrics(
                    where: { metricKeys: [
                        "staking_ratio",
                        "reward_rate",
                        "staked_tokens",
                        "inflation_rate",
                        "circulating_supply",
                        "total_supply",
                        "price",
                        "marketcap",
                        "active_validators",
                        "block_time"
                    ] },
                    limit: $limit
                ) {
                    defaultValue
                    metricKey
                    createdAt
                }
            }
        }
        """
        variables = {"chains": self.chains, "limit": 10}
        headers = {"X-API-KEY": self.api_key}

        try:
            response = self.fetch_with_retry_post(self.base_url, headers=headers, data={"query": query, "variables": variables})
            if response and "data" in response and response["data"]["assets"]:
                return response["data"]["assets"]
        except Exception as e:
            print(f"Error fetching StakingRewards data: {e}")
        return None

    def fetch_with_retry_post(self, url: str, headers: Dict[str, str], data: Dict, max_retries: int = 3) -> Optional[Dict]:
        """Fetch data from API with retry for POST requests."""
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=data, headers=headers)
                if response.status_code == 429:  # Rate limiting
                    print(f"Rate limit hit. Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data from {url} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to fetch data after {max_retries} attempts.")
        return None

    def collect(self) -> None:
        """Collect staking data for all chains and save in a single JSON file."""
        combined_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "staking_data": {}
        }

        staking_data = self.fetch_staking_data()
        if staking_data:
            for asset in staking_data:
                metrics = {metric["metricKey"]: metric["defaultValue"] for metric in asset["metrics"]}
                if "block_time" not in metrics:
                    print(f"Warning: 'block_time' metric is missing for chain: {asset['slug']}")

                # Store data under its respective blockchain key
                combined_data["staking_data"][asset["slug"]] = metrics

            # Save the combined JSON file to S3
            self.save_output(combined_data, "stakingrewards")
        else:
            print("No staking data found.")





class GasFeeCollector(BaseCollector):  # ✅ Inherits from BaseCollector to use save_to_s3()
    """Optimized Collector for blockchain gas fees and transaction costs."""

    def __init__(self, s3_bucket: str, s3_path: str, etherscan_api_key: str, bscscan_api_key: str):
        super().__init__(s3_bucket, s3_path)  # ✅ Initialize BaseCollector

        # ✅ API Keys are now passed from Lambda Handler
        self.etherscan_api_key = etherscan_api_key
        self.bscscan_api_key = bscscan_api_key

        # ✅ Updated Public RPC Endpoints
        self.rpc_urls = {
            "ethereum": "https://eth.llamarpc.com",
            "avalanche": "https://api.avax.network/ext/bc/C/rpc",
            "bsc": "https://bsc-dataseed.binance.org/",
            "solana": "https://api.mainnet-beta.solana.com"
        }

    def fetch_gas_fees(self, chain_name: str, rpc_url: str, api_key: Optional[str] = None) -> Optional[Dict]:
        """Fetch gas fees for Ethereum, BSC, and Avalanche."""
        try:
            if chain_name == "ethereum" and api_key:
                # ✅ Fix: Use the correct Etherscan URL
                gas_url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={api_key}"
            elif chain_name == "bsc" and api_key:
                # ✅ Fix: Use the correct BscScan URL
                gas_url = f"https://api.bscscan.com/api?module=gastracker&action=gasoracle&apikey={api_key}"
            else:
                # ✅ For Avalanche and other chains, use their RPC directly
                gas_price_payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_gasPrice", "params": []}
                gas_price_response = requests.post(rpc_url, json=gas_price_payload).json()
                avg_gas_price_gwei = int(gas_price_response["result"], 16) / 1e9  # Convert Wei to Gwei

                # ✅ Fetch latest block number
                latest_block_payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}
                latest_block_response = requests.post(rpc_url, json=latest_block_payload).json()
                latest_block_number = int(latest_block_response["result"], 16)

                # ✅ Fetch gas used in last 4 blocks
                gas_used_list = []
                for i in range(4):
                    block_number = hex(latest_block_number - i)
                    block_payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getBlockByNumber", "params": [block_number, True]}
                    block_response = requests.post(rpc_url, json=block_payload).json()

                    if "result" not in block_response:
                        continue

                    transactions = block_response["result"]["transactions"]
                    gas_used_values = [int(tx["gas"], 16) for tx in transactions if "gas" in tx]
                    valid_gas_values = [g for g in gas_used_values if 21_000 <= g <= 5_000_000]

                    if valid_gas_values:
                        gas_used_list.append(sum(valid_gas_values) / len(valid_gas_values))

                if not gas_used_list:
                    return None

                avg_gas_used = sum(gas_used_list) / len(gas_used_list)
                transaction_fee_native = (avg_gas_used * avg_gas_price_gwei * 1e9) / 10**18

                return {
                    "chain": chain_name,
                    "avg_gas_price_gwei": avg_gas_price_gwei,
                    "avg_gas_used": avg_gas_used,
                    "transaction_fee": transaction_fee_native
                }

            # ✅ Fetch gas price from API (for Ethereum and BSC only)
            gas_response = requests.get(gas_url).json()
            if gas_response["status"] != "1":
                return None
            avg_gas_price_gwei = float(gas_response["result"]["ProposeGasPrice"])

            # ✅ Fetch latest block number
            latest_block_payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}
            latest_block_response = requests.post(rpc_url, json=latest_block_payload).json()
            latest_block_number = int(latest_block_response["result"], 16)

            # ✅ Fetch gas used in last 4 blocks
            gas_used_list = []
            for i in range(4):
                block_number = hex(latest_block_number - i)
                block_payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getBlockByNumber", "params": [block_number, True]}
                block_response = requests.post(rpc_url, json=block_payload).json()

                if "result" not in block_response:
                    continue

                transactions = block_response["result"]["transactions"]
                gas_used_values = [int(tx["gas"], 16) for tx in transactions if "gas" in tx]
                valid_gas_values = [g for g in gas_used_values if 21_000 <= g <= 5_000_000]

                if valid_gas_values:
                    gas_used_list.append(sum(valid_gas_values) / len(valid_gas_values))

            if not gas_used_list:
                return None

            avg_gas_used = sum(gas_used_list) / len(gas_used_list)
            transaction_fee_native = (avg_gas_used * avg_gas_price_gwei * 1e9) / 10**18

            return {
                "chain": chain_name,
                "avg_gas_price_gwei": avg_gas_price_gwei,
                "avg_gas_used": avg_gas_used,
                "transaction_fee": transaction_fee_native
            }
        except Exception as e:
            print(f"❌ Error fetching {chain_name} fees: {e}")
            return None

    def fetch_solana_fees(self, num_slots: int = 4, max_retries: int = 3) -> Optional[Dict]:
        """Fetch real-time Solana fees using latest slots with retry mechanism, excluding 'slot' from final JSON."""
        try:
            solana_rpc = self.rpc_urls["solana"]
            
            # ✅ Step 1: Get latest Solana slot
            slot_payload = {"jsonrpc": "2.0", "id": 1, "method": "getSlot", "params": [{"commitment": "finalized"}]}
            slot_response = requests.post(solana_rpc, json=slot_payload).json()

            if "result" not in slot_response:
                print("❌ Error: Could not fetch Solana slot.")
                return None

            latest_slot = slot_response["result"]
            print(f"🔹 Latest Solana Slot: {latest_slot}")

            compute_units_used = []
            prioritization_fees = []

            # ✅ Step 2: Fetch transaction details for last `num_slots`
            for i in range(num_slots):
                slot_number = latest_slot - i
                retries = 0

                while retries < max_retries:
                    block_payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "getBlock",
                        "params": [
                            slot_number,
                            {"encoding": "jsonParsed", "transactionDetails": "full", "maxSupportedTransactionVersion": 0}
                        ]
                    }
                    block_response = requests.post(solana_rpc, json=block_payload).json()

                    if "result" in block_response:
                        break  # ✅ Exit retry loop if block fetch is successful
                    
                    retries += 1
                    print(f"⚠️ Retry {retries}/{max_retries} fetching Solana block {slot_number}...")

                if "result" not in block_response:
                    print(f"❌ Failed to fetch Solana block {slot_number} after {max_retries} retries.")
                    continue  # Move to the next slot instead of failing

                transactions = block_response["result"].get("transactions", [])
                if not transactions:
                    print(f"⚠️ Warning: No transactions found in slot {slot_number}.")
                    continue

                for tx in transactions:
                    meta = tx.get("meta", {})
                    compute_units = meta.get("computeUnitsConsumed", 0)
                    fee = meta.get("fee", 0)

                    if compute_units > 0:
                        compute_units_used.append(compute_units)

                    if fee > 0:
                        prioritization_fees.append(fee)

            if not compute_units_used:
                print("❌ No compute unit data found for Solana. Check RPC availability.")
                return None

            avg_cu_per_tx = sum(compute_units_used) / len(compute_units_used)
            if prioritization_fees:
                avg_prioritization_fee = sum(prioritization_fees) / len(prioritization_fees)
            else:
                avg_prioritization_fee = 5000  # Default when no fees exist (debugging)

            avg_cost_per_cu = avg_prioritization_fee / avg_cu_per_tx
            total_tx_cost = avg_cu_per_tx * avg_cost_per_cu / 10**9  # Convert Lamports to SOL

            print(f"✅ Solana Fees Collected: CU={avg_cu_per_tx}, Fee={total_tx_cost}")

            return {
                "chain": "solana",
                "avg_cu_per_tx": avg_cu_per_tx,
                "avg_cost_per_cu_lamports": avg_cost_per_cu,
                "total_transaction_cost": total_tx_cost  # ✅ 'slot' is removed from the final JSON output
            }

        except Exception as e:
            print(f"❌ Error fetching Solana fees: {e}")
            return None


    def collect(self) -> None:
        """Collect gas fee data for all chains and store in S3."""
        fees_data = {"timestamp": datetime.now(timezone.utc).isoformat(), "fees": []}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}

            # ✅ Fetch gas fees dynamically for all chains
            for chain, rpc_url in self.rpc_urls.items():
                if chain == "ethereum":
                    futures[executor.submit(self.fetch_gas_fees, chain, rpc_url, self.etherscan_api_key)] = chain
                elif chain == "bsc":
                    futures[executor.submit(self.fetch_gas_fees, chain, rpc_url, self.bscscan_api_key)] = chain
                elif chain == "solana":
                    futures[executor.submit(self.fetch_solana_fees, 4)] = chain  # ✅ Special handling for Solana
                else:
                    futures[executor.submit(self.fetch_gas_fees, chain, rpc_url)] = chain  # ✅ Default case

            # ✅ Collect results
            for future in futures:
                result = future.result()
                if result:
                    fees_data["fees"].append(result)

        # ✅ Save to S3 as gas_fees&prices.json
        self.save_output(fees_data, "gas_fees&prices")
        print("✅ Gas fee data collected and stored in S3.")



class TPSCollector(BaseCollector):
    """Collector for fetching real-time TPS data for multiple blockchains and storing in S3."""

    RPC_URLS = {
        "solana": "https://api.mainnet-beta.solana.com",
        "avalanche": "https://api.avax.network/ext/bc/C/rpc"
    }

    def __init__(self, etherscan_api_key: str, bscscan_api_key: str, s3_bucket: str, s3_path: str):
        super().__init__(s3_bucket, s3_path)
        self.etherscan_api_key = etherscan_api_key
        self.bscscan_api_key = bscscan_api_key
        self.EXPLORER_APIS = {
            "ethereum": f"https://api.etherscan.io/api?module=proxy&action=eth_blockNumber&apikey={self.etherscan_api_key}",
            "bsc": f"https://api.bscscan.com/api?module=proxy&action=eth_blockNumber&apikey={self.bscscan_api_key}"
        }

    def fetch_json(self, url: str) -> Optional[Dict]:
        """Fetch and validate JSON response"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")
            return None

    def fetch_solana_tps(self) -> Optional[float]:
        """Fetch Solana's real-time TPS using recent performance samples"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getRecentPerformanceSamples",
            "params": [4]  # Fetch last 4 slots
        }
        try:
            response = requests.post(self.RPC_URLS["solana"], json=payload).json()
            if "result" not in response or not response["result"]:
                return None

            samples = response["result"]
            tx_counts = [sample["numTransactions"] for sample in samples]
            time_periods = [sample["samplePeriodSecs"] for sample in samples]

            if not tx_counts or not time_periods:
                return None

            return sum(tx_counts) / sum(time_periods)

        except Exception as e:
            print(f"❌ Error fetching Solana TPS: {e}")
            return None

    def fetch_tps_etherscan_bscscan(self, chain: str) -> Optional[float]:
        """Fetch real-time TPS for Ethereum & BSC using Etherscan/BscScan APIs"""
        try:
            latest_block_response = self.fetch_json(self.EXPLORER_APIS[chain])
            if not latest_block_response or "result" not in latest_block_response:
                return None

            latest_block = int(latest_block_response["result"], 16)

            api_base_url = "https://api.etherscan.io" if chain == "ethereum" else "https://api.bscscan.com"
            api_key = self.etherscan_api_key if chain == "ethereum" else self.bscscan_api_key

            block_details_url = f"{api_base_url}/api?module=proxy&action=eth_getBlockByNumber&tag={hex(latest_block)}&boolean=true&apikey={api_key}"
            block_details_response = self.fetch_json(block_details_url)

            if not block_details_response or "result" not in block_details_response:
                return None

            block_details = block_details_response["result"]
            num_transactions = len(block_details["transactions"])
            timestamp_latest = int(block_details["timestamp"], 16)

            # Fetch previous block
            previous_block = latest_block - 1
            prev_block_details_url = f"{api_base_url}/api?module=proxy&action=eth_getBlockByNumber&tag={hex(previous_block)}&boolean=true&apikey={api_key}"
            prev_block_response = self.fetch_json(prev_block_details_url)

            if not prev_block_response or "result" not in prev_block_response:
                return None

            timestamp_previous = int(prev_block_response["result"]["timestamp"], 16)
            block_time = timestamp_latest - timestamp_previous

            if block_time <= 0:
                return None

            return num_transactions / block_time

        except Exception as e:
            print(f"❌ Error fetching {chain.upper()} TPS: {e}")
            return None

    def fetch_avalanche_tps(self) -> Optional[float]:
        """Fetch real-time TPS for Avalanche using its RPC API"""
        try:
            payload_block = {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}
            block_response = requests.post(self.RPC_URLS["avalanche"], json=payload_block).json()
            latest_block = int(block_response.get("result", "0"), 16)

            if not latest_block:
                return None

            payload_block_details = {"jsonrpc": "2.0", "id": 1, "method": "eth_getBlockByNumber", "params": [hex(latest_block), True]}
            block_details_response = requests.post(self.RPC_URLS["avalanche"], json=payload_block_details).json()

            if "result" not in block_details_response:
                return None

            block_details = block_details_response["result"]
            num_transactions = len(block_details["transactions"])
            timestamp_latest = int(block_details["timestamp"], 16)

            previous_block = latest_block - 1
            payload_prev_block = {"jsonrpc": "2.0", "id": 1, "method": "eth_getBlockByNumber", "params": [hex(previous_block), True]}
            prev_block_response = requests.post(self.RPC_URLS["avalanche"], json=payload_prev_block).json()

            if "result" not in prev_block_response:
                return None

            timestamp_previous = int(prev_block_response["result"]["timestamp"], 16)
            block_time = timestamp_latest - timestamp_previous

            if block_time <= 0:
                return None

            return num_transactions / block_time

        except Exception as e:
            print(f"❌ Error fetching Avalanche TPS: {e}")
            return None

    def collect(self) -> None:
        """Collect TPS data and save to S3."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tps": {
                "solana": self.fetch_solana_tps(),
                "ethereum": self.fetch_tps_etherscan_bscscan("ethereum"),
                "bsc": self.fetch_tps_etherscan_bscscan("bsc"),
                "avalanche": self.fetch_avalanche_tps(),
            }
        }
        
        # Save to S3 instead of local storage
        self.save_output(data, "tps_values")



        

def lambda_handler(event: Dict, context: Any) -> Dict:
    """Main Lambda handler."""

    collectors = [
        CoinGeckoCollector(S3_BUCKET, S3_PATH),
        DefiLlamaCollector(S3_BUCKET, S3_PATH),
        StakingRewardsCollector(STAKINGREWARDS_API_KEY, S3_BUCKET, S3_PATH),
        GasFeeCollector(S3_BUCKET, S3_PATH, ETHERSCAN_API_KEY, BSCSCAN_API_KEY),
        TPSCollector(
            etherscan_api_key=ETHERSCAN_API_KEY,
            bscscan_api_key=BSCSCAN_API_KEY,
            s3_bucket=S3_BUCKET,
            s3_path=S3_PATH
        )
    ]

    for collector in collectors:
        try:
            collector.collect()
        except Exception as e:
            print(f"Error in collector {collector.__class__.__name__}: {e}")

    return {"statusCode": 200, "body": "Data collection complete."}


if __name__ == "__main__":
    # Run the same workflow locally
    print(f"LOCAL_MODE={LOCAL_MODE}, OUTPUT_DIR={OUTPUT_DIR}")
    lambda_handler({}, None)


