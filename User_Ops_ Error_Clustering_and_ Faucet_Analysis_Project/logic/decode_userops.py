# decode_userops.py

import pandas as pd
import os
import asyncio
import nest_asyncio
from web3 import Web3
from typing import List, Dict
import re
from eth_abi import decode
from tqdm import tqdm

nest_asyncio.apply()

# --- UserOp Fields ---
userop_fields = [
    "address", "uint256", "bytes", "bytes", "uint256", "uint256",
    "uint256", "uint256", "uint256", "bytes", "bytes"
]

# --- Utils ---
def decode_native_value_blind(call_data_bytes):
    try:
        if len(call_data_bytes) < 100:
            return 0.0
        _, value_wei, _ = decode(["address", "uint256", "bytes"], call_data_bytes[4:])
        return value_wei / 1e18
    except Exception:
        return 0.0

def init_web3(rpc_url: str) -> Web3:
    return Web3(Web3.HTTPProvider(rpc_url))

def get_event_topic() -> str:
    event_signature = "UserOperationEvent(bytes32,address,address,uint256,bool,uint256,uint256)"
    return Web3.keccak(text=event_signature).hex()

# --- STEP 1: Decode Logs ---
async def process_tx(tx_hash: str, w3: Web3, event_topic: str) -> List[Dict]:
    decoded_results = []
    try:
        receipt = await asyncio.to_thread(w3.eth.get_transaction_receipt, tx_hash)
        tx = await asyncio.to_thread(w3.eth.get_transaction, tx_hash)
        raw_input = tx["input"]
        bundler = tx["from"]  # 🆕 Extract bundler address
        input_data = bytes.fromhex(raw_input[2:]) if isinstance(raw_input, str) else bytes(raw_input)

        for log in receipt.logs:
            if log["topics"][0].hex() == event_topic:
                userOpHash = log["topics"][1].hex()
                sender = Web3.to_checksum_address(log["topics"][2].hex()[-40:])
                paymaster = Web3.to_checksum_address(log["topics"][3].hex()[-40:])
                nonce, success, actualGasCost, actualGasUsed = w3.codec.decode(
                    ["uint256", "bool", "uint256", "uint256"],
                    log["data"]
                )

                decoded_results.append({
                    "tx_hash": tx_hash,
                    "userOpHash": userOpHash,
                    "sender": sender,
                    "paymaster": paymaster,
                    "bundler": bundler,  # 🆕 Add bundler to output
                    "nonce": nonce,
                    "success": success,
                    "actualGasCost": actualGasCost,
                    "actualGasUsed": actualGasUsed
                })

    except Exception as e:
        print(f"⚠️ Error processing tx {tx_hash}: {e}")

    return decoded_results

async def decode_userops_from_csv(input_csv: str, rpc_url: str, output_dir: str, batch_size: int = 100):
    os.makedirs(output_dir, exist_ok=True)

    w3 = init_web3(rpc_url)
    event_topic = get_event_topic()

    df = pd.read_csv(input_csv)
    tx_hashes = df["hash"].tolist()
    print(f"📦 Loaded {len(tx_hashes)} transactions from {input_csv}")
    print(f"🔍 Looking for UserOperationEvent with topic: {event_topic}")

    decoded_results = []

    for i in range(0, len(tx_hashes), batch_size):
        batch = tx_hashes[i:i + batch_size]
        tasks = [process_tx(tx, w3, event_topic) for tx in batch]
        batch_results = await asyncio.gather(*tasks)
        for res in batch_results:
            decoded_results.extend(res)

        print(f"✅ Processed batch {i // batch_size + 1} ({i + len(batch)}/{len(tx_hashes)})")

    results_df = pd.DataFrame(decoded_results)
    chunk_num = input_csv.split("_chunk_")[-1].split(".")[0]
    output_path = os.path.join(output_dir, f"user_operations_chunk_{chunk_num}.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved decoded UserOps to {output_path}")

# --- STEP 2: Append native_value ---
def append_native_value_to_userops(folder: str, rpc_url: str):
    print("\n--- Appending native_value to each user_operations file ---")
    w3 = init_web3(rpc_url)

    for file in sorted(os.listdir(folder)):
        if not file.startswith("user_operations_chunk_") or not file.endswith(".csv"):
            continue

        path = os.path.join(folder, file)
        df = pd.read_csv(path)

        if "native_value" not in df.columns:
            df["native_value"] = 0.0

        tx_cache = {}

        for idx in tqdm(df.index, desc=f"🧮 {file}"):
            try:
                tx_hash = df.at[idx, "tx_hash"]
                sender = df.at[idx, "sender"].lower()
                nonce = int(df.at[idx, "nonce"])

                if tx_hash not in tx_cache:
                    tx = w3.eth.get_transaction(tx_hash)
                    raw_input = tx["input"]
                    input_data = bytes.fromhex(raw_input[2:]) if isinstance(raw_input, str) else bytes(raw_input)
                    tx_cache[tx_hash] = input_data
                else:
                    input_data = tx_cache[tx_hash]

                payload = input_data[4:]
                userops, _ = decode([f"({','.join(userop_fields)})[]", "address"], payload)

                for userop in userops:
                    if userop[0].lower() == sender and int(userop[1]) == nonce:
                        call_data = userop[3]
                        df.at[idx, "native_value"] = decode_native_value_blind(call_data)
                        break

            except Exception as e:
                print(f"❌ Error row {idx} (tx {tx_hash[:10]}): {e}")

        df.to_csv(path, index=False)
        print(f"✅ Saved updated file: {path}")

# --- Main Pipeline ---
async def decode_all_userops_from_dir(input_dir: str, rpc_url: str, output_dir: str = "./userops", batch_size: int = 100):
    files = sorted(f for f in os.listdir(input_dir) if f.startswith("blocks_chunk_") and f.endswith(".csv"))
    if not files:
        print("❌ No block chunk files found.")
        return

    for f in files:
        full_path = os.path.join(input_dir, f)
        await decode_userops_from_csv(full_path, rpc_url, output_dir, batch_size)

    append_native_value_to_userops(output_dir, rpc_url)

