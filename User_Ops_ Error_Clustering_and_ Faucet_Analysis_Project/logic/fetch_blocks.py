# src/fetch_blocks.py

import nest_asyncio
import asyncio
from web3.middleware import ExtraDataToPOAMiddleware
from web3 import AsyncWeb3, AsyncHTTPProvider
import pandas as pd
from datetime import datetime
from typing import Optional

nest_asyncio.apply()


def init_web3(rpc_url: str) -> AsyncWeb3:
    """
    Initialize AsyncWeb3 connection with middleware.
    """
    w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w3


async def fetch_tx_details(w3: AsyncWeb3, tx: dict, block_timestamp: int, block_num: int) -> dict:
    """
    Fetch and parse transaction receipt, return enriched transaction details.
    """
    try:
        receipt = await w3.eth.get_transaction_receipt(tx['hash'])
        tx_fee = receipt['gasUsed'] * tx.get('gasPrice', 0) / 1e18
        tx_type = tx.get('type', 'unknown')

        return {
            "blockNumber": block_num,
            "hash": tx['hash'].hex(),
            "from": tx['from'],
            "to": tx['to'],
            "value": w3.from_wei(tx['value'], 'ether'),
            "gasUsed": receipt['gasUsed'],
            "gasLimit": tx['gas'],
            "gasPriceGwei": w3.from_wei(tx['gasPrice'], 'gwei') if tx.get('gasPrice') else None,
            "nonce": tx['nonce'],
            "status": receipt['status'],
            "transactionFee": tx_fee,
            "timestamp": pd.to_datetime(block_timestamp, unit='s'),
            "transaction_type": tx_type,
            "result": "success" if receipt['status'] == 1 else "error",
        }

    except Exception as e:
        print(f"⚠️ Error on tx {tx['hash'].hex()}: {e}")
        return {}


async def fetch_block(w3: AsyncWeb3, block_num: int) -> list:
    """
    Fetch full block with all transactions and enrich each transaction.
    """
    try:
        block = await w3.eth.get_block(block_num, full_transactions=True)
        tasks = [fetch_tx_details(w3, tx, block['timestamp'], block_num) for tx in block['transactions']]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]
    except Exception as e:
        print(f"⚠️ Error on block {block_num}: {e}")
        return []


async def process_chunk(w3: AsyncWeb3, start_block: int, end_block: int, chunk_number: int,
                        batch_size: int = 10, output_dir: Optional[str] = ".") -> None:
    """
    Fetch blocks in a range and save the results to a CSV.
    """
    all_rows = []

    for i in range(start_block, end_block + 1, batch_size):
        batch = list(range(i, min(i + batch_size, end_block + 1)))
        tasks = [fetch_block(w3, bn) for bn in batch]
        results = await asyncio.gather(*tasks)

        for block_rows in results:
            all_rows.extend(block_rows)

        print(f"✅ Processed blocks {batch[0]} to {batch[-1]} in chunk {chunk_number}")

    df = pd.DataFrame(all_rows)
    filename = f"{output_dir}/blocks_chunk_{chunk_number}.csv"
    df.to_csv(filename, index=False)
    print(f"💾 Saved chunk {chunk_number} to {filename}")


async def fetch_and_save_chunks(rpc_url: str,
                                start_block: Optional[int] = None,
                                end_block: Optional[int] = None,
                                chunk_size: int = 10000,
                                batch_size: int = 10,
                                output_dir: Optional[str] = ".",
                                initial_chunk_number: int = 1) -> None:
    """
    Main function to initialize web3 and process all chunks from start to end block.
    """
    w3 = init_web3(rpc_url)
    current_end = end_block or await w3.eth.block_number
    current_start = start_block or (current_end - 200_000)

    chunk_number = initial_chunk_number

    for chunk_start in range(current_start, current_end + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, current_end)
        print(f"\n🧱 Chunk {chunk_number}: Blocks {chunk_start:,} → {chunk_end:,}")
        await process_chunk(w3, chunk_start, chunk_end, chunk_number,
                            batch_size=batch_size, output_dir=output_dir)
        chunk_number += 1

    print("🎉 All chunks processed!")
