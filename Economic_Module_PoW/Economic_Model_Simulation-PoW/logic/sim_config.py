#EPOCHS = 5
#FEE_RATE = 0.001

# Network model constants
PRICE_IMPACT_FACTOR = 0.2
PRICE_ADJUSTMENT_SPEED = 0.1
STAKING_APY_THRESHOLD = 8
LP_APY_THRESHOLD = 2
STAKING_GROWTH_RATE = 0.1
STAKING_BASE_UNSTAKE_RATE = 0.02
STAKING_HIGH_UNSTAKE_RATE = 0.1
LP_APY_SMOOTHING_LAMBDA = 0.3
STAKING_MAX_FROM_FLOAT = 0.2
LP_GROWTH_RATE = 0.15
LP_BASE_UNSTAKE_RATE = 0.03
LP_HIGH_UNSTAKE_RATE = 0.12
LP_MAX_FROM_FLOAT = 0.15
MAX_MINERS = 20e3  # Based on hardware analysis
GAS_PER_USEROP = 110e3  # Average gas per user operation
BLOCKS_PER_DAY = 5040  # Number of blocks per day (assuming 4s block time)
MAX_TX_PER_DAY = int((30_000_000 // GAS_PER_USEROP) * BLOCKS_PER_DAY)  # ~1.37M tx/day
MAX_DAILY_VOLUME = 2e9  # 10B daily volume is between Solana and ETH
MAX_BUNDLERS = 10 # Is this a realistic expectation?
DAILY_TX_PER_BUNDLER = MAX_TX_PER_DAY // MAX_BUNDLERS  # Divide by expected mature number of bundlers
MAX_BLOCKS_MINED = 10
MAX_BUNDLES = 20
INITIAL_TOKEN_PRICE = 1.0


STGF_PARAMS = {
    "a": 0.010,
    "b": 30e9,
    "c": 0.00054166667,
    "d": 300e6
}

ELIGIBILITY_CRITERIA = {
    "developer": {"min_maw": 100},
    "miner": {"min_uptime": 0.95, "no_slashing": True},
    "liquidity": {"min_epochs_locked": 1, "max_slippage": 10},
    "user": {"min_tx_count": 1},
    "bundler": {"min_success_rate": 0.90, "min_uptime": 0.95}
}

ELIGIBILITY_FILTERS = {
    "user": lambda m: m.get("tx_count", 0) >= 1,
    "developer": lambda m: m.get("maw", 0) >= 100,
    "liquidity": lambda m: m.get("epochs_locked", 0) >= 1 and m.get("slippage", 0) <= 10,
    #"bundler": lambda m: m.get("success_rate", 0) >= 0.90 and m.get("uptime", 0) >= 0.95,
    "miner": lambda m: m.get("uptime", 0) >= 0.95 and not m.get("slashed", False),
}

#ROLE_DISTRIBUTION = {
#<<<<<<< HEAD
#   "user": 0.70,
#    "liquidity": 0.20,
#    "developer": 0.07,
#    "bundler": 0.01,
#    "miner": 0.02
#=======
#    "user": 500,
#    "liquidity": 108,
#    "developer": 35,
#    "miner": 300,
#    "bundler": 2,
#>>>>>>> 1d8f14f (runs and qa)
#}

REWARD_SPLITS = {
    "user": 0.45,
    "developer": 0.25,
    "miner": 0.25,
    "bundler": 0.05
}

#SCORE_WEIGHTS = {
#    "user": {
#        "tx_count": 0.20,
#        "volume": 0.20,
#        "gas": 0.10,
#        "cent_held": 0.20,
#        "dapps_used": 0.30
#    },
#    "developer": {
#        "txs": 0.20,
#        "maw": 0.35,
#        "volume": 0.35,
#        "cent_held": 0.10
#    },
#    "liquidity": {
#        "pool_volume": 0.20,
#        "tx_count": 0.15,
#        "epochs_locked": 0.15,
#        "slippage": 0.30,
#        "capital": 0.20
#    },
#    "bundler": {
#        "user_ops": 0.30,
#        "gas_used": 0.15,
#        "success_rate": 0.15,
#        "delay": 0.20,
#        "diversity": 0.20
#    },
#    "miner": {
#        "blocks": 0.25,
#        "uptime": 0.20,
#        "accuracy": 0.20,
#        "gas": 0.10,
#        "slashed": -0.10,
#        "hashrate": 0.15,
#        "cent_held": 0.20
#    }
#}

# Define the reward split among tiers (must sum to 1.0)
TIER_REWARD_SHARES = {
    "Tier0": 0.30,  
    "Tier1": 0.25,  
    "Tier2": 0.20,  
    "Tier3": 0.15,   
    "Tier4": 0.10 
}


TIER_THRESHOLDS = {
    "user": [0.03, 0.12, 0.25, 0.60],
    "developer": [0.20, 0.40, 0.80],
    "bundler": [0.20, 0.40, 0.80],
    "miner": [0.20, 0.40, 0.80]
}

CONFIG = {
    #"epochs": EPOCHS,
    #"fee_rate": FEE_RATE,
    "stgf_params": STGF_PARAMS,
    "eligibility_criteria": ELIGIBILITY_CRITERIA,
    "reward_splits": REWARD_SPLITS,
    #"score_weights": SCORE_WEIGHTS,
    "tier_thresholds": TIER_THRESHOLDS
}
