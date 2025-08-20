# allocation_config.py

from network_config import TokenAllocation

# ===============================
# Token Allocations for the Simulation (Adjusted to 36 months max vesting)
# ===============================

allocations = [
    # ✅ Fully unlocked at TGE
    TokenAllocation(
        category="public_sale",
        initial_allocation=20,  # 20% of total supply
        vesting_months=0,
        initial_amount=0,
        cliff_months=0
    ),
    TokenAllocation(
        category="airdrop",
        initial_allocation=9,  # 9% unlocked immediately at TGE
        vesting_months=18,
        initial_amount=0,
        cliff_months=0
    ),

    # 🔒 Locked, with cliffs and vesting (max 36 months)
    TokenAllocation(
        category="team",
        initial_allocation=15,  # 15% for team
        vesting_months=24,      # 24 months linear vesting
        initial_amount=0,
        cliff_months=12         # 12 months cliff
    ),
    TokenAllocation(
        category="investors",
        initial_allocation=11,  # 11% for investors
        vesting_months=18,      # 18 months linear vesting
        initial_amount=0,
        cliff_months=6          # 6 months cliff
    ),
    TokenAllocation(
        category="treasury",
        initial_allocation=15,  # 15% treasury
        vesting_months=36,      # 36 months linear vesting (max)
        initial_amount=0,
        cliff_months=0
    ),
    TokenAllocation(
        category="ecosystem",
        initial_allocation=15,  # 15% ecosystem incentives
        vesting_months=36,      # 36 months linear vesting (adjusted)
        initial_amount=0,
        cliff_months=0
    ),

    # Optional reserve/future use case
    TokenAllocation(
        category="reserves",
        initial_allocation=15,  # 15% reserve pool
        vesting_months=36,      # 36 months vesting (adjusted)
        initial_amount=0,
        cliff_months=0
    )
]
