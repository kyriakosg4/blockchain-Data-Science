from seeded_random import SeededRandom
from dataclasses import dataclass, field
from typing import Dict, List, DefaultDict, Optional, Tuple
from collections import defaultdict, Counter
from enum import Enum
from agents import Agent, StrategySet, Strategy, ActivityLevel
import score_model
import sim_config
from allocation_config import allocations
from calculator import NetworkCalculator
from network_config import NetworkConfig, TokenAllocation, RewardPool
from adoption_model import compute_micro_utility, compute_micro_adoption_growth, get_network_growth_rate
from sim_config import INITIAL_TOKEN_PRICE
import random






# retain the strategies
#class ActivityLevel(Enum):
#    HIGH = 0.8
#    MEDIUM = 0.5
#    LOW = 0.2
#    INACTIVE = 0.0

# dynamic score and reputation score is included 
class AgentState(Agent):
    def __init__(
        self,
        agent_id: str,
        initial_balance: float,
        choice_engine_seed: int,
        role: Optional[str],
        activity_level: ActivityLevel,
        strategy: Strategy,
    ):
        super().__init__(agent_id, initial_balance, choice_engine_seed, role, strategy)
        self.activity_level = activity_level
        self.strategy = strategy
        self.transaction_log = []
        self.rewards_earned_last_epoch = 0.0
        self.rewards_earned = 0.0
        self.fees_paid_last_epoch = 0.0
        self.activities_performed = defaultdict(float)
        self.fees_paid = 0.0
        self.value_generated = 0.0
        self.score = 0.0
        self.reputation_score = 1.0
        self.fee_breakdown = defaultdict(float)
        self.previous_score = 0.0
        self.normalized_score = 0.0
        self.total_liquidity_provided = 0.0
        self.reward_breakdown = defaultdict(lambda: defaultdict(float))
        self.bundled_transactions = 0  
        self.transactions_bundled_this_epoch = 0 



 

@dataclass
class RoleDistribution:
    """Track role-related metrics"""
    count: int = 0
    total_rewards: float = 0.0
    total_liquidity: float = 0.0
    total_activities: DefaultDict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )



# keeps historical data for fees
@dataclass
class FeeMetrics:
    transfer_tokens_fees: List[float] = field(default_factory=list)
    provide_liquidity_fees: List[float] = field(default_factory=list)

    def record_fees(self, activity_fee_totals: Dict[str, float]):
        """
        Stores the total actual fees collected per activity during the epoch.
        This data is used to compute activity weights and fee evolution.
        """
        self.transfer_tokens_fees.append(activity_fee_totals.get("transfer_tokens", 0.0))
        self.provide_liquidity_fees.append(activity_fee_totals.get("provide_liquidity", 0.0))



@dataclass
class SystemState:
    """For cadCAD tracking"""
    config: NetworkConfig
    total_supply: float
    circulating_supply: float
    agents: Dict[str, AgentState]
    role_distribution: Dict[str, RoleDistribution]
    reward_pools: RewardPool
    choice_engine: SeededRandom
    governance_multipliers: Dict[str, float] = field(default_factory=dict)
    current_epoch: int = 0
    epochs: int = 36  
    block_height: int = 0
    fee_metrics: FeeMetrics = field(default_factory=FeeMetrics)
    activity_weights: Dict[str, float] = field(default_factory=dict)
    past_activity_weights: List[Dict[str, float]] = field(default_factory=list)
    lambda_smoothing: float = 0.6
    beta_fixed: float = 0.5
    fixed_D_value: float = 0.6
    adoption_level: float = 0.05  # Starting value (5% of potential market)
    adoption_growth_rate: float = 0.0  # For optional tracking/logging
    utility: float = 0.0
    adoption_history: List[float] = field(default_factory=list)
    agent_entry_history: List[int] = field(default_factory=list)
    adoption_delta_history: List[float] = field(default_factory=list)
    released_tokens: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    utility_history: list = field(default_factory=list)
    adoption_history: list = field(default_factory=list)
    adoption_delta_history: list = field(default_factory=list)
    macro_utility_history: List[float] = field(default_factory=list)
    micro_utility_history: List[float] = field(default_factory=list)
    macro_growth_history: List[float] = field(default_factory=list)
    micro_growth_history: List[float] = field(default_factory=list)
    metrics: Dict[str, Dict[int, float]] = field(default_factory=lambda: defaultdict(dict)) 



    
    @classmethod
    def initialize(cls, config: NetworkConfig) -> 'SystemState':
        """Create initial system state from configuration"""
    
        # Valid governance activities only
        governance_activities = ["transfer_tokens", "provide_liquidity"]
    
        obj = cls(
            config=config,
            current_epoch=0,
            block_height=0,
            total_supply=config.initial_supply,
            circulating_supply=config.initial_supply,
            agents={},
            role_distribution={role: RoleDistribution() for role in config.roles},
            reward_pools=RewardPool(),
            choice_engine=SeededRandom(config.seed),
            governance_multipliers={
                activity: SeededRandom(config.seed).uniform(0.5, 1.5)
                for activity in governance_activities
            },
            activity_weights=config.activity_weights.copy(),
            utility_history=[],
            adoption_history=[],
            adoption_delta_history=[],
            agent_entry_history=[],
            macro_utility_history=[],
            micro_utility_history=[],
            macro_growth_history=[],
            micro_growth_history=[]
        )
    
        # Manually initialize required attribute to avoid AttributeError later
        obj.previous_T_observed = {}
        obj.epochs = config.epochs
        obj.epoch_released_tokens = 0.0
        obj.tokens_distributed_this_epoch = 0.0
        obj.simulation_complete = False
        obj.metrics = defaultdict(dict)

    
        return obj


    
    
    def validate_state(self) -> None:
        """Validate system state invariants"""
    
        # Supply checks
        if self.total_supply < 0 or self.circulating_supply < 0:
            raise ValueError("Supply cannot be negative")
        if self.circulating_supply > self.total_supply:
            raise ValueError("Circulating supply cannot exceed total supply")
    
        # Check if total assigned balances match circulating supply
        total_agent_balance = sum(agent.balance for agent in self.agents.values())

        total_pool_balance = self.reward_pools.total()

        liquidity_locked = sum(agent.total_liquidity_provided for agent in self.agents.values())
        used_tokens = total_agent_balance + liquidity_locked + self.reward_pools.total()
        
        if abs(used_tokens - self.circulating_supply) > 1e-3:
            raise ValueError(
    f"Used tokens ({used_tokens}) do not match circulating supply ({self.circulating_supply})"
)


        # Role distribution check
        agent_role_counts = defaultdict(int)
        for agent in self.agents.values():
            if agent.role:
                agent_role_counts[agent.role] += 1
    
        for role, dist in self.role_distribution.items():
            if dist.count != agent_role_counts[role]:
                raise ValueError(f"Role distribution mismatch for {role}")
        
        print(f"Validation Details:")
        print(f"Block Height: {self.block_height}")
        print(f"Total Supply: {self.total_supply}")
        print(f"Circulating Supply: {self.circulating_supply}")
        print(f"Total Agent Balance: {total_agent_balance}")
        print(f"Total Pool Balance: {total_pool_balance}")
        print(f"Sum: {total_agent_balance + total_pool_balance}")
        print(f"Difference: {abs(total_agent_balance + total_pool_balance - self.circulating_supply)}")
        
        """if abs(total_agent_balance + total_pool_balance - self.circulating_supply) > 1e-3:
            raise ValueError(
                f"Sum of balances ({total_agent_balance + total_pool_balance}) "
                f"does not match circulating supply ({self.circulating_supply})"
            )"""
        
        # Role distribution check
        agent_role_counts = defaultdict(int)
        for agent in self.agents.values():
            if agent.role:
                agent_role_counts[agent.role] += 1
        
        for role, dist in self.role_distribution.items():
            if dist.count != agent_role_counts[role]:
                raise ValueError(f"Role distribution mismatch for {role}")

    def update_metrics(self) -> Dict[str, float]:
        """Calculate current system metrics for analysis"""
        metrics = {}
        
        # Participation metrics
        total_agents = len(self.agents)
        metrics['total_agents'] = total_agents
    
        for role, dist in self.role_distribution.items():
            metrics[f'{role}_ratio'] = dist.count / total_agents if total_agents > 0 else 0
            # Removed stake_ratio since staking is deprecated
    
        # Activity metrics
        for role, dist in self.role_distribution.items():
            for activity, count in dist.total_activities.items():
                metrics[f'{role}_{activity}_rate'] = count / dist.count if dist.count > 0 else 0
    
        # Reward metrics
        metrics['total_rewards'] = sum(agent.rewards_earned for agent in self.agents.values())
    
        return metrics

    
    def get_value_generated(self, role: str, strategy_name: str) -> float:
        """Get total value generated by agents using this strategy"""
        value = 0.0
        for agent in self.agents.values():
            if agent.role == role:
                # For now, approximate value as balance growth
                value += agent.balance - agent.fees_paid + agent.rewards_earned
        return value

    def get_fees_paid(self, role: str, strategy_name: str) -> float:
        """Get total fees paid by agents using this strategy"""
        fees = 0.0
        for agent in self.agents.values():
            if agent.role == role:
                fees += agent.fees_paid
        return fees

    def get_rewards_earned(self, role: str, strategy_name: str) -> float:
        """Get total rewards earned by agents using this strategy"""
        rewards = 0.0
        for agent in self.agents.values():
            if agent.role == role:
                rewards += agent.rewards_earned
        return rewards


    def get_activities_performed(self, role: str, strategy_name: str) -> Dict[str, int]:
        """Get activity counts for agents using this strategy"""
        activities = defaultdict(int)
        for agent in self.agents.values():
            if agent.role == role:
                for activity, count in agent.activities_performed.items():
                    activities[activity] += count
        return dict(activities)


    # # D equally values for all activities 
# def update_governance_multipliers(self):
#     """
#     Dynamically updates governance multipliers for each activity using:
#     αₐ^(t+1) = αₐ^(t) × (1 + β × (Dₐ - Tₐ))

#     Where:
#     - Dₐ = desired share (random, normalized to sum to 1)
#     - Tₐ = actual observed share of activity (normalized to sum to 1)
#     - β = sensitivity = abs(Tₐ - Tₐ_last)
#     """
#     activity_names = list(self.governance_multipliers.keys())

#     # Step 1: Calculate total performed amount for normalization
#     total_activity_amount = 0.0
#     activity_totals = {}

#     for activity in activity_names:
#         amount = sum(agent.activities_performed.get(activity, 0.0) for agent in self.agents.values())
#         activity_totals[activity] = amount
#         total_activity_amount += amount

#     # Avoid division by zero
#     if total_activity_amount == 0:
#         print("⚠️ No activity performed this epoch — governance multipliers unchanged.")
#         return {
#             "D": {},
#             "T": {},
#             "beta": {},
#             "alpha": self.governance_multipliers.copy()
#         }

#     # Step 2: Normalize Tₐ (observed shares)
#     T_observed = {a: activity_totals[a] / total_activity_amount for a in activity_names}

#     # Step 3: Use fixed D value (e.g., 0.6) and normalize it across activities
#     fixed_value = self.fixed_D_value
#     desired_raw = {a: fixed_value for a in activity_names}
#     total_desired = sum(desired_raw.values())
#     D_desired = {a: desired_raw[a] / total_desired for a in activity_names}

#     # Step 4: Capture previous T before overwriting it
#     previous_T = self.previous_T_observed.copy() if hasattr(self, "previous_T_observed") else {}

#     # Step 5: Update governance multipliers
#     beta_dict = {}
#     for a in activity_names:
#         alpha_t = self.governance_multipliers[a]
#         T_a = T_observed[a]
#         D_a = D_desired[a]

#         # β = abs(Tₐ - Tₐ_last)
#         prev_T_a = previous_T.get(a, 0.0)
#         beta = self.beta_fixed
#         beta_dict[a] = beta

#         # Update α
#         new_alpha = alpha_t * (1 + beta * (D_a - T_a))
#         self.governance_multipliers[a] = max(0.5, min(1.5, new_alpha))  # Clamp to [0.5, 1.5]

#     # Step 6: Store T for use in the next epoch
#     self.previous_T_observed = T_observed

#     print(f"✅ Governance multipliers updated: {self.governance_multipliers}")

#     return {
#         "D": D_desired,
#         "T": T_observed,
#         "beta": beta_dict,
#         "alpha": self.governance_multipliers.copy()
#     }


        
    def update_governance_multipliers(self):
        """
        Dynamically updates governance multipliers for valid activities only:
        transfer_tokens, provide_liquidity.
        """
        valid_governance_activities = {"transfer_tokens", "provide_liquidity"}
        activity_names = [a for a in self.governance_multipliers.keys() if a in valid_governance_activities]
    
        for agent in self.agents.values():
            for key in list(agent.activities_performed.keys()):
                if key not in activity_names:
                    del agent.activities_performed[key]
            if hasattr(agent, "activity_fee_history"):
                for epoch in agent.activity_fee_history:
                    agent.activity_fee_history[epoch] = {
                        k: v for k, v in agent.activity_fee_history[epoch].items() if k in activity_names
                    }
    
        total_activity_amount = 0.0
        activity_totals = {}
        for activity in activity_names:
            amount = sum(agent.activities_performed.get(activity, 0.0) for agent in self.agents.values())
            activity_totals[activity] = amount
            total_activity_amount += amount
    
        if total_activity_amount == 0:
            print("⚠️ No activity performed this epoch — governance multipliers unchanged.")
            return {
                "D": {},
                "T": {},
                "beta": {},
                "alpha": self.governance_multipliers.copy()
            }
    
        current_T = {a: activity_totals[a] / total_activity_amount for a in activity_names}
    
        if hasattr(self.config, "custom_D_values"):
            desired_raw = {
                k: v for k, v in self.config.custom_D_values.items() if k in activity_names
            }
        else:
            fixed_value = self.fixed_D_value
            desired_raw = {a: fixed_value for a in activity_names}
    
        total_desired = sum(desired_raw.values())
        D_desired = {a: desired_raw[a] / total_desired for a in activity_names}
    
        if self.current_epoch == 0:
            self.previous_T_observed = current_T
            print("⏸️ Epoch 0 — Skipping governance update (no previous T).")
            return {
                "D": D_desired,
                "T": {},
                "beta": {},
                "alpha": self.governance_multipliers.copy()
            }
    
        T_observed = self.previous_T_observed.copy()
        beta_dict = {}
    
        for a in activity_names:
            alpha_t = self.governance_multipliers[a]
            T_a = T_observed.get(a, 0.0)
            D_a = D_desired[a]
            beta = self.beta_fixed
            beta_dict[a] = beta
    
            print(f"[DEBUG] Epoch {self.current_epoch} | activity: {a} | α_prev: {alpha_t:.6f} | Tₐ(prev): {T_a:.6f} | Dₐ: {D_a:.6f} | β: {beta}")
            new_alpha = alpha_t * (1 + beta * (D_a - T_a))
            self.governance_multipliers[a] = max(0.5, min(5, new_alpha))
    
        self.previous_T_observed = current_T
        print(f"✅ Governance multipliers updated: {self.governance_multipliers}")
    
        return {
            "D": D_desired,
            "T": T_observed,
            "beta": beta_dict,
            "alpha": self.governance_multipliers.copy()
        }



    def get_previous_epoch_fees(self) -> Dict[str, float]:
        """
        Retrieves the total fees collected in the last epoch for each activity.
        This is used in `NetworkConfig.get_activity_fee()` to calculate activity weights.
        """
        return {
            "transfer_tokens": sum(self.fee_metrics.transfer_tokens_fees[-1:]),
            "provide_liquidity": sum(self.fee_metrics.provide_liquidity_fees[-1:]),
        }


    def update_reputation_discounts(self):
        """Updates reputation-based discounts R_u^(t) for each agent.
           Ensures a max discount of 20% and prevents division by zero.
        """
        for agent in self.agents.values():
            previous_rewards = agent.rewards_earned_last_epoch
            previous_fees = agent.fees_paid_last_epoch
    
            if previous_fees > 0:
                new_discount = 1 - (previous_rewards / previous_fees)
                agent.reputation_score = max(0.8, min(1.0, 1 - new_discount))  # Cap max discount at 20%
            else:
                agent.reputation_score = 1.0  # ✅ No discount if no fees paid
    
            # Store values for next epoch
            agent.rewards_earned_last_epoch = agent.rewards_earned
            agent.fees_paid_last_epoch = agent.fees_paid
    
        print(f"Reputation discounts updated.")



    def release_tokens_from_vesting(self, epoch: int, total_supply: float):
        """
        Update released_tokens dict by releasing monthly vesting per allocation.
        Only applies after cliff has passed.
        Tracks total newly released tokens this epoch.
        Stores per-epoch release details to `self.released_per_epoch`.
    
        NOTE: This does NOT modify `circulating_supply` directly to avoid double-counting.
        That should be handled separately in update_epoch().
        """
        from allocation_config import allocations  # contains list of TokenAllocation
    
        total_newly_released = 0.0
        epoch_releases = {}
    
        for alloc in allocations:
            if epoch < alloc.cliff_months:
                continue  # Cliff not passed — nothing released
    
            months_since_cliff = epoch - alloc.cliff_months
            months_vested = min(months_since_cliff, alloc.vesting_months)
    
            allocation_base = (alloc.initial_allocation / 100) * total_supply

            if alloc.vesting_months == 0:
                print(f"⚠️ Skipping vesting for {alloc.category}: vesting_months is 0")
                continue  # Skip this allocation to avoid crashing
            
            monthly_release_amount = allocation_base / alloc.vesting_months
            unlocked_tokens = months_vested * monthly_release_amount

    
            # Compute newly released tokens this epoch
            previous_unlocked = self.released_tokens.get(alloc.category, 0.0)
            newly_released = unlocked_tokens - previous_unlocked
    
            if newly_released > 0:
                total_newly_released += newly_released
                epoch_releases[alloc.category] = newly_released
    
            # Update record of total released for this category
            self.released_tokens[alloc.category] = unlocked_tokens
    
            # Debug output
            print(f"🔓 Epoch {epoch} | {alloc.category} unlocked {newly_released:.2f} tokens")
    
        # Track how much was released this epoch for onboarding
        self.epoch_released_tokens = total_newly_released
        self.tokens_distributed_this_epoch = 0.0  # Reset consumption tracker
    
        # Save per-category breakdown for debugging/reporting
        if not hasattr(self, "released_per_epoch"):
            self.released_per_epoch = {}
        self.released_per_epoch[epoch] = epoch_releases
    
        print(f"📦 Epoch {epoch} | Total newly released tokens available for onboarding: {total_newly_released:.2f}")







    def spawn_new_agents(self, count: int):
        """
        Spawns new agents based on adoption, using tokens from the current epoch's released pool.
        Note: Does NOT update circulating_supply directly to avoid double-counting.
        """
        # Allow onboarding agents using ALL past released tokens (not just this epoch)
        onboarding_categories = {
            "airdrop", "team", "investors", "treasury", "ecosystem", "reserves"
        }
        total_released_tokens = sum(
            self.released_tokens.get(cat, 0.0) for cat in onboarding_categories
        )
        
        available_pool = total_released_tokens - self.tokens_distributed_this_epoch

    
        for i in range(count):
            if available_pool <= 0:
                print(f"⚠️ No tokens available to onboard agent {i}, skipping.")
                break
    
            # Select role using weights
            role_choices = list(self.config.roles.keys())
            role_weights = [self.config.roles[r].allocation_percent for r in role_choices]
            role = self.choice_engine.choice(role_choices, p=role_weights)

    
            agent_id = f"adopted_agent_{self.current_epoch}_{i}"
            activity_level = self.choice_engine.choice(list(ActivityLevel))
    
            # Choose strategy from correct role set
            strategy_pool = {
                "user": StrategySet().user_strategies,
                "developer": StrategySet().developer_strategies,
                "bundler": StrategySet().bundler_strategies,
                "miner": StrategySet().miner_strategies
            }.get(role, [])
    
            strategy = self.choice_engine.choice(strategy_pool)
    
            # Users & developers get balances
            balance = 0.0
            if role in ["user", "developer"]:
                existing_balances = [agent.balance for agent in self.agents.values()]
                max_existing_balance = max(existing_balances) if existing_balances else 200
                min_balance = self.config.roles[role].min_balance
                balance = self.choice_engine.uniform(min_balance, max_existing_balance)
                balance = min(balance, available_pool)
                available_pool -= balance
    
            agent = AgentState(
                agent_id=agent_id,
                initial_balance=balance,
                choice_engine_seed=self.config.seed + len(self.agents) + i,
                role=role,
                activity_level=activity_level,
                strategy=strategy
            )

            agent.created_in_epoch = self.current_epoch
    
            agent.set_agent_state(self)
            self.agents[agent_id] = agent
            self.role_distribution[role].count += 1
    
            print(f"🆕 Spawned {role} agent {agent_id} with balance: {balance:.2f}")





    def update_epoch(self):
        print(f"\n📢 update_epoch called at epoch {self.current_epoch}")
    
        if self.current_epoch >= self.config.epochs:
            print(f"🛑 Reached epoch limit ({self.config.epochs}) — stopping.")
            return
    
        self.current_epoch += 1
        self.release_tokens_from_vesting(epoch=self.current_epoch, total_supply=self.total_supply)
    
        global_mempool = []
        for agent in self.agents.values():
            agent.start_of_epoch_balance = agent.balance
            agent.set_agent_state(self)
            agent.update_activity_level(self.config)
    
            print(f"[{agent.id}] Updated activity levels: {agent.updated_activity_levels}")
    
            tx_level = agent.updated_activity_levels.get("transfer_tokens", 0.0)
            liq_level = agent.updated_activity_levels.get("provide_liquidity", 0.0)
            agent.max_possible_tx = int(tx_level * 20)
            agent.max_possible_liq = int(liq_level * 12)
            agent.max_possible_activities = agent.max_possible_tx + agent.max_possible_liq
    
            agent.epoch_transactions = agent.generate_transactions(
                self.current_epoch, list(self.agents.keys()), self.config
            )
            agent.transaction_log.append(agent.epoch_transactions)
            if agent.role == "user":
                global_mempool.extend(agent.epoch_transactions)
    
        for agent in self.agents.values():
            if agent.role != "user":
                continue
            for tx in agent.epoch_transactions:
                if tx.get("type") == "transfer_tokens":
                    amount = tx.get("amount", 0.0)
                    agent.activities_performed["transfer_tokens"] += amount
    
        bundlers = [a for a in self.agents.values() if a.role == "bundler"]
        total_bundler_activity = sum(b.updated_activity_levels.get("bundle_transactions", 0.0) for b in bundlers)
        for b in bundlers:
            activity_level = b.updated_activity_levels.get("bundle_transactions", b.activity_level)
            share = activity_level / total_bundler_activity if total_bundler_activity > 0 else 1 / len(bundlers)
            tx_count = min(int(share * len(global_mempool)), len(global_mempool))
            bundled = [global_mempool.pop() for _ in range(tx_count)]
            b.transactions_bundled_this_epoch = len(bundled)
            b.bundled_transactions += len(bundled)
    
        for agent in self.agents.values():
            agent.execute_activity("provide_liquidity", self.config.roles[agent.role], self.config)
    
        for agent in self.agents.values():
            if agent.role == "user":
                agent.execute_activity("transfer_tokens", self.config.roles[agent.role], self.config)
    
        print(f"\n📦 Epoch {self.current_epoch} | Agent Activity Summary:")
        for agent in self.agents.values():
            acts = agent.activity_count_history.get(self.current_epoch, {})
            max_possible = getattr(agent, "max_possible_activities", 0)
            total_acts = sum(acts.values())
            print(f"  - {agent.id[:12]} | Role: {agent.role:9s} | Activity: {total_acts:2d}/{max_possible:2d} | Balance: {agent.balance:.2f}")
    
        total_activity = sum(
            sum(agent.activity_count_history.get(self.current_epoch, {}).values())
            for agent in self.agents.values()
        )
        total_max = sum(
            getattr(agent, "max_possible_activities", 0)
            for agent in self.agents.values()
        )
        utility = total_activity / total_max if total_max > 0 else 0.0
        self.utility = utility
    
        print(f"\n📊 Epoch {self.current_epoch} | Utility Numerator: {total_activity}")
        print(f"📊 Epoch {self.current_epoch} | Utility Denominator: {total_max}")
        print(f"📊 Epoch {self.current_epoch} | Computed Utility: {utility:.6f}")
    
        self.utility_history.append(utility)
    
        prev_adoption = self.adoption_level
        alpha = 0.3
        growth = alpha * utility * (1 - self.adoption_level)
        self.adoption_growth_rate = growth
        self.adoption_level = min(self.adoption_level * (1 + growth), 1.0)
    
        delta_adoption = self.adoption_level - prev_adoption
        print(f"\n🌱 Epoch {self.current_epoch} | Previous Adoption: {prev_adoption:.6f} → New: {self.adoption_level:.6f} | Δ: {delta_adoption:.6f}")
    
        new_agents = int(self.config.total_potential_agents * delta_adoption)
        print(f"🧪 Agents to Spawn (Δadoption × max_agents): {new_agents}")
        self.spawn_new_agents(new_agents)
    
        self.adoption_history.append(self.adoption_level)
        self.adoption_delta_history.append(delta_adoption)
        self.agent_entry_history.append(new_agents)
    
        if not hasattr(self, "epoch_summary"):
            self.epoch_summary = {}
        self.epoch_summary[self.current_epoch] = {
            "utility": utility,
            "adoption": self.adoption_level,
            "delta_adoption": delta_adoption,
            "agents_spawned": new_agents
        }

        # ✅ Track agent count by role per epoch
        if not hasattr(self, "agent_count_history"):
            self.agent_count_history = {}
        
        role_counts = Counter(a.role for a in self.agents.values())
        self.agent_count_history[self.current_epoch] = {
            "total_agents": len(self.agents),
            **{role: role_counts.get(role, 0) for role in ["user", "developer", "bundler", "miner"]}
        }

 
        if self.current_epoch == 1:
            activity_weights = self.config.activity_weights.copy()
        else:
            prev_epoch = self.current_epoch - 1
            activity_fee_totals = defaultdict(float)
            for agent in self.agents.values():
                for activity, fee in agent.activity_fee_history.get(prev_epoch, {}).items():
                    activity_fee_totals[activity] += fee
    
            fixed_weights = {
                "bundle_transactions": self.config.activity_weights["bundle_transactions"],
                "mine_block": self.config.activity_weights["mine_block"]
            }
    
            dynamic_total = (
                activity_fee_totals.get("transfer_tokens", 0.0) +
                activity_fee_totals.get("provide_liquidity", 0.0)
            )
            dynamic_weights = {}
            if dynamic_total > 0:
                dynamic_weights["transfer_tokens"] = activity_fee_totals.get("transfer_tokens", 0.0) / dynamic_total
                dynamic_weights["provide_liquidity"] = activity_fee_totals.get("provide_liquidity", 0.0) / dynamic_total
            else:
                dynamic_weights["transfer_tokens"] = self.config.activity_weights["transfer_tokens"]
                dynamic_weights["provide_liquidity"] = self.config.activity_weights["provide_liquidity"]
    
            activity_weights = {**dynamic_weights, **fixed_weights}
    
        self.activity_weights = activity_weights.copy()
        self.past_activity_weights.append(activity_weights.copy())
    
        for agent in self.agents.values():
            epoch = self.current_epoch
            agent.activity_fee_history[epoch] = dict(agent.fee_breakdown)
            agent.activity_amount_history[epoch] = dict(agent.activities_performed)
    
        prev_epoch = self.current_epoch - 1
        prev_token_price = self.macro_metrics['token_price'].get(prev_epoch, INITIAL_TOKEN_PRICE)
        self.token_price = prev_token_price
    
        if hasattr(self, "macro_metrics"):
            self.market_cap = self.macro_metrics["market_cap"].get(prev_epoch, 0.0)
            self.total_velocity = self.macro_metrics["total_velocity"].get(prev_epoch, 0.0)
            self.lp_amount = self.macro_metrics["lp_amount"].get(prev_epoch, 0.0)
            self.apy_lp = self.macro_metrics["apy_lp"].get(prev_epoch, 0.0)
            self.security_ratio = self.macro_metrics["security_ratio"].get(prev_epoch, 0.0)
            self.fee_sustainability = self.macro_metrics["fee_sustainability"].get(prev_epoch, 0.0)
            self.monthly_fees = self.macro_metrics["monthly_fees"].get(prev_epoch, 0.0)
            self.monthly_rewards = self.macro_metrics["monthly_rewards"].get(prev_epoch, 0.0)
    
        reward_pool = RewardPool()
        rewards_info = reward_pool.calculate_monthly_rewards(
            self.agents, self.config, token_price=self.token_price
        )
        total_reward_tokens = rewards_info['total_rewards_cent']
    
        fixed_role_weights = {
            "bundle_transactions": self.config.activity_weights["bundle_transactions"],
            "mine_block": self.config.activity_weights["mine_block"]
        }
    
        score_model.compute_all_agent_scores(
            agents=self.agents,
            activity_weights=self.activity_weights,
            role_fixed_weights=fixed_role_weights,
            smoothing_lambda=self.lambda_smoothing
        )
    
        score_model.distribute_rewards(
            agents=self.agents,
            total_reward_pool=total_reward_tokens,
            role_reward_shares=sim_config.REWARD_SPLITS,
            epoch=self.current_epoch  # ✅ Add this
        )

        for role in sim_config.REWARD_SPLITS:
            total = sum(agent.rewards_earned for agent in self.agents.values() if agent.role == role)
            count = sum(1 for agent in self.agents.values() if agent.role == role and agent.score > 0)
            print(f"{role.title():<10} | Agents receiving rewards: {count} | Total rewards: {total:.2f}")

        # ✅ Add: Role reward summary tracking
        if not hasattr(self, "role_rewards_summary"):
            self.role_rewards_summary = {}
        
        epoch = self.current_epoch
        role_rewards = defaultdict(float)
        for agent in self.agents.values():
            reward = agent.rewards_earned_history.get(epoch, 0.0)
            role_rewards[agent.role] += reward
        
        self.role_rewards_summary[epoch] = role_rewards

        self.reward_pools = RewardPool()
    
        if hasattr(self, "calculator"):
            epoch = self.current_epoch
            real_tx_volume = sum(agent.activities_performed.get("transfer_tokens", 0.0) for agent in self.agents.values())
            real_lp_amount = sum(getattr(agent, "total_liquidity_provided", 0.0) for agent in self.agents.values())
            num_miners = sum(1 for agent in self.agents.values() if agent.role == "miner")
            num_bundlers = sum(1 for agent in self.agents.values() if agent.role == "bundler")
            real_tx_volume_usd = real_tx_volume * self.token_price
    
            external_state = {
                "monthly_volume_tokens": real_tx_volume,
                "monthly_volume_usd": real_tx_volume_usd,
                "lp_amount": real_lp_amount,
                "total_fees": sum(agent.fees_paid for agent in self.agents.values()),
                "total_rewards": sum(agent.rewards_earned for agent in self.agents.values()),
                "num_miners": num_miners,
                "num_bundlers": num_bundlers,
                "agent_objects": list(self.agents.values())
            }
    
            self.calculator.update_metrics(epoch=epoch, external_state=external_state)
            self.current_lp_apy = self.calculator.metrics['apy_lp'].get(epoch, 0.0)
            self.monthly_tx_volume_tokens = real_tx_volume
    
        self.update_reputation_discounts()
        gov_debug = self.update_governance_multipliers()
        self.governance_debug_log = getattr(self, "governance_debug_log", [])
        
        # ✅ Only store the updated alpha multipliers
        multiplier_snapshot = {"epoch": self.current_epoch}
        multiplier_snapshot.update(gov_debug.get("alpha", {}))
        self.governance_debug_log.append(multiplier_snapshot)
  
        if hasattr(self, "macro_metrics"):
            real_tx_volume = sum(agent.activities_performed.get("transfer_tokens", 0.0) for agent in self.agents.values())
            self.macro_metrics['transaction_volume_usd'][self.current_epoch] = real_tx_volume * self.token_price
            self.macro_metrics['monthly_tx_volume_tokens'][self.current_epoch] = real_tx_volume
    
        for agent in self.agents.values():
            agent.rewards_earned = 0.0
            agent.fees_paid = 0.0
            agent.reset_activity_tracking()
            agent.fee_breakdown = defaultdict(float)
    
        self.circulating_supply = sum(self.released_tokens.values())

        maturity = self.circulating_supply / self.total_supply if self.total_supply > 0 else 0.0
        adjusted_lp_threshold = sim_config.LP_APY_THRESHOLD * (1.5 - maturity)
        self.metrics["adjusted_lp_threshold"][self.current_epoch] = adjusted_lp_threshold

        self.effective_supply = self.circulating_supply - self.lp_amount
    
        self.metrics["circulating_supply"][self.current_epoch] = self.circulating_supply
        self.metrics["effective_supply"][self.current_epoch] = self.effective_supply
    
        if not hasattr(self, "supply_metrics"):
            self.supply_metrics = {}
        self.supply_metrics[self.current_epoch] = {
            "circulating_supply": self.circulating_supply,
            "effective_supply": self.effective_supply
        }
        self.metrics["supply_metrics"] = dict(self.supply_metrics)
    
        self.metrics["circulating_supply"] = dict(self.metrics["circulating_supply"])
        self.metrics["effective_supply"] = dict(self.metrics["effective_supply"])
    
        print(f"🧾 Epoch {self.current_epoch} | Utility History: {self.utility_history}")
        print(f"🧾 Epoch {self.current_epoch} | Adoption History: {self.adoption_history}")
        print(f"✅ Epoch {self.current_epoch} updates applied.")










def initialize_system_state(config: NetworkConfig, allocations: list[TokenAllocation]) -> Tuple[SystemState, Dict[str, AgentState]]:
    """Create and initialize system state based on TGE release and real vesting schedule."""

    # 1️⃣ Initialize SystemState object
    initial_state = SystemState.initialize(config)
    
    # 2️⃣ Initialize the calculator using config
    calculator = NetworkCalculator(config, allocations)
    
    # 3️⃣ Calculate TGE circulating supply from vesting logic
    tge_circulating_supply = calculator._calculate_initial_circulating()
    initial_state.circulating_supply = tge_circulating_supply

    # 4️⃣ Connect calculator to the simulation state
    initial_state.calculator = calculator
    initial_state.macro_metrics = calculator.metrics  # used by update_epoch()
    initial_state.metrics = calculator.metrics  # ✅ Needed by agents for metrics access


    print(f"✅ TGE Circulating Supply computed: {tge_circulating_supply}")
    calculator = NetworkCalculator(config, allocations)


    # 5️⃣ Determine agent count per role
    role_targets = {
        role: int(config.num_agents * config.roles[role].allocation_percent)
        for role in config.roles
    }

    agents = {}

    # 6️⃣ Initialize users and developers with balances
    tge_role_allocations = {
        "user": 0.15,        # 15% of total supply goes to users at TGE
        "developer": 0.05    # 5% of total supply goes to developers at TGE
    }
    
    for role in ["user", "developer"]:
        num_agents = role_targets[role]
        min_balance = config.roles[role].min_balance
    
        total_tokens = config.initial_supply * tge_role_allocations[role]
    
        raw_balances = [
            initial_state.choice_engine.uniform(min_balance, min_balance * 2)
            for _ in range(num_agents)
        ]
        scaling_factor = total_tokens / sum(raw_balances)
        final_balances = [b * scaling_factor for b in raw_balances]

        for i in range(num_agents):
            agent_id = f"{role}_agent_{i}"
            activity_level = initial_state.choice_engine.choice(list(ActivityLevel))
            strategy_pool = getattr(StrategySet(), f"{role}_strategies")
            strategy = initial_state.choice_engine.choice(strategy_pool)

            agents[agent_id] = AgentState(
                agent_id=agent_id,
                initial_balance=final_balances[i],
                choice_engine_seed=config.seed + i,
                role=role,
                activity_level=activity_level,
                strategy=strategy
            )

            initial_state.agents[agent_id] = agents[agent_id]
            initial_state.role_distribution[role].count += 1
            initial_state.role_distribution[role].total_liquidity += final_balances[i]

    # 7️⃣ Initialize bundlers and miners WITHOUT balances
    for role in ["bundler", "miner"]:
        num_agents = role_targets[role]
        for i in range(num_agents):
            agent_id = f"{role}_agent_{i}"
            activity_level = initial_state.choice_engine.choice(list(ActivityLevel))
            strategy_pool = getattr(StrategySet(), f"{role}_strategies")
            strategy = initial_state.choice_engine.choice(strategy_pool)

            agents[agent_id] = AgentState(
                agent_id=agent_id,
                initial_balance=0.0,
                choice_engine_seed=config.seed + 10000 + i,
                role=role,
                activity_level=activity_level,
                strategy=strategy
            )

            initial_state.agents[agent_id] = agents[agent_id]
            initial_state.role_distribution[role].count += 1

    # 8️⃣ Initialize reward pool and governance multipliers
    initial_state.reward_pools = RewardPool()

    for activity_name, activity_cfg in config.activities.items():
        initial_state.governance_multipliers[activity_name] = activity_cfg.governance_multiplier

    # ✅ Manually track the TGE tokens as already released
    initial_state.released_tokens = {
        role: config.initial_supply * share
        for role, share in tge_role_allocations.items()
    }


    # 9️⃣ Logs
    print(f"System State Initialized.")
    print(f"Total Supply (fixed): {initial_state.total_supply}")
    print(f"TGE Circulating Supply: {initial_state.circulating_supply}")
    for role, count in role_targets.items():
        print(f"{role}: {count} agents ({count / config.num_agents:.1%})")

    initial_state.validate_state()
    return initial_state, agents





