from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from seeded_random import SeededRandom
from network_config import ActivityConfig, RoleConfig, NetworkConfig
from collections import defaultdict
from sim_config import LP_APY_THRESHOLD


class ActivityLevel(Enum):
    HIGH = (0.6, 0.9)   # Random between 70-90%
    MEDIUM = (0.3, 0.7) # Random between 40-60%
    LOW = (0.1, 0.4)    # Random between 10-30%
    INACTIVE = (0.0, 0.0) # Always inactive

    def get_range(self) -> Tuple[float, float]:
        """Returns the min-max range for the activity level."""
        return self.value
    
class Agent:
    def __init__(self, agent_id: str, initial_balance: float, choice_engine_seed: int, role: Optional[str] = None, strategy: Optional['Strategy'] = None):
        self.id = agent_id
        self.balance = initial_balance
        self.role = role
        self.rewards_earned = 0.0  #should be in state or here? 
        self.fees_paid = 0.0 #should be in state or here? 
        self.choice_engine = SeededRandom(choice_engine_seed)
        self.reputation_score = 1.0
        self._system_state = None
        self.activities_performed = defaultdict(int)
        self.previous_score = 0.0
        self.normalized_score = 0.0
        self.liquidity_add_count = 0 
        self.liquidity_add_count = 0
        self.liquidity_remove_count = 0
        self.strategy = strategy or self.assign_strategy(role)
        self.strategy_name = self.strategy.name  
        self.fee_breakdown = defaultdict(float)
        self.activity_fee_history = defaultdict(lambda: defaultdict(float))
        self.activity_amount_history = defaultdict(lambda: defaultdict(int))
        self.activity_count_history = defaultdict(lambda: defaultdict(int))
        self.fee_snapshot = defaultdict(lambda: defaultdict(list))
        self.reward_breakdown = defaultdict(lambda: defaultdict(float))
        self.liquidity_add_count_history = defaultdict(int)
        self.liquidity_remove_count_history = defaultdict(int)
        self.liquidity_added_amount_history = defaultdict(float)
        self.liquidity_removed_amount_history = defaultdict(float)
        self.rewards_earned_history = defaultdict(float)
        self.updated_activity_levels = {}
        
        min_activity, max_activity = self.strategy.activity_level  # Get tuple
        self.activity_level = self.choice_engine.uniform(min_activity, max_activity)  # Assign float

    def set_agent_state(self, system_state: 'SystemState') -> None:
        """Set system state for agent to interact with"""
        self._system_state = system_state

    @property
    def state(self):
        """Always fetches current state"""
        if not self._system_state:
            raise ValueError("Network state not initialized")
        return self._system_state.agents[self.id]


    def update_activity_level(self, config: NetworkConfig) -> None:
        """
        Dynamically adjusts agent's per-activity participation levels based on:
        - Initial strategy-defined range (e.g., HIGH = 0.6–0.9)
        - Network signals via alpha = Wₐ × Gₐ
        - If alpha < threshold (e.g., 0.75), reduce enthusiasm by narrowing range downward
        - If alpha ≥ threshold, reward activity with higher range
    
        The adjusted value is stored and reused throughout the epoch.
        """
        self.updated_activity_levels = {}
    
        # Get full strategy-defined bounds
        strategy_min, strategy_max = self.strategy.activity_level
    
        for activity in config.activities:
            is_allowed = activity in config.roles[self.role].activities
            is_required = activity in ["mine_block", "bundle_transactions"]
    
            if not is_allowed and not is_required:
                continue
    
            # Sample once from full strategy range (if not already set)
            if activity not in self.updated_activity_levels:
                base_value = self.choice_engine.uniform(strategy_min, strategy_max)
            else:
                base_value = self.updated_activity_levels[activity]
    
            # Compute αₐ = Wₐ × Gₐ
            activity_weight = config.activity_weights.get(activity, 0.0)
            governance_multiplier = self._system_state.governance_multipliers.get(activity, 1.0)
            alpha = activity_weight * governance_multiplier
    
            # Adjust subrange dynamically
            ALPHA_THRESHOLD = 0.3
            if alpha < ALPHA_THRESHOLD:
                # Scale down → discourage activity
                new_min = strategy_min
                new_max = base_value
            else:
                # Scale up → encourage activity
                new_min = base_value
                new_max = strategy_max
    
            # Draw final activity level from adjusted range
            adjusted_level = self.choice_engine.uniform(new_min, new_max)
            self.updated_activity_levels[activity] = adjusted_level
    
        # Set max activity bounds for utility computation (using upper bound of strategy range)
        level_value = strategy_max
        self.max_possible_tx = int(level_value * 20)
        self.max_possible_liq = int(level_value * 12)
        self.max_possible_activities = self.max_possible_tx + self.max_possible_liq




    def get_fee_tolerance(self, level: float) -> Tuple[float, float]:
        if 0.7 <= level <= 0.9:
            return (0.05, 0.10)  # HIGH
        elif 0.4 <= level <= 0.6:
            return (0.02, 0.05)  # MEDIUM
        elif 0.1 <= level <= 0.3:
            return (0.01, 0.03)  # LOW
        else:
            return (0.00, 0.00)  # INACTIVE



    def generate_transactions(
    self,
    epoch: int,
    possible_recipients: List[str],
    config: NetworkConfig
) -> List[Dict]:
        if not isinstance(possible_recipients, list) or not possible_recipients:
            return []
        if self.balance <= 20 or self.role != "user":
            return []
    
        self.update_activity_level(config)
        activity_name = "transfer_tokens"
    
        # Safeguard: default to minimum strategy value if not set (shouldn't happen)
        effective_level = self.updated_activity_levels.get(
            activity_name,
            self.strategy.activity_level[0]  # fallback to min of strategy
        )
    
        # Skip if not active enough this round
        if self.choice_engine.random() >= effective_level:
            print(f"⚠️ Agent {self.id} skipped transfer_tokens due to low activity chance ({effective_level:.2f})")
            return []
    
        # Compute how many transactions to attempt
        min_tx = int(effective_level * 10)
        max_tx = int(effective_level * 20)
        num_transactions = self.choice_engine.randint(min_tx, max_tx)
        if num_transactions < 1:
            return []
    
        # 💰 Budget for all tx
        min_pct, max_pct = self.strategy.transaction_preference
        total_tx_budget = self.balance * self.choice_engine.uniform(min_pct, max_pct)
        amount_per_tx = total_tx_budget / num_transactions
    
        min_tol, max_tol = self.get_fee_tolerance(effective_level)
        fee_threshold = self.choice_engine.uniform(min_tol, max_tol) * self.balance
    
        transactions = []
        valid_recipients = [r for r in possible_recipients if r != self.id]
        if not valid_recipients:
            return []
    
        for _ in range(num_transactions):
            recipient = self.choice_engine.choice(valid_recipients)
    
            fee = float(config.get_activity_fee(
                activity_name=activity_name,
                user_reputation=self.reputation_score,
                system_state=self._system_state,
                agent_id=self.id
            ))
    
            if fee > fee_threshold and self.choice_engine.random() > 0.2:
                continue
    
            if amount_per_tx + fee <= self.balance:
                self.balance -= (amount_per_tx + fee)
                print(f"💸 Agent {self.id} sent {amount_per_tx:.2f} with fee {fee:.4f} at epoch {epoch}")
                self._system_state.agents[recipient].balance += amount_per_tx
                self.fees_paid += fee
                self.fee_breakdown[activity_name] += fee
                self.activities_performed[activity_name] += amount_per_tx
                self._system_state.reward_pools.transfer_fees += fee
    
                # ⏺ Track
                self.activity_amount_history[epoch][activity_name] += amount_per_tx
                self.activity_count_history[epoch][activity_name] += 1
                self.activity_fee_history[epoch][activity_name] += fee
                self.fee_snapshot[epoch][activity_name].append(fee)
    
                transactions.append({
                    'sender': self.id,
                    'recipient': recipient,
                    'amount': amount_per_tx,
                    'fee': fee,
                    'epoch': epoch,
                    'activity': activity_name,
                    'balance_before': self.balance + amount_per_tx + fee,
                    'balance_after': self.balance
                })
    
        self.epoch_transactions = transactions
        return transactions



    def execute_activity(self, activity_name: str, role_config: RoleConfig, config: NetworkConfig) -> bool:
        if activity_name != "provide_liquidity":
            return False
    
        if activity_name not in role_config.activities:
            return False
    
        if self.balance < 100 and self.total_liquidity_provided <= 0:
            return False
    
        activity_level = self.updated_activity_levels.get(activity_name, self.strategy)
        if self.choice_engine.random() > activity_level:
            print(f"⚠️ Agent {self.id} skipped provide_liquidity due to low activity chance ({activity_level:.2f})")
            return False
    
        min_actions = int(activity_level * 5)
        max_actions = int(activity_level * 12)
        num_actions = self.choice_engine.randint(min_actions, max_actions)
        if num_actions < 1:
            return False
    
        min_liq, max_liq = self.strategy.liquidity_preference
        total_budget = self.balance * self.choice_engine.uniform(min_liq, max_liq)
        amount_per_action = total_budget / num_actions
    
        epoch = self._system_state.current_epoch
        apy_lp = getattr(self._system_state, "current_lp_apy", 0.0)
    
        released = self._system_state.metrics.get("total_released", {}).get(epoch, 0.0)
        total_supply = self._system_state.metrics.get("total_supply", {}).get(epoch, 1.0)
        maturity = released / total_supply if total_supply > 0 else 0.0
        adjusted_lp_threshold = LP_APY_THRESHOLD * (1.5 - maturity)
    
        success = False
    
        # Ensure epoch keys exist in history
        self.activity_amount_history[epoch]  
        self.activity_count_history[epoch]
        self.liquidity_add_count_history[epoch]  
        self.liquidity_remove_count_history[epoch]  
    
        for _ in range(num_actions):
            # ------------------------
            # 🔻 REMOVE LIQUIDITY CASE
            # ------------------------
            if (apy_lp < adjusted_lp_threshold and self.choice_engine.random() > 0.05) or \
               (apy_lp >= adjusted_lp_threshold and self.choice_engine.random() < 0.05):
    
                if self.total_liquidity_provided < amount_per_action:
                    continue
    
                min_tol, max_tol = self.get_fee_tolerance(activity_level)
                fee_threshold = self.choice_engine.uniform(min_tol, max_tol) * self.balance
    
                fee = float(config.get_activity_fee(
                    activity_name=activity_name,
                    user_reputation=self.reputation_score,
                    system_state=self._system_state,
                    agent_id=self.id
                ))
    
                if fee > fee_threshold and self.choice_engine.random() > 0.2:
                    continue
    
                if fee > self.balance:
                    continue
    
                self.total_liquidity_provided -= amount_per_action
                self.balance += amount_per_action - fee
    
                self.fees_paid += fee
                self.fee_breakdown[activity_name] += fee
                self.activity_fee_history[epoch][activity_name] += fee
                self.fee_snapshot[epoch][activity_name].append(fee)
                self.activities_performed["remove_liquidity"] += amount_per_action
    
                # Add epoch tracking for removals
                self.liquidity_remove_count += 1
                self.liquidity_remove_count_history[epoch] += 1
                self.liquidity_removed_amount_history[epoch] += amount_per_action
    
                print(f"🧯 Agent {self.id} removed {amount_per_action:.2f} liquidity with fee {fee:.4f} at epoch {epoch}")
                self._system_state.reward_pools.liquidity_fees += fee
                success = True
                continue
    
            # ------------------------
            # 🔺 ADD LIQUIDITY CASE
            # ------------------------
            min_tol, max_tol = self.get_fee_tolerance(activity_level)
            fee_threshold = self.choice_engine.uniform(min_tol, max_tol) * self.balance
    
            fee = float(config.get_activity_fee(
                activity_name=activity_name,
                user_reputation=self.reputation_score,
                system_state=self._system_state,
                agent_id=self.id
            ))
    
            if fee > fee_threshold and self.choice_engine.random() > 0.2:
                continue
    
            if amount_per_action + fee > self.balance:
                break
    
            self.balance -= (amount_per_action + fee)
            self.fees_paid += fee
            self.fee_breakdown[activity_name] += fee
            self.activity_fee_history[epoch][activity_name] += fee
            self.activities_performed[activity_name] += amount_per_action
            self.activity_amount_history[epoch][activity_name] += 1
            self.activity_count_history[epoch][activity_name] += 1
            self.fee_snapshot[epoch][activity_name].append(fee)
            self.total_liquidity_provided += amount_per_action
    
            # ✅ Add epoch tracking for adds
            self.liquidity_add_count += 1
            self.liquidity_add_count_history[epoch] += 1
            self.liquidity_added_amount_history[epoch] += amount_per_action
    
            print(f"💧 Agent {self.id} added {amount_per_action:.2f} liquidity with fee {fee:.4f} at epoch {epoch}")
            self._system_state.reward_pools.liquidity_fees += fee
            success = True
    
        return success


    

       

    # def compute_raw_activity_score(self, activity_weights: Dict[str, float]) -> float:
#     """
#     Computes the raw activity score A_u^(t) for the agent.
#     This score is a weighted sum of the agent's activities and resources, including:
#     - The number of token transfers performed during the current epoch
#     - The total tokens staked across all epochs
#     - The total liquidity provided across all epochs
#     
#     The weights for each activity are supplied as a dictionary with keys:
#     "transfer_tokens", "stake_tokens", and "provide_liquidity".
#     
#     Returns:
#         float: The computed raw activity score A_u^(t)
#     """

#     # Compute the weighted score for token transfers performed this epoch
#     transfer_score = activity_weights.get("transfer_tokens", 0.0) * self.activities_performed.get("transfer_tokens", 0.0)

#     # Compute the weighted score for the total amount of tokens staked (cumulative)
#     staking_score = activity_weights.get("stake_tokens", 0.0) * self.total_staked

#     # Compute the weighted score for the total liquidity provided (cumulative)
#     liquidity_score = activity_weights.get("provide_liquidity", 0.0) * self.total_liquidity_provided

#     # Return the sum of all component scores as the final raw activity score
#     return transfer_score + staking_score + liquidity_score




    # def compute_user_score(
#     self,
#     activity_weights: Dict[str, float],
#     total_score: float,
#     lambda_smoothing: float = 0.6
# ) -> float:
#     """
#     Computes the normalized and exponentially smoothed user score S_u^(t+1),
#     which is used for reward distribution in the next epoch.
#     
#     The formula is:
#         S_u^(t+1) = λ * S_u^(t) + (1 - λ) * (A_u^(t) / ΣA)
#     where:
#         - λ (lambda_smoothing) is the smoothing factor (default 0.6)
#         - S_u^(t) is the previous smoothed score
#         - A_u^(t) is the raw activity score of the user for the current epoch
#         - ΣA is the total raw activity score of all users for the current epoch
#     
#     Args:
#         activity_weights (Dict[str, float]): Weights for each activity type (e.g., transfers, staking, liquidity).
#         total_score (float): Sum of all users’ raw activity scores in the current epoch (ΣA).
#         lambda_smoothing (float): Smoothing factor to control weight of previous score vs. current normalized score.
#     
#     Returns:
#         float: The updated and smoothed user score S_u^(t+1).
#     """

#     # Compute this user's raw activity score for the current epoch
#     raw_score = self.compute_raw_activity_score(activity_weights)

#     # Normalize the user's raw score by the total score of all users
#     normalized_score = raw_score / total_score if total_score > 0 else 0.0

#     # Store the normalized score for reference or logging
#     self.normalized_score = normalized_score

#     # Apply exponential smoothing to blend previous and current scores
#     smoothed_score = (lambda_smoothing * self.previous_score) + ((1 - lambda_smoothing) * normalized_score)

#     # Update the stored previous score to the new smoothed score for future use
#     self.previous_score = smoothed_score

#     # Save the final smoothed score for this epoch
#     self.score = smoothed_score

#     # Return the final score
#     return self.score

    

    def collect_reward(self, reward_amount):
        self.balance += reward_amount
        self.rewards_earned = reward_amount
        self.rewards_earned_last_epoch = reward_amount
    
        epoch = self._system_state.current_epoch
        self.rewards_earned_history[epoch] = reward_amount
        
        
        weights = self._system_state.activity_weights
    
        for activity in weights:
            share = reward_amount * weights[activity]
            self.reward_breakdown[epoch][activity] = share



    def reset_activity_tracking(self):
        self.activities_performed = defaultdict(float)
        #self.epoch_transactions = []




"""Σ
class Role:
    def __init__(self, config: RoleConfig):
        self.config = config
        self.total_activities = 0
        self.total_rewards = 0.0

    def calculate_reward(self, activity: ActivityConfig) -> float:
        "Calculate reward for an activity based on base rate and role multiplier"
        return activity.base_reward * self.config.reward_multiplier  # proportion of reward

    def validate_activity(self, agent: Agent, activity: ActivityConfig) -> bool:
        "Validate if agent can perform this activity"
        # Check minimum balance requirement
        if agent.balance < self.config.min_balance:
            return False
        # Check if activity is allowed for this role
        return activity.name in self.config.activities

    def process_activity(self, agent: Agent, activity: ActivityConfig) -> float:
        "Process activity execution and return reward amount"
        if not self.validate_activity(agent, activity):
            return 0.0

        if agent.execute_activity(activity.name, self.config):
            reward = self.calculate_reward(activity)
            self.total_activities += 1
            self.total_rewards += reward
            return reward

        return 0.0
"""




@dataclass
class Strategy:
    """Base strategy configuration"""
    name: str
    role: str
    activity_level: Tuple[float, float] 
    transaction_preference: Tuple[float, float]  
    liquidity_preference: Tuple[float, float]

@dataclass
class StrategySet:
    """Predefined strategy sets for different roles based on JSON role configuration"""
    
    user_strategies: List[Strategy] = field(default_factory=lambda: [
        Strategy(
            name="active_trader",
            role="user",
            activity_level= ActivityLevel.HIGH.get_range(),
            transaction_preference=(0.7, 0.9),  # Prefers frequent transactions
            liquidity_preference=(0.1, 0.2)  # Uses 10-20% for liquidity providing
        ),
        Strategy(
            name="hodler",
            role="user",
            activity_level= ActivityLevel.LOW.get_range(),  # Low activity, rarely trades
            transaction_preference=(0.2, 0.4),  # Prefers fewer transactions
            liquidity_preference=(0.0, 0.1)  # May provide liquidity occasionally
        )
    ])   
    developer_strategies: List[Strategy] = field(default_factory=lambda: [
        Strategy(
            name="liquidity_provider",
            role="developer",
            activity_level= ActivityLevel.MEDIUM.get_range(),  # Medium-high activity in providing liquidity
            transaction_preference=(0.0, 0.0),
            liquidity_preference=(0.6, 0.8)  # Uses 60-80% of balance for liquidity
        ),
        Strategy(
            name="protocol_builder",
            role="developer",
            activity_level= ActivityLevel.MEDIUM.get_range(),  # Medium activity, interacts with staking & liquidity
            transaction_preference=(0.0, 0.0),
            liquidity_preference=(0.3, 0.5)  # Provides some liquidity but not primary
        )
    ])
    
    bundler_strategies: List[Strategy] = field(default_factory=lambda: [
        Strategy(
            name="efficient_bundler",
            role="bundler",
            activity_level=ActivityLevel.HIGH.get_range(),
            transaction_preference=(0.0, 0.0),
            liquidity_preference=(0.0, 0.0)
        ),
        Strategy(
            name="casual_bundler",
            role="bundler",
            activity_level=ActivityLevel.MEDIUM.get_range(),
            transaction_preference=(0.0, 0.0),
            liquidity_preference=(0.0, 0.0)
        ),
        Strategy(
            name="passive_bundler",
            role="bundler",
            activity_level=ActivityLevel.LOW.get_range(),
            transaction_preference=(0.0, 0.0),
            liquidity_preference=(0.0, 0.0)
        )
    ])

    miner_strategies: List[Strategy] = field(default_factory=lambda: [
        Strategy(
            name="power_miner",
            role="miner",
            activity_level=ActivityLevel.HIGH.get_range(),
            transaction_preference=(0.0, 0.0),
            liquidity_preference=(0.0, 0.0)
        ),
        Strategy(
            name="steady_miner",
            role="miner",
            activity_level=ActivityLevel.MEDIUM.get_range(),
            transaction_preference=(0.0, 0.0),
            liquidity_preference=(0.0, 0.0)
        ),
        Strategy(
            name="light_miner",
            role="miner",
            activity_level=ActivityLevel.LOW.get_range(),
            transaction_preference=(0.0, 0.0),
            liquidity_preference=(0.0, 0.0)
        )
    ])

  