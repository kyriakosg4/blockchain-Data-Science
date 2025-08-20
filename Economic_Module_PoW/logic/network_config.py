from dataclasses import dataclass
from typing import Dict, List, Optional
from decimal import Decimal, getcontext
import json

class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors"""
    pass


@dataclass
class ActivityConfig:
    """Configuration for network activities"""
    name: str
    base_reward_rate: float
    fee: float
    governance_multiplier: float = 1.0  # Default multiplier if not provided

    def validate(self) -> None:
        """Validate the activity configuration values."""
        if not self.name:
            raise ConfigValidationError("Activity name cannot be empty")

        for value, name in [
            (self.base_reward_rate, "base_reward_rate"),
            (self.fee, "fee"),
            (self.governance_multiplier, "governance_multiplier")
        ]:
            if not isinstance(value, (int, float)):
                raise ConfigValidationError(f"Activity {name} must be a number")
            if value < 0:
                raise ConfigValidationError(f"Activity {name} cannot be negative")

@dataclass
class TokenAllocation:
    category: str
    initial_allocation: int  
    vesting_months: int
    initial_amount: int
    cliff_months: int
    
    @property
    def monthly_release(self) -> float:
        """Calculate monthly release of % allocated for vesting schedules"""
        if self.vesting_months == 0 and self.cliff_months == 0:
            return self.initial_allocation / 100  # Immediate release
        elif self.cliff_months > 0:
            if self.cliff_months >= self.vesting_months:
                return 0
            return self.initial_allocation / 100 / (self.vesting_months - self.cliff_months)
        else:
            return self.initial_allocation / 100 / self.vesting_months if self.vesting_months > 0 else 0

@dataclass
class RewardPool:
    """Agent-centric monthly reward calculator and fee aggregator."""
    transfer_fees: float = 0.0
    liquidity_fees: float = 0.0

    def total(self) -> float:
        """Return the total of all fee pools."""
        return self.transfer_fees + self.liquidity_fees

    def calculate_monthly_rewards(self, agents, params, token_price):
        """
        Calculate total fees collected in the current month in USD and tokens (cent).
        Assumes no inflation/emissions.

        Args:
            agents: dict of all agents
            params: global simulation parameters (token price, etc.)

        Returns:
            dict with monthly_fees_usd, monthly_fees_cent, total_rewards_cent
        """
        total_fees_cent = 0.0

        for agent in agents.values():
            for activity, fee in agent.fee_breakdown.items():
                total_fees_cent += fee

        monthly_fees_usd = total_fees_cent * token_price
        total_rewards_cent = total_fees_cent

        return {
            'monthly_fees_usd': monthly_fees_usd,
            'monthly_fees_cent': total_fees_cent,
            'total_rewards_cent': total_rewards_cent
        }

@dataclass
class RoleConfig:
    """Configuration for network roles"""
    name: str
    activities: List[str]
    min_balance: float
    allocation_percent: float  

    def validate(self) -> None:
        """Validate role configuration parameters"""
        if not self.name:
            raise ConfigValidationError("Role name cannot be empty")

        if not self.activities or not isinstance(self.activities, list):
            raise ConfigValidationError(f"Role {self.name} must have at least one valid activity")

        if not isinstance(self.min_balance, (int, float)):
            raise ConfigValidationError(f"Role {self.name} min_balance must be a number")
        if self.min_balance < 0:
            raise ConfigValidationError(f"Role {self.name} min_balance cannot be negative")

        if not isinstance(self.allocation_percent, (int, float)):
            raise ConfigValidationError(f"Role {self.name} allocation_percent must be a number")
        if not 0 <= self.allocation_percent <= 1:
            raise ConfigValidationError(f"Role {self.name} allocation_percent must be between 0 and 1")
            

    def target_count(self, total_agents: int) -> int:
        """Calculate target number of agents for this role"""
        return int(total_agents * self.allocation_percent)

@dataclass
class NetworkConfig:
    # Network core resources
    initial_supply: float
    seed: int
    
    # Transaction processing parameters
    min_batch_size: int
    max_batch_size: int 
    base_reward_rate: float
    base_block_reward: float
    #fee_rate: float 
    
    # Fee structure (must sum to 1.0)
    bundler_fee_share: float
    validator_fee_share: float 
    network_fee_share: float
    fee_transactions: float

    # Agents and roles
    num_agents: int
    roles: Dict[str, RoleConfig]
    activities: Dict[str, ActivityConfig]
    total_potential_agents: int

    # Weights
    activity_weights: Dict[str, float]
    epochs: int = 36 


    

    

    def get_activity_fee(self, activity_name: str, user_reputation: float = 1.0, system_state: 'SystemState' = None, agent_id=None):
        if activity_name not in self.activities:
            raise ValueError(f"Unknown activity: {activity_name}")
    
        # 🔄 Convert to Decimal for precision
        base_fee = Decimal(str(self.activities[activity_name].fee))
        reputation = Decimal(str(user_reputation))
        governance_multiplier = Decimal("1.0")
        activity_weight = Decimal("1.0")
    
        # ✅ Get governance multiplier
        if system_state:
            governance_multiplier = Decimal(str(system_state.governance_multipliers.get(activity_name, 1.0)))
    
        # ✅ Get weight for previous epoch
        if system_state and system_state.current_epoch > 0:
            try:
                previous_weights = system_state.past_activity_weights[system_state.current_epoch - 1]
                activity_weight = Decimal(str(previous_weights.get(activity_name, 1.0)))
            except IndexError:
                activity_weight = Decimal("1.0")
        elif system_state and system_state.current_epoch == 0:
            activity_weight = Decimal(str(self.config.activity_weights.get(activity_name, 1.0)))
    
        # ✅ Compute precise fee
        fee = base_fee * activity_weight * governance_multiplier * reputation
    
        # ✅ Clamp to [0.0001, 0.1]
        fee = max(Decimal("0.0001"), min(fee, Decimal("0.1")))
    
        # 🧾 Optional debug print
        #if (
            #system_state is not None
            #and system_state.current_epoch in [4, 5, 6]
            #and agent_id == "user_agent_10"
        #):
            #print(
                #f"[Epoch {system_state.current_epoch}] Fee calc for {activity_name} (Agent: {agent_id}):\n"
                #f"  • Base fee (Bₐ): {base_fee:.6f}\n"
                #f"  • Weight (Wₐ): {activity_weight:.6f}\n"
                #f"  • Governance multiplier (αₐ): {governance_multiplier:.6f}\n"
                #f"  • Reputation (Rᵤ): {reputation:.6f}\n"
                #f"  → Fee (Decimal): {fee:.6f}\n"
            #)
    
        return fee  # Return as Decimal
    




    def validate(self) -> None:
        """Validate configuration parameters"""
        # Validate fee structure
        total_fee_share = (self.bundler_fee_share + 
                          self.validator_fee_share + 
                          self.network_fee_share)
        if abs(total_fee_share - 1.0) > 1e-10:
            raise ValueError(
                f"Fee shares must sum to 1.0, got {total_fee_share} "
                f"(bundler: {self.bundler_fee_share}, "
                f"validator: {self.validator_fee_share}, "
                f"network: {self.network_fee_share})"
            )

        # Validate role allocations sum to 1
        total_allocation = sum(role.allocation_percent for role in self.roles.values())
        if abs(total_allocation - 1.0) > 1e-10:
            raise ValueError(
                f"Role allocations must sum to 1.0, got {total_allocation}"
            )

        # Validate all activities
        for activity in self.activities.values():
            activity.validate()
            
        # Validate all roles
        for role in self.roles.values():
            role.validate()
            
        # Validate role activities exist
        all_activities = set(self.activities.keys())
        for role in self.roles.values():
            invalid_activities = set(role.activities) - all_activities
            if invalid_activities:
                raise ConfigValidationError(
                    f"Role {role.name} references undefined activities: {invalid_activities}"
                )
    
         # ✅ New activity_weights validation
        if not self.activity_weights:
            raise ValueError("Missing activity_weights in configuration.")
    
        for activity, weight in self.activity_weights.items():
            if activity not in self.activities:
                raise ValueError(f"Unknown activity in activity_weights: {activity}")
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"Invalid weight value for activity {activity}: {weight}")
                

def load_configuration(config_file: str) -> NetworkConfig:
    """Load network configuration from JSON file"""
    with open(config_file, 'r') as f:
        data = json.load(f)
    
    # Convert activities dict
    activities = {
        name: ActivityConfig(**config)
        for name, config in data['activities'].items()
    }
    
    # Convert roles dict
    roles = {
        name: RoleConfig(**config)
        for name, config in data['roles'].items()
    }

    activity_weights = data.get("activity_weights", {})
    
    # Create and validate config
    config = NetworkConfig(
        initial_supply=data['initial_supply'],
        base_reward_rate=data['base_reward_rate'],
        num_agents=data['num_agents'],
        epochs=36,
        total_potential_agents=data["total_potential_agents"],
        seed=data['seed'],
        min_batch_size=data['min_batch_size'],
        max_batch_size=data['max_batch_size'],
        base_block_reward=data['base_block_reward'],
        fee_transactions = data['activities']['transfer_tokens']['fee'],
        bundler_fee_share=data['bundler_fee_share'],
        validator_fee_share=data['validator_fee_share'],
        network_fee_share=data['network_fee_share'],
        activities=activities,
        roles=roles,
        activity_weights=activity_weights
    )
    
    config.validate()
    return config