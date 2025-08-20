from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np

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
#class Activity:
    type: Literal['count', 'amount']
    reward_share: float # currently acts as multiplier
    reward_type: Literal['yield', 'salary', 'score']
    min_value: float
    description: str
    display_name: str
    current_value: float = 0
    
    def __post_init__(self):
        if self.type == 'count':
            self.min_value = int(self.min_value)
            self.current_value = int(self.current_value)

@dataclass
#class ActivityConfig:
    activities: Dict[str, Activity]

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ActivityConfig':
        activities = {
            name: Activity(
                type=params['type'],
                reward_share=params['reward_share'],
                reward_type=params['reward_type'],
                min_value=0,
                description=params.get('description', ''),
                display_name=params.get('display_name', name)
            )
            for name, params in config_dict.items()
        }
        return cls(activities=activities)

@dataclass
class NetworkParams:
    timeframe: int # months
    total_supply: float # tokens
    initial_circulating: Optional[float]
    staking_rate: float
    lp_rate: float
    num_miners: int
    num_bundlers: int
    num_wallets: int
    sc_deployed: int
    daily_transaction_volume: int
    max_daily_volume: float 
    native_volume: float
    avg_transaction_value: float
    fee_rate: float
    inflation_rate: float
    inflation_reduction_rate: float
    inflation_reduction_months: int
    growth_scaling_factor: float
    initial_market_penetration: float
    token_price: float
    total_addressable_market: float = 0.97

    def __post_init__(self):
        self.max_total_supply = self.calculate_max_total_supply()

    def calculate_max_total_supply(self) -> float:
        total_emission = 0
        month = 0
        min_emission = 1
        
        while True:
            emission = self.calculate_monthly_emission(month)
            if emission < min_emission:
                break
            total_emission += emission
            month += 1
        
        return self.total_supply + total_emission

    def calculate_monthly_emission(self, month: int) -> float:
        reduction_power = np.floor(month / self.inflation_reduction_months)
        return (self.total_supply * 
                self.inflation_rate * 
                (1 - self.inflation_reduction_rate) ** reduction_power / 12)



@dataclass
class RewardPool:
    """Agent-centric monthly reward calculator based on actual fees collected."""

    def calculate_monthly_rewards(self, agents, params) -> dict:
        """
        Calculate total fees collected in the current month in USD and tokens (cent).
        Assumes no inflation/emissions.
        
        Args:
            agents: dict of all agents
            params: global simulation parameters (token price, etc.)

        Returns:
            dict with monthly_fees_usd, monthly_fees_cent, total_rewards_cent
        """
        # Step 1: Aggregate fees from all agents across all activities
        total_fees_cent = 0.0
        
        for agent in agents.values():
            for activity, fee in agent.fee_breakdown.items():
                total_fees_cent += fee

        # Step 2: Convert to USD (if needed)
        monthly_fees_usd = total_fees_cent * params.token_price
        
        # Step 3: Since no emissions, total rewards = total fees collected
        total_rewards_cent = total_fees_cent

        return {
            'monthly_fees_usd': monthly_fees_usd,
            'monthly_fees_cent': total_fees_cent,
            'total_rewards_cent': total_rewards_cent
        }
        

    def calculate_yields(self, agents: dict, auto_restake: bool = True) -> Dict[str, float]:
        total_liquidity = 0.0
        rewards_from_liquidity = 0.0
    
        for agent in agents.values():
            total_liquidity += agent.total_liquidity_provided
            rewards_from_liquidity += agent.reward_breakdown.get("provide_liquidity", 0.0)
    
        yields = {}
    
        if total_liquidity > 0:
            monthly_rate_liquidity = rewards_from_liquidity / total_liquidity
            liquidity_apy = ((1 + monthly_rate_liquidity) ** 12 - 1) * 100 if auto_restake else monthly_rate_liquidity * 12 * 100
            yields['liquidity_apy'] = liquidity_apy
        else:
            yields['liquidity_apy'] = 0.0
    
        return yields




#@dataclass
#class RewardPool:
    """Stateless calculator of the reward pool"""

    #def calculate_monthly_rewards(self, network_state: dict, params) -> float:
        monthly_fees_usd = network_state['monthly_volume'] * params.fee_rate
        monthly_fees_cent = monthly_fees_usd / params.token_price
        total_rewards = monthly_fees_cent + network_state.get('emission', 0)
        return total_rewards

    #def calculate_distributions(self, rewards: float, activity_config: ActivityConfig) -> Dict[str, float]:
        return {
            name: rewards * activity.reward_share
            for name, activity in activity_config.activities.items()
        }

    #def calculate_yields(self, distributions: dict, network_state: dict, activity_config: ActivityConfig, auto_restake: bool = True) -> Dict[str, float]:
        yields = {}

        for name, activity in activity_config.activities.items():
            if activity.reward_type != 'yield' or name not in network_state['locked_amounts']:
                continue

            monthly_reward = distributions[name]
            locked_amount = network_state['locked_amounts'][name]

            if locked_amount <= 0:
                yields[name] = 0
                continue

            monthly_rate = monthly_reward / locked_amount
            
            if auto_restake:
                yields[name] = ((1 + monthly_rate) ** 12 - 1) * 100
            else:
                yields[name] = monthly_rate * 12 * 100

        return yields

    #def project_dapp_rewards(self, daily_volume: float, network_state: dict, activity_config: ActivityConfig) -> dict:
        monthly_volume_usd = daily_volume * 30
        monthly_fees_usd = monthly_volume_usd * network_state['fee_rate'] 
        monthly_fees_cent = monthly_fees_usd / network_state['token_price']

        dapp_share = activity_config.activities['sc_deployment'].reward_share

        monthly_rewards_cent = monthly_fees_cent * dapp_share
        monthly_rewards_usd = monthly_rewards_cent * network_state['token_price']

        return {
            'monthly_volume_usd': monthly_volume_usd,
            'monthly_fees_cent': monthly_fees_cent,
            'monthly_rewards_cent': monthly_rewards_cent,
            'monthly_rewards_usd': monthly_rewards_usd
        }
