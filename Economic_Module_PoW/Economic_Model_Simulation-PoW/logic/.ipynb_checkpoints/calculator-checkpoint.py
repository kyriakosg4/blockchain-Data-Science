import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import Dict, List, Optional
import numpy as np
#from simulation_framework.network_config import *
from network_config import NetworkConfig, TokenAllocation, ActivityConfig, RewardPool
#from simulation_framework.metrics import HealthMetrics
from collections import defaultdict



from sim_config import (
    PRICE_IMPACT_FACTOR,
    PRICE_ADJUSTMENT_SPEED,
    STAKING_APY_THRESHOLD,
    LP_APY_THRESHOLD,
    STAKING_GROWTH_RATE,
    STAKING_BASE_UNSTAKE_RATE,
    STAKING_HIGH_UNSTAKE_RATE,
    STAKING_MAX_FROM_FLOAT,
    LP_GROWTH_RATE,
    LP_BASE_UNSTAKE_RATE,
    LP_HIGH_UNSTAKE_RATE,
    LP_MAX_FROM_FLOAT,
    LP_APY_SMOOTHING_LAMBDA,
    MAX_MINERS,
    MAX_BUNDLERS, 
    GAS_PER_USEROP,
    BLOCKS_PER_DAY,
    MAX_TX_PER_DAY,
    DAILY_TX_PER_BUNDLER, 
    MAX_DAILY_VOLUME,
    INITIAL_TOKEN_PRICE
)


class NetworkCalculator:
    def __init__(self, config: NetworkConfig, allocations: List[TokenAllocation]):
        self.config = config
        self.allocations = allocations
        self.num_months = config.epochs 
        self.activity_config = config.activities

        # Initialize macro metrics container
        self.metrics = defaultdict(dict)

        # Pre-initialize per-activity reward metrics
        self.metrics['activity_rewards_tokens'] = {a: {} for a in self.activity_config}
        self.metrics['activity_rewards_dollars'] = {a: {} for a in self.activity_config}

        self.token_price = 1.0
        self.state = {}




## Surely being used in latest version

    def _calculate_token_state(self, month: int, state: dict) -> Dict[str, float]:
        """
        Calculate token price, demand, and velocity using agent-driven activity data.
        Includes both token and USD volume metrics.
        """
        prev_price = INITIAL_TOKEN_PRICE if month == 0 else state['token_price']
        effective_supply = state['circulating_supply'] - state['lp_amount']
    
        if effective_supply <= 0:
            return {
                'token_price': prev_price,
                'token_demand': 0,
                'monthly_tx_volume_tokens': 0,
                'monthly_tx_volume_usd': 0,
                'effective_supply': 0,
                'tx_velocity': 0,
                'reward_velocity': 0,
                'fee_velocity': 0,
                'total_velocity': 0
            }
    
        # Extract both volumes
        monthly_volume_tokens = state.get('monthly_volume_tokens', 0.0)
        monthly_volume_usd = state.get('monthly_volume_usd', monthly_volume_tokens * prev_price)
    
        # Total token demand = fees paid + liquidity locked
        monthly_demand = state['total_fees'] + state['lp_amount'] + state.get('monthly_volume_tokens', 0.0)
    
        # Price impact model based on demand-supply imbalance
        supply_demand_imbalance = (monthly_demand - effective_supply) / effective_supply
        market_price = prev_price * (1 + PRICE_IMPACT_FACTOR * supply_demand_imbalance)
        new_price = prev_price + PRICE_ADJUSTMENT_SPEED * (market_price - prev_price)
    
        # Velocity calculations
        circulating_supply = state['circulating_supply']
        tx_velocity = monthly_volume_tokens / circulating_supply if circulating_supply > 0 else 0
        reward_velocity = state['total_rewards'] / circulating_supply if circulating_supply > 0 else 0
        fee_velocity = state['total_fees'] / circulating_supply if circulating_supply > 0 else 0
        total_velocity = tx_velocity + reward_velocity + fee_velocity
    
        return {
            'token_price': new_price,
            'token_demand': monthly_demand,
            'monthly_tx_volume_tokens': monthly_volume_tokens,
            'monthly_tx_volume_usd': monthly_volume_usd,
            'effective_supply': effective_supply,
            'tx_velocity': tx_velocity,
            'reward_velocity': reward_velocity,
            'fee_velocity': fee_velocity,
            'total_velocity': total_velocity
        }


# Network Operations, Initial State
    # TODO: review fix and add    
    # def calculate_stgf(epoch, category_name="STGF-Ecosystem"):
    #     for alloc in allocations:
    #         if alloc.category == category_name:
    #             stgf = TOTAL_SUPPLY * alloc.initial_allocation / 100
    #             alpha = sim_config.STGF_PARAMS.get(category_name, {}).get("alpha", 1)
    #             total_months = EPOCHS
    #             denominator = sum([1 / (1 + m)**alpha for m in range(total_months)])
    #             weight = 1 / (1 + epoch)**alpha / denominator
    #             return weight * stgf / EPOCHS
    #     raise ValueError(f"STGF category '{category_name}' not found in allocations.")


    
    def _calculate_release_schedule(self, total_supply: float, num_months: int = 36) -> Dict[str, np.ndarray]:
        """
        Calculate token release schedule including cliffs and vesting.
        Returns a dict mapping each allocation category to a release array.
        """
        releases = {}
    
        for alloc in self.allocations:
            releases[alloc.category] = np.zeros(num_months)
    
            if alloc.vesting_months == 0 and alloc.cliff_months == 0:
                continue
    
            monthly_release = (total_supply * alloc.initial_allocation / 100) / alloc.vesting_months
    
            for month in range(num_months):
                if month < alloc.cliff_months:
                    releases[alloc.category][month] = releases[alloc.category][month - 1] if month > 0 else 0
                elif month < (alloc.cliff_months + alloc.vesting_months):
                    previous = releases[alloc.category][month - 1] if month > 0 else 0
                    releases[alloc.category][month] = previous + monthly_release
                else:
                    releases[alloc.category][month] = releases[alloc.category][month - 1]
    
        return releases


    def _calculate_initial_circulating(self, role: Optional[str] = None) -> float:
        if role:
            return sum(
                self.config.initial_supply * alloc.initial_allocation / 100
                for alloc in self.allocations
                if alloc.vesting_months == 0 and alloc.cliff_months == 0 and alloc.category == role
            )
        return sum(
            self.config.initial_supply * alloc.initial_allocation / 100
            for alloc in self.allocations
            if alloc.vesting_months == 0 and alloc.cliff_months == 0
        )


    def _calculate_overall_monthly_release(self, month: int, releases: Dict[str, np.ndarray]) -> float:
        """Safely computes incremental token release across all allocation categories."""
        
        if month == 0:
            return 0
    
        # Determine the max number of months available in release schedules
        max_month = max(len(r) for r in releases.values()) - 1
        if month > max_month:
            return 0  # No more release data to access
    
        return sum(
            (
                releases[alloc.category][month] - releases[alloc.category][month - 1]
                if month < len(releases[alloc.category])
                else 0
            )
            for alloc in self.allocations
        )

    
    def _calculate_vesting_release(self, month: int) -> float: # Pending verification
        """Calculate tokens released from vesting schedules, subject to cliff and vesting period constraints."""
        return sum(
            alloc.monthly_release for alloc in self.allocations 
            if (month >= alloc.cliff_months and  # Past cliff period
                month < (alloc.vesting_months + alloc.cliff_months))  # Still in vesting period
        )

# # Network Operations, State Updates
# def _calculate_token_emission(self, month: int) -> float:  # Verified but hopefully not needed, since we'll do without inflation
#     """
#     Calculate monthly token distribution from pre-minted reserve.
#     Despite "inflation" nomenclature, this actually releases from a fixed supply
#     at a rate of initial_supply * inflation_rate, reduced over time.
#     """
#     reduction_power = np.floor(month / self.config.inflation_reduction_months)
#     monthly_emission = (
#         self.config.initial_supply * 
#         self.config.inflation_rate * 
#         (1 - self.config.inflation_reduction_rate) ** reduction_power / 12
#     )
    
#     # return monthly_emission
#     raise NotImplementedError("This method is deprecated, use NetworkParams.calculate_monthly_emission instead")


    #def _calculate_monthly_inflation(self, current_supply: float) -> float: #deprecate! 
#    """Calculate monthly inflation"""
#        #return current_supply * (self.config.inflation_rate / 12)
#        raise NotImplementedError("This method is deprecated, use NetworkParams.calculate_monthly_emission instead") 
    
    def _update_supply_state(self, month: int, state: dict) -> dict:
        """
        Update supply metrics based on vesting and agent activity.
    
        - Tracks vesting and total released tokens
        - Calculates circulating supply
        - Calculates staked and LP amounts based on agents
        - Calculates effective floating supply
        """
    
        total_supply = self.config.initial_supply  # fixed total supply, no emissions
    
        # Step 1: Calculate vested tokens from release schedule
        if month > 0:
            monthly_release = self._calculate_overall_monthly_release(month, state['releases'])
            total_released = state['total_released'] + monthly_release
        else:
            # Initial circulating only from immediate (0 vesting) releases
            total_released = sum(
                self.config.initial_supply * alloc.initial_allocation / 100 / alloc.vesting_months
                if alloc.vesting_months > 0 else self.config.initial_supply * alloc.initial_allocation / 100
                for alloc in self.allocations
                if alloc.cliff_months == 0
            )
    
        circulating_supply = total_released
    
        # Step 2: Aggregate from agents (required for staking and LP logic)
        if 'agent_objects' not in state:
            raise ValueError("agent_objects missing in state. Pass agents into state dict.")
    
        agent_list = state['agent_objects']
        lp_amount = sum(agent.total_liquidity_provided for agent in agent_list)
    
        # Step 3: Effective supply = circulating minus locked amounts
        effective_supply = circulating_supply - lp_amount
    
        return {
            'total_supply': total_supply,
            'total_released': total_released,
            'circulating_supply': circulating_supply,
            'lp_amount': lp_amount,
            'effective_supply': effective_supply
        }

    
# Network Dynamics
   # def get_network_utility(self, adoption: float, alpha: float = 2) -> float:
#     """Calculate utility using Metcalfe's Law."""
#     return adoption ** alpha

# def get_network_growth_rate(self, adoption, max_adoption, beta=0.8, alpha=2) -> float:
#     if adoption <= 0 or adoption > max_adoption:
#         raise ValueError("Adoption out of bounds, must be higher than 0 and lower than max.")
#     utility = self.get_network_utility(adoption, alpha)
#     return beta * utility * (1 - adoption / max_adoption)

# def calculate_adoption(self, initial_adoption: float, max_adoption: float,
#                        growth_rate: float, month: int, midpoint: int = 12) -> float:
#     if month <= 0:
#         return initial_adoption
#     t = month - midpoint
#     adoption = max_adoption / (1 + np.exp(-growth_rate * t))
#     min_adoption = max_adoption / (1 + np.exp(growth_rate * midpoint))
#     growth = max(adoption - min_adoption, 0)
#     adoption = initial_adoption + growth
#     return max(initial_adoption, min(adoption, max_adoption))

# def _calculate_adoption(self, month: int) -> dict:
#     users = self.calculate_adoption(
#         initial_adoption=self.config.num_wallets,
#         max_adoption=self.config.total_addressable_market,
#         growth_rate=self.config.growth_scaling_factor,
#         month=month
#     )

#     normalized_adoption = users / self.config.total_addressable_market
#     utility = self.get_network_utility(normalized_adoption)

       # # Use sigmoid to cap volume growth (configurable)
# def bounded_volume(epoch, cap=self.config.max_daily_volume, midpoint=12, steepness=0.5):
#     return cap / (1 + np.exp(-steepness * (epoch - midpoint)))

# daily_volume = bounded_volume(month)
# transaction_volume = daily_volume * 30

# # FIXED counts for now
# # TODO: update to dynamic growth
# miners = self.config.num_miners
# bundlers = self.config.num_bundlers

# return {
#     'users': users,
#     'miners': miners,
#     'bundlers': bundlers,
#     'transaction_volume': transaction_volume
# }

# # Fee Collection  
# def _calculate_monthly_fees_in_tokens(self) -> float:
#     """Calculate monthly fees from daily transactions"""
#     daily_volume = self.config.daily_transaction_volume * (self.config.avg_transaction_value / self.config.token_price)
#     return daily_volume * self.config.fee_rate * 30

# # Reward Distribution. Refactor into dedicated engine class   
# def calculate_distributions(self, month: int) -> Dict[str, float]:  # deprecate, handle in RewardPool class
#     monthly_pool = self.reward_pool.monthly_rewards(month)
#     return {
#         name: monthly_pool * activity.reward_share
#         for name, activity in self.activity_config.activities.items()
#     }

# def get_distribution_summary(self) -> Dict[str, Dict[str, float]]: 
#     distributions = self.calculate_distributions(0)  # this is just for the initial state, the summary should return a yearly average or something 
#     total_pool = sum(distributions.values())
    
#     return {
#         name: {
#             'monthly_reward': reward,
#             'share': reward / total_pool if total_pool > 0 else 0, 
#             'annual_reward': reward * 12,
#             'apy': self.calculate_apy(name, reward)
#         }
#         for name, reward in distributions.items()
#     }

# def calculate_apy(self, activity_name: str, monthly_reward: float) -> float:
#     activity = self.activity_config.activities[activity_name]
#     if activity.current_value > 0:
#         return (monthly_reward * 12 / activity.current_value) * 100
#     return 0.0

# def _calculate_monthly_metrics(self, month: int) -> Dict[str, float]:  # Pending separation of native transactions vs dApp volume
#     """Calculate all metrics for a given month based on adoption"""
#     adoption = self._calculate_adoption(month)
    
#     return {
#         'daily_volume': adoption['transaction_volume'] / 30,
#         'monthly_fees_tokens': (adoption['transaction_volume'] * self.config.fee_rate) / self.config.token_price,
#         'monthly_fees_usd': adoption['transaction_volume'] * self.config.fee_rate,
#         'monthly_inflation': self._calculate_monthly_inflation(self.config.initial_supply),
#         'network_growth_rate': self.get_network_growth_rate(
#             adoption=self.config.initial_market_penetration,
#             max_adoption=1.0,
#             beta=self.config.growth_scaling_factor
#         )
#     }


    # def _calculate_activity_rewards(self, monthly_fees: float, monthly_inflation: float, month: int) -> Dict[str, float]:  # deprecate, handle in RewardPool class
#     """Calculate rewards for each activity"""
#     total_rewards = monthly_fees + monthly_inflation
#     return {
#         name: total_rewards * activity.reward_share
#         for name, activity in self.activity_config.activities.items()
#     }

# def _calculate_reward_distribution(self, month: int) -> float:
#     """Calculate reward distribution ratio"""
#     if month >= len(self.metrics['activity_rewards_tokens']):
#         return 0
#     total_rewards = sum(
#         rewards[month] 
#         for rewards in self.metrics['activity_rewards_tokens'].values()
#     )
#     available_rewards = self.metrics['monthly_inflation'][month] + self.metrics['monthly_fees']
#     return total_rewards / available_rewards if available_rewards > 0 else 0

# # Health Metrics, OKRs and KPIs 
# def _calculate_velocity_metrics(self, state: dict) -> dict:  # deprecate, handle in token_state update
#     """
#     Calculate velocity components:
#     V_tx(t) = native_volume(t)/(S_c(t) * P(t))
#     V_rewards(t) = rewards_distributed(t)/S_c(t)
#     V_total(t) = V_tx(t) + V_rewards(t)
#     """
#     if state['circulating_supply'] == 0:
#         return {
#             'tx_velocity': 0,
#             'reward_velocity': 0,
#             'total_velocity': 0
#         }
        
#     tx_velocity = (state['native_volume'] * 30) / (
#         state['circulating_supply'] * state['token_price']
#     )
    
#     reward_velocity = state['total_rewards'] / state['circulating_supply']
    
#     # return {
#     #     'tx_velocity': tx_velocity,
#     #     'reward_velocity': reward_velocity,
#     #     'total_velocity': tx_velocity + reward_velocity
#     # }
#     raise NotImplementedError("This method is deprecated, use _calculate_token_state instead")

# def _calculate_security_ratio(self, staked: float, total: float) -> float:
#     """Calculate security ratio"""
#     return staked / total if total > 0 else 0

# def _calculate_fee_sustainability(self, fees: float, inflation: float) -> float:
#     """Calculate fee sustainability ratio"""
#     return fees / inflation if inflation > 0 else 0

    
    def _calculate_token_velocity(self, monthly_volume_tokens: float, circulating_supply: float) -> float:
        """
        Calculate monthly token velocity
        Args:
            monthly_volume_tokens: Monthly transaction volume in tokens (already adjusted for growth)
            circulating_supply: Circulating supply for current month
        Returns:
            Monthly velocity ratio (volume/supply)
        """
        return monthly_volume_tokens / circulating_supply if circulating_supply > 0 else 0

    # def _calculate_token_metrics(self, month: int, circulating_supply: float, token_prices: np.ndarray, monthly_volume_usd: float) -> Dict[str, float]:
#     """Calculate token metrics based on adoption metrics"""
#     prev_price = self.config.token_price if month == 0 else token_prices[month - 1]
#     effective_supply = circulating_supply * (1 - self.config.staking_rate - self.config.lp_rate)
    
#     if effective_supply <= 0:
#         return {
#             'token_price': prev_price,
#             'token_demand': 0,
#             'native_volume': 0,
#             'sc_volume': monthly_volume_usd * (1 - self.config.native_volume),
#             'monthly_tx_volume_tokens': 0,
#             'monthly_tx_volume_usd': monthly_volume_usd,
#             'effective_supply': 0
#         }
        
#     # Calculate native transaction demand
#     daily_native_volume_tokens = (monthly_volume_usd / 30 * self.config.native_volume) / prev_price
#     monthly_native_volume_tokens = daily_native_volume_tokens * 30
    
#     # Total token demand includes transaction demand + staking + LP
#     monthly_demand = (
#         monthly_native_volume_tokens +
#         circulating_supply * self.config.staking_rate +
#         circulating_supply * self.config.lp_rate
#     )
    
#     # Price change based on supply/demand
#     demand_ratio = monthly_demand / effective_supply
#     price_change = (demand_ratio - 1) * 0.1
#     price_change = max(min(price_change, 0.2), -0.2)
#     new_price = prev_price * (1 + price_change)

        # return {
        #     'token_price': new_price,
        #     'token_demand': monthly_demand,
        #     'native_volume': daily_native_volume_tokens,
        #     'sc_volume': monthly_volume_usd * (1 - self.config.native_volume),
        #     'monthly_tx_volume_tokens': monthly_volume_usd / prev_price,
        #     'monthly_tx_volume_usd': monthly_volume_usd,
        #     'effective_supply': effective_supply
        # }
        #raise NotImplementedError("This method is deprecated, use _calculate_token_state instead")

    # def _calculate_fee_generation_rate(self, month: int) -> float:
#     """Calculate fee generation rate as percentage of transaction volume"""
#     if month >= len(self.metrics['monthly_tx_volume_usd']):
#         return 0
        
#     monthly_fees = self._calculate_monthly_fees()
#     monthly_volume = self.metrics['monthly_tx_volume_usd'][month]
    
#     return monthly_fees / monthly_volume if monthly_volume > 0 else 0

# def _calculate_concentration_ratio(self, month: int) -> float:
#     """Calculate largest holder concentration"""
#     if month >= len(self.metrics['token_releases']):
#         return 0
#     largest_holding = max(
#         releases[month] 
#         for releases in self.metrics['token_releases'].values()
#     )
#     return largest_holding / self.metrics['total_supply'][month]

# def _calculate_circulating_ratio(self, month: int) -> float:
#     """Calculate ratio of circulating supply to total supply"""
#     if month >= len(self.metrics['total_supply']) or self.metrics['total_supply'][month] == 0:
#         return 0
#     return self.metrics['circulating_supply'][month] / self.metrics['total_supply'][month]

# def _calculate_economic_security(self, month: int) -> float:
#     """Calculate economic security ratio"""
#     if month >= len(self.metrics['staked_amount']) or self.metrics['total_supply'][month] == 0:
#         return 0
#     return (self.metrics['staked_amount'][month] * self.metrics['token_price'][month]) / \
#         (self.metrics['total_supply'][month] * self.metrics['token_price'][month])

# def _calculate_staking_ratio(self, month: int) -> float:
#     """Calculate ratio of staked tokens to circulating supply"""
#     if month >= len(self.metrics['circulating_supply']) or self.metrics['circulating_supply'][month] == 0:
#         return 0
#     return self.metrics['staked_amount'][month] / self.metrics['circulating_supply'][month]

# def _calculate_supply_growth(self, month: int) -> float:  # = to inflation rate if no burns and not a dynamic parameter
#     """Calculate month-over-month supply growth rate"""
#     if month == 0 or month >= len(self.metrics['total_supply']):
#         return 0
#     prev_supply = self.metrics['total_supply'][month-1]
#     if prev_supply == 0:
#         return 0
#     return (self.metrics['total_supply'][month] - prev_supply) / prev_supply

# def _calculate_release_rate(self, month: int) -> float:
#     """Calculate token release rate"""
#     if month >= len(self.metrics['total_supply']) or self.metrics['total_supply'][month] == 0:
#         return 0
#     monthly_release = sum(
#         releases[month] - (releases[month-1] if month > 0 else 0)
#         for releases in self.metrics['token_releases'].values()
#     )
#     return monthly_release / self.metrics['total_supply'][month]

    # def get_health_metrics(self, month: int) -> HealthMetrics:
    #     """Calculate health metrics using state data"""
    #     if month >= len(self.metrics['total_supply']):
    #         return HealthMetrics(0,0,0,0,0,0,0,0,0,0,0)  # Return zeros if month out of range
            
    #     return HealthMetrics(
    #         circulating_ratio=self.metrics['circulating_supply'][month] / self.metrics['total_supply'][month] if self.metrics['total_supply'][month] > 0 else 0,
    #         staking_ratio=self.metrics['staked_amount'][month] / self.metrics['circulating_supply'][month] if self.metrics['circulating_supply'][month] > 0 else 0,
    #         supply_growth_rate=self._calculate_supply_growth(month),
    #         token_release_rate=self._calculate_release_rate(month),
    #         fee_generation_rate=self.metrics['monthly_fees'][month] / self.metrics['monthly_tx_volume_usd'][month] if self.metrics['monthly_tx_volume_usd'][month] > 0 else 0,
    #         fee_sustainability=self.metrics['monthly_fees'][month] / self.metrics['emission'][month] if self.metrics['emission'][month] > 0 else 0,
    #         concentration_ratio=self._calculate_concentration_ratio(month),
    #         reward_distribution=self.metrics['total_rewards'][month] / self.metrics['emission'][month] if self.metrics['emission'][month] > 0 else 0,
    #         economic_security_ratio=self._calculate_economic_security(month),
    #         staking_apy=self.metrics['yields'].get('staking', 0),
    #         lp_apy=self.metrics['yields'].get('liquidity', 0)
    #     )

# # Yield Calculations
# def _calculate_apy_series(self, monthly_rewards: np.ndarray, stake_amounts: np.ndarray) -> List[float]:  # Compounded APY
#     """Calculate APY with proper monthly compounding based on category's stake
#     
#     Args:
#         monthly_rewards: Array of monthly rewards for this category
#         stake_amounts: Array of stake amounts for this category (not total stake)
#     """
#     # Calculate monthly rate as rewards/stake for non-zero stakes
#     monthly_rates = np.divide(
#         monthly_rewards,
#         stake_amounts,
#         out=np.zeros_like(monthly_rewards), 
#         where=stake_amounts > 0
#     )
    
#     # Convert to compounded APY: (1 + r)^12 - 1
#     return (((1 + monthly_rates) ** 12 - 1) * 100).tolist()

# def calculate_activity_yields(self) -> Dict[str, List[float]]:  # Refactor reward_pool class. 
#     """Calculate yields over timeframe only for yield-type activities"""
#     months = range(self.timeframe)
#     yields = {}
    
#     # Only track yields for relevant activities
#     for name, activity in self.activity_config.activities.items():
#         if activity.reward_type == 'yield':
#             yields[name] = []
    
#     for month in months:
#         monthly_pool = self._calculate_monthly_fees_in_tokens(month)
        
#         for name, activity in self.activity_config.activities.items():
#             if activity.reward_type != 'yield':
#                 continue
                
#             if activity.current_value > activity.min_value:
#                 monthly_reward = monthly_pool * activity.reward_share
                
#                 # Get correct denominator based on activity
#                 if name == 'staking':
#                     denominator = self.reward_pool.total_supply * self.reward_pool.staking_rate
#                 elif name == 'liquidity':
#                     denominator = self.reward_pool.total_supply * self.reward_pool.lp_rate
#                 else:
#                     denominator = activity.current_value
                
#                 # Calculate monthly rate first
#                 monthly_rate = monthly_reward / denominator
                
#                 # Convert to APY with monthly compounding: (1 + r)^12 - 1
#                 apy = ((1 + monthly_rate) ** 12 - 1) * 100
#                 yields[name].append(apy)
#             else:
#                 yields[name].append(0)
                
#     return yields

# def _calculate_yield(self, monthly_reward: float, relevant_pool: float, month: int) -> float:
#     """Calculate monthly yield rate 
#     Args:
#         monthly_reward: Monthly rewards in tokens
#         relevant_pool: Pool size (staked or LP amount) for that month (when auto-restaking is enabled, this should include the rewards that have been restaked)
#         month: Current month in projection
#     Returns:
#         Monthly yield rate as percentage 
#     """
#     if relevant_pool == 0:
#         return 0
#     return monthly_reward / relevant_pool

# # USD Value Calculations
# def _calculate_dollar_value_rewards(self, token_rewards: Dict[str, float], token_price: float) -> Dict[str, float]:
#     """Convert token rewards to dollar value"""
#     return {
#         activity: reward * token_price 
#         for activity, reward in token_rewards.items()
#     }


    def update_metrics(self, epoch: int, external_state: Dict):
        """
        Update calculator metrics for a given epoch using real agent-driven values.
    
        Parameters:
            epoch (int): The current simulation epoch.
            external_state (Dict): Dictionary containing agent-level outputs for the epoch.
                Expected keys:
                    - 'monthly_volume': total tokens transferred (float)
                    - 'lp_amount': total tokens added to liquidity (float)
                    - 'num_miners': number of agents with role 'miner' (int)
                    - 'num_bundlers': number of agents with role 'bundler' (int)
                    - 'agent_objects': list of agent instances for liquidity/staking (List[Agent])
        """
        # Unpack external input
        volume = external_state.get("monthly_volume", 0.0)
        lp_amount = external_state.get("lp_amount", 0.0)
        num_miners = external_state.get("num_miners", 0)
        num_bundlers = external_state.get("num_bundlers", 0)
        agent_objects = external_state.get("agent_objects", [])
    
        # Ensure required previous state entries exist
        prev_epoch = epoch - 1
        prev_total_released = self.metrics.get('total_released', {}).get(prev_epoch, 0.0)
        if 'releases' not in self.state:
            # Compute full vesting schedule once
            self.state['releases'] = self._calculate_release_schedule(self.config.initial_supply)

        # Use the correct release matrix
        releases = self.state['releases']

    
        # Prepare state dict for supply update
        supply_input = {
            'total_released': prev_total_released,
            'releases': releases,
            'agent_objects': agent_objects
        }
    
        # Get updated supply values (no emissions)
        supply_state = self._update_supply_state(month=epoch, state=supply_input)
        self.state.update(supply_state)
    
        # Store vesting values directly
        for key, value in supply_state.items():
            if key not in self.metrics:
                self.metrics[key] = {}
            self.metrics[key][epoch] = value
    
        # Extract from supply_state
        circulating_supply = supply_state['circulating_supply']
        total_supply = supply_state['total_supply']
        total_released = supply_state['total_released']
    
        # Compute derived metrics
        token_state = self._calculate_token_state(epoch, {
            'epoch': epoch,
            'monthly_volume_tokens': external_state['monthly_volume_tokens'],
            'monthly_volume_usd': external_state['monthly_volume_usd'],
            'total_fees': external_state['total_fees'],
            'lp_amount': lp_amount,
            'num_miners': num_miners,
            'num_bundlers': num_bundlers,
            'circulating_supply': circulating_supply,
            'total_rewards': external_state['total_rewards'],
            'total_supply': total_supply,
            'token_price': self.metrics['token_price'].get(epoch - 1, INITIAL_TOKEN_PRICE)
        })
        token_price = token_state['token_price']
        market_cap = circulating_supply * token_price
        fdv = total_supply * token_price
        total_velocity = (volume / circulating_supply) if circulating_supply > 0 else 0.0
        monthly_fees = external_state["total_fees"]  
        monthly_rewards = monthly_fees
    
        # APY Calculation
        if lp_amount > 0:
            raw_apy_lp = ((1 + (monthly_rewards / lp_amount)) ** 12 - 1) * 100 if lp_amount > 0 else 0.0

            # Apply exponential smoothing using λ from config
            prev_apy = self.metrics['apy_lp'].get(epoch - 1, raw_apy_lp)
            lambda_ = LP_APY_SMOOTHING_LAMBDA
            apy_lp = (1 - lambda_) * prev_apy + lambda_ * raw_apy_lp

        else:
            apy_lp = 0.0
    
        # Store all metrics
        self.metrics['monthly_tx_volume_tokens'][epoch] = volume
        self.metrics['lp_amount'][epoch] = lp_amount
        self.metrics['miners'][epoch] = num_miners
        self.metrics['bundlers'][epoch] = num_bundlers
        self.metrics['token_price'][epoch] = token_price
        self.metrics['market_cap'][epoch] = market_cap
        self.metrics['fdv'][epoch] = fdv
        print(f"✅ Writing circ_supply to metrics at epoch {epoch}: {circulating_supply}")
        self.metrics['circulating_supply'][epoch] = circulating_supply
        self.metrics['total_supply'][epoch] = total_supply
        self.metrics['total_velocity'][epoch] = total_velocity
        self.metrics['monthly_fees'][epoch] = monthly_fees
        self.metrics['monthly_rewards'][epoch] = monthly_rewards
        self.metrics['apy_lp'][epoch] = apy_lp
        self.metrics['fee_sustainability'][epoch] = 1.0
        self.metrics['effective_supply'][epoch] = circulating_supply - lp_amount
        self.metrics['effective_supply_ratio'][epoch] = (
            self.metrics['effective_supply'][epoch] / circulating_supply if circulating_supply > 0 else 0.0
        )
        maturity = total_released / total_supply if total_supply > 0 else 0.0
        self.metrics['maturity'][epoch] = maturity



    
     
    # All-in-one
# def projections(self) -> Dict[str, List[float]]:
#     """Calculate all network metrics over time with adoption-based growth"""

#     # Initialize state tracking
#     state = {
#         'total_supply': self.config.initial_supply,
#         'cumulative_emission': 0,
#         'total_released': 0,
#         'circulating_supply': 0,
#         'staked_amount': 0,
#         'lp_amount': 0,
#         'total_rewards': 0,
#         'token_price': self.config.token_price,
#         'releases': self._calculate_release_schedule(self.config.initial_supply),
#         'yields': {'staking': 0, 'liquidity': 0},
#     }

#     # Initialize RewardPool
#     reward_pool = RewardPool()

#     # Initialize metrics arrays
#     metrics = {
#         'total_supply': [],
#         'circulating_supply': [],
#         'effective_supply': [],
#         'staked_amount': [],
#         'lp_amount': [], 
#         'token_price': [],
#         'token_demand': [],
#         'native_volume': [],
#         'sc_volume': [],
#         'monthly_tx_volume_tokens': [],
#         'monthly_tx_volume_usd': [],
#         'monthly_fees': [],
#         'security_ratio': [],
#         'fee_sustainability': [],
#         'tx_velocity': [],
#         'reward_velocity': [],
#         'total_velocity': [],
#         # 'network_growth_rate': [],
#         'emission': [],
#         'monthly_rewards': [],
#         'total_rewards': [],
#         'market_cap': [],
#         'fdv': [],
#         'miners': [],
#         'bundlers': [],
#         'wallets': [], 
#         'effective_supply_ratio': [],
#         'transaction_growth': [],
#         'apy_lp': [],
#         'activity_rewards_tokens': {name: [] for name in self.activity_config.activities.keys()},
#         'activity_rewards_dollars': {name: [] for name in self.activity_config.activities.keys()},
#         'yields': {'staking': [], 'liquidity': []},
#     }

#     # Main simulation loop
#     for month in range(self.num_months):
#         # 1. Calculate current adoption state
#         adoption = self._calculate_adoption(month)
#         metrics['miners'].append(adoption['miners'])
#         metrics['bundlers'].append(adoption['bundlers'])
#         metrics['wallets'].append(adoption['users'])
#         state['monthly_volume'] = None

#         # 2. Update supply state (vesting + inflation)
#         supply_state = self._update_supply_state(month, state)
#         state.update(supply_state)

#         # 3. Calculate token metrics with new supply
#         token_state = self._calculate_token_state(month, state)
#         state.update(token_state)

#         # 4. Calculate rewards based on transactions
#         rewards = float(reward_pool.calculate_monthly_rewards(state, self.config))
#         state['monthly_rewards'] = rewards
#         state['total_rewards'] += rewards
#         state['monthly_fees'] = state['monthly_rewards'] - state['emission']

#         # 5. Calculate distributions and update staking/LP
#         distributions = reward_pool.calculate_distributions(rewards, self.activity_config)
#         yields = reward_pool.calculate_yields(distributions, state, self.activity_config)

#         for activity_name, reward_tokens in distributions.items():
#             reward_usd = reward_tokens * token_price
#             self.metrics['activity_rewards_tokens'][activity_name][epoch] = reward_tokens
#             self.metrics['activity_rewards_dollars'][activity_name][epoch] = reward_usd

#         metrics['yields']['liquidity'].append(state['yields'].get('liquidity', 0))

#         # 6. Store post-activity updates
#         state.update({
#             'distributions': distributions,
#             'yields': yields,
#             'monthly_fees': state['monthly_volume'] * self.config.fee_rate
#         })

#         # 7. Store all metrics for this month
#         for key in ['total_supply', 'circulating_supply', 'effective_supply',
#                     'lp_amount', 'token_price', 'token_demand', 
#                     'native_volume', 'sc_volume', 'monthly_tx_volume_tokens', 
#                     'monthly_tx_volume_usd', 'tx_velocity', 'reward_velocity', 'monthly_fees',
#                     'total_velocity', 'monthly_rewards', 'total_rewards']:
#             metrics[key].append(state[key])

#         for name in self.activity_config.activities.keys():
#             metrics['activity_rewards_tokens'][name].append(state['distributions'][name])
#             metrics['activity_rewards_dollars'][name].append(
#                 state['distributions'][name] * state['token_price']
#             )

#         metrics['fee_sustainability'].append(
#             state['monthly_fees'] / state['emission'] if state['emission'] > 0 else 0
#         )

#         metrics['market_cap'].append(state['circulating_supply'] * state['token_price'])
#         metrics['fdv'].append(state['total_supply'] * state['token_price'])
#         metrics['effective_supply_ratio'].append(state['effective_supply'] / state['circulating_supply'])
#         metrics['transaction_growth'].append(
#             (state['monthly_volume'] - metrics['monthly_tx_volume_usd'][-1]) / metrics['monthly_tx_volume_usd'][-1] 
#             if month > 0 else 0
#         )
#         metrics['apy_lp'].append(state['yields'].get('liquidity', 0))

#     return {
#         **{k: metrics[k] for k in metrics.keys() - {'activity_rewards_tokens', 'activity_rewards_dollars', 'yields'}},
#         'activity_rewards_tokens': {k: v for k, v in metrics['activity_rewards_tokens'].items()},
#         'activity_rewards_dollars': {k: v for k, v in metrics['activity_rewards_dollars'].items()},
#         'token_releases': {k: v.tolist() for k, v in state['releases'].items()},
#         'yields': metrics['yields']
#     }


    # def project_health_metrics_over_time(self) -> List[HealthMetrics]:
    #     """Project health metrics over simulation period"""
    #     return [self.calculate_health_metrics(month) 
    #             for month in range(self.num_months)]


    

def validate_reward_shares():
    pass
          
def validate_supply():
    pass
