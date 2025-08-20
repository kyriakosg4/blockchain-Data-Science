from dataclasses import asdict, dataclass, field
import itertools
from warnings import warn
from typing import Dict, DefaultDict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from states import SystemState


MAX_MATRIX_ELEMENTS = 7500 # Safe limit for most local computers

@dataclass 
class RolePerformanceMetrics:
    """Track performance metrics for each strategy within a role"""
    total_value_generated: float = 0.0
    total_fees_paid: float = 0.0
    total_rewards_earned: float = 0.0
    transactions_processed: int = 0
    activities_completed: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    efficiency_score: float = 0.0

@dataclass
class NetworkMetrics:
    """Tracks key performance indicators from Incentiv model"""
    # Token Velocity Metrics
    transaction_volume: float = 0.0
    unique_senders: set = field(default_factory=set)
    unique_recipients: set = field(default_factory=set)
    _history_counter: int = field(default=0)
    token_velocity: float = 0.0
    
    # Activity Metrics  
    activities_by_role: DefaultDict[str, DefaultDict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    fees_collected: float = 0.0
    rewards_distributed: float = 0.0
    
    # Role Participation
    role_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    role_balances: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    
    # Historical Metrics (per epoch/timestep)
    historical_metrics: DefaultDict[int, Dict] = field(
        default_factory=lambda: defaultdict(dict)
    )

    strategy_performance: DefaultDict[str, Dict[str, RolePerformanceMetrics]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(RolePerformanceMetrics))
    )
    
    parameter_variations: DefaultDict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    
    def update_from_cadcad_policy_input(self, policy_input: Dict, system_state: SystemState, substep: int) -> None:
        """FOR CADCAD ONLY!
        Update metrics based on policy outputs before state updates
        Complements update_from_state by tracking metrics at each timestep"""
        # Process current timestep transactions
        processed_txs = policy_input.get('processed_transactions', [])
        if processed_txs:
            self.transaction_volume += sum(tx['amount'] for tx in processed_txs)
            self.fees_collected += sum(tx['fee'] for tx in processed_txs)
            
            # Track participants
            self.unique_senders.update(tx['sender'] for tx in processed_txs)
            self.unique_recipients.update(tx['recipient'] for tx in processed_txs)
        
        # Process current timestep rewards
        rewards = policy_input.get('rewards', {})
        if rewards:
            self.rewards_distributed += sum(rewards.values())
        
        # Update metrics from current state
        self.update_from_state(system_state, substep)

    def track_strategy_performance(self, role: str, strategy_name: str, state: SystemState) -> None:
        """Track performance metrics for a specific strategy"""
        perf = self.strategy_performance[role][strategy_name]
        
        # Update base metrics
        perf.total_value_generated += state.get_value_generated(role, strategy_name)
        perf.total_fees_paid += state.get_fees_paid(role, strategy_name)
        perf.total_rewards_earned += state.get_rewards_earned(role, strategy_name)
        
        # Update activity counts
        for activity, count in state.get_activities_performed(role, strategy_name).items():
            perf.activities_completed[activity] += count
            
        # Calculate efficiency
        perf.efficiency_score = self._calculate_efficiency_score(perf)

    def update_from_state(self, state: SystemState, epoch: int) -> None:
        """Update metrics from current system state"""
        self._update_role_metrics(state)
        self._update_activity_metrics(state)
        self._update_token_metrics(state)
        self._store_historical(epoch)

    def _update_role_metrics(self, state: SystemState) -> None:
        """Update role participation and balance metrics"""
        self.role_counts.clear()
        self.role_balances.clear()
        
        for role_name, distribution in state.role_distribution.items():
            self.role_counts[role_name] = distribution.count
            self.role_balances[role_name] = distribution.total_staked
    
    def _update_activity_metrics(self, state: SystemState) -> None:
        """Update activity-related metrics"""
        self.activities_by_role.clear()
        
        for agent_id, agent_state in state.agents.items():
            if agent_state.role:
                for activity, count in agent_state.activities_performed.items():
                    self.activities_by_role[agent_state.role][activity] += count
    
    def _update_token_metrics(self, state: SystemState) -> None:
        """Update token velocity and distribution metrics"""
        total_balance = sum(agent.balance for agent in state.agents.values())
        if total_balance > 0:
            self.token_velocity = self.transaction_volume / total_balance
    
    def _store_historical(self, epoch: Optional[int] = None) -> None:
        """Store current metrics for historical analysis"""

        tracking_id = epoch if epoch is not None else self._history_counter
        self._history_counter += 1 if epoch is None else 0

        base_metrics = {
            'token_velocity': self.token_velocity,
            'total_fees': self.fees_collected,
            'total_rewards': self.rewards_distributed,
            'role_counts': dict(self.role_counts),
            'role_balances': dict(self.role_balances),
            'activities': {role: dict(activities) 
                        for role, activities in self.activities_by_role.items()}
        }


        
        strategy_metrics = {
            'strategy_performance': {
                role: {strategy: perf.__dict__ if hasattr(perf, '__dict__') else {'value': perf} for strategy, perf in role_perf.items()}for role, role_perf in self.strategy_performance.items()
                },
            'parameter_state': {
                param: values[-1] if values else None
                for param, values in self.parameter_variations.items()
            }
        }
        
        self.historical_metrics[tracking_id] = {**base_metrics, **strategy_metrics}

    def get_current_metrics(self) -> Dict:
        """Get current network metrics"""
        return {
            # Value Fee Metrics (from Incentiv model)
            'fee_distribution_efficiency': self.rewards_distributed / self.fees_collected if self.fees_collected > 0 else 0,
            
            # Participation Metrics
            'role_participation': {
                role: count / sum(self.role_counts.values())
                for role, count in self.role_counts.items()
            },
            
            # Token Velocity
            'token_velocity': self.token_velocity,
            'unique_participants': len(self.unique_senders | self.unique_recipients),
            
            # Activity Metrics
            'activities_per_role': {
                role: dict(activities)
                for role, activities in self.activities_by_role.items()
            }
        }

    def get_historical_analysis(self, num_epochs: int = 10) -> Dict:
        """Get trend analysis for recent epochs"""
        recent_epochs = sorted(self.historical_metrics.keys())[-num_epochs:]
        if not recent_epochs:
            return {}
            
        return {
            'velocity_trend': [
                self.historical_metrics[epoch]['token_velocity']
                for epoch in recent_epochs
            ],
            'fee_efficiency_trend': [
                self.historical_metrics[epoch]['total_rewards'] /
                self.historical_metrics[epoch]['total_fees']
                if self.historical_metrics[epoch]['total_fees'] > 0 else 0
                for epoch in recent_epochs
            ],
            'role_distribution_trend': {
                role: [
                    self.historical_metrics[epoch]['role_counts'].get(role, 0)
                    for epoch in recent_epochs
                ]
                for role in self.role_counts
            }
        }
    
    def generate_payoff_matrices(self, state: SystemState) -> Dict[str, np.ndarray]:
        """Generate payoff matrices for Nash equilibrium analysis"""
        matrices = {}
        
        # Get all strategies for each role
        role_strategies = {
            role: list(self.strategy_performance[role].keys())
            for role in self.role_counts.keys()
        }
        
        # Generate matrix dimensions and check for large strategy space
        dims = [len(strategies) for strategies in role_strategies.values()]
 
        if np.prod(dims) > MAX_MATRIX_ELEMENTS:
            warn(f"Large strategy space detected: {np.prod(dims)} elements")
        
        # Create payoff matrix for each role
        for role in self.role_counts.keys():
            payoff_matrix = np.zeros(dims)
            
            # Fill matrix with normalized performance metrics
            for idx, strategy_combo in enumerate(itertools.product(*role_strategies.values())):
                idx_multi = np.unravel_index(idx, dims)
                payoff = self._calculate_strategy_payoff(role, strategy_combo)
                payoff_matrix[idx_multi] = payoff
                
            matrices[role] = payoff_matrix
            
        return matrices
    
    def track_parameter_variation(self, parameter: str, value: float) -> None:
        """Track changes in system parameters"""
        self.parameter_variations[parameter].append(value)
        
    def get_parameter_response(self, parameter: str) -> Dict[str, List[float]]:
        """Get system response metrics for parameter variations"""
        responses = {}
        param_values = self.parameter_variations[parameter]
        
        # Calculate key metrics for each parameter value
        for metric in ['efficiency', 'participation', 'rewards']:
            metric_values = [
                self.historical_metrics[t].get(f'{metric}_score', 0.0)
                for t in range(len(param_values))
            ]
            responses[metric] = metric_values
            
        return responses
    
    def _calculate_efficiency_score(self, perf: RolePerformanceMetrics) -> float:
        """Calculate efficiency score for a strategy"""
        if perf.total_value_generated == 0:
            return 0.0
            
        return (perf.total_rewards_earned - perf.total_fees_paid) / perf.total_value_generated
        
    def _calculate_strategy_payoff(self, role: str, strategy_combo: Tuple[str, ...]) -> float:
        """Calculate payoff for a role's strategy given other roles' strategies"""
        perf = self.strategy_performance[role][strategy_combo[list(self.role_counts.keys()).index(role)]]
        
        # Base payoff on efficiency and total value
        payoff = perf.efficiency_score * perf.total_value_generated
        
        # Add role-specific adjustments
        if role == 'validator':
            payoff *= (1 + perf.activities_completed.get('validate_block', 0) / 100)
        elif role == 'bundler':
            payoff *= (1 + perf.activities_completed.get('bundle_transactions', 0) / 1000)
            
        return payoff