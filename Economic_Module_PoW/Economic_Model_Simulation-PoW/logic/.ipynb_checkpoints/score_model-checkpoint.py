from typing import Dict
from sim_config import TIER_THRESHOLDS, REWARD_SPLITS, TIER_REWARD_SHARES
import numpy as np
from collections import defaultdict



# Ensure tier shares sum to 1.0
assert abs(sum(TIER_REWARD_SHARES.values()) - 1.0) < 1e-6, "Tier reward shares must sum to 1.0"


# === 1. Compute raw activity score for a single agent ===
def compute_raw_activity_score(agent, activity_weights, role_fixed_weights):
    """
    Computes the raw activity score A_u^(t) based on:
    - This epoch's transfer activity
    - Cumulative liquidity added
    - Miner/bundler activity using fixed weights for those roles
    """

    # Standard roles
    transfer_score = activity_weights.get("transfer_tokens", 0.0) * agent.activities_performed.get("transfer_tokens", 0.0)
    liquidity_score = activity_weights.get("provide_liquidity", 0.0) * agent.total_liquidity_provided

    bundling_score = 0.0
    mining_score = 0.0

    if agent.role == "bundler":
        bundled_tx_count = getattr(agent, "transactions_bundled_this_epoch", 0)
        bundling_score = role_fixed_weights.get("bundle_transactions", 0.0) * bundled_tx_count

    if agent.role == "miner":
        activity_level = getattr(agent, "updated_activity_levels", {}).get("mine_block", getattr(agent, "activity_level", 0.0))
        max_blocks = getattr(agent.strategy, "max_blocks", 24)  # fallback default
        try:
            activity_value = float(getattr(activity_level, "value", activity_level))
        except Exception:
            print(f"❌ Invalid activity_level for agent {agent.id}: {activity_level}")
            activity_value = 0.0

        mining_score = role_fixed_weights.get("mine_block", 0.0) * activity_value * max_blocks


    return transfer_score + liquidity_score + bundling_score + mining_score



def normalize_scores(score_dict):
    total = sum(score_dict.values())
    if total == 0:
        return {agent_id: 0.0 for agent_id in score_dict}
    return {agent_id: score / total for agent_id, score in score_dict.items()}

# === Apply smoothing ===
def apply_smoothing(prev_score, new_score, smoothing_lambda):
    return smoothing_lambda * prev_score + (1 - smoothing_lambda) * new_score

# SOS check again this function
#def is_eligible(agent, role):
    #filters = ELIGIBILITY_CRITERIA.get(role, {})
    #for k, expected in filters.items():
     #   actual = agent.get(k)
     #   if actual is None:
      #      return False
       # if isinstance(expected, (int, float)) and actual < expected:
       #     return False
       # if isinstance(expected, bool) and actual != expected:
        #    return False
   # return True
    

def assign_percentile_tier(sorted_scores, agent_id, role):
    """
    Assigns tier to an agent based on their percentile and the thresholds defined for their role in TIER_THRESHOLDS.
    Supports roles with 2 to 4 thresholds.
    """

    index = sorted_scores.index(agent_id)
    percentile = (index + 1) / len(sorted_scores)

    thresholds = TIER_THRESHOLDS.get(role)

    if thresholds is None:
        raise ValueError(f"No tier thresholds defined for role: {role}")

    num_thresholds = len(thresholds)

    if num_thresholds == 4:
        if percentile <= thresholds[0]:
            return "Tier0"
        elif percentile <= thresholds[1]:
            return "Tier1"
        elif percentile <= thresholds[2]:
            return "Tier2"
        elif percentile <= thresholds[3]:
            return "Tier3"
        else:
            return "Tier4"

    elif num_thresholds == 3:
        if percentile <= thresholds[0]:
            return "Tier0"
        elif percentile <= thresholds[1]:
            return "Tier1"
        elif percentile <= thresholds[2]:
            return "Tier2"
        else:
            return "Tier3"

    elif num_thresholds == 2:
        if percentile <= thresholds[0]:
            return "Tier0"
        elif percentile <= thresholds[1]:
            return "Tier1"
        else:
            return "Tier2"

    else:
        raise ValueError(f"Threshold list for role {role} must have 2, 3, or 4 thresholds.")



#def assign_percentile_tier(score, percentiles):
#    for i, p in enumerate(percentiles):
#        if score <= p:
#            return f"T{i+1}"
#    return f"T{len(percentiles)+1}"

# === Main function: compute scores for all agents ===

def compute_all_agent_scores(
    agents,
    activity_weights,
    role_fixed_weights,
    smoothing_lambda=0.6
):
    """
    Computes raw scores, normalizes per role (user, bundler, miner),
    applies smoothing, assigns tiers.
    """

    roles = set(agent.role for agent in agents.values())

    for role in roles:
        agents_in_role = {aid: agent for aid, agent in agents.items() if agent.role == role}

        raw_scores = {}
        for agent_id, agent in agents_in_role.items():
            raw_score = compute_raw_activity_score(agent, activity_weights, role_fixed_weights)
            raw_scores[agent_id] = raw_score

        normalized_scores = normalize_scores(raw_scores)

        for agent_id, agent in agents_in_role.items():
            smoothed_score = apply_smoothing(agent.score, normalized_scores.get(agent_id, 0.0), smoothing_lambda)
            agent.previous_score = agent.score
            agent.score = smoothed_score

        sorted_agents = sorted(normalized_scores.keys(), key=lambda aid: normalized_scores[aid], reverse=True)
        for agent_id, agent in agents_in_role.items():
            agent.tier = assign_percentile_tier(sorted_agents, agent_id, role)

# def compute_all_scores(agent_pool):
#     results = []
#     for role, agents in agent_pool.items():
#         scores = []
#         for agent in agents:
#             score = compute_score(agent, role)
#             agent['score'] = score
#             agent['eligible'] = is_eligible(agent, role)
#             scores.append(score)
# 
#         if scores:
#             percentiles = np.percentile(scores, [p*100 for p in ELIGIBILITY_CRITERIA.get(role, [])])
#             for agent in agents:
#                 agent['tier'] = assign_percentile_tier(agent['score'], percentiles)
#                 results.append(agent)
# 
#     return results


# --------------------------------------------
# Legacy scoring functions (no longer used)
# --------------------------------------------

# # def compute_agent_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
# #     """
# #     Compute a weighted score for an agent based on input metrics.
# #
# #     Args:
# #         metrics: Dict of agent features (e.g., tx_count, volume)
# #         weights: Dict of feature weights (must match keys in metrics)
# #
# #     Returns:
# #         Weighted score (float)
# #     """
# #     score = 0.0
# #     for k, weight in weights.items():
# #         value = metrics.get(k, 0)
# #         score += value * weight
# #     return score

# # def normalize_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
# #     """
# #     Normalize a dict of scores so that values sum to 1.
# #     Useful for proportional reward splits.
# #
# #     Args:
# #         score_dict: {agent_id: raw_score}
# #
# #     Returns:
# #         {agent_id: normalized_score}
# #     """
# #     total = sum(score_dict.values())
# #     if total == 0:
# #         return {k: 0 for k in score_dict}
# #     return {k: v / total for k, v in score_dict.items()}

# # def compute_scores_for_role(agent_metrics: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Dict[str, float]:
# #     """
# #     Compute scores for all agents within a role.
# #
# #     Args:
# #         agent_metrics: {agent_id: {metric_name: value}}
# #         weights: {metric_name: weight}
# #
# #     Returns:
# #         {agent_id: weighted_score}
# #     """
# #     return {
# #         agent_id: compute_agent_score(metrics, weights)
# #         for agent_id, metrics in agent_metrics.items()
# #     }


def distribute_rewards(
    agents,
    total_reward_pool,
    role_reward_shares=REWARD_SPLITS,
    epoch=None
):
        """
        Distributes rewards to agents by role and by tier, based on their smoothed scores.
        - role_reward_shares: e.g., {'user': 0.4, 'bundler': 0.3, 'miner': 0.3}
        - TIER_REWARD_SHARES: e.g., {'Tier0': 0.35, 'Tier1': 0.30, ...}
        """
        from collections import defaultdict
    
        roles = set(agent.role for agent in agents.values())
        role_totals = defaultdict(float)
    
        for role in roles:
            # Get all agents in this role
            agents_in_role = {
                aid: agent for aid, agent in agents.items()
                if agent.role == role
            }
    
            role_pool = total_reward_pool * role_reward_shares.get(role, 0.0)
    
            # Filter agents with score > 0 by tier
            active_tier_agents = defaultdict(list)
            for agent in agents_in_role.values():
                if agent.score > 0:
                    active_tier_agents[agent.tier].append(agent)
    
            # Compute initial tier pools (only for tiers with active agents)
            tier_pools = {}
            for tier, share in TIER_REWARD_SHARES.items():
                if active_tier_agents[tier]:
                    tier_pools[tier] = role_pool * share
                else:
                    tier_pools[tier] = 0.0
    
            # Redistribute unused tier pool to active tiers
            unused_pool = role_pool - sum(tier_pools.values())
            active_tiers = [t for t in TIER_REWARD_SHARES if active_tier_agents[t]]
            if active_tiers:
                total_active_share = sum(TIER_REWARD_SHARES[t] for t in active_tiers)
                for t in active_tiers:
                    tier_pools[t] += unused_pool * (TIER_REWARD_SHARES[t] / total_active_share)
    
            # Distribute rewards to agents
            for tier, agents_in_tier in active_tier_agents.items():
                tier_pool = tier_pools[tier]
                total_score = sum(agent.score for agent in agents_in_tier)
                for agent in agents_in_tier:
                    reward_share = (agent.score / total_score) * tier_pool if total_score > 0 else 0.0
                    agent.collect_reward(reward_share)
                    role_totals[role] += reward_share
                    if epoch is not None:
                        agent.rewards_earned_history[epoch] = reward_share
    
        # Save role totals (optional summary)
        if epoch is not None:
            system = next(iter(agents.values()))._system_state
            if not hasattr(system, "role_rewards_summary"):
                system.role_rewards_summary = defaultdict(dict)
            system.role_rewards_summary[epoch] = dict(role_totals)
    
