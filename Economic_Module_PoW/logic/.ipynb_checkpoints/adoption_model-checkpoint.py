import numpy as np
import pandas as pd


def compute_micro_utility(agent_list):
    total_activity = sum(
        sum(agent.activity_count_history.get(agent._system_state.current_epoch, {}).values())
        for agent in agent_list
    )
    total_max = sum(
        getattr(agent, "max_possible_activities", 0)
        for agent in agent_list
    )
    return total_activity / total_max if total_max > 0 else 0.0
    


def compute_micro_adoption_growth(utility, current_adoption, alpha=0.3, max_adoption=1.0):
    growth = alpha * utility * (1 - current_adoption / max_adoption)
    updated_adoption = min(current_adoption * (1 + growth), max_adoption)
    return growth, updated_adoption


def get_network_growth_rate(adoption, beta, max_adoption, alpha=2):
    """
    Calculate the network growth rate (dA(t)/dt) based on adoption dynamics and utility derived from adoption.

    Parameters:
        adoption (float): Current adoption level, A(t).
        beta (float): Growth sensitivity constant, affects the rate of adoption.
        max_adoption (float): Maximum potential adoption level (A_max).
        alpha (float): Exponent to calculate utility (default is 2 for Metcalfe's Law).

    Returns:
        utility (float): Utility derived from adoption.
        growth_rate (float): Rate of change of adoption (dA(t)/dt).
    """
    if adoption <= 0 or adoption > max_adoption:
        raise ValueError("Adoption out of bounds, must be higher than 0 and lower than max.")

    utility = adoption ** alpha
    growth_rate = beta * utility * (1 - adoption / max_adoption)
    return utility, growth_rate

def simulate_growth(epochs=36, initial_adoption=0.05, max_adoption=1.0, beta=0.5, alpha=2, mode="exponential"):
    """
    Simulate network adoption growth over time.

    Parameters:
        epochs (int): Number of time steps to simulate.
        initial_adoption (float): Initial adoption level.
        max_adoption (float): Maximum adoption possible.
        beta (float): Sensitivity constant.
        alpha (float): Utility exponent.
        mode (str): 'linear' or 'exponential'

    Returns:
        pd.DataFrame: DataFrame with columns for epoch, adoption, utility, and growth_rate
    """
    adoption = initial_adoption
    adoption_history = []
    utility_history = []
    growth_rate_history = []
    epoch = []

    for t in range(epochs):
        epoch.append(t)
        utility, growth_rate = get_network_growth_rate(adoption, beta, max_adoption, alpha)

        adoption_history.append(adoption)
        utility_history.append(utility)
        growth_rate_history.append(growth_rate)

        if mode == "linear":
            adoption += growth_rate
        elif mode == "exponential":
            adoption *= (1 + growth_rate)
        else:
            raise ValueError("Mode must be 'linear' or 'exponential'")

        # Prevent overshooting
        adoption = min(adoption, max_adoption)

    return pd.DataFrame({
        "epoch": epoch,
        "adoption": adoption_history,
        "utility": utility_history,
        "growth_rate": growth_rate_history
    })
