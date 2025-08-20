from typing import Dict, List
from simulation_framework.agents import Agent, ActivityLevel
import numpy as np

# Define archetypes for each role with input metrics
def generate_agents_from_archetypes(
    archetype_definitions: Dict[str, List[Dict]],
    number_of_agents: int,
    role_distribution: Dict[str, int],
    seed_offset: int = 0,
    variation_std: float = 0.08
) -> Dict[str, Agent]:
    """
    Generate seeded agents based on predefined archetypes with slight variation.

    Args:
        archetype_definitions: {role: [{name, metrics}]}
        role_distribution: {role: number of agents}
        number_of_agents: total number of agents to create
        seed_offset: base seed for reproducibility
        variation_std: std deviation as % of each metric (default 5%)

    Returns:
        {agent_id: Agent instance}
    """
    agents = {}
    agent_counter = 0

    for role, archetypes in archetype_definitions.items():
        share_or_count = role_distribution.get(role, 0)
        count = int(share_or_count * number_of_agents) if isinstance(share_or_count, float) and 0 < share_or_count <= 1 else int(share_or_count)
        if count == 0:
            continue

        num_archetypes = len(archetypes)
        per_archetype = count // num_archetypes


        for i, archetype in enumerate(archetypes):
            for j in range(per_archetype):
                agent_id = f"{role}_{i}_{j}"
                seed = seed_offset + agent_counter
                rng = np.random.default_rng(seed) #replace with seeded random later
                name = archetype.get("name", None)

                # Apply slight variation to metrics
                noisy_metrics = {
                    k: v * (1 + rng.normal(0, variation_std))
                    for k, v in archetype.items()
                    if isinstance(v, (int, float)) and k != 'initial_balance'
                }
                noisy_metrics['initial_balance'] = archetype.get("initial_balance", 100)

                agent = Agent(
                    agent_id=agent_id,
                    initial_balance=noisy_metrics['initial_balance'],
                    choice_engine_seed=seed,
                    role=role,
                    name=name
                )
                agent.metrics = noisy_metrics  # Attach metrics for scoring

                agents[agent_id] = agent
                agent_counter += 1

    return agents
