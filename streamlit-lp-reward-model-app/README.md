# LP Reward Model

An interactive Streamlit app to simulate and visualize Liquidity Provider (LP) reward programs in DeFi, covering **locked APR rewards** and **flexible-liquidity multipliers** across staged timelines.

## What this app models
- **Locked APR (for lockers):** APR specifies the share of the reward fund someone earns **if they keep funds locked for the full stage**. Rewards unlock over time using a **quadratic curve**:
  - For month *i* in a stage of *M* months, unlocked APR = `APR * (i/M)^2`.
  - Early exit earns less; full APR is reached only at month *M*.
- **Penalty (optional, for lockers only):** If a user unlocks before the stage ends, a penalty factor can reduce the unlocked APR, reinforcing commitment.
- **Flexible LP rewards (for non-lockers):** Rewards are scaled by **bucket-based multipliers** (e.g., 0–500, 500–5K, 5K–100K, …) that reflect contribution size during each stage.
- **Staged timeline:** Four stages (Genesis, Bootstrap, Growth, Mature) with configurable durations (must sum to 112 weeks) and their own parameters.

> **Mindset:** Encourage long-term participation (via time-based APR for lockers) while fairly compensating flexible liquidity using transparent, bucketed multipliers.

## Features
- Configure stage durations (must total 112 epochs/weeks).
- Set **APR per stage** (for lockers).
- Define **liquidity bucket multipliers** and a **fixed reward share** per stage (for flexible LPs).
- Optional **penalty** on early unlocks (applies to locked APR only).
- Interactive tables and Plotly charts for APR unlocks and flexible LP rewards.
- Scenario exploration with separate tabs for APR, Liquidity, and combined Reward Simulation.

## Tech stack
- **UI:** Streamlit  
- **Data:** Pandas  
- **Charts:** Plotly

## Dependencies
Install directly:
```bash
pip install streamlit pandas plotly

