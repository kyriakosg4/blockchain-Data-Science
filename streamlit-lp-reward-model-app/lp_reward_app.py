# Libraries required
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def main():

    # Penalty is a proposal that can drive the behavior of the lps to stay longer and help the ecosystem in general
    st.sidebar.header("Penalty Configuration")
    penalty_percent = st.sidebar.slider("Penalty for ATP", min_value = 0.0, max_value=50.0, value=0.15, step = 5.0)
    penalty_factor = 1 - (penalty_percent/100)

    st.sidebar.markdown("---")
    st.sidebar.header("Locked amount + Liquidity")

    # Define the total amount (locked + liquidity)
    locked_liq_amount = st.sidebar.number_input(
        "Total amount",
        min_value=0,
        value=100_000_000,
        step = 100_000,
        format="%d"
    )

    # locked amount, % of the total amount 
    locked_percent = st.sidebar.slider(
        "Locked Share (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )

    # the remaining equals to the liquidity pool 
    lp_percent = 100.0 - locked_percent

    st.title("Liquidity Rewwards Strucure")

    st.markdown("""
    Adjust the number of months for each reward stage. 
    Each month is equal to 4 weeks. Total must sum up to 112 epochs.
    """)

    TOTAL_EPOCHS = 112
    DEFAULT_EPOCH = 28
    WEEKS = 4
    
    def show_multiplier_note(stage_name):
        st.markdown(
            f"<span style='font-size: 0.9em; color: gray;'>"
            f"Note: This multiplier applies only to participants who were active in the <b>{stage_name}</b> stage."
            "</span>",
            unsafe_allow_html=True
        )
    
    # project is splitted into different timeframes
    with st.expander("Timeframe Configuration", expanded=False):
        genesis_weeks = st.slider("Genesis", min_value = 0, max_value = TOTAL_EPOCHS, value = DEFAULT_EPOCH, step = 4)
        bootstrap_weeks = st.slider("Bootstrap",min_value = 0, max_value = TOTAL_EPOCHS, value = DEFAULT_EPOCH, step = 4)
        growth_weeks = st.slider("Growth",min_value = 0, max_value = TOTAL_EPOCHS, value = DEFAULT_EPOCH, step = 4)
        mature_weeks = st.slider("Mature",min_value = 0, max_value = TOTAL_EPOCHS, value = DEFAULT_EPOCH, step = 4)

    sum_epochs = genesis_weeks + bootstrap_weeks + growth_weeks + mature_weeks

    genesis_months = genesis_weeks / WEEKS  # returns float (for integer use //)
    bootstrap_months = bootstrap_weeks/WEEKS
    growth_months = growth_weeks/WEEKS
    mature_months = mature_weeks/WEEKS

    st.markdown(f"The sum of epochs is {sum_epochs}")

    if sum_epochs != TOTAL_EPOCHS:
        st.error("Total number of epochs must be 112")
    else:
        st.success("Epoch Distribution is valid")

    # APR defines the % of the total reward that the agent will gain and is dependent to the amount of the investment 
    # concerns the agensts that will lock their funds
    # multipliers is a fixed parameter which differs according to the timestage and the total investment (funds buckets)
    with st.expander("APR Parameters", expanded = False):
        genesis_apr = st.number_input("Genesis - APR (%):", min_value=0.0, max_value=100.0, value=25.0, step = 0.5)

        bootstrap_apr = st.number_input("Bootstrap - APR (%):", min_value=0.0, max_value=100.0, value=15.0, step = 0.5)
        enable_boot_mult = st.checkbox("Enable APR Multiplier for Bootstrap Stage")
        if enable_boot_mult:
            multiplier_boot = st.number_input(
                "APR Multiplier:",
                min_value=0.25,
                max_value=3.0,
                value=1.5,
                step=0.25,
                key="boostrap_multiplier"
            )
            show_multiplier_note("Genesis")


        growth_apr = st.number_input("Growth - APR (%):", min_value=0.0, max_value=100.0, value=10.0, step = 0.5)
        enable_growth_mult = st.checkbox("Enable APR Multiplier for Growth Stage")
        if enable_growth_mult:
            multiplier_growth = st.number_input(
                "APR Multiplier:",
                min_value=0.25,
                max_value=3.0,
                value=1.25,
                step=0.25,
                key="growth_multiplier"
            )
            show_multiplier_note("Bootstrap")

        mature_apr = st.number_input("Mature - APR (%):", min_value=0.0, max_value=100.0, value=7.5, step = 0.5)
        enable_mature_mult = st.checkbox("Enable APR Multiplier for Mature Stage")
        if enable_mature_mult:
            multiplier_mature = st.number_input(
                "APR Multiplier:",
                min_value=0.25,
                max_value=3.0,
                value=1.25,
                step=0.25,
                key="mature_multiplier"
            )
            show_multiplier_note("Growth")


    st.subheader("Liquidity Pool Parameters")
    stages = ["Genesis", "Bootstrap", "Growth", "Mature"]
    buckets = ["0-500", "500-5K", "5K-100K", "100K-1M", "1M+"]
    default_val = [0.75, 1.0, 1.25, 1.5, 1.75]
    

    multipliers_by_stage = {}
    fixed_reward_share = {}


    for stage in stages:
        with st.expander(f"{stage} Reward Multipliers", expanded=False):
            multipliers_by_stage[stage] = []
            for i, label in enumerate(buckets):
                mult = st.number_input(f"{stage} Bucket {label}", min_value=0.25, max_value=3.0, value=default_val[i], step = 0.05)
                multipliers_by_stage[stage].append(mult)

            fixed_reward_share[stage] = st.slider(f"Fixed Reward share for {stage}:", min_value=0.2, max_value=5.0, value=1.5, step = 0.1)


    apr_tab, liquidity_tab, rewards_tab = st.tabs(['APR', "Liquidity", "Rewards"])
    with apr_tab:
        
        # APR tables showing the % that someone will be assigned according to the duration that locked, with or without penalty
        def generate_apr_tables(months, apr, multiplier=None, penalty_factor=1.0):
            """
            Generate an APR table with and without multiplier and penalty.
            - months: number of months
            - apr: base APR
            - multiplier: optional stage multiplier (defaults to 1.0)
            - penalty_factor: penalty reduction factor (e.g. 0.85 if 15% penalty)
            """
            if multiplier is None:
                multiplier = 1.0

            data = []

            for i in range(1, int(months) + 1):
                # Base APR formula (quadratic curve)
                base_apr = apr * (i / months) ** 2
                base_apr = round(base_apr, 2)

                # Multiplied APR
                multiplied_apr = round(base_apr * multiplier, 2)

                # Penalty APR
                if i == 1:
                    penalty_apr = 0
                    penalty_multiplied_apr = 0
                elif i == int(months) or penalty_factor == 1.0:
                    penalty_apr = base_apr
                    penalty_multiplied_apr = multiplied_apr
                else:
                    # Apply penalty factor to same formula (recompute from (i-1) month)
                    penalty_base = apr * ((i - 1) / months) ** 2
                    penalty_apr = round(penalty_base * penalty_factor, 2)
                    penalty_multiplied_apr = round(penalty_base * multiplier * penalty_factor, 2)

                data.append([
                    i,
                    base_apr,
                    multiplied_apr,
                    penalty_apr,
                    penalty_multiplied_apr
                ])

            return pd.DataFrame(data, columns=[
                "Month",
                "APR Unlocked (%)",
                "APR (Multiplier Applied) (%)",
                "APR Unlocked (%) (PENALTY)",
                "APR (Multiplier Applied) (%) (PENALTY)"
            ])

        st.subheader("Genesis APR Table")
        df_genesis = generate_apr_tables(genesis_months, genesis_apr, None, penalty_factor)
        st.dataframe(df_genesis, use_container_width=True, hide_index=True)

        st.subheader("Bootstap APR Table")
        boot_multiplier = multiplier_boot if "enable_boot_mult" in locals() and enable_boot_mult else None
        df_boot = generate_apr_tables(bootstrap_months, bootstrap_apr, boot_multiplier, penalty_factor)
        st.dataframe(df_boot, use_container_width=True, hide_index=True)

        st.subheader("Growth APR Table:")
        gr_multiplier = multiplier_growth if "enable_growth_mult" in locals() and enable_growth_mult else None
        df_growth = generate_apr_tables(growth_months, growth_apr, gr_multiplier, penalty_factor)
        st.dataframe(df_growth, use_container_width=True, hide_index=True)

        st.subheader("Mature APR Table:")
        mat_multiplier = multiplier_mature if "enable_mature_mult" in locals() and enable_mature_mult else None
        df_mature = generate_apr_tables(mature_months, mature_apr, mat_multiplier, penalty_factor)
        st.dataframe(df_mature, use_container_width=True, hide_index=True)

    
    # showing the tables for the multipliers that were defined by the user
    with liquidity_tab:
    

        def build_liquidity_table(stage, months):
            multipliers = multipliers_by_stage[stage]
            reward_share = fixed_reward_share[stage]/100
            rewards = [round(mult * reward_share * months * 100, 2) for mult in multipliers]
            return pd.DataFrame(
                {"Buckets" : buckets,
                 "Reward Multipliers" : multipliers,
                 "Rewards (%)" : rewards
                }
            )
        
        st.subheader("Genesis Liquidity Table")
        df_genesis_liq = build_liquidity_table("Genesis", genesis_months)
        st.dataframe(df_genesis_liq, use_container_width=True, hide_index=True)

        st.subheader("Bootstrap Liquidity Table")
        df_bootstrap_liq = build_liquidity_table("Bootstrap", bootstrap_months)
        st.dataframe(df_bootstrap_liq, use_container_width=True, hide_index=True)

        st.subheader("Growth Liquidity Table")
        df_growth_liq = build_liquidity_table("Growth", growth_months)
        st.dataframe(df_growth_liq, use_container_width=True, hide_index=True)

        st.subheader("Mature Liquidity Table")
        df_mature_liq = build_liquidity_table("Mature", mature_months)
        st.dataframe(df_mature_liq, use_container_width=True, hide_index=True)

    

    # with rewards_tab:
    #     st.markdown("### Reward Simulation")

    #     st.markdown(
    #         "<span style='font-size: 0.9em; color: gray;'>"
    #         "ℹ️ Minimum amount required to participate in locked rewards is <b>5,000</b> tokens."
    #         "</span>",
    #         unsafe_allow_html=True
    #     )


    #     stage_months_map = {
    #         "Genesis": int(genesis_months),
    #         "Bootstrap": int(bootstrap_months),
    #         "Growth": int(growth_months),
    #         "Mature": int(mature_months),
    #     }

    #     stage_apr_map = {
    #         "Genesis": df_genesis,
    #         "Bootstrap": df_boot,
    #         "Growth": df_growth,
    #         "Mature": df_mature,
    #     }

    #     bucket_range_map = {
    #         "0-500": (0, 500),
    #         "500-5K": (500, 5000),
    #         "5K-100K": (5000, 100000),
    #         "100K-1M": (100000, 1_000_000),
    #         "1M+": (1_000_000, 2_000_000),
    #     }

    #     reward_configs = []
    #     for i in range(5):
    #         with st.expander(f"Reward Set {i+1}", expanded=(i == 0)):
    #             col1, col2, col3 = st.columns(3)
    #             with col1:
    #                 stage = st.selectbox(f"Stage {i+1}", stages, key=f"stage_{i}")
    #             with col2:
    #                 bucket = st.selectbox(f"Bucket {i+1}", buckets, key=f"bucket_{i}")
    #             with col3:
    #                 pct = st.slider(f"Allocation % {i+1}", 0.0, 100.0, 20.0, step=1.0, key=f"alloc_{i}") / 100
    #             reward_configs.append((stage, bucket, pct))

    #     for idx, (stage, bucket, selected_percentage) in enumerate(reward_configs):
    #         st.markdown(f"#### 📊 Reward Breakdown for Set {idx+1} — {stage}, {bucket}")

    #         months = stage_months_map[stage]
    #         apr_table = stage_apr_map[stage]
    #         apr_cols = [
    #             "APR Unlocked (%)",
    #             "APR (Multiplier Applied) (%)",
    #             "APR Unlocked (%) (PENALTY)",
    #             "APR (Multiplier Applied) (%) (PENALTY)"
    #         ]

    #         bucket_min, bucket_max = bucket_range_map[bucket]
    #         avg_amount = (bucket_min + bucket_max) / 2
    #         bucket_index = buckets.index(bucket)
    #         multiplier = multipliers_by_stage[stage][bucket_index]
    #         reward_share = fixed_reward_share[stage]

    #         fig = go.Figure()

    #         # Locked rewards (4 types, if avg ≥ 5000)
    #         if avg_amount >= 5000:
    #             for col in apr_cols:
    #                 apr_series = apr_table[col].tolist()
    #                 rewards = []
    #                 for m in range(months):
    #                     apr_percent = apr_series[m] / 100
    #                     reward = locked_liq_amount * (locked_percent / 100) * selected_percentage * apr_percent
    #                     rewards.append(reward)
    #                 fig.add_trace(go.Bar(
    #                     x=list(range(1, months + 1)),
    #                     y=rewards,
    #                     name=col,
    #                     marker=dict(opacity=0.8)
    #                 ))

    #         # Select the correct liquidity table based on stage
    #         liq_table = {
    #             "Genesis": df_genesis_liq,
    #             "Bootstrap": df_bootstrap_liq,
    #             "Growth": df_growth_liq,
    #             "Mature": df_mature_liq
    #         }[stage]

    #         # Extract the final reward percent from the table for the selected bucket
    #         final_reward_percent = liq_table.loc[liq_table['Buckets'] == bucket, "Rewards (%)"].values[0] / 100

    #         # Compute per-month liquidity rewards using a linear ramp-up
    #         liquidity_rewards = []
    #         for m in range(1, months + 1):
    #             month_ratio = m / months
    #             reward = (
    #                 locked_liq_amount *
    #                 (lp_percent / 100) *
    #                 selected_percentage *
    #                 month_ratio *
    #                 final_reward_percent
    #             )
    #             liquidity_rewards.append(reward)

    #         fig.add_trace(go.Bar(
    #             x=list(range(1, months + 1)),
    #             y=liquidity_rewards,
    #             name="Liquidity Reward",
    #             marker=dict(opacity=0.6)
    #         ))

    #         fig.update_layout(
    #             title=f"Reward Distribution for Set {idx+1}",
    #             xaxis_title="Month",
    #             yaxis_title="Reward (tokens)",
    #             barmode="group",
    #             height=500,
    #             legend_title="Reward Type"
    #         )

    #         st.plotly_chart(fig, use_container_width=True)

    
    # Inside that tab, user can try multiple combinations of liquidity based on the bucket category and observe the bar plots
    # Each bar plot corresponds to a bucket category and presents the rewards the lp provider can obtain for both liquidity and locked occasions
    with rewards_tab:
        st.markdown("### Reward Simulation")

        st.markdown(
            "<span style='font-size: 0.9em; color: gray;'>"
            "ℹ️ Minimum amount required to participate in locked rewards is <b>5,000</b> tokens."
            "</span>",
            unsafe_allow_html=True
        )

        # Select stage
        stage = st.selectbox("Select Stage", stages, key="reward_stage")

        # Stage maps
        stage_months_map = {
            "Genesis": int(genesis_months),
            "Bootstrap": int(bootstrap_months),
            "Growth": int(growth_months),
            "Mature": int(mature_months),
        }
        stage_apr_map = {
            "Genesis": df_genesis,
            "Bootstrap": df_boot,
            "Growth": df_growth,
            "Mature": df_mature,
        }
        stage_liq_table_map = {
            "Genesis": df_genesis_liq,
            "Bootstrap": df_bootstrap_liq,
            "Growth": df_growth_liq,
            "Mature": df_mature_liq,
        }

        # Bucket data
        bucket_range_map = {
            "0-500": (0, 500),
            "500-5K": (500, 5000),
            "5K-100K": (5000, 100000),
            "100K-1M": (100000, 1_000_000),
            "1M+": (1_000_000, 2_000_000),
        }

        # Expander: Liquidity Allocation
        with st.expander("Liquidity Allocation", expanded=True):
            st.caption("Allocate % across liquidity buckets (uses total_amount × (1 - locked share))")
            liquidity_allocations = {}
            total_liq_pct = 0
            for bucket in buckets:
                pct = st.slider(f"Liquidity % for {bucket}", 0.0, 100.0, 0.0, step=1.0, key=f"liq_{bucket}")
                liquidity_allocations[bucket] = pct
                total_liq_pct += pct
            st.markdown(f"**Total Liquidity Allocation:** {total_liq_pct}%")
            if total_liq_pct != 100:
                st.error("Liquidity allocation must sum to 100%.")

        # Expander: Locked Allocation
        with st.expander("Locked Allocation", expanded=True):
            st.caption("Allocate % across locked buckets (uses total_amount × locked share)")
            locked_allocations = {}
            total_locked_pct = 0
            for bucket in buckets[2:]:
                pct = st.slider(f"Locked % for {bucket}", 0.0, 100.0, 0.0, step=1.0, key=f"locked_{bucket}")
                locked_allocations[bucket] = pct
                total_locked_pct += pct
            st.markdown(f"**Total Locked Allocation:** {total_locked_pct}%")
            if total_locked_pct != 100:
                st.error("Locked allocation must sum to 100%.")

        if total_liq_pct == 100 and total_locked_pct == 100:
            st.success("✅ Allocation configuration is valid.")
            months = stage_months_map[stage]
            apr_table = stage_apr_map[stage]
            liq_table = stage_liq_table_map[stage]

            for bucket in buckets:
                st.markdown(f"#### 📊 Reward Distribution for {bucket}")
                fig = go.Figure()

                # Locked reward (if eligible and configured)
                if bucket in locked_allocations:
                    bucket_min, bucket_max = bucket_range_map[bucket]
                    avg_amount = (bucket_min + bucket_max) / 2
                    if avg_amount >= 5000:
                        locked_pct = locked_allocations[bucket] / 100
                        for col in [
                            "APR Unlocked (%)",
                            "APR (Multiplier Applied) (%)",
                            "APR Unlocked (%) (PENALTY)",
                            "APR (Multiplier Applied) (%) (PENALTY)"
                        ]:
                            rewards = []
                            for m in range(months):
                                apr_percent = apr_table[col].iloc[m] / 100
                                reward = (
                                    locked_liq_amount *
                                    (locked_percent / 100) *
                                    locked_pct *
                                    apr_percent
                                )
                                rewards.append(reward)
                            fig.add_trace(go.Bar(
                                x=list(range(1, months + 1)),
                                y=rewards,
                                name=col,
                                marker=dict(opacity=0.8)
                            ))

                # Liquidity reward (always applicable)
                if bucket in liquidity_allocations:
                    final_reward_percent = liq_table.loc[liq_table['Buckets'] == bucket, "Rewards (%)"].values[0] / 100
                    liq_pct = liquidity_allocations[bucket] / 100
                    liquidity_rewards = []
                    for m in range(1, months + 1):
                        month_ratio = m / months
                        reward = (
                            locked_liq_amount *
                            (lp_percent / 100) *
                            liq_pct *
                            month_ratio *
                            final_reward_percent
                        )
                        liquidity_rewards.append(reward)

                    fig.add_trace(go.Bar(
                        x=list(range(1, months + 1)),
                        y=liquidity_rewards,
                        name="Liquidity Reward",
                        marker=dict(opacity=0.6)
                    ))

                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Reward (tokens)",
                    barmode="group",
                    height=500,
                    legend_title="Reward Type"
                )
                st.plotly_chart(fig, use_container_width=True)




if __name__ == "__main__":
    main()
        

