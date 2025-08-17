import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os


def visualize_userop_failures(
    sender_csv="data/summary_failures_per_sender.csv",
    bundler_csv="data/summary_failures_per_bundler.csv",
    paymaster_csv="data/summary_failures_per_paymaster.csv",
    output_prefix="final",  # used for filename suffixes
    start_block=None,
    end_block=None,
    start_time=None,
    end_time=None,
    output_dir="visualizations"
):
   

    os.makedirs(output_dir, exist_ok=True)  # ✅ Ensure folder exists

    # Load data
    sender_df = pd.read_csv(sender_csv)
    bundler_df = pd.read_csv(bundler_csv)
    paymaster_df = pd.read_csv(paymaster_csv)

    # Shorten address utility
    def shorten_address(addr):
        if isinstance(addr, str):
            if addr.lower() == "0x0000000000000000000000000000000000000000":
                return "No Paymasters"
            return addr[:6] + "..." + addr[-4:] if len(addr) > 12 else addr
        return addr

    color_palette = px.colors.qualitative.Plotly

    # Optional block range info
    title_range = f"Blocks {start_block:,}–{end_block:,} ({start_time} to {end_time})" \
        if all([start_block, end_block, start_time, end_time]) else "UserOp Failure Summary"

    # ============ Bundler Plot ============
    bundler_df = bundler_df.sort_values(by="fail_count", ascending=False)
    bundler_df["short"] = bundler_df["bundler"].apply(shorten_address)

    threshold = 1.0
    big_df = bundler_df[bundler_df["percent_of_total_failures"] >= threshold]
    small_df = bundler_df[bundler_df["percent_of_total_failures"] < threshold]
    others_sum = small_df["percent_of_total_failures"].sum()
    if others_sum > 0:
        others_row = pd.DataFrame({
            "bundler": ["Others"],
            "total_ops": [small_df["total_ops"].sum()],
            "fail_count": [small_df["fail_count"].sum()],
            "failure_rate_percent": [None],
            "percent_of_total_failures": [others_sum],
            "rank": [None],
            "short": ["Others"]
        })
        big_df = pd.concat([big_df, others_row], ignore_index=True)

    fig_bundler = make_subplots(
        rows=2, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.6, 0.4],
        subplot_titles=("Fail Count", "Failure Share", "Failure Rate (%)"),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"colspan": 2, "type": "bar"}, None]]
    )

    fig_bundler.add_trace(go.Bar(
        x=bundler_df["fail_count"],
        y=bundler_df["short"],
        orientation='h',
        text=bundler_df["fail_count"],
        textposition='auto',
        marker_color=color_palette[:len(bundler_df)],
        hovertext=bundler_df["bundler"],
        hoverinfo="text"
    ), row=1, col=1)

    fig_bundler.add_trace(go.Pie(
        labels=big_df["short"],
        values=big_df["percent_of_total_failures"],
        textinfo='percent',
        marker_colors=color_palette[:len(big_df)],
        showlegend=True
    ), row=1, col=2)

    fig_bundler.add_trace(go.Bar(
        x=bundler_df["short"],
        y=bundler_df["failure_rate_percent"],
        text=bundler_df["failure_rate_percent"].round(1).astype(str) + "%",
        textposition='auto',
        marker_color=color_palette[:len(bundler_df)],
        hovertext=bundler_df["bundler"],
        hoverinfo="text"
    ), row=2, col=1)

    fig_bundler.update_layout(
        height=900,
        width=1600,
        title_text=f"Bundler Error Analysis — {title_range}",
        margin=dict(l=80, r=80, t=80, b=80),
        font=dict(size=12)
    )

    fig_bundler.write_image(os.path.join(output_dir, f"bundler_analysis_{output_prefix}.png"))
    fig_bundler.write_html(os.path.join(output_dir, f"bundler_analysis_{output_prefix}.html"))
    
    # ============ Sender Plot ============
    fig_sender = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Fail Count", "Failure Rate (%)"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    sender_df = sender_df.sort_values(by="fail_count", ascending=False).head(10)
    sender_df["short"] = sender_df["sender"].apply(shorten_address)

    fig_sender.add_trace(go.Bar(
        x=sender_df["fail_count"],
        y=sender_df["short"],
        orientation='h',
        text=sender_df["fail_count"],
        textposition='auto',
        marker_color=color_palette[:len(sender_df)],
        hovertext=sender_df["sender"],
        hoverinfo="text"
    ), row=1, col=1)

    fig_sender.add_trace(go.Bar(
        x=sender_df["short"],
        y=sender_df["failure_rate_percent"],
        text=sender_df["failure_rate_percent"].round(1).astype(str) + "%",
        textposition='auto',
        marker_color=color_palette[:len(sender_df)],
        hovertext=sender_df["sender"],
        hoverinfo="text"
    ), row=1, col=2)

    fig_sender.update_layout(
        height=600,
        width=1400,
        title_text=f"Sender Error Analysis — {title_range}",
        margin=dict(l=80, r=80, t=80, b=80),
        font=dict(size=12)
    )

    fig_sender.write_image(os.path.join(output_dir, f"sender_analysis_{output_prefix}.png"))
    fig_sender.write_html(os.path.join(output_dir, f"sender_analysis_{output_prefix}.html"))

    # ============ Paymaster Plot ============
    paymaster_df = paymaster_df.sort_values(by="fail_count", ascending=False)
    paymaster_df["short"] = paymaster_df["paymaster"].apply(shorten_address)

    threshold_pm = 1.0
    big_pm_df = paymaster_df[paymaster_df["percent_of_total_failures"] >= threshold_pm]
    small_pm_df = paymaster_df[paymaster_df["percent_of_total_failures"] < threshold_pm]
    others_sum_pm = small_pm_df["percent_of_total_failures"].sum()
    if others_sum_pm > 0:
        others_row_pm = pd.DataFrame({
            "paymaster": ["Others"],
            "total_ops": [small_pm_df["total_ops"].sum()],
            "fail_count": [small_pm_df["fail_count"].sum()],
            "failure_rate_percent": [None],
            "percent_of_total_failures": [others_sum_pm],
            "rank": [None],
            "short": ["Others"]
        })
        big_pm_df = pd.concat([big_pm_df, others_row_pm], ignore_index=True)

    fig_paymaster = make_subplots(
        rows=2, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.6, 0.4],
        subplot_titles=("Fail Count", "Failure Share", "Failure Rate (%)"),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"colspan": 2, "type": "bar"}, None]]
    )

    fig_paymaster.add_trace(go.Bar(
        x=paymaster_df["fail_count"],
        y=paymaster_df["short"],
        orientation='h',
        text=paymaster_df["fail_count"],
        textposition='auto',
        marker_color=color_palette[:len(paymaster_df)],
        hovertext=paymaster_df["paymaster"],
        hoverinfo="text"
    ), row=1, col=1)

    fig_paymaster.add_trace(go.Pie(
        labels=big_pm_df["short"],
        values=big_pm_df["percent_of_total_failures"],
        textinfo='percent',
        marker_colors=color_palette[:len(big_pm_df)],
        showlegend=True
    ), row=1, col=2)

    fig_paymaster.add_trace(go.Bar(
        x=paymaster_df["short"],
        y=paymaster_df["failure_rate_percent"],
        text=paymaster_df["failure_rate_percent"].round(1).astype(str) + "%",
        textposition='auto',
        marker_color=color_palette[:len(paymaster_df)],
        hovertext=paymaster_df["paymaster"],
        hoverinfo="text"
    ), row=2, col=1)

    fig_paymaster.update_layout(
        height=900,
        width=1600,
        title_text=f"Paymaster Error Analysis — {title_range}",
        margin=dict(l=80, r=80, t=80, b=80),
        font=dict(size=12)
    )

    fig_paymaster.write_image(os.path.join(output_dir, f"paymaster_analysis_{output_prefix}.png"))
    fig_paymaster.write_html(os.path.join(output_dir, f"paymaster_analysis_{output_prefix}.html"))

    print("✅ All plots saved to PNG and HTML.")

