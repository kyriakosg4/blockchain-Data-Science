# src/faucet_visualize.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_faucet_metrics(input_csv="data/faucet_clicks_fulldata_sorted.csv", output_dir="faucet_visuals"):
    df = pd.read_csv(input_csv)
    sns.set(style="whitegrid")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Histogram of total faucet clicks
    plt.figure(figsize=(10, 6))
    sns.histplot(df["faucet_clicks"], bins=30, kde=False)
    plt.title("Distribution of Faucet Clicks (7 Days)")
    plt.xlabel("Faucet Clicks")
    plt.ylabel("Number of Wallets")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hist_total_faucet_clicks.png"))
    plt.close()

    # 2. Boxplot of average clicks per day
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df["avg_clicks_per_day"])
    plt.title("Boxplot of Average Clicks Per Day")
    plt.xlabel("Average Clicks Per Day")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "box_avg_clicks_per_day.png"))
    plt.close()

    # 3. Bar plot comparing averages
    avg_all = df["avg_clicks_per_day"].mean()
    avg_5p = df[df["is_top_5_percent"]]["avg_clicks_per_day"].mean()
    avg_2p = df[df["is_top_2_percent"]]["avg_clicks_per_day"].mean()

    plt.figure(figsize=(7, 5))
    sns.barplot(x=["All", "Top 5%", "Top 2%"], y=[avg_all, avg_5p, avg_2p])
    plt.title("Avg Clicks Per Day: All vs Top Wallets")
    plt.ylabel("Average Clicks Per Day")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bar_avg_clicks_percentiles.png"))
    plt.close()

    # 4. Histogram of average time between clicks
    plt.figure(figsize=(10, 6))
    sns.histplot(df["avg_time_between_clicks_hr"].dropna(), bins=30)
    plt.title("Avg Time Between Clicks (Hours)")
    plt.xlabel("Time in Hours")
    plt.ylabel("Number of Wallets")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hist_avg_time_between_clicks.png"))
    plt.close()

    # 5. Scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="faucet_clicks",
        y="avg_clicks_per_day",
        hue="is_top_2_percent",
        style="is_top_5_percent",
        palette="Set1"
    )
    plt.title("Total Faucet Clicks vs Avg Clicks Per Day")
    plt.xlabel("Faucet Clicks (7 Days)")
    plt.ylabel("Average Clicks Per Day")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_clicks_vs_avg.png"))
    plt.close()

    print(f"✅ All faucet plots saved to: {output_dir}/")
