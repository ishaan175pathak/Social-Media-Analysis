import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs_full"
INPUT_FILE = os.path.join(OUTPUT_DIR, "clustered_full_dataset.csv")

df = pd.read_csv(INPUT_FILE)

df["target_label"] = df["target"].map({
    0: "Negative",
    1: "Positive"
})

# =========================
# COUNTS
# =========================
cluster_target_counts = (
    df.groupby(["cluster", "target", "target_label"])
    .size()
    .reset_index(name="tweet_count")
    .sort_values(["cluster", "target"])
    .reset_index(drop=True)
)

cluster_target_counts.to_csv(
    os.path.join(OUTPUT_DIR, "cluster_target_counts.csv"),
    index=False
)

# =========================
# PERCENTAGES
# =========================
cluster_target_percentages = cluster_target_counts.copy()

cluster_target_percentages["percentage"] = (
    cluster_target_percentages["tweet_count"]
    / cluster_target_percentages.groupby("cluster")["tweet_count"].transform("sum")
) * 100

cluster_target_percentages.to_csv(
    os.path.join(OUTPUT_DIR, "cluster_target_percentages.csv"),
    index=False
)

# =========================
# DOMINANT SENTIMENT
# =========================
dominant_sentiment = (
    cluster_target_percentages.loc[
        cluster_target_percentages.groupby("cluster")["percentage"].idxmax()
    ]
    .sort_values("cluster")
    .reset_index(drop=True)
)

dominant_sentiment.to_csv(
    os.path.join(OUTPUT_DIR, "dominant_sentiment_per_cluster.csv"),
    index=False
)

# =========================
# PIVOTS
# =========================
pivot_counts = cluster_target_counts.pivot_table(
    index="cluster",
    columns="target_label",
    values="tweet_count",
    fill_value=0
)

pivot_percentages = cluster_target_percentages.pivot_table(
    index="cluster",
    columns="target_label",
    values="percentage",
    fill_value=0
)

pivot_counts.to_csv(os.path.join(OUTPUT_DIR, "cluster_target_counts_pivot.csv"))
pivot_percentages.to_csv(os.path.join(OUTPUT_DIR, "cluster_target_percentages_pivot.csv"))

# =========================
# PLOT
# =========================
plt.figure(figsize=(10, 6))
pivot_percentages.plot(kind="bar", stacked=True, ax=plt.gca())
plt.title("Sentiment Distribution Across Clusters")
plt.xlabel("Cluster")
plt.ylabel("Percentage")
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_sentiment_distribution.png"), dpi=300)
plt.show()

# =========================
# PRINT
# =========================
print("\n=== Cluster Target Counts ===")
print(cluster_target_counts)

print("\n=== Cluster Target Percentages ===")
print(cluster_target_percentages)

print("\n=== Dominant Sentiment Per Cluster ===")
print(dominant_sentiment)

print("\nSaved files:")
print(f"- {os.path.join(OUTPUT_DIR, 'cluster_target_counts.csv')}")
print(f"- {os.path.join(OUTPUT_DIR, 'cluster_target_percentages.csv')}")
print(f"- {os.path.join(OUTPUT_DIR, 'dominant_sentiment_per_cluster.csv')}")
print(f"- {os.path.join(OUTPUT_DIR, 'cluster_target_counts_pivot.csv')}")
print(f"- {os.path.join(OUTPUT_DIR, 'cluster_target_percentages_pivot.csv')}")
print(f"- {os.path.join(OUTPUT_DIR, 'cluster_sentiment_distribution.png')}")