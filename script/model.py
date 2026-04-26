import os
import re
import glob
import csv
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# =========================
# CONFIG
# =========================
INPUT_DIR = "exports/cleaned_tweets_csv"
OUTPUT_DIR = "outputs_full"

MODEL_NAME = "distilbert-base-uncased"
TEXT_COL = "text"

CSV_CHUNKSIZE = 20000
EMBED_BATCH_SIZE = 64
MAX_LEN = 128

N_CLUSTERS = 5
KMEANS_BATCH_SIZE = 4096
PCA_COMPONENTS = 2

SAMPLE_TWEETS_PER_CLUSTER = 5
PLOT_SAMPLE_PER_CHUNK = 300
RANDOM_STATE = 42

USECOLS = ["target", "id", "date", "user", "text", "year", "month", "day", "hour"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)


# =========================
# HELPERS
# =========================
def list_csv_parts(input_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    return files


def iter_chunks(input_dir: str, chunksize: int):
    for file_path in list_csv_parts(input_dir):
        for chunk in pd.read_csv(file_path, usecols=USECOLS, chunksize=chunksize):
            yield chunk


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df[TEXT_COL] != ""]
    df = df.drop_duplicates(subset=[TEXT_COL])
    df = df.reset_index(drop=True)
    return df


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = torch.sum(masked_embeddings, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(texts: list[str], tokenizer, model, device, batch_size: int, max_len: int) -> np.ndarray:
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)

            pooled = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = pooled.cpu().numpy()
            all_embeddings.append(pooled)

    embeddings = np.vstack(all_embeddings)
    embeddings = normalize(embeddings)
    return embeddings


def tokenize_for_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]


# =========================
# MODEL SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    batch_size=KMEANS_BATCH_SIZE,
    n_init="auto"
)

ipca = IncrementalPCA(n_components=PCA_COMPONENTS)

# =========================
# PASS 1: FIT KMEANS + PCA
# =========================
print("\n=== PASS 1: Fitting MiniBatchKMeans and IncrementalPCA ===")

total_rows = 0
chunk_count = 0

for chunk in iter_chunks(INPUT_DIR, CSV_CHUNKSIZE):
    chunk = clean_chunk(chunk)
    if chunk.empty:
        continue

    texts = chunk[TEXT_COL].tolist()
    embeddings = embed_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=EMBED_BATCH_SIZE,
        max_len=MAX_LEN
    )

    kmeans.partial_fit(embeddings)

    if embeddings.shape[0] >= PCA_COMPONENTS:
        ipca.partial_fit(embeddings)

    total_rows += len(chunk)
    chunk_count += 1

    if chunk_count % 10 == 0:
        print(f"Processed chunks: {chunk_count}, rows so far: {total_rows}")

print(f"Pass 1 complete. Total rows processed: {total_rows}")


# =========================
# PASS 2: ASSIGN CLUSTERS + SAVE OUTPUTS
# =========================
print("\n=== PASS 2: Predicting clusters and collecting outputs ===")

cluster_counts = Counter()
cluster_samples = defaultdict(list)
cluster_keywords = defaultdict(Counter)

plot_points = []
plot_labels = []

clustered_output_path = os.path.join(OUTPUT_DIR, "clustered_full_dataset.csv")
write_header = True

for chunk in iter_chunks(INPUT_DIR, CSV_CHUNKSIZE):
    chunk = clean_chunk(chunk)
    if chunk.empty:
        continue

    texts = chunk[TEXT_COL].tolist()
    embeddings = embed_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=EMBED_BATCH_SIZE,
        max_len=MAX_LEN
    )

    labels = kmeans.predict(embeddings)
    pca_points = ipca.transform(embeddings)

    chunk["cluster"] = labels
    chunk["pca_1"] = pca_points[:, 0]
    chunk["pca_2"] = pca_points[:, 1]

    # update counts
    for label in labels:
        cluster_counts[int(label)] += 1

    # collect sample tweets + keywords
    for _, row in chunk.iterrows():
        label = int(row["cluster"])
        text = str(row[TEXT_COL])

        if len(cluster_samples[label]) < SAMPLE_TWEETS_PER_CLUSTER:
            cluster_samples[label].append(text)

        cluster_keywords[label].update(tokenize_for_keywords(text))

    # save clustered rows
    chunk.to_csv(
        clustered_output_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False
    )
    write_header = False

    # sample points for visualization
    if len(chunk) <= PLOT_SAMPLE_PER_CHUNK:
        sampled = chunk
    else:
        sampled = chunk.sample(PLOT_SAMPLE_PER_CHUNK, random_state=RANDOM_STATE)

    plot_points.append(sampled[["pca_1", "pca_2"]].to_numpy())
    plot_labels.append(sampled["cluster"].to_numpy())

print("Pass 2 complete.")


# =========================
# SAVE SUMMARIES
# =========================
cluster_dist_df = pd.DataFrame(
    [{"cluster": c, "tweet_count": cluster_counts[c]} for c in sorted(cluster_counts)]
)
cluster_dist_df.to_csv(os.path.join(OUTPUT_DIR, "cluster_distribution.csv"), index=False)

summary = {}
for cluster_id in sorted(cluster_counts):
    top_words = [w for w, _ in cluster_keywords[cluster_id].most_common(15)]
    summary[cluster_id] = {
        "tweet_count": int(cluster_counts[cluster_id]),
        "top_words": top_words,
        "sample_tweets": cluster_samples[cluster_id]
    }

with open(os.path.join(OUTPUT_DIR, "cluster_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "cluster_summary.txt"), "w", encoding="utf-8") as f:
    for cluster_id in sorted(summary):
        f.write(f"\n=== Cluster {cluster_id} ===\n")
        f.write(f"Tweet count: {summary[cluster_id]['tweet_count']}\n")
        f.write("Top words: " + ", ".join(summary[cluster_id]["top_words"]) + "\n")
        f.write("Sample tweets:\n")
        for i, tweet in enumerate(summary[cluster_id]["sample_tweets"], 1):
            f.write(f"{i}. {tweet}\n")


# =========================
# SAVE MODEL ARTIFACTS
# =========================
np.save(os.path.join(OUTPUT_DIR, "cluster_centers.npy"), kmeans.cluster_centers_)
np.save(os.path.join(OUTPUT_DIR, "pca_components.npy"), ipca.components_)


# =========================
# PCA PLOT
# =========================
all_plot_points = np.vstack(plot_points) if plot_points else np.empty((0, 2))
all_plot_labels = np.concatenate(plot_labels) if plot_labels else np.empty((0,))

if len(all_plot_points) > 0:
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        all_plot_points[:, 0],
        all_plot_points[:, 1],
        c=all_plot_labels,
        alpha=0.6,
        s=10
    )
    plt.title("Full Dataset Tweet Clusters (sampled PCA view)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_cluster_plot_full.png"), dpi=300)
    plt.show()

print("\nSaved:")
print(f"- {clustered_output_path}")
print(f"- {os.path.join(OUTPUT_DIR, 'cluster_distribution.csv')}")
print(f"- {os.path.join(OUTPUT_DIR, 'cluster_summary.txt')}")
print(f"- {os.path.join(OUTPUT_DIR, 'cluster_summary.json')}")
print(f"- {os.path.join(OUTPUT_DIR, 'pca_cluster_plot_full.png')}")
print(f"- {os.path.join(OUTPUT_DIR, 'cluster_centers.npy')}")
print(f"- {os.path.join(OUTPUT_DIR, 'pca_components.npy')}")