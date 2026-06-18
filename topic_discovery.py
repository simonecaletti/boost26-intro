#!/usr/bin/env python3
"""
Unsupervised topic discovery on a CSV of arXiv jet-physics papers.

Primary method : BERTopic (sentence-transformers + UMAP + HDBSCAN)
Secondary method: TF-IDF + TruncatedSVD + HDBSCAN (toggled via RUN_BASELINE)

Dependencies:
    pip install bertopic scikit-learn pandas umap-learn hdbscan sentence-transformers matplotlib

Usage:
    python topic_discovery.py
"""

import sys
import os

# ---------------------------------------------------------------------------
# Configuration — edit here, not inline
# ---------------------------------------------------------------------------
INPUT_CSV          = "arxiv_jet_papers.csv"
OUTPUT_DIR         = "output"

EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
MIN_TOPIC_SIZE     = 4      # lower = more (smaller) topics; merged down via NR_TOPICS
NR_TOPICS          = 15     # target after merging similar topics (excludes outlier -1)
N_REPRESENTATIVE   = 3      # papers to print per topic
RUN_BASELINE       = True   # also run TF-IDF + SVD + HDBSCAN

# BERTopic UMAP settings
UMAP_N_NEIGHBORS   = 15
UMAP_N_COMPONENTS  = 5
UMAP_METRIC        = "cosine"

# Custom stopwords (English boilerplate + physics-arXiv filler)
EXTRA_STOPWORDS = [
    "we", "us", "our", "using", "use", "used", "show", "shown",
    "study", "studied", "result", "results", "present", "presented",
    "paper", "approach", "method", "methods", "model", "models",
    "based", "new", "also", "however", "thus", "therefore", "within",
    "via", "given", "obtain", "obtained", "find", "found", "consider",
    "considered", "propose", "proposed", "provide", "provided",
    "case", "cases", "different", "various", "well", "may", "can",
    "one", "two", "three", "first", "second", "order", "high", "low",
    "large", "small", "set", "function", "functions", "value", "values",
    "data", "analysis", "respectively", "compared", "comparison",
    "et", "al", "e", "g", "i", "ii", "iii", "important", "calculate",
    "calculated", "demonstrate", "demonstrated", "shown", "work",
    "recent", "known", "new", "arxiv", "preprint", "physical", "review",
    "letter", "letters", "journal", "proceedings",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np


def out(filename: str) -> str:
    """Return path inside OUTPUT_DIR, creating the directory if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, filename)


def load_corpus(path: str) -> pd.DataFrame:
    """Load the CSV and build a 'text' column = title + abstract."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"ERROR: input file not found: {path}")

    required = {"arxiv_id", "title", "abstract"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV is missing columns: {missing}")

    df["title"]    = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    df["text"]     = df["title"] + ". " + df["abstract"]
    print(f"Loaded {len(df)} papers from {path}.")
    return df


# ---------------------------------------------------------------------------
# Primary method: BERTopic
# ---------------------------------------------------------------------------

def run_bertopic(df: pd.DataFrame):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from umap import UMAP
    import hdbscan
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    docs = df["text"].tolist()

    # --- Stage 1: Embed -------------------------------------------------------
    print("\n[BERTopic] Embedding documents with", EMBEDDING_MODEL, "...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=64)

    # --- Stage 2: Reduce dimensionality (5-D for clustering) -----------------
    print("[BERTopic] Reducing with UMAP ...")
    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        metric=UMAP_METRIC,
        random_state=42,
        low_memory=False,
    )

    # --- Stage 3: Cluster -----------------------------------------------------
    print("[BERTopic] Clustering with HDBSCAN ...")
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # --- Stage 4: Label via c-TF-IDF ------------------------------------------
    all_stopwords = list(ENGLISH_STOP_WORDS) + EXTRA_STOPWORDS
    vectorizer = CountVectorizer(
        stop_words=all_stopwords,
        ngram_range=(1, 2),
        min_df=2,
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    # --- Merge fine-grained topics down to NR_TOPICS --------------------------
    print(f"[BERTopic] Merging topics down to {NR_TOPICS} ...")
    topic_model.reduce_topics(docs, nr_topics=NR_TOPICS)
    topics = topic_model.topics_
    # probs not updated by reduce_topics; reuse original max-prob per doc
    probs  = topic_model.probabilities_ if hasattr(topic_model, "probabilities_") else probs

    # --- Results: print & save ------------------------------------------------
    topic_info = topic_model.get_topic_info()
    print("\n" + "="*70)
    print("BERTopic — discovered topics")
    print("="*70)
    # Concise table: id, size, top-5 terms
    for _, row in topic_info.iterrows():
        top5 = ", ".join(row["Representation"][:5])
        print(f"  Topic {row['Topic']:3d}  ({row['Count']:4d} papers)  {top5}")

    topic_info.to_csv(out("topics.csv"), index=False)
    print(f"\nTopic table saved to {out('topics.csv')}")

    # Per-paper assignment table
    topic_labels = topic_model.topic_labels_
    paper_df = df[["arxiv_id", "title"]].copy()
    paper_df["topic"]       = topics
    paper_df["topic_label"] = [topic_labels.get(t, str(t)) for t in topics]
    paper_df["probability"] = [
        float(np.max(p)) if hasattr(p, "__len__") else float(p)
        for p in probs
    ]
    paper_df.to_csv(out("paper_topics.csv"), index=False)
    print(f"Per-paper assignments saved to {out('paper_topics.csv')}")

    # Representative papers per topic
    print("\n" + "="*70)
    print(f"Top-{N_REPRESENTATIVE} representative papers per topic")
    print("="*70)
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        label = topic_labels.get(tid, str(tid))
        reps  = topic_model.get_representative_docs(tid)
        print(f"\nTopic {tid} — {label}  (size={row['Count']})")
        for doc_text in reps[:N_REPRESENTATIVE]:
            match = df[df["text"] == doc_text]
            if not match.empty:
                r = match.iloc[0]
                print(f"  [{r['arxiv_id']}] {r['title'][:90]}")
            else:
                print(f"  {doc_text[:100]}")

    # --- HTML visualizations --------------------------------------------------
    print("\n[BERTopic] Saving HTML visualizations ...")
    try:
        topic_model.visualize_topics().write_html(out("bertopic_intertopic.html"))
        print(f"  {out('bertopic_intertopic.html')}")

        topic_model.visualize_barchart(
            top_n_topics=min(20, len(topic_info) - 1)
        ).write_html(out("bertopic_barchart.html"))
        print(f"  {out('bertopic_barchart.html')}")

        # 2-D reduction for document map
        reduced_2d = UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            n_components=2,
            metric=UMAP_METRIC,
            random_state=42,
        ).fit_transform(embeddings)

        topic_model.visualize_documents(
            docs,
            reduced_embeddings=reduced_2d,
            hide_annotations=True,
        ).write_html(out("bertopic_documents.html"))
        print(f"  {out('bertopic_documents.html')}")
    except Exception as exc:
        print(f"  WARNING: HTML visualization failed ({exc})")
        reduced_2d = None

    # --- Static matplotlib scatter plot --------------------------------------
    print("\n[BERTopic] Saving static 2-D cluster scatter plot ...")
    try:
        if reduced_2d is None:
            reduced_2d = UMAP(
                n_neighbors=UMAP_N_NEIGHBORS,
                n_components=2,
                metric=UMAP_METRIC,
                random_state=42,
            ).fit_transform(embeddings)

        topic_ids  = np.array(topics)
        unique_ids = sorted(set(topic_ids))
        # Assign colors: grey for noise (-1), distinct colors for real topics
        palette = cm.get_cmap("tab20", max(len(unique_ids), 1))
        color_map = {}
        color_idx = 0
        for tid in unique_ids:
            if tid == -1:
                color_map[tid] = (0.7, 0.7, 0.7, 0.4)   # translucent grey
            else:
                color_map[tid] = palette(color_idx)
                color_idx += 1

        fig, ax = plt.subplots(figsize=(12, 8))
        # Draw noise first so real clusters sit on top
        for tid in unique_ids:
            mask   = topic_ids == tid
            label  = "noise" if tid == -1 else topic_labels.get(tid, str(tid))
            # Shorten label for legend: keep first 4 words
            short  = " / ".join(label.split("_")[1:5]) if "_" in label else label
            ax.scatter(
                reduced_2d[mask, 0], reduced_2d[mask, 1],
                s=10 if tid == -1 else 20,
                alpha=0.35 if tid == -1 else 0.75,
                color=color_map[tid],
                label=f"T{tid}: {short} (n={mask.sum()})" if tid != -1 else f"noise (n={mask.sum()})",
                zorder=1 if tid == -1 else 2,
            )

        # Annotate each topic cluster with its id at the centroid
        for tid in unique_ids:
            if tid == -1:
                continue
            mask = topic_ids == tid
            cx, cy = reduced_2d[mask, 0].mean(), reduced_2d[mask, 1].mean()
            ax.text(cx, cy, str(tid), fontsize=8, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"))

        ax.set_title("BERTopic — 2-D UMAP document projection", fontsize=13)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7,
                  frameon=True, title="Topics")
        fig.tight_layout()
        plot_path = out("bertopic_clusters.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {plot_path}")
    except Exception as exc:
        print(f"  WARNING: static plot failed ({exc})")

    return topic_model, paper_df


# ---------------------------------------------------------------------------
# Secondary method: TF-IDF + TruncatedSVD + HDBSCAN (baseline)
# ---------------------------------------------------------------------------

def run_baseline(df: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer
    from sklearn.pipeline import make_pipeline
    import hdbscan

    docs = df["text"].tolist()

    print("\n" + "="*70)
    print("Baseline: TF-IDF + TruncatedSVD + HDBSCAN")
    print("="*70)

    # --- Stage 1: Vectorize ---------------------------------------------------
    all_sw = list(ENGLISH_STOP_WORDS) + EXTRA_STOPWORDS
    vectorizer = TfidfVectorizer(
        stop_words=all_sw,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(docs)
    print(f"TF-IDF matrix: {X.shape}")

    # --- Stage 2: Reduce (LSA) ------------------------------------------------
    n_components = min(100, X.shape[1] - 1)
    svd  = TruncatedSVD(n_components=n_components, random_state=42)
    pipe = make_pipeline(svd, Normalizer(copy=False))
    X_reduced = pipe.fit_transform(X)
    print(f"Reduced to {X_reduced.shape[1]} LSA components.")

    # --- Stage 3: Cluster -----------------------------------------------------
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels     = clusterer.fit_predict(X_reduced)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"Clusters found: {n_clusters}  |  noise points: {n_noise}")

    # --- Stage 4: Label via top TF-IDF terms per cluster ---------------------
    terms = vectorizer.get_feature_names_out()
    print("\nBaseline cluster top terms:")
    for cid in sorted(set(labels)):
        mask      = labels == cid
        label_str = "NOISE" if cid == -1 else f"Cluster {cid}"
        if cid == -1:
            print(f"  {label_str}: {mask.sum()} papers (unlabelled)")
            continue
        centroid  = np.asarray(X[mask].mean(axis=0)).ravel()
        top_terms = ", ".join(terms[i] for i in centroid.argsort()[::-1][:10])
        print(f"  {label_str} ({mask.sum()} papers): {top_terms}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    df = load_corpus(INPUT_CSV)
    run_bertopic(df)
    if RUN_BASELINE:
        run_baseline(df)


if __name__ == "__main__":
    main()
