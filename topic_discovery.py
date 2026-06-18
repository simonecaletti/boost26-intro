#!/usr/bin/env python3
"""
Hierarchical topic discovery on arXiv jet-physics papers.

Pass 1  — cluster full corpus (coarse).
Pass 2  — groom off-topic clusters; save groomed CSV.
Pass 3  — cluster groomed corpus; identify top-N clusters.
Pass 4  — recursively sub-cluster each top cluster while its UMAP radius
           exceeds RADIUS_THRESHOLD; labels grow as T3 → T3a/T3b → T3ba/T3bb.
Plot    — single 2-D UMAP of all papers:
            • groomed papers → light grey
            • Pass-3 cluster boundary → thin colored convex-hull ring
            • recursive leaf dots → shaded colors inside the ring

Dependencies:
    pip install bertopic scikit-learn pandas umap-learn hdbscan
                sentence-transformers matplotlib scipy

Usage:
    python topic_discovery.py
"""

import sys, os, warnings
from dataclasses import dataclass, field
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CSV         = "arxiv_jet_papers.csv"
OUTPUT_DIR        = "output"
CLUSTERS_DIR      = os.path.join(OUTPUT_DIR, "clusters")

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
MIN_TOPIC_SIZE    = 4
N_REPRESENTATIVE  = 3
N_TOP_CLUSTERS    = 5     # how many Pass-3 clusters to recurse into

PASS1_NR_TOPICS   = 15
PASS3_NR_TOPICS   = 12
PASS4_NR_TOPICS   = 8     # target topics at each recursion level

# Recursion stopping criteria
RADIUS_THRESHOLD  = 0.8   # mean UMAP distance from centroid; split if larger
MAX_CLUSTER_DEPTH = 3     # maximum depth below Pass-3 (1 = old single-level behaviour)

# Topics whose top terms match any of these are groomed out after Pass 1
OFFTOPIC_KEYWORDS = [
    "dark matter", "dark", "inflation", "inflaton", "reheating",
    "gravitational wave", "gravitational", "string", "moduli", "ads",
    "conformal", "bootstrap", "axion", "cosmological",
]

UMAP_N_NEIGHBORS  = 15
UMAP_N_COMPONENTS = 5    # for clustering
UMAP_METRIC       = "cosine"

# Base colors for Pass-3 top clusters (one per cluster, up to N_TOP_CLUSTERS)
CLUSTER_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]

N_LABEL_TERMS = 3   # how many top BERTopic terms to show as the cluster name

# Papers to highlight individually (each gets its own marker color and legend entry).
# Each entry: (legend_label, search_string) — search_string is matched
# case-insensitively against title + authors + abstract.
HIGHLIGHT_PAPERS = [
    ("Lund b-jet plane (Ghira et al.)", "Ghira"),
]
# Colors for highlighted papers (cycled if more entries than colors)
HIGHLIGHT_COLORS = ["#ff1493", "#00ced1", "#ffd700", "#39ff14", "#ff6600"]

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
import pandas as pd
import numpy as np

LETTERS = "abcdefghijklmnop"


# ---------------------------------------------------------------------------
# Cluster tree node
# ---------------------------------------------------------------------------

@dataclass
class ClusterNode:
    """One node in the recursive cluster tree.

    label      — hierarchical name, e.g. "T3", "T3a", "T3ba"
    global_idx — indices into the *full* corpus (df_full / xy_all)
    children   — child ClusterNodes (empty ⟹ leaf)
    is_noise   — True for BERTopic -1 noise buckets (plotted transparently)
    """
    label:      str
    global_idx: np.ndarray
    top_terms:  list = field(default_factory=list)   # top BERTopic terms
    children:   list = field(default_factory=list)
    is_noise:   bool = False

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def n(self):
        return len(self.global_idx)


def collect_leaves(node):
    """Depth-first; returns list of ClusterNode leaves."""
    if node.is_leaf:
        return [node]
    result = []
    for child in node.children:
        result.extend(collect_leaves(child))
    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def out(filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, filename)


def cout(filename):
    os.makedirs(CLUSTERS_DIR, exist_ok=True)
    return os.path.join(CLUSTERS_DIR, filename)


def load_corpus(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"ERROR: input file not found: {path}")
    for col in ("arxiv_id", "title", "abstract"):
        if col not in df.columns:
            sys.exit(f"ERROR: CSV missing column '{col}'")
    df["title"]    = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    df["text"]     = df["title"] + ". " + df["abstract"]
    print(f"Loaded {len(df)} papers from {path}.")
    return df


# ---------------------------------------------------------------------------
# Embedding & UMAP
# ---------------------------------------------------------------------------

def embed_all(df):
    from sentence_transformers import SentenceTransformer
    print(f"\n[Embed] Embedding {len(df)} documents with {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    emb   = model.encode(df["text"].tolist(), show_progress_bar=True, batch_size=64)
    print(f"[Embed] Done. Shape: {emb.shape}")
    return emb


def umap_2d(embeddings):
    """2-D projection for plotting (computed once on the full corpus)."""
    from umap import UMAP
    print("[UMAP-2D] Projecting to 2-D for final plot ...")
    xy = UMAP(n_neighbors=UMAP_N_NEIGHBORS, n_components=2,
              metric=UMAP_METRIC, random_state=42).fit_transform(embeddings)
    print("[UMAP-2D] Done.")
    return xy


# ---------------------------------------------------------------------------
# BERTopic wrapper
# ---------------------------------------------------------------------------

def _is_offtopic(representation):
    rep = " ".join(representation).lower()
    return any(kw in rep for kw in OFFTOPIC_KEYWORDS)


def run_bertopic(embeddings, docs, nr_topics, label, min_topic_size=None):
    """Cluster pre-computed embeddings with BERTopic, merge to nr_topics."""
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from umap import UMAP
    import hdbscan

    mts = min_topic_size or MIN_TOPIC_SIZE
    print(f"\n[{label}] Clustering {len(docs)} docs (min_topic_size={mts}) ...")

    umap_model = UMAP(n_neighbors=min(UMAP_N_NEIGHBORS, len(docs) - 2),
                      n_components=min(UMAP_N_COMPONENTS, len(docs) - 2),
                      metric=UMAP_METRIC, random_state=42, low_memory=False)
    hdb_model  = hdbscan.HDBSCAN(min_cluster_size=mts, metric="euclidean",
                                  cluster_selection_method="eom", prediction_data=True)
    all_sw     = list(ENGLISH_STOP_WORDS) + EXTRA_STOPWORDS
    vectorizer = CountVectorizer(stop_words=all_sw, ngram_range=(1, 2), min_df=2)
    ctfidf     = ClassTfidfTransformer(reduce_frequent_words=True)

    model = BERTopic(embedding_model=SentenceTransformer(EMBEDDING_MODEL),
                     umap_model=umap_model, hdbscan_model=hdb_model,
                     vectorizer_model=vectorizer, ctfidf_model=ctfidf,
                     calculate_probabilities=False, verbose=False)

    model.fit_transform(docs, embeddings=embeddings)

    actual_topics = len(model.get_topic_info()) - 1  # exclude -1
    target = min(nr_topics, max(actual_topics, 1))
    if actual_topics > target:
        print(f"[{label}] Merging {actual_topics} → {target} topics ...")
        model.reduce_topics(docs, nr_topics=target)

    return model


def print_topics(model, label, csv_path):
    ti = model.get_topic_info()
    print(f"\n{'='*70}\nTopics — {label}\n{'='*70}")
    for _, row in ti.iterrows():
        top5 = ", ".join(row["Representation"][:5])
        print(f"  T{row['Topic']:3d}  ({row['Count']:4d} papers)  {top5}")
    ti.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")
    return ti


def print_reps(model, df):
    labels = model.topic_labels_
    for _, row in model.get_topic_info().iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        reps = model.get_representative_docs(tid)
        print(f"\n  T{tid} — {labels.get(tid, '')}  (n={row['Count']})")
        for doc in reps[:N_REPRESENTATIVE]:
            match = df[df["text"] == doc]
            if not match.empty:
                r = match.iloc[0]
                print(f"    [{r['arxiv_id']}] {r['title'][:85]}")


def top_n_topics(ti, n):
    """Return list of topic IDs for the n largest non-noise topics."""
    real = ti[ti["Topic"] != -1].sort_values("Count", ascending=False)
    return [int(r["Topic"]) for _, r in real.head(n).iterrows()]


# ---------------------------------------------------------------------------
# Recursive sub-clustering (Pass 4+)
# ---------------------------------------------------------------------------

def cluster_radius(xy):
    """Mean distance from centroid in 2-D UMAP — our splitting criterion."""
    if len(xy) < 2:
        return 0.0
    c = xy.mean(axis=0)
    return float(np.linalg.norm(xy - c, axis=1).mean())


def recursive_cluster(emb_sub, df_sub, global_idx, xy_all, label, depth):
    """
    Recursively sub-cluster a group of papers.

    Returns a ClusterNode that is either a leaf (when the cluster is compact
    enough, too small, or we hit MAX_CLUSTER_DEPTH) or an internal node whose
    children are themselves ClusterNodes.

    Parameters
    ----------
    emb_sub    : (n, d) embedding slice for this group
    df_sub     : DataFrame slice (reset index), same order as emb_sub
    global_idx : 1-D int array of indices into df_full (and xy_all)
    xy_all     : (N_full, 2) 2-D UMAP of the entire corpus
    label      : hierarchical label string, e.g. "T3", "T3a", "T3ba"
    depth      : current recursion depth (0 = called from Pass 4 loop)
    """
    xy_sub = xy_all[global_idx]
    node   = ClusterNode(label=label, global_idx=global_idx)

    r = cluster_radius(xy_sub)
    print(f"  [{label}]  n={len(df_sub)}  radius={r:.3f}")

    # Stopping conditions
    if len(df_sub) < 8 or r < RADIUS_THRESHOLD or depth >= MAX_CLUSTER_DEPTH:
        print(f"  [{label}]  → leaf")
        return node

    # --- Sub-cluster this group ---
    mts = max(4, len(df_sub) // 20)
    tm  = run_bertopic(emb_sub, df_sub["text"].tolist(), PASS4_NR_TOPICS, label, mts)

    topics    = np.array(tm.topics_)
    real_sids = sorted(s for s in set(topics) if s != -1)

    if len(real_sids) < 2:
        print(f"  [{label}]  Only {len(real_sids)} real sub-topic(s) → leaf")
        return node

    # Save CSVs and topic table for this level
    ti = print_topics(tm, label, cout(f"{label}_topics.csv"))
    print_reps(tm, df_sub)
    df_sub.to_csv(cout(f"{label}.csv"), index=False)

    # Build children, attaching top terms from this level's BERTopic model
    ti_map = {int(r["Topic"]): r["Representation"] for _, r in ti.iterrows()}
    for i, sid in enumerate(real_sids):
        mask  = np.where(topics == sid)[0]
        terms = ti_map.get(sid, [])[:N_LABEL_TERMS]
        child = recursive_cluster(
            emb_sub[mask],
            df_sub.iloc[mask].reset_index(drop=True),
            global_idx[mask],
            xy_all,
            label + LETTERS[i],
            depth + 1,
        )
        child.top_terms = terms
        node.children.append(child)

    # Noise bucket (BERTopic -1): kept as a non-recursive leaf child
    noise_mask = np.where(topics == -1)[0]
    if len(noise_mask) > 0:
        node.children.append(
            ClusterNode(label=label + "?",
                        global_idx=global_idx[noise_mask],
                        is_noise=True)
        )

    return node


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def subcolors(base_hex, n):
    """n perceptually distinct colors that stay in the hue family of base_hex.

    Hue is spread ±25° around the base; saturation and value are varied
    so adjacent entries look different even when n is large.
    """
    import colorsys
    import matplotlib.colors as mc
    if n <= 0:
        return []
    r, g, b = mc.to_rgb(base_hex)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)                   # 0 → 1
        dh   = (t - 0.5) * 0.14                 # ±0.07 hue shift (~25°)
        new_h = (h + dh) % 1.0
        new_s = 0.55 + 0.35 * (0.5 + 0.5 * np.cos(np.pi * t))   # 0.55–0.90
        new_v = 0.55 + 0.35 * t                 # 0.55–0.90, darker→lighter
        colors.append(colorsys.hsv_to_rgb(new_h, new_s, new_v))
    return colors


# ---------------------------------------------------------------------------
# Final composite plot
# ---------------------------------------------------------------------------

def _find_highlighted(df_full):
    """Return list of (label, color, global_idx_array) for HIGHLIGHT_PAPERS entries."""
    results = []
    for i, (label, search) in enumerate(HIGHLIGHT_PAPERS):
        col  = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
        low  = search.lower()
        mask = df_full.apply(
            lambda r: low in str(r.get("title", "")).lower()
                   or low in str(r.get("authors", "")).lower()
                   or low in str(r.get("abstract", "")).lower(),
            axis=1,
        )
        idx = np.where(mask.values)[0]
        if len(idx) == 0:
            print(f"  [Highlight] WARNING: no papers matched '{search}'")
        else:
            print(f"  [Highlight] '{label}': {len(idx)} paper(s) matched")
            for j in idx:
                print(f"    [{df_full.iloc[j]['arxiv_id']}] "
                      f"{df_full.iloc[j]['title'][:80]}")
        results.append((label, col, idx))
    return results


def final_plot(xy_all, df_full, groomed_mask, pass3_topics, pass4_trees, ti3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon as MplPolygon

    top_ids   = top_n_topics(ti3, N_TOP_CLUSTERS)
    rank_of   = {tid: rank for rank, tid in enumerate(top_ids)}
    color_for = {tid: CLUSTER_COLORS[rank_of[tid]] for tid in top_ids}

    # Pre-compute highlighted paper indices
    highlights = _find_highlighted(df_full) if HIGHLIGHT_PAPERS else []

    fig, ax = plt.subplots(figsize=(16, 10))
    cluster_handles   = []
    highlight_handles = []

    # 1. Groomed papers — light grey
    groom_idx = np.where(~groomed_mask)[0]
    ax.scatter(xy_all[groom_idx, 0], xy_all[groom_idx, 1],
               s=8, color="#cccccc", alpha=0.4, zorder=1)
    cluster_handles.append(
        mpatches.Patch(color="#cccccc", label=f"groomed (n={len(groom_idx)})")
    )

    # 2. Convex-hull rings for top Pass-3 clusters
    for tid in top_ids:
        col = color_for[tid]
        idx = np.where(pass3_topics == tid)[0]
        pts = xy_all[idx]
        if len(pts) < 3:
            continue
        hull  = ConvexHull(pts)
        verts = pts[hull.vertices]
        c     = verts.mean(axis=0)
        verts = c + 1.08 * (verts - c)
        ax.add_patch(MplPolygon(verts, closed=True, fill=True,
                                facecolor=col, alpha=0.07,
                                edgecolor=col, linewidth=1.8, zorder=2))

    # 3. Recursive leaf dots + thin sub-cluster hull outlines + legend
    for rank_i, tid in enumerate(top_ids):
        rank   = rank_of[tid]
        col    = color_for[tid]
        p3_idx = np.where(pass3_topics == tid)[0]
        root   = pass4_trees.get(tid)
        cterms = ", ".join(root.top_terms) if (root and root.top_terms) else f"T{rank}"

        cluster_handles.append(
            mpatches.Patch(color=col, alpha=0.7,
                           label=f"T{rank}  [{cterms}]  (n={len(p3_idx)})")
        )

        if root is None:
            ax.scatter(xy_all[p3_idx, 0], xy_all[p3_idx, 1],
                       s=18, color=col, alpha=0.85, zorder=3)
        else:
            leaves = collect_leaves(root)
            real_leaves  = [l for l in leaves if not l.is_noise]
            noise_leaves = [l for l in leaves if l.is_noise]

            sc_list = subcolors(col, len(real_leaves))

            for i, leaf in enumerate(real_leaves):
                sc     = sc_list[i]
                sterms = ", ".join(leaf.top_terms) if leaf.top_terms else ""
                lbl    = (f"  {leaf.label}  [{sterms}]  (n={leaf.n})" if sterms
                          else f"  {leaf.label}  (n={leaf.n})")
                ax.scatter(xy_all[leaf.global_idx, 0], xy_all[leaf.global_idx, 1],
                           s=24, color=sc, alpha=0.92, zorder=4)
                cluster_handles.append(mpatches.Patch(color=sc, label=lbl))

                # Thin dashed hull outline separating this sub-cluster's dots
                pts_leaf = xy_all[leaf.global_idx]
                if len(pts_leaf) >= 3:
                    try:
                        hull  = ConvexHull(pts_leaf)
                        verts = pts_leaf[hull.vertices]
                        c     = verts.mean(axis=0)
                        verts = c + 1.04 * (verts - c)
                        ax.add_patch(MplPolygon(
                            verts, closed=True, fill=False,
                            edgecolor=sc, linewidth=0.9,
                            linestyle="--", alpha=0.75, zorder=5,
                        ))
                    except Exception:
                        pass

            for leaf in noise_leaves:
                ax.scatter(xy_all[leaf.global_idx, 0], xy_all[leaf.global_idx, 1],
                           s=10, color=col, alpha=0.25, zorder=3)

        # Cluster centroid label
        pts    = xy_all[p3_idx]
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        ax.text(cx, cy, f"T{rank}\n{cterms}", fontsize=7.5, fontweight="bold",
                ha="center", va="center", color=col, multialignment="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75,
                          ec=col, lw=1.2))

    # 4. Non-top Pass-3 clusters — neutral dots
    for _, row in ti3.iterrows():
        tid = int(row["Topic"])
        if tid == -1 or tid in top_ids:
            continue
        idx = np.where(pass3_topics == tid)[0]
        ax.scatter(xy_all[idx, 0], xy_all[idx, 1],
                   s=9, color="#aaaaaa", alpha=0.45, zorder=3)

    # 5. Highlighted papers — drawn on top, each its own color
    markers = ["*", "D", "^", "P", "X"]
    for i, (label, hcol, idx) in enumerate(highlights):
        if len(idx) == 0:
            continue
        mk = markers[i % len(markers)]
        ax.scatter(xy_all[idx, 0], xy_all[idx, 1],
                   s=180, color=hcol, marker=mk,
                   edgecolors="black", linewidths=0.6, zorder=6)
        highlight_handles.append(
            Line2D([0], [0], marker=mk, color="w", markerfacecolor=hcol,
                   markeredgecolor="black", markersize=10, label=label)
        )

    # Cluster legend — inside axes, two columns, semi-transparent background
    leg1 = ax.legend(handles=cluster_handles, loc="upper left",
                     bbox_to_anchor=(0.01, 0.99), bbox_transform=ax.transAxes,
                     fontsize=7, frameon=True, framealpha=0.85,
                     title="Topics", title_fontsize=8,
                     ncol=2, borderpad=0.5, labelspacing=0.3,
                     columnspacing=0.8, handlelength=1.0)
    ax.add_artist(leg1)
    # Highlighted papers legend — bottom-left, inside axes
    if highlight_handles:
        ax.legend(handles=highlight_handles, loc="lower left",
                  bbox_to_anchor=(0.01, 0.01), bbox_transform=ax.transAxes,
                  fontsize=8, frameon=True, framealpha=0.85,
                  title="Highlighted", title_fontsize=8)

    ax.set_title("Hierarchical topic map — jet-physics arXiv corpus", fontsize=13)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    path = out("final_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df_full   = load_corpus(INPUT_CSV)
    docs_full = df_full["text"].tolist()

    # Embed once, reuse across all passes
    emb_full = embed_all(df_full)

    # 2-D projection for final plot (computed once on full corpus)
    xy_all = umap_2d(emb_full)

    # =========================================================================
    # PASS 1 — full corpus, coarse clustering
    # =========================================================================
    print("\n" + "#"*70)
    print("# PASS 1 — full corpus")
    print("#"*70)

    tm1 = run_bertopic(emb_full, docs_full, PASS1_NR_TOPICS, "Pass1")
    print_topics(tm1, "Pass 1", out("pass1_topics.csv"))

    # =========================================================================
    # PASS 2 — groom off-topic clusters, save groomed CSV
    # =========================================================================
    print("\n" + "#"*70)
    print("# PASS 2 — grooming")
    print("#"*70)

    ti1 = tm1.get_topic_info()
    offtopic_ids = {int(row["Topic"]) for _, row in ti1.iterrows()
                    if row["Topic"] != -1 and _is_offtopic(row["Representation"])}

    print(f"[Groom] Off-topic topics detected: {sorted(offtopic_ids)}")
    for tid in sorted(offtopic_ids):
        top5 = ", ".join(ti1[ti1["Topic"] == tid]["Representation"].iloc[0][:5])
        print(f"  T{tid}: {top5}")

    topics1      = np.array(tm1.topics_)
    groomed_mask = np.array([t not in offtopic_ids for t in topics1])  # True = kept
    df_groomed   = df_full[groomed_mask].reset_index(drop=True)
    emb_groomed  = emb_full[groomed_mask]

    groomed_csv = out("groomed_papers.csv")
    df_groomed.to_csv(groomed_csv, index=False)
    print(f"[Groom] Kept {len(df_groomed)} / {len(df_full)} papers → {groomed_csv}")

    # =========================================================================
    # PASS 3 — cluster groomed corpus
    # =========================================================================
    print("\n" + "#"*70)
    print("# PASS 3 — groomed corpus")
    print("#"*70)

    docs_groomed = df_groomed["text"].tolist()
    tm3 = run_bertopic(emb_groomed, docs_groomed, PASS3_NR_TOPICS, "Pass3")
    ti3 = print_topics(tm3, "Pass 3", out("pass3_topics.csv"))
    print("\nRepresentative papers:")
    print_reps(tm3, df_groomed)

    # Map Pass-3 topic assignments back to full-corpus indices
    topics3_groomed = np.array(tm3.topics_)
    pass3_topics    = np.full(len(df_full), -2, dtype=int)
    pass3_topics[groomed_mask] = topics3_groomed

    # Global indices for groomed papers (into df_full / xy_all)
    groomed_global_idx = np.where(groomed_mask)[0]

    # =========================================================================
    # PASS 4+ — recursive sub-clustering of top-N Pass-3 clusters
    # =========================================================================
    print("\n" + "#"*70)
    print(f"# PASS 4+ — recursive sub-clustering (radius_threshold={RADIUS_THRESHOLD}, "
          f"max_depth={MAX_CLUSTER_DEPTH})")
    print("#"*70)

    top_ids     = top_n_topics(ti3, N_TOP_CLUSTERS)
    pass4_trees = {}   # tid → ClusterNode (root of recursion tree)

    for rank, tid in enumerate(top_ids):
        # Indices within df_groomed that belong to this top cluster
        sub_idx = np.where(topics3_groomed == tid)[0]
        df_sub  = df_groomed.iloc[sub_idx].reset_index(drop=True)
        emb_sub = emb_groomed[sub_idx]
        global_idx = groomed_global_idx[sub_idx]

        label = f"T{rank}"
        print(f"\n{'─'*60}")
        print(f"Entering recursive_cluster for {label}  (n={len(df_sub)})")

        root = recursive_cluster(emb_sub, df_sub, global_idx, xy_all, label, depth=0)
        # Attach Pass-3 top terms to the root node
        p3_row = ti3[ti3["Topic"] == tid]
        if not p3_row.empty:
            root.top_terms = list(p3_row.iloc[0]["Representation"])[:N_LABEL_TERMS]
        pass4_trees[tid] = root

    # =========================================================================
    # FINAL PLOT
    # =========================================================================
    print("\n" + "#"*70)
    print("# Final composite plot")
    print("#"*70)

    final_plot(xy_all, df_full, groomed_mask, pass3_topics, pass4_trees, ti3)


if __name__ == "__main__":
    main()
