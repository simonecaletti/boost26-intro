#!/usr/bin/env python3
"""
Jet-physics-inspired topic clustering on arXiv jet-physics papers.

Analogy
-------
  particle  = paper
  pt        = count(ONTOPIC_KEYWORDS) − count(OFFTOPIC_KEYWORDS)  (floored at PT_EPSILON)
  position  = 2-D UMAP coordinates of title + abstract embedding

Pass 1  — generalized-kt clustering (p=P_EXPONENT, default 0 = C/A) with R=R_JET.
           Produces coarse "topic jets"; papers too far from any jet become beam remnants.
Pass 2  — SoftDrop grooming of each jet (z_cut, beta) to remove off-topic papers.
Pass 3  — re-cluster each groomed jet with R=R_SUB to find sub-jets (sub-topics).
Plot    — 2-D UMAP; jets as colored convex-hull rings; sub-jets as shaded dots;
          groomed / beam-remnant papers in grey.  Legend mirrors topic_discovery.py.

Dependencies:
    pip install scikit-learn pandas umap-learn sentence-transformers matplotlib scipy

Usage:
    python topic_clustering.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CSV        = "arxiv_jet_papers.csv"
OUTPUT_DIR       = "output_genkt"

# Set to True to reload UMAP coordinates and pt from a previous run's
# umap_coords.csv instead of re-embedding and re-projecting.
# Useful when tuning clustering / grooming parameters.
# Set to False (or delete umap_coords.csv) to force a full recomputation.
REUSE_UMAP       = True

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

# Generalized-kt exponent: p=0 → C/A, p=1 → kt, p=-1 → anti-kt
# anti-kt (p=-1) builds jets outward from the hardest (most on-topic) papers,
# giving compact, well-defined topic cores — best for finding the 5 target regions.
P_EXPONENT       = 1
R_JET            = 2.0   # Pass-1 jet radius in UMAP units; ~2 gives ~5-8 large jets
R_SUB            = 0.8   # Pass-3 sub-jet radius
MIN_JET_PAPERS   = 8     # discard jets smaller than this after grooming

# SoftDrop parameters
# z_cut=0.15 with beta=0 (mass-drop-like) aggressively drops low-pt papers;
# beta=1 is softer grooming — keep at 0.1 to preserve borderline papers.
Z_CUT            = 0.3
BETA             = 0.0   # beta=0: pure z-cut (mass-drop grooming), angle-independent

# Minimum pt (avoids zero/negative issues in the clustering distances)
PT_EPSILON       = 0.1

N_TOP_JETS       = 5     # how many groomed jets to sub-cluster in Pass 3
N_LABEL_TERMS    = 3     # TF-IDF terms shown as jet/sub-jet label
N_SUB_MIN        = 3     # minimum papers per sub-jet to keep it

UMAP_N_NEIGHBORS = 15
UMAP_METRIC      = "cosine"

# pt scoring keywords
# Generic jet-physics baseline
ONTOPIC_KEYWORDS = [
    "jet", "jets", "qcd", "lund", "quark", "gluon", "parton", "shower",
    "fragmentation", "substructure", "splitting", "collider", "lhc",
    "transverse momentum", "rapidity", "fastjet",
    # ML / tagging cluster
    "neural network", "deep learning", "machine learning", "transformer",
    "graph neural", "graph network", "bdt", "boosted decision", "autoencoder",
    "normalizing flow", "anomaly detection", "tagger", "tagging", "classifier",
    # Energy correlator cluster
    "energy correlator", "energy-energy correlator", "eec", "eeec",
    "projected correlator", "n-point correlator",
    # Heavy flavour / resummation cluster
    "b-jet", "b jet", "heavy quark", "bottom quark", "charm quark",
    "resummation", "sudakov", "nnll", "nnnlo", "next-to-next", "dglap",
    "soft-collinear", "soft collinear", "thrust", "event shape",
    # Non-perturbative cluster
    "hadronization", "power correction", "string model", "cluster model",
    "underlying event", "colour reconnection", "color reconnection",
    # Higgs cluster
    "higgs", "higgs boson", "boosted higgs", "vh production",
    # Jet substructure / grooming cluster
    "soft drop", "softdrop", "lund plane", "n-subjettiness",
    "mass drop", "trimming", "grooming", "angularity",
    "energy correlation", "jet charge", "fat jet", "boosted",
]

OFFTOPIC_KEYWORDS = [
    # Cosmology / BSM unrelated to jets
    "dark matter", "inflation", "inflaton", "reheating",
    "gravitational wave", "string theory", "moduli", "ads/cft",
    "conformal field", "bootstrap", "axion", "cosmological",
    # Heavy-ion / QGP (remove next two lines if you want this as a cluster)
    "quark-gluon plasma", "qgp", "jet quenching", "heavy-ion",
    "quenching weight", "nuclear modification",
    # extra 
    "majorana", "hyperons", "instantons", "ray", "reheating", 
    "inflation", "inflaton", "ell",
]

# Seed colors — extended automatically via golden-ratio HSV wheel
_JET_COLORS_SEED = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3",
                    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62"]

HIGHLIGHT_PAPERS = [
    ("Lund b-jet plane (Ghira et al.)", "2512.17408"),
]
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
    "letter", "letters", "journal", "proceedings", "mathrm", "pb",
    "scriptscriptstyle", "_s rightarrow", "pm0", "d_s", "p_", "tev",
]

# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np


CLUSTERS_DIR = os.path.join(OUTPUT_DIR, "clusters")


def out(filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, filename)


def cout(filename):
    os.makedirs(CLUSTERS_DIR, exist_ok=True)
    return os.path.join(CLUSTERS_DIR, filename)


# ---------------------------------------------------------------------------
# Corpus I/O
# ---------------------------------------------------------------------------

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
# pt — keyword-based transverse momentum
# ---------------------------------------------------------------------------

def compute_pt(df):
    """
    pt(paper) = Σ occurrences(ONTOPIC_KEYWORDS)  −  Σ occurrences(OFFTOPIC_KEYWORDS)
    Floored at PT_EPSILON so no paper has zero or negative pt.
    Papers that score PT_EPSILON are "soft" and will be easily groomed away.
    """
    import re
    pts = np.zeros(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        text  = row["text"].lower()
        score = 0.0
        for kw in ONTOPIC_KEYWORDS:
            score += len(re.findall(r'\b' + re.escape(kw) + r'\b', text))
        for kw in OFFTOPIC_KEYWORDS:
            score -= len(re.findall(r'\b' + re.escape(kw) + r'\b', text))
        pts[i] = max(PT_EPSILON, score)
    soft = (pts == PT_EPSILON).sum()
    print(f"[pt] range {pts.min():.1f} – {pts.max():.1f},  "
          f"mean {pts.mean():.1f},  soft (=ε) papers: {soft}")
    return pts


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
    from umap import UMAP
    print("[UMAP-2D] Projecting to 2-D ...")
    xy = UMAP(n_neighbors=UMAP_N_NEIGHBORS, n_components=2,
              metric=UMAP_METRIC, random_state=42).fit_transform(embeddings)
    print(f"[UMAP-2D] Done.  coord ranges: "
          f"x=[{xy[:,0].min():.2f},{xy[:,0].max():.2f}]  "
          f"y=[{xy[:,1].min():.2f},{xy[:,1].max():.2f}]")
    return xy


# ---------------------------------------------------------------------------
# PseudoJet — clustering history node
# ---------------------------------------------------------------------------

class PseudoJet:
    """
    A particle (single paper) or merged cluster of papers.

    Attributes
    ----------
    pt      : transverse momentum (keyword score, ≥ PT_EPSILON)
    x, y    : pt-weighted centroid in 2-D UMAP space
    indices : list of global paper indices in this pseudojet
    left    : harder child after the last merge (None for a leaf)
    right   : softer child after the last merge (None for a leaf)
    """
    __slots__ = ("pt", "x", "y", "indices", "left", "right")

    def __init__(self, pt, x, y, indices):
        self.pt      = float(pt)
        self.x       = float(x)
        self.y       = float(y)
        self.indices = list(indices)
        self.left    = None   # harder child
        self.right   = None   # softer child

    def merge_with(self, other):
        total_pt = self.pt + other.pt
        m        = PseudoJet(
            total_pt,
            (self.pt * self.x + other.pt * other.x) / total_pt,
            (self.pt * self.y + other.pt * other.y) / total_pt,
            self.indices + other.indices,
        )
        # Convention: left = harder (higher pt), right = softer
        if self.pt >= other.pt:
            m.left, m.right = self, other
        else:
            m.left, m.right = other, self
        return m

    @property
    def is_leaf(self):
        return self.left is None


# ---------------------------------------------------------------------------
# Generalized-kt clustering
# ---------------------------------------------------------------------------

def generalized_kt(pseudojets, R=1.0, p=0):
    """
    Generalized-kt family clustering algorithm (exact, O(n²) per step).

    Distance measures
    -----------------
    d_ij = min(pt_i^{2p}, pt_j^{2p}) × ΔR_ij² / R²
    d_iB = pt_i^{2p}

    Special cases
    -------------
    p = 0  →  Cambridge / Aachen : d_ij = ΔR²/R²,  d_iB = 1
    p = 1  →  kt                 : d_ij = min(pt²)×ΔR²/R²
    p = -1 →  anti-kt            : d_ij = min(pt⁻²)×ΔR²/R²

    Algorithm
    ---------
    Repeat until no pseudojets remain:
      • if  min(d_iB) ≤ min(d_ij)  →  promote i to a final jet
      • else                        →  merge closest pair (i, j)

    Returns a list of PseudoJet objects (the final jets), preserving the
    full merge tree in each jet's .left / .right attributes for SoftDrop.
    """
    pjs  = list(pseudojets)
    jets = []
    R2   = R * R

    def _pt2p(pt):
        return 1.0 if p == 0 else float(pt) ** (2 * p)

    while pjs:
        n     = len(pjs)
        pos   = np.array([[pj.x, pj.y] for pj in pjs])
        pt2p  = np.array([_pt2p(pj.pt) for pj in pjs])

        # Beam distances
        d_iB      = pt2p.copy()
        min_bi    = int(d_iB.argmin())
        min_beam  = d_iB[min_bi]

        # Pairwise distances (vectorised)
        diff  = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n,n,2)
        dr2   = (diff * diff).sum(axis=2)                       # (n,n)
        ptfac = np.minimum(pt2p[:, np.newaxis], pt2p[np.newaxis, :])
        d_ij  = ptfac * dr2 / R2
        np.fill_diagonal(d_ij, np.inf)

        min_pair = d_ij.min()

        if min_beam <= min_pair:
            # Particle min_bi is farther than R from all others → becomes a jet
            jets.append(pjs[min_bi])
            pjs.pop(min_bi)
        else:
            # Merge closest pair
            flat   = int(d_ij.argmin())
            i, j   = divmod(flat, n)
            merged = pjs[i].merge_with(pjs[j])
            pjs    = [pjs[k] for k in range(n) if k != i and k != j]
            pjs.append(merged)

    return jets


# ---------------------------------------------------------------------------
# SoftDrop grooming
# ---------------------------------------------------------------------------

def softdrop_indices(jet, z_cut=Z_CUT, beta=BETA, R0=R_JET):
    """
    Apply SoftDrop to the merge history of `jet`.

    Starting from the jet root, iteratively undo the last clustering step:
      harder, softer ← undo(current node)
      z   = pt_softer / (pt_harder + pt_softer)
      ΔR  = UMAP distance between harder and softer centroids
      if z ≥ z_cut × (ΔR/R0)^beta  →  PASS: return all indices in current node
      else                           →  FAIL: drop softer, continue on harder

    Returns the list of global paper indices that survive grooming.
    The iterative (non-recursive) implementation avoids Python stack limits
    for large jets.
    """
    node = jet
    while not node.is_leaf:
        harder, softer = node.left, node.right   # left is harder by convention
        z   = softer.pt / (harder.pt + softer.pt)
        dr  = np.sqrt((harder.x - softer.x) ** 2 + (harder.y - softer.y) ** 2)
        if z >= z_cut * (dr / R0) ** beta:
            break          # condition passes → keep this node's full content
        node = harder      # drop softer branch, descend into harder
    return list(node.indices)


# ---------------------------------------------------------------------------
# TF-IDF label extraction
# ---------------------------------------------------------------------------

def extract_labels(df_subset, n=N_LABEL_TERMS):
    """Top TF-IDF n-grams for a single subset of papers (used as fallback)."""
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    texts = df_subset["text"].tolist()
    if len(texts) < 2:
        return []
    sw = list(ENGLISH_STOP_WORDS) + EXTRA_STOPWORDS
    try:
        vec    = TfidfVectorizer(stop_words=sw, ngram_range=(1, 2),
                                  max_features=2000, min_df=1)
        X      = vec.fit_transform(texts)
        scores = np.asarray(X.mean(axis=0)).flatten()
        top    = scores.argsort()[::-1][:n]
        return [vec.get_feature_names_out()[i] for i in top]
    except Exception:
        return []


def extract_discriminative_labels(groups_df, n=N_LABEL_TERMS):
    """
    Extract discriminative labels for multiple groups simultaneously.

    Strategy
    --------
    1. Fit TF-IDF on ALL individual papers from all groups combined.
       IDF is therefore computed over hundreds of documents — meaningful.
    2. Compute the average TF-IDF score per term for each group.
    3. Discriminativeness score for group i, term t:
           disc(i, t) = avg_tfidf(i, t) / (Σ_j avg_tfidf(j, t) + ε)
       This is the group's *share* of the term's total weight.
       A term common to all groups scores ≈ 1/N; a term exclusive to one
       group scores ≈ 1.  Multiply by avg_tfidf to suppress very rare terms.

    groups_df : list of DataFrames (one per group, each with a 'text' column)
    Returns   : list of lists of top-n term strings, one list per group
    """
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    if not groups_df:
        return []
    if len(groups_df) == 1:
        return [extract_labels(groups_df[0], n)]

    sw         = list(ENGLISH_STOP_WORDS) + EXTRA_STOPWORDS
    all_texts  = []
    boundaries = []
    for df in groups_df:
        texts = df["text"].tolist()
        boundaries.append((len(all_texts), len(all_texts) + len(texts)))
        all_texts.extend(texts)

    if len(all_texts) < 2:
        return [[] for _ in groups_df]
    try:
        vec        = TfidfVectorizer(stop_words=sw, ngram_range=(1, 2),
                                      max_features=3000, min_df=2)
        X          = vec.fit_transform(all_texts)          # (n_papers, n_terms)
        feat_names = vec.get_feature_names_out()

        # Average TF-IDF per group → (n_groups, n_terms)
        group_avg = np.zeros((len(groups_df), X.shape[1]))
        for i, (s, e) in enumerate(boundaries):
            if e > s:
                group_avg[i] = np.asarray(X[s:e].mean(axis=0)).flatten()

        # Discriminativeness: group's share of each term's total weight
        total        = group_avg.sum(axis=0) + 1e-10   # (n_terms,)
        disc         = group_avg / total                # (n_groups, n_terms)
        # Final score: discriminativeness × actual TF-IDF presence
        final_scores = disc * group_avg

        result = []
        for i in range(len(groups_df)):
            scores = final_scores[i]
            top    = scores.argsort()[::-1][:n]
            result.append([feat_names[j] for j in top if scores[j] > 0])
        return result
    except Exception:
        return [[] for _ in groups_df]


# ---------------------------------------------------------------------------
# Color helpers  (mirrors topic_discovery.py)
# ---------------------------------------------------------------------------

def _jet_colors(n):
    import colorsys
    colors = list(_JET_COLORS_SEED)
    while len(colors) < n:
        i = len(colors)
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.85)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors[:n]


def subcolors(base_hex, n):
    """n perceptually distinct colors that stay in the hue family of base_hex."""
    import colorsys
    import matplotlib.colors as mc
    if n <= 0:
        return []
    r, g, b = mc.to_rgb(base_hex)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    colors  = []
    for i in range(n):
        t     = i / max(n - 1, 1)
        dh    = (t - 0.5) * 0.14
        new_h = (h + dh) % 1.0
        new_s = 0.55 + 0.35 * (0.5 + 0.5 * np.cos(np.pi * t))
        new_v = 0.55 + 0.35 * t
        colors.append(colorsys.hsv_to_rgb(new_h, new_s, new_v))
    return colors


# ---------------------------------------------------------------------------
# Highlight helper  (mirrors topic_discovery.py)
# ---------------------------------------------------------------------------

def _find_highlighted(df_full):
    results = []
    for i, (label, arxiv_id) in enumerate(HIGHLIGHT_PAPERS):
        col  = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
        # Match on arxiv_id column; strip version suffix (e.g. "2407.08158v2" → "2407.08158")
        mask = df_full["arxiv_id"].astype(str).str.split("v").str[0] == arxiv_id.strip()
        idx  = np.where(mask.values)[0]
        if len(idx) == 0:
            print(f"  [Highlight] WARNING: arXiv ID '{arxiv_id}' not found in corpus")
        else:
            print(f"  [Highlight] '{label}': matched arXiv:{arxiv_id}")
            for j in idx:
                print(f"    [{df_full.iloc[j]['arxiv_id']}] "
                      f"{df_full.iloc[j]['title'][:80]}")
        results.append((label, col, idx))
    return results


# ---------------------------------------------------------------------------
# Final composite plot
# ---------------------------------------------------------------------------

def final_plot(xy_all, df_full, jet_results, all_raw_indices):
    """
    Parameters
    ----------
    jet_results     : list of dicts (see main() for structure)
    all_raw_indices : set of all global indices that ended up in *any* jet
                      (used to identify beam remnants)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon as MplPolygon

    n_jets  = len(jet_results)
    palette = _jet_colors(n_jets)

    # Identify beam remnants (papers not in any raw jet)
    beam_idx = np.array([i for i in range(len(df_full))
                         if i not in all_raw_indices], dtype=int)

    # Collect all SoftDrop-groomed-out indices across jets
    all_dropped = set()
    for jr in jet_results:
        all_dropped.update(jr["dropped_idx"].tolist())
    dropped_arr = np.array(sorted(all_dropped), dtype=int)

    highlights = _find_highlighted(df_full) if HIGHLIGHT_PAPERS else []

    fig, ax = plt.subplots(figsize=(16, 10))
    cluster_handles   = []
    highlight_handles = []

    # 1. Beam remnants — very light grey (smallest, drawn first)
    if len(beam_idx) > 0:
        ax.scatter(xy_all[beam_idx, 0], xy_all[beam_idx, 1],
                   s=5, color="#e0e0e0", alpha=0.35, zorder=1)

    # 2. SoftDrop-groomed papers — light grey
    if len(dropped_arr) > 0:
        ax.scatter(xy_all[dropped_arr, 0], xy_all[dropped_arr, 1],
                   s=8, color="#cccccc", alpha=0.55, zorder=1)

    total_soft = len(beam_idx) + len(dropped_arr)
    cluster_handles.append(
        mpatches.Patch(color="#cccccc",
                       label=f"groomed / beam remnants  (n={total_soft})")
    )

    # 3. Jet hulls + leaf dots + sub-jet dashed hulls + legend entries
    for rank, jr in enumerate(jet_results):
        col      = palette[rank]
        all_idx  = jr["all_idx"]
        kept_idx = jr["kept_idx"]
        terms    = ", ".join(jr["top_terms"]) if jr["top_terms"] else f"J{rank}"

        cluster_handles.append(
            mpatches.Patch(color=col, alpha=0.7,
                           label=f"J{rank}  [{terms}]  (n={len(kept_idx)})")
        )

        # Convex hull of the full (pre-groom) jet extent
        pts = xy_all[all_idx]
        if len(pts) >= 3:
            try:
                hull  = ConvexHull(pts)
                verts = pts[hull.vertices]
                c     = verts.mean(axis=0)
                verts = c + 1.08 * (verts - c)
                ax.add_patch(MplPolygon(verts, closed=True, fill=True,
                                        facecolor=col, alpha=0.07,
                                        edgecolor=col, linewidth=1.8, zorder=2))
            except Exception:
                pass

        sub_jets = jr["sub_jets"]
        if not sub_jets:
            # No sub-clustering: all kept papers as one solid color
            ax.scatter(xy_all[kept_idx, 0], xy_all[kept_idx, 1],
                       s=18, color=col, alpha=0.85, zorder=3)
        else:
            sc_list = subcolors(col, len(sub_jets))
            for si, sj in enumerate(sub_jets):
                sc    = sc_list[si]
                sterms = ", ".join(sj["top_terms"]) if sj["top_terms"] else ""
                lbl   = (f"  {sj['label']}  [{sterms}]  (n={len(sj['idx'])})"
                         if sterms else f"  {sj['label']}  (n={len(sj['idx'])})")
                sidx  = sj["idx"]

                ax.scatter(xy_all[sidx, 0], xy_all[sidx, 1],
                           s=22, color=sc, alpha=0.90, zorder=4)
                cluster_handles.append(mpatches.Patch(color=sc, label=lbl))

                # Dashed convex hull for this sub-jet
                pts_sj = xy_all[sidx]
                if len(pts_sj) >= 3:
                    try:
                        hull  = ConvexHull(pts_sj)
                        verts = pts_sj[hull.vertices]
                        c     = verts.mean(axis=0)
                        verts = c + 1.04 * (verts - c)
                        ax.add_patch(MplPolygon(
                            verts, closed=True, fill=False,
                            edgecolor=sc, linewidth=0.9,
                            linestyle="--", alpha=0.75, zorder=5,
                        ))
                    except Exception:
                        pass

            # Sub-jet noise (papers in jet but not in any sub-jet)
            sub_covered = set()
            for sj in sub_jets:
                sub_covered.update(sj["idx"].tolist())
            noise_idx = np.array([i for i in kept_idx if i not in sub_covered], dtype=int)
            if len(noise_idx) > 0:
                ax.scatter(xy_all[noise_idx, 0], xy_all[noise_idx, 1],
                           s=10, color=col, alpha=0.25, zorder=3)

        # Centroid label
        cx = xy_all[kept_idx, 0].mean()
        cy = xy_all[kept_idx, 1].mean()
        ax.text(cx, cy, f"J{rank}\n{terms}", fontsize=7.5, fontweight="bold",
                ha="center", va="center", color=col, multialignment="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75,
                          ec=col, lw=1.2))

    # 4. Highlighted papers — drawn on top
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

    # Legend — mirrors topic_discovery.py layout
    leg1 = ax.legend(handles=cluster_handles, loc="upper left",
                     bbox_to_anchor=(0.01, 0.99), bbox_transform=ax.transAxes,
                     fontsize=7, frameon=True, framealpha=0.85,
                     title="Jets / Sub-jets", title_fontsize=8,
                     ncol=2, borderpad=0.5, labelspacing=0.3,
                     columnspacing=0.8, handlelength=1.0)
    ax.add_artist(leg1)
    if highlight_handles:
        ax.legend(handles=highlight_handles, loc="lower left",
                  bbox_to_anchor=(0.01, 0.01), bbox_transform=ax.transAxes,
                  fontsize=8, frameon=True, framealpha=0.85,
                  title="Highlighted", title_fontsize=8)

    title = (f"Jet-physics topic map — arXiv jet corpus\n"
             f"C/A  R={R_JET}  (p={P_EXPONENT})   |   "
             f"SoftDrop  z_cut={Z_CUT}  β={BETA}   |   "
             f"sub-jet  R={R_SUB}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    path = out("final_clustering.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df_full = load_corpus(INPUT_CSV)

    # --- pt (keyword score) ---
    print("\n[pt] Computing keyword-based transverse momenta ...")
    pts = compute_pt(df_full)

    # --- Embeddings & 2-D UMAP ---
    umap_path   = out("umap_coords.csv")
    cached_umap = REUSE_UMAP and os.path.exists(umap_path)

    if cached_umap:
        print(f"\n[UMAP] Loading cached coordinates from {umap_path} ...")
        umap_df = pd.read_csv(umap_path)
        # Align to df_full by arxiv_id in case row order differs
        umap_df = umap_df.set_index("arxiv_id").reindex(df_full["arxiv_id"].values)
        if umap_df[["umap_x", "umap_y"]].isnull().any().any():
            print("[UMAP] WARNING: some papers missing from cache — falling back to recompute")
            cached_umap = False
        else:
            xs     = umap_df["umap_x"].values.astype(float)
            ys     = umap_df["umap_y"].values.astype(float)
            xy_all = np.column_stack([xs, ys])
            print(f"[UMAP] Loaded {len(xs)} rows.  coord ranges: "
                  f"x=[{xs.min():.2f},{xs.max():.2f}]  y=[{ys.min():.2f},{ys.max():.2f}]")

    if not cached_umap:
        emb_full = embed_all(df_full)
        xy_all   = umap_2d(emb_full)
        xs, ys   = xy_all[:, 0], xy_all[:, 1]
        umap_out = df_full[["arxiv_id", "title"]].copy()
        umap_out["umap_x"] = xs
        umap_out["umap_y"] = ys
        umap_out["pt"]     = pts
        umap_out.to_csv(umap_path, index=False)
        print(f"[UMAP] Coordinates saved → {umap_path}")

    # =========================================================================
    # PASS 1 — generalized-kt clustering (C/A by default)
    # =========================================================================
    print("\n" + "#" * 70)
    algo = {0: "Cambridge/Aachen", 1: "kt", -1: "anti-kt"}.get(P_EXPONENT,
           f"gen-kt(p={P_EXPONENT})")
    print(f"# PASS 1 — {algo}  R={R_JET}")
    print("#" * 70)

    pseudojets = [PseudoJet(pts[i], xs[i], ys[i], [i]) for i in range(len(df_full))]
    print(f"  Clustering {len(pseudojets)} papers ...")
    raw_jets = generalized_kt(pseudojets, R=R_JET, p=P_EXPONENT)
    raw_jets.sort(key=lambda j: len(j.indices), reverse=True)
    print(f"  → {len(raw_jets)} raw jets")
    for k, j in enumerate(raw_jets[:8]):
        print(f"    J{k}: n={len(j.indices)}  pt={j.pt:.1f}")

    # Global index set of all papers that ended up in any jet
    all_raw_indices = set()
    for j in raw_jets:
        all_raw_indices.update(j.indices)
    beam_count = len(df_full) - len(all_raw_indices)
    print(f"  → {beam_count} beam remnants (single-paper jets outside R={R_JET})")

    # =========================================================================
    # PASS 2 — SoftDrop grooming
    # =========================================================================
    print("\n" + "#" * 70)
    print(f"# PASS 2 — SoftDrop grooming  z_cut={Z_CUT}  β={BETA}  R₀={R_JET}")
    print("#" * 70)

    jet_results = []
    for k, jet in enumerate(raw_jets):
        kept_idx    = np.array(softdrop_indices(jet, Z_CUT, BETA, R_JET), dtype=int)
        all_idx     = np.array(jet.indices, dtype=int)
        dropped_idx = np.setdiff1d(all_idx, kept_idx)
        frac        = len(kept_idx) / max(len(all_idx), 1) * 100
        print(f"  J{k}: {len(all_idx):4d} → {len(kept_idx):4d} kept  "
              f"({len(dropped_idx)} groomed,  {frac:.0f}%)")

        if len(kept_idx) < MIN_JET_PAPERS:
            print(f"       below MIN_JET_PAPERS={MIN_JET_PAPERS}, skipping")
            continue

        jet_results.append({
            "all_idx":     all_idx,
            "kept_idx":    kept_idx,
            "dropped_idx": dropped_idx,
            "top_terms":   [],   # filled below after all jets are known
            "sub_jets":    [],
        })

    # Sort by groomed-jet size and assign ranks
    jet_results.sort(key=lambda jr: len(jr["kept_idx"]), reverse=True)
    for rank, jr in enumerate(jet_results):
        jr["jet_idx"] = rank

    # Discriminative labels across all jets (IDF penalises terms common to many jets)
    jet_dfs   = [df_full.iloc[jr["kept_idx"]].reset_index(drop=True) for jr in jet_results]
    jet_terms = extract_discriminative_labels(jet_dfs)
    for jr, terms in zip(jet_results, jet_terms):
        jr["top_terms"] = terms

    print(f"\n  {len(jet_results)} jets survive grooming + size threshold")
    for jr in jet_results:
        rank  = jr["jet_idx"]
        terms = ", ".join(jr["top_terms"])
        print(f"    J{rank}: n={len(jr['kept_idx'])}  [{terms}]")
        jet_csv = df_full.iloc[jr["kept_idx"]].copy()
        jet_csv["umap_x"] = xs[jr["kept_idx"]]
        jet_csv["umap_y"] = ys[jr["kept_idx"]]
        jet_csv["pt"]     = pts[jr["kept_idx"]]
        jet_csv.to_csv(cout(f"J{rank}.csv"), index=False)

    # =========================================================================
    # PASS 3 — sub-clustering within each top jet  (R = R_SUB)
    # =========================================================================
    print("\n" + "#" * 70)
    print(f"# PASS 3 — sub-clustering top-{N_TOP_JETS} jets  R={R_SUB}")
    print("#" * 70)

    for jr in jet_results[:N_TOP_JETS]:
        kidx  = jr["kept_idx"]
        rank  = jr["jet_idx"]
        print(f"\n  J{rank}  n={len(kidx)}")

        sub_pjs = [PseudoJet(pts[i], xs[i], ys[i], [i]) for i in kidx]
        sub_raw = generalized_kt(sub_pjs, R=R_SUB, p=P_EXPONENT)
        sub_raw = [sj for sj in sub_raw if len(sj.indices) >= N_SUB_MIN]
        sub_raw.sort(key=lambda sj: len(sj.indices), reverse=True)

        if len(sub_raw) < 2:
            print(f"  J{rank}: only {len(sub_raw)} qualifying sub-jet(s) — skipping")
            continue

        # Collect all sub-jets first, then compute discriminative labels in one shot
        sub_entries = []
        for si, sj in enumerate(sub_raw):
            sidx  = np.array(sj.indices, dtype=int)
            label = f"J{rank}{chr(ord('a') + si)}"
            sub_entries.append({"label": label, "idx": sidx})

        sub_dfs   = [df_full.iloc[e["idx"]].reset_index(drop=True) for e in sub_entries]
        sub_terms = extract_discriminative_labels(sub_dfs)

        for entry, sterms in zip(sub_entries, sub_terms):
            entry["top_terms"] = sterms
            jr["sub_jets"].append(entry)
            print(f"    {entry['label']}: n={len(entry['idx']):4d}  [{', '.join(sterms)}]")
            sjet_csv = df_full.iloc[entry["idx"]].copy()
            sjet_csv["umap_x"] = xs[entry["idx"]]
            sjet_csv["umap_y"] = ys[entry["idx"]]
            sjet_csv["pt"]     = pts[entry["idx"]]
            sjet_csv.to_csv(cout(f"{entry['label']}.csv"), index=False)

    # =========================================================================
    # FINAL PLOT
    # =========================================================================
    print("\n" + "#" * 70)
    print("# Final composite plot")
    print("#" * 70)

    final_plot(xy_all, df_full, jet_results, all_raw_indices)


if __name__ == "__main__":
    main()
