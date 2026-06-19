# genkt.py — Logic and Design

## Overview

`genkt.py` clusters a corpus of arXiv jet-physics papers by drawing a direct analogy between
papers and particles in a collider event.  Each paper is treated as a particle whose position
is its location in a 2-D UMAP embedding of its title + abstract, and whose transverse momentum
(**pt**) measures how on-topic it is.  The jet-finding and grooming algorithms from collider
physics — generalized-kt and SoftDrop — are then run on this space to discover topic clusters
("jets") and remove off-topic papers ("grooming").

---

## The Particle-Physics Analogy

| Collider concept | Script concept |
|-----------------|---------------|
| Particle | Paper |
| Transverse momentum pt | Keyword relevance score (IDF-weighted) |
| Position (η, φ) | 2-D UMAP coordinates of title+abstract embedding |
| Jet | Topic cluster |
| Beam remnant | Paper too isolated to join any jet |
| SoftDrop grooming | Removal of low-relevance papers from a jet |
| Sub-jet | Sub-topic within a jet |

---

## Pipeline

### Step 0 — Corpus loading

Reads `arxiv_jet_papers.csv` (columns: `arxiv_id`, `title`, `abstract`, ...).
The `text` field used for all downstream NLP is `title + ". " + abstract`.

---

### Step 1 — pt scoring (keyword-based transverse momentum)

Each paper receives a scalar **pt** score that measures how on-topic it is.

**Method: IDF-weighted keyword counting**

1. For each keyword `kw` in `ONTOPIC_KEYWORDS` and `OFFTOPIC_KEYWORDS`, compute its
   document frequency `df(kw)` = number of papers containing it at least once.

2. Compute the inverse document frequency with add-1 smoothing:
   ```
   idf(kw) = log( N / (1 + df(kw)) )
   ```
   Keywords that appear in almost every paper (e.g. "jet") get low IDF weight.
   Rare, specific keywords (e.g. "Lund plane") get high weight.

3. For each paper, sum IDF-weighted occurrence counts:
   ```
   pt = Σ count(kw) × idf(kw)   for kw in ONTOPIC_KEYWORDS
      − Σ count(kw) × idf(kw)   for kw in OFFTOPIC_KEYWORDS
   pt = max(PT_EPSILON, pt)
   ```
   The floor `PT_EPSILON = 0.1` avoids zero or negative pt values, which would
   cause numerical issues in the distance formulas.

---

### Step 2 — Embedding and 2-D UMAP projection

**Embedding**: Each paper's `text` is encoded with a sentence-transformer model
(`allenai/specter2_base` by default, a model pre-trained on scientific paper
citation networks).  This produces a high-dimensional vector (768-d) per paper
that captures semantic meaning.

**2-D UMAP**: The embeddings are projected to 2 dimensions using UMAP
(`n_neighbors=15`, `metric="cosine"`).  These 2-D coordinates serve as both
the "position" for jet clustering and the axes of the final plot.

The 2-D coordinates are cached in `<output_dir>/umap_coords.csv` (controlled
by `--no-reuse-umap`).

---

### Step 3 (Pass 1) — Generalized-kt jet clustering

The core clustering step runs the **generalized-kt family** algorithm
(Catani et al., Ellis & Soper) on the 2-D UMAP space.

#### Distance measures

For two pseudojets *i* and *j*:
```
d_ij = min(pt_i^{2p}, pt_j^{2p}) × ΔR_ij² / R²
d_iB = pt_i^{2p}          (beam distance for particle i)
```
where `ΔR_ij` is the Euclidean distance in UMAP space and `R` is the jet radius.

The exponent **p** selects the algorithm variant:

| p | Algorithm | Behaviour |
|---|-----------|-----------|
| 0 | Cambridge/Aachen | Pure geometry; merges closest pairs regardless of pt |
| +1 | kt | Merges softest particles first (bottom-up) |
| −1 | anti-kt | Hardest particles cluster outward; gives circular, well-defined jets |

#### Algorithm (O(n²) per step)

```
repeat until no pseudojets remain:
    compute all d_ij and all d_iB
    if min(d_iB) ≤ min(d_ij):
        promote particle i to a final jet  (it is isolated)
    else:
        merge the closest pair (i, j) into a new pseudojet
            new pt  = pt_i + pt_j
            new x,y = pt-weighted centroid
            merge tree: left = harder, right = softer
```

Particles that never get close enough to any neighbour are promoted individually
as **beam remnants** — papers too isolated to belong to any topic.

The full merge history is preserved in each jet's `.left` / `.right` tree,
which is required by SoftDrop.

---

### Step 4 (Pass 2) — SoftDrop grooming

SoftDrop (Larkoski, Marzani, Salam, Soyez) removes soft, wide-angle radiation
from each jet by walking down its merge tree.

#### Algorithm (iterative, single branch)

```
node ← jet root
while node is not a leaf:
    harder, softer ← node.left, node.right   (left is always harder)
    z   = pt_softer / (pt_harder + pt_softer)
    ΔR  = UMAP distance between harder and softer centroids
    if z ≥ z_cut × (ΔR / R₀)^β:
        PASS — keep all papers in current node; stop
    else:
        FAIL — drop the softer branch; descend into harder
return node.indices
```

The condition `z ≥ z_cut × (ΔR/R₀)^β` is the SoftDrop criterion.

- **z_cut** controls the minimum momentum fraction required to keep a branch.
  Higher values = more aggressive grooming.
- **β** (beta) modulates the angular dependence.
  `β = 0`: pure z-cut (mass-drop grooming), angle-independent.
  `β > 0`: large-angle branches are less penalised; softer grooming.
  `β < 0`: large-angle branches are penalised more.

Papers surviving grooming = the `node.indices` at the point where the
condition first passes.  Papers dropped = the softer branches that failed.

Jets with fewer than `MIN_JET_PAPERS` papers after grooming are discarded entirely.

---

### Step 5 (Pass 3) — Sub-jet finding

For each of the top `N_PLOT_JETS` groomed jets, the same generalized-kt
algorithm is re-run with a smaller radius `R_SUB < R_JET`.  This finds
**sub-jets** — finer topic clusters within each jet.

Sub-jets with fewer than `N_SUB_MIN` papers are discarded.  If fewer than 2
qualifying sub-jets are found, the jet is left unsplit.

Sub-jets are labelled `J{rank}{letter}`, e.g. `J0a`, `J0b`, `J1a`, etc.

---

### Step 6 — Discriminative label extraction

Topic labels (shown in the legend and centroid boxes) are extracted via
**discriminative TF-IDF** across all jets simultaneously.

1. Pool all papers from all jets into one TF-IDF fit.  Because IDF is computed
   over the joint corpus, terms that appear across many jets get penalised.
2. Compute the average TF-IDF score per term for each group:
   `group_avg[i, t]`.
3. Compute the discriminativeness score:
   ```
   disc[i, t] = group_avg[i, t] / (Σ_j group_avg[j, t] + ε)
   ```
   A term exclusive to one jet scores ≈ 1; a term common to all scores ≈ 1/N.
4. Final score = `disc × group_avg` (suppresses very rare terms).

This produces labels that are *specific* to each jet rather than generic
jet-physics vocabulary.

---

### Step 7 — Final plot

A single 2-D UMAP scatter plot showing the full corpus:

| Layer | Content |
|-------|---------|
| Very light grey dots (zorder 1) | Beam remnants (never joined any jet) |
| Light grey dots (zorder 1) | SoftDrop-groomed papers |
| Shaded convex hull polygon (zorder 2) | Full extent of each jet (pre-groom), 7% opacity |
| Coloured dots (zorder 4) | Sub-jet papers, each sub-jet in a distinct hue variant |
| Dashed convex hull outlines (zorder 5) | Sub-jet boundaries |
| `J{n}` box at jet centroid (zorder 7) | Jet label, white box with coloured border |
| `{a/b/c}` box at sub-jet centroid (zorder 6) | Sub-jet letter label |
| Highlighted star/diamond markers (zorder 6) | Individual papers from `HIGHLIGHT_PAPERS` |

**Legend structure**:
- One white-bordered rectangle per jet (no fill), showing terms and paper count.
- Coloured patches for each sub-jet, indented under their parent jet.

---

## Alternative Clustering Modes (`--clustering`)

The `--clustering` flag replaces Pass 1 + Pass 2 + Pass 3 entirely.
The same embedding, UMAP projection, pt scoring, label extraction, and plot are reused.

### `agglomerative`

Runs `sklearn.AgglomerativeClustering` (Ward linkage) on the 2-D UMAP coordinates.
`N_PLOT_JETS + 4` initial clusters are computed; clusters smaller than `MIN_JET_PAPERS`
become beam remnants.  Sub-clustering also uses agglomerative clustering.
No SoftDrop grooming step.

### `nmf`

Runs NMF (Non-negative Matrix Factorization) on a TF-IDF matrix of all papers.
Each NMF component is a topic.  Papers in the lowest-weight percentile (15%)
of their dominant component are treated as groomed.  Sub-topics are found by
running NMF again within each jet.  Labels come directly from the NMF basis vectors.

---

## Key Configuration Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--p` | −1 | Generalized-kt exponent (0=C/A, 1=kt, −1=anti-kt) |
| `--r-jet` | 1.0 | Pass-1 jet radius in UMAP units |
| `--r-sub` | 0.6 | Pass-3 sub-jet radius |
| `--z-cut` | 0.3 | SoftDrop momentum fraction threshold |
| `--beta` | 2.0 | SoftDrop angular exponent |
| `--n-jets` | 7 | Number of jets to plot and sub-cluster |
| `--n-label` | 3 | TF-IDF terms per label |
| `--min-jet` | 8 | Minimum papers per jet after grooming |
| `--n-sub-min` | 3 | Minimum papers per sub-jet |
| `--clustering` | genkt | Algorithm: `genkt`, `agglomerative`, `nmf` |
| `--no-reuse-umap` | — | Force UMAP recomputation |
| `--output-dir` | output_genkt | Directory for all outputs |

---

## Output Files

| File | Content |
|------|---------|
| `<output_dir>/umap_coords.csv` | 2-D UMAP coordinates + pt per paper |
| `<output_dir>/groomed_papers.csv` | Papers removed by SoftDrop + beam remnants |
| `<output_dir>/final_clustering.png` | Final composite plot |
| `<output_dir>/clusters/J{n}.csv` | Papers in jet n (post-grooming) |
| `<output_dir>/clusters/J{n}{letter}.csv` | Papers in sub-jet n{letter} |
