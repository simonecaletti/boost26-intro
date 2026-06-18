#!/usr/bin/env python3
"""
Build a word cloud from the arXiv jet-physics CSV produced by the scraper.

Reads titles + abstracts, strips common English and physics-boilerplate
stopwords, and renders a PNG word cloud.

Dependencies:
    pip install wordcloud matplotlib nltk
"""

import csv
import re
import sys
from collections import Counter, defaultdict

import nltk
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK data silently on first run.
nltk.download("punkt", quiet=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CSV = "arxiv_jet_papers.csv"
OUTPUT_PNG = "arxiv_jet_wordcloud.png"

# Which columns to pull text from.
TEXT_COLUMNS = ["title", "abstract"]

# Extra stopwords on top of the wordcloud library's built-in English set.
# Mostly generic academic / arXiv boilerplate that drowns out the signal.
EXTRA_STOPWORDS = {
    "we", "us", "our", "using", "use", "used", "show", "shown", "study",
    "studied", "result", "results", "present", "presented", "paper",
    "approach", "method", "methods", "model", "models", "based", "new",
    "also", "however", "thus", "therefore", "within", "via", "given",
    "obtain", "obtained", "find", "found", "consider", "considered",
    "propose", "proposed", "provide", "provided", "case", "cases",
    "different", "various", "well", "may", "can", "one", "two", "three",
    "first", "second", "order", "high", "low", "large", "small", "set",
    "function", "functions", "value", "values", "data", "analysis",
    "respectively", "compared", "comparison", "et", "al", "e", "g",
    "i", "ii", "iii", "left", "right", "mathrm", "rm", "text",
    "leq", "geq", "sim", "approx", "times", "x", "k", "n",
}


def load_text(path):
    chunks = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = [c for c in TEXT_COLUMNS if c in reader.fieldnames]
            if not cols:
                sys.exit(f"None of {TEXT_COLUMNS} found in {path}. "
                         f"Columns are: {reader.fieldnames}")
            for row in reader:
                for c in cols:
                    if row.get(c):
                        chunks.append(row[c])
    except FileNotFoundError:
        sys.exit(f"Input file not found: {path}")
    return " ".join(chunks)


def clean(text):
    text = text.lower()
    # Drop LaTeX-ish tokens: $...$, \commands, braces.
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[{}^_]", " ", text)
    # Keep words and intra-word hyphens (so "quark-jet" survives).
    tokens = re.findall(r"[a-z]+(?:-[a-z]+)*", text)
    # Drop very short tokens.
    tokens = [t for t in tokens if len(t) > 2]
    return tokens


def stem_merge(tokens):
    """
    Collapse tokens that share a Porter stem, then represent each stem by
    the most frequent surface form (e.g. stem 'jet' → display 'jets' if
    'jets' appeared more often than 'jet').

    Returns a {display_word: merged_count} Counter.
    """
    stemmer = PorterStemmer()

    # Count raw surface forms first.
    surface_counts = Counter(tokens)

    # Group surface forms by stem.
    stem_to_surfaces = defaultdict(Counter)
    for word, n in surface_counts.items():
        stem = stemmer.stem(word)
        stem_to_surfaces[stem][word] += n

    # For each stem: pick the most common surface form, sum all counts.
    merged = Counter()
    for stem, surfaces in stem_to_surfaces.items():
        display = surfaces.most_common(1)[0][0]
        merged[display] = sum(surfaces.values())

    return merged


def main():
    raw = load_text(INPUT_CSV)
    tokens = clean(raw)

    stopwords = set(WordCloud().stopwords) | EXTRA_STOPWORDS
    tokens = [t for t in tokens if t not in stopwords]

    freqs = stem_merge(tokens)
    if not freqs:
        sys.exit("No words left after filtering.")

    print("Top 25 terms (after stem-merging):")
    for word, n in freqs.most_common(25):
        print(f"  {n:5d}  {word}")

    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        colormap="viridis",
        max_words=200,
        collocations=False,   # we already tokenized; avoid double-counting
        prefer_horizontal=0.9,
    ).generate_from_frequencies(freqs)

    wc.to_file(OUTPUT_PNG)
    print(f"\nSaved word cloud to {OUTPUT_PNG}")

    # Optional: show it interactively.
    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


if __name__ == "__main__":
    main()
