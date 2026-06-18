#!/usr/bin/env python3
"""
Scrape arXiv for papers published between July 2025 and today whose title or
abstract contains at least one of a set of jet-physics keywords.

Uses the official arXiv API (export.arxiv.org/api/query), which is the
sanctioned way to query arXiv programmatically. Results are saved to CSV.
"""

import csv
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KEYWORDS = [
    "jet",
    "lund plane",
    "jet substructure",
    "quark-jet",
    "gluon-jet",
    "boosted object",
    "energy-energy correlator",
    "energy correlator",
    "event shape",
    "soft drop",
    "grooming",
    "jet grooming",
    "resummation",
    "parton shower",
    "flavour tagging",
    "non-perturbative",
    "power correction",
    "jet quenching",
]

START_DATE = datetime(2025, 7, 1, tzinfo=timezone.utc)
END_DATE = datetime.now(timezone.utc)

# arXiv categories to restrict to (hep-ph, hep-ex, hep-th, nucl-th, nucl-ex).
# Set to None to search all categories.
CATEGORIES = ["hep-ph", "hep-ex"]

API_URL = "http://export.arxiv.org/api/query"
BATCH_SIZE = 100          # max 2000, but smaller batches are gentler
MAX_RESULTS = 5000        # hard cap on total fetched
SLEEP_BETWEEN = 3.0       # seconds; arXiv requests >=3s between calls
OUTPUT_CSV = "arxiv_jet_papers.csv"

ATOM = "{http://www.w3.org/2005/Atom}"


def build_query():
    """Build the arXiv search_query string from keywords and categories."""
    kw_terms = []
    for kw in KEYWORDS:
        # Quote multi-word phrases; search both title (ti) and abstract (abs).
        phrase = f'"{kw}"' if " " in kw else kw
        kw_terms.append(f"ti:{phrase}")
        kw_terms.append(f"abs:{phrase}")
    kw_clause = "(" + " OR ".join(kw_terms) + ")"

    if CATEGORIES:
        cat_clause = "(" + " OR ".join(f"cat:{c}" for c in CATEGORIES) + ")"
        return f"{kw_clause} AND {cat_clause}"
    return kw_clause


def fetch_batch(search_query, start):
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": BATCH_SIZE,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "jet-scraper/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def parse_entries(xml_bytes):
    root = ET.fromstring(xml_bytes)
    entries = []
    for entry in root.findall(f"{ATOM}entry"):
        published_str = entry.findtext(f"{ATOM}published", "")
        try:
            published = datetime.strptime(
                published_str, "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        title = " ".join(entry.findtext(f"{ATOM}title", "").split())
        summary = " ".join(entry.findtext(f"{ATOM}summary", "").split())
        arxiv_id = entry.findtext(f"{ATOM}id", "").rsplit("/", 1)[-1]
        authors = [a.findtext(f"{ATOM}name", "")
                   for a in entry.findall(f"{ATOM}author")]
        primary = entry.find(f"{ATOM}category")
        category = primary.get("term") if primary is not None else ""

        entries.append({
            "arxiv_id": arxiv_id,
            "published": published,
            "title": title,
            "authors": "; ".join(authors),
            "category": category,
            "abstract": summary,
            "url": entry.findtext(f"{ATOM}id", ""),
        })
    return entries


def keyword_match(text):
    low = text.lower()
    return [kw for kw in KEYWORDS if kw.lower() in low]


def main():
    search_query = build_query()
    print("Query:", search_query)

    collected = []
    seen = set()
    start = 0

    while start < MAX_RESULTS:
        print(f"Fetching results {start}–{start + BATCH_SIZE} ...")
        try:
            xml_bytes = fetch_batch(search_query, start)
        except Exception as e:
            print("  request failed:", e, "— retrying once")
            time.sleep(SLEEP_BETWEEN)
            xml_bytes = fetch_batch(search_query, start)

        batch = parse_entries(xml_bytes)
        if not batch:
            print("  no more entries.")
            break

        stop = False
        for e in batch:
            if e["published"] < START_DATE:
                # Results are sorted newest-first, so we can stop.
                stop = True
                break
            if e["published"] > END_DATE:
                continue
            if e["arxiv_id"] in seen:
                continue
            matched = keyword_match(e["title"] + " " + e["abstract"])
            if not matched:
                continue
            seen.add(e["arxiv_id"])
            e["matched_keywords"] = ", ".join(matched)
            collected.append(e)

        if stop:
            print("  reached papers older than July 2025 — stopping.")
            break

        start += BATCH_SIZE
        time.sleep(SLEEP_BETWEEN)

    collected.sort(key=lambda x: x["published"], reverse=True)
    print(f"\nMatched {len(collected)} papers.")

    fields = ["arxiv_id", "published", "title", "authors",
              "category", "matched_keywords", "url", "abstract"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for e in collected:
            row = dict(e)
            row["published"] = e["published"].strftime("%Y-%m-%d")
            w.writerow(row)

    print("Saved to", OUTPUT_CSV)


if __name__ == "__main__":
    main()
