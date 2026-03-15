"""
prepare_dataset.py — Download and prepare TREC-COVID dataset
=============================================================
Run this ONCE before run.py.

Downloads from HuggingFace (no beir server needed):
  corpus.jsonl.gz   — 171K biomedical paper abstracts
  queries.jsonl.gz  — 50 official TREC queries
  test.tsv          — official NIST relevance judgements (qrels)

Then selects a manageable subset and saves to data/trec_covid/:
  corpus_subset.json   — ~800 documents (relevant + random negatives)
  queries.json         — 50 queries
  qrels.json           — relevance judgements for subset docs

Usage:
  python3 prepare_dataset.py
  python3 prepare_dataset.py --docs 1000   # larger subset
"""

import gzip
import json
import csv
import random
import argparse
from pathlib import Path

try:
    import requests
except ImportError:
    raise SystemExit("pip install requests")

# ── Download helpers ───────────────────────────────────────────────────────────

HF_BASE = "https://huggingface.co/datasets"

URLS = {
    "corpus":  f"{HF_BASE}/BeIR/trec-covid/resolve/main/corpus.jsonl.gz",
    "queries": f"{HF_BASE}/BeIR/trec-covid/resolve/main/queries.jsonl.gz",
    "qrels":   f"{HF_BASE}/BeIR/trec-covid-qrels/resolve/main/test.tsv",
}

def download(url: str, dest: Path, desc: str):
    if dest.exists():
        print(f"  ✓ {desc} already downloaded ({dest})")
        return
    print(f"  Downloading {desc}...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"    {pct:.0f}%  ({downloaded/1e6:.1f} MB)", end="\r")
    print(f"  ✓ {desc} saved ({downloaded/1e6:.1f} MB)          ")

# ── Load raw files ─────────────────────────────────────────────────────────────

def load_corpus(path: Path) -> dict:
    print("  Parsing corpus...")
    corpus = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {
                "title": doc.get("title", ""),
                "text":  doc.get("text", ""),
            }
    print(f"  Loaded {len(corpus):,} documents")
    return corpus

def load_queries(path: Path) -> dict:
    queries = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    print(f"  Loaded {len(queries)} queries")
    return queries

def load_qrels(path: Path) -> dict:
    qrels = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query-id"]
            did = row["corpus-id"]
            rel = int(row["score"])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = rel
    total_pairs = sum(len(v) for v in qrels.values())
    print(f"  Loaded {len(qrels)} queries with {total_pairs:,} relevance judgements")
    return qrels

# ── Subset selection ───────────────────────────────────────────────────────────

def build_subset(corpus: dict, queries: dict, qrels: dict,
                 n_relevant: int = 500, n_noise: int = 300,
                 seed: int = 42) -> tuple[dict, dict, dict]:
    """
    Select a manageable subset:
      - All docs that appear as relevant (score >= 1) in qrels, up to n_relevant
      - n_noise random non-relevant docs (makes retrieval harder — forces real ranking)
      - Only keep qrels entries where the doc is in our subset
    """
    random.seed(seed)

    # Collect relevant doc IDs
    relevant_ids = set()
    for rels in qrels.values():
        for did, score in rels.items():
            if score >= 1 and did in corpus:
                relevant_ids.add(did)

    print(f"  Relevant docs in corpus: {len(relevant_ids):,}")

    # Sample relevant docs
    sampled_relevant = set(random.sample(sorted(relevant_ids),
                                         min(n_relevant, len(relevant_ids))))

    # Sample noise docs (not in any qrel)
    all_ids = set(corpus.keys())
    noise_pool = list(all_ids - relevant_ids)
    sampled_noise = set(random.sample(noise_pool, min(n_noise, len(noise_pool))))

    selected_ids = sampled_relevant | sampled_noise
    print(f"  Subset: {len(sampled_relevant)} relevant + {len(sampled_noise)} noise = {len(selected_ids)} total docs")

    # Build subset corpus
    subset_corpus = {
        did: corpus[did]
        for did in selected_ids
    }

    # Filter qrels to subset docs only, keep queries that still have >= 1 relevant doc
    subset_qrels = {}
    for qid, rels in qrels.items():
        filtered = {did: score for did, score in rels.items() if did in selected_ids and score >= 1}
        if filtered:
            subset_qrels[qid] = filtered

    print(f"  Queries with relevant docs in subset: {len(subset_qrels)}")

    return subset_corpus, queries, subset_qrels

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs",  type=int, default=800,  help="Max relevant docs to include")
    parser.add_argument("--noise", type=int, default=500,  help="Non-relevant docs to add")
    parser.add_argument("--raw",   type=str, default="data/trec_raw",  help="Raw download dir")
    parser.add_argument("--out",   type=str, default="data/trec_covid", help="Output dir")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download ────────────────────────────────────────────────────
    print("\n=== Step 1: Downloading TREC-COVID from HuggingFace ===")
    corpus_gz  = raw_dir / "corpus.jsonl.gz"
    queries_gz = raw_dir / "queries.jsonl.gz"
    qrels_tsv  = raw_dir / "test.tsv"

    download(URLS["corpus"],  corpus_gz,  "corpus (~170MB)")
    download(URLS["queries"], queries_gz, "queries (~5KB)")
    download(URLS["qrels"],   qrels_tsv,  "qrels (~500KB)")

    # ── Step 2: Parse ───────────────────────────────────────────────────────
    print("\n=== Step 2: Parsing raw files ===")
    corpus  = load_corpus(corpus_gz)
    queries = load_queries(queries_gz)
    qrels   = load_qrels(qrels_tsv)

    # ── Step 3: Build subset ────────────────────────────────────────────────
    print("\n=== Step 3: Building subset ===")
    subset_corpus, subset_queries, subset_qrels = build_subset(
        corpus, queries, qrels,
        n_relevant=args.docs,
        n_noise=args.noise,
    )

    # ── Step 4: Save ────────────────────────────────────────────────────────
    print("\n=== Step 4: Saving to", out_dir, "===")

    # corpus_subset.json: list of dicts for RAGPipeline.ingest()
    documents = [
        {
            "doc_id":   did,
            "text":     (d["title"] + "\n\n" + d["text"]).strip(),
            "metadata": {"title": d["title"], "doc_id": did},
        }
        for did, d in subset_corpus.items()
    ]
    (out_dir / "corpus_subset.json").write_text(json.dumps(documents, indent=2))
    print(f"  ✓ corpus_subset.json ({len(documents)} docs)")

    (out_dir / "queries.json").write_text(json.dumps(subset_queries, indent=2))
    print(f"  ✓ queries.json ({len(subset_queries)} queries)")

    (out_dir / "qrels.json").write_text(
        json.dumps({k: dict(v) for k, v in subset_qrels.items()}, indent=2)
    )
    print(f"  ✓ qrels.json ({len(subset_qrels)} queries with relevance judgements)")

    print(f"\nDone. Run: python3 run.py")

if __name__ == "__main__":
    main()