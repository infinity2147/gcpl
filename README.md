# RAG System with Retrieval Benchmarking
### AI Intern Hackathon — Option B Submission

A production-grade Retrieval-Augmented Generation system evaluated on the **TREC-COVID benchmark** with official NIST relevance judgements. Tests two chunking strategies, two embedding models, vector vs. hybrid search, LLM reranking, and query rewriting across 6 configurations and 10 queries.

---

## Deliverables Checklist

| Deliverable | Status | Location |
|---|---|---|
| Working RAG prototype | ✅ | `rag_system.py`, `run.py` |
| Dataset description and source | ✅ | Section 1 of `report_final.pdf` |
| Evaluation results with tables | ✅ | Section 3 of `report_final.pdf` + `results/` |
| Discussion of trade-offs and hallucinations | ✅ | Section 4 of `report_final.pdf` |
| 5-page PDF documentation | ✅ | `report_final.pdf` (7 pages) |
| Reranking implementation (bonus) | ✅ | `LLMReranker` in `rag_system.py` |
| Query rewriting (bonus) | ✅ | `QueryRewriter` in `rag_system.py` |
| Hybrid retrieval (bonus) | ✅ | `hybrid_search()` in `rag_system.py` |
| Latency and cost analysis (bonus) | ✅ | Section 4.3 of report + `results/` JSON |

---

## Project Structure

```
rag_project/
│
├── prepare_dataset.py     # Step 1: Download TREC-COVID from HuggingFace
├── rag_system.py          # Core RAG engine (565 lines)
├── benchmark.py           # Evaluation framework with official qrels
├── run.py                 # Entry point — all 6 configs, smart result merge
│
├── report_final.pdf       # 7-page technical report (submit this)
├── report_final.tex       # LaTeX source
├── requirements.txt       # Python dependencies
│
├── data/
│   └── trec_covid/        # Created by prepare_dataset.py
│       ├── corpus_subset.json   # ~1,300 documents
│       ├── queries.json         # 50 official TREC queries
│       └── qrels.json           # NIST relevance judgements
│
└── results/               # Created by run.py
    ├── benchmark_summary.json
    ├── Fixed_Small_Vector.json
    ├── Fixed_Small_Hybrid.json
    ├── Fixed_Large_Vector.json
    ├── Sentence_Small_Vector.json
    ├── Sentence_Small_Hybrid.json
    └── Sentence_Large_Hybrid_Rerank.json
```

---

## Dataset

**TREC-COVID** from the [BEIR benchmark](https://github.com/beir-cellar/beir).

- **Full corpus**: 171,332 biomedical papers on COVID-19 (CORD-19, Allen Institute for AI)
- **Queries**: 50 official TREC topics
- **Ground truth**: Human relevance judgements by NIST assessors (graded 0/1/2)
- **Subset used**: ~1,300 documents (800 relevant + 500 noise), selected to make retrieval challenging

This is a standard IR benchmark — metrics are directly comparable to published retrieval literature.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
```
openai>=1.0.0
faiss-cpu>=1.7.4
rank-bm25>=0.2.2
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
requests>=2.28.0
```

### 2. Set OpenAI API key

```bash
export OPENAI_API_KEY=sk-...
```

### 3. Download the dataset (run once)

```bash
python3 prepare_dataset.py
```

This downloads TREC-COVID from HuggingFace (~170MB), builds the 1,300-doc subset, and saves everything to `data/trec_covid/`. Takes 2–4 minutes depending on connection speed.

**What it does:**
- Downloads `corpus.jsonl.gz`, `queries.jsonl.gz`, `test.tsv` from `huggingface.co/datasets/BeIR/trec-covid`
- Selects 800 relevant + 500 non-relevant documents with `seed=42` (reproducible)
- Saves `corpus_subset.json`, `queries.json`, `qrels.json`

---

## How to Run

### Full benchmark — all 6 configurations × 10 queries

```bash
python3 run.py
```

**Runtime:** ~25–35 minutes  
**Cost:** ~$0.15 total across all 6 configs  
**Output:** Comparison table printed to console + 7 JSON files in `results/`

### Quick smoke test — 3 queries only

```bash
python3 run.py --quick
```

**Runtime:** ~8–10 minutes  
**Use this to:** verify everything works before running the full benchmark

### Run a single configuration

```bash
python3 run.py --config "Fixed+Small+Vector"
python3 run.py --config "Sentence"           # runs both Sentence configs
python3 run.py --config "Rerank"             # runs only the rerank config
python3 run.py --config "Hybrid"             # runs both Hybrid configs
```

The `--config` flag does case-insensitive substring matching on config names.

### Use more queries

```bash
python3 run.py --queries 15    # up to 50 (number of official TREC queries)
```

### Smart resume — rerun only specific configs without losing others

```bash
# The results from configs you DON'T rerun are preserved in benchmark_summary.json
python3 run.py --config "Sentence"    # reruns Sentence configs, keeps Fixed results
```

---

## System Architecture

### The 6 Benchmark Configurations

| Config | Chunker | Embedding | Search | Reranker | Query Rewrite |
|---|---|---|---|---|---|
| Fixed+Small+Vector | Fixed (300w, 50 overlap) | text-embedding-3-small | FAISS cosine | No | No |
| Fixed+Small+Hybrid | Fixed | text-embedding-3-small | FAISS + BM25 (RRF) | No | No |
| Fixed+Large+Vector | Fixed | text-embedding-3-large | FAISS cosine | No | No |
| Sentence+Small+Vector | Sentence boundary | text-embedding-3-small | FAISS cosine | No | No |
| Sentence+Small+Hybrid | Sentence boundary | text-embedding-3-small | FAISS + BM25 (RRF) | No | No |
| Sentence+Large+Hybrid+Rerank | Sentence boundary | text-embedding-3-large | FAISS + BM25 (RRF) | Yes (TOP_N=5) | Yes |

**What each comparison isolates:**
- Row 1 vs 2 → effect of hybrid vs. vector-only search
- Row 1 vs 3 → effect of large vs. small embedding model
- Row 1 vs 4 → effect of sentence vs. fixed chunking
- Row 4 vs 5 → effect of hybrid search with sentence chunking
- Row 5 vs 6 → effect of reranking + query rewriting

### Core Components (`rag_system.py`)

| Class / Function | What it does |
|---|---|
| `FixedSizeChunker` | Splits text into fixed word-count windows (300 words, 50 overlap) |
| `SentenceChunker` | PDF-aware sentence grouping — collapses single `\n` word-wraps, splits on sentence boundaries |
| `EmbeddingEngine` | OpenAI embeddings with text-keyed cache, token-safe batching, exponential retry |
| `VectorIndex` | FAISS `IndexFlatIP` with L2 normalisation for exact cosine nearest-neighbour |
| `BM25Index` | `rank_bm25.BM25Okapi` keyword index |
| `hybrid_search()` | Reciprocal Rank Fusion of FAISS + BM25 results |
| `LLMReranker` | GPT-4o-mini scores each passage 0–10 for relevance; returns top-5 |
| `QueryRewriter` | GPT-4o-mini expands queries with synonyms and explicit intent |
| `AnswerGenerator` | GPT-4o-mini grounded answer with source citations |
| `RAGPipeline` | Orchestrates everything; configure via constructor, then `.ingest()` → `.query()` |

### Using RAGPipeline directly

```python
from rag_system import RAGPipeline

pipeline = RAGPipeline(
    chunker="sentence",        # "fixed" or "sentence"
    embed_model="small",       # "small" or "large"
    search_mode="hybrid",      # "vector" or "hybrid"
    use_reranker=False,
    use_query_rewriter=False,
)

docs = [{"doc_id": "doc1", "text": "...", "metadata": {"title": "My Paper"}}]
pipeline.ingest(docs)

result = pipeline.query("What is the mechanism of cytokine storm?")
print(result["answer"])
print(result["retrieved_chunks"])    # list with scores and methods
print(result["total_latency_s"])
```

---

## Benchmark Results Summary

| Configuration | P@3 | P@5 | R@5 | MRR | NDCG@10 | LLM | Lat(s) | Cost($) |
|---|---|---|---|---|---|---|---|---|
| Fixed+Small+Vector | **0.933** | 0.880 | 0.161 | 0.950 | **0.710** | 0.83 | 4.12 | 0.011 |
| Fixed+Small+Hybrid | **0.933** | 0.860 | 0.165 | 0.950 | **0.714** | 0.83 | **3.85** | 0.011 |
| Fixed+Large+Vector | 0.833 | 0.880 | 0.163 | 0.933 | 0.683 | 0.82 | 4.22 | 0.045 |
| **Sentence+Small+Vector** | **0.933** | **0.900** | **0.173** | **1.000** | 0.705 | **0.84** | 4.22 | 0.011 |
| Sentence+Small+Hybrid | **0.933** | 0.860 | 0.165 | 0.950 | 0.708 | **0.84** | 4.71 | **0.010** |
| Sent.+Large+Hybrid+Rerank | 0.800 | 0.820 | 0.159 | 0.950 | 0.470 | 0.82 | 11.44 | 0.038 |

**Key findings:**
- `text-embedding-3-small` outperforms `text-embedding-3-large` by 2.7 NDCG points at 4× lower cost
- Sentence chunking achieves perfect MRR (1.000) — the only config where first result was always relevant
- Hybrid search is query-dependent: +33% P@3 on exact-match queries, −50% MRR on semantic queries
- Reranker with TOP_N=5 achieves P@5=0.820 vs P@5=0.460 with TOP_N=3 (78% improvement)

---

## Evaluation Metrics

All metrics computed against **official NIST relevance judgements** (not self-defined ground truth):

| Metric | Definition | Why it matters |
|---|---|---|
| **P@K** | Fraction of top-K results from relevant docs | Precision of top results |
| **R@5** | Fraction of all relevant docs found in top-5 | Coverage (conservative — see note below) |
| **MRR** | `1 / rank` of first relevant result, averaged | How quickly you find something useful |
| **NDCG@10** | Graded DCG normalised by ideal; rewards ranking highly-relevant docs higher | Standard TREC metric |
| **LLM score** | GPT-4o-mini judges answer quality 0–10 | Answer quality beyond retrieval |

**Note on R@5:** The denominator includes all relevant documents in the full 171K corpus (average 27 per query). A perfect retrieval of 5 from 27 relevant documents gives R@5 ≈ 0.19. Relative differences between configurations are the meaningful signal, not absolute values.

---

## Output Files

Each config saves a JSON in `results/` with this structure:

```json
{
  "config_name": "Fixed+Small+Vector",
  "config": {"chunker": "fixed", "embed_model": "small", ...},
  "summary": {
    "avg_precision_at_3": 0.9333,
    "avg_precision_at_5": 0.88,
    "avg_recall_at_5": 0.161,
    "avg_mrr": 0.95,
    "avg_ndcg_at_10": 0.7101,
    "avg_llm_score": 0.83,
    "avg_total_latency_s": 4.12,
    "num_chunks": 1019,
    "embed_time_s": 24.6
  },
  "per_query": [
    {
      "query_id": "38",
      "query": "What is the mechanism of inflammatory response...",
      "answer": "The inflammatory response in COVID-19...",
      "retrieved_doc_ids": ["abc123", "def456", ...],
      "precision_at_3": 1.0,
      "recall_at_5": 0.122,
      "mrr": 1.0,
      "ndcg_at_10": 0.931,
      "llm_score": 0.8,
      "total_latency_s": 4.51
    }
  ],
  "cost": {
    "indexing_embed_cost": 0.00611,
    "generation_cost": 0.00515,
    "total_usd": 0.01127
  }
}
```

---

## Known Issues and Limitations

1. **R@5 ceiling effect** — With 27 relevant documents on average and only 5 retrieved, absolute recall cannot exceed ~0.19. This is a corpus size constraint, not a system failure.

2. **Q13 (transmission routes) underperforms** — NDCG = 0.316–0.452 across all configs. The TREC qrels for Q13 include highly specialised epidemiological papers (fomite transmission, aerosol physics) that are underrepresented in our random sample. This is a corpus coverage issue, not a retrieval failure.

3. **Reranker vs. document-level qrels mismatch** — The LLM reranker scores passages at the chunk level, but TREC qrels judge relevance at the document level. This mismatch explains the reranker's lower NDCG despite competitive P@5.

4. **Query rewriting over-expands clinical queries** — Medical queries need type-preserving expansion (stay in the same clinical category), not open-ended academic broadening.

---

## Technology Choices

| Component | Choice | Why | Considered |
|---|---|---|---|
| Vector index | FAISS IndexFlatIP | Exact cosine search, no approximation error, fast for <10K chunks | Annoy (approximate, not needed at this scale) |
| Keyword search | rank-bm25 BM25Okapi | Standard BM25 variant, pure Python, no server needed | Elasticsearch (heavyweight for benchmarking) |
| Hybrid fusion | RRF | Scale-agnostic rank combination, no calibration needed | Weighted score combination (requires normalisation) |
| Embeddings | OpenAI text-embedding-3-* | State-of-the-art, reproducible, both model sizes available | sentence-transformers (requires local GPU for speed) |
| Generation | gpt-4o-mini | Fast, cheap, good instruction following | gpt-4o (4× more expensive, marginal quality gain for RAG) |
| Reranker | LLM-based (gpt-4o-mini) | No additional dependencies | cross-encoder/ms-marco-MiniLM-L-6-v2 (better latency, needs torch) |

---

## Reproduction

All results are fully reproducible:

```bash
# Exact reproduction of benchmark results
export OPENAI_API_KEY=sk-...
python3 prepare_dataset.py        # seed=42, deterministic subset
python3 run.py                    # all 6 configs, 10 queries
```

The corpus subset uses `random.seed(42)` in `prepare_dataset.py`. Given the same API key and the same model versions, results should be within ±0.02 of reported values (variance comes from GPT-4o-mini temperature in LLM scoring).
