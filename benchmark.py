"""
Benchmark & Evaluation Module — TREC-COVID Edition
====================================================
Uses official NIST relevance judgements (qrels) instead of self-defined ground truth.

Metrics:
  - Precision@K, Recall@K  (standard IR metrics against qrels)
  - MRR (Mean Reciprocal Rank)
  - NDCG@10 (graded relevance — rewards partial matches, standard in IR)
  - LLM answer quality score (0-1) via gpt-4o-mini judge
  - Latency and token cost per query
"""

import os
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

from rag_system import RAGPipeline

DATA_DIR = Path("data/trec_covid")

# ── Load TREC-COVID queries ────────────────────────────────────────────────────

def load_trec_queries(n: int = 10) -> list[dict]:
    """
    Load queries and qrels saved by prepare_dataset.py.
    Selects queries with the most relevant docs in the subset — these are the
    hardest queries where retrieval ranking actually matters.
    """
    queries = json.loads((DATA_DIR / "queries.json").read_text())
    qrels   = json.loads((DATA_DIR / "qrels.json").read_text())

    test_queries = []
    for qid, qtext in queries.items():
        if qid not in qrels:
            continue
        relevant_docs = [did for did, score in qrels[qid].items() if score >= 1]
        if len(relevant_docs) >= 2:
            test_queries.append({
                "id":            qid,
                "query":         qtext,
                "relevant_docs": relevant_docs,
                "qrels":         qrels[qid],
            })

    test_queries.sort(key=lambda x: len(x["relevant_docs"]), reverse=True)
    selected = test_queries[:n]
    print(f"  Selected {len(selected)} queries "
          f"(avg {np.mean([len(q['relevant_docs']) for q in selected]):.1f} relevant docs each)")
    return selected

# ── Metrics ────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / k if k > 0 else 0.0

def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in relevant if d in retrieved[:k])
    return hits / len(relevant)

def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    for i, d in enumerate(retrieved):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved: list[str], qrels: dict, k: int = 10) -> float:
    """
    Normalised Discounted Cumulative Gain at K.
    Uses graded relevance: 0=not relevant, 1=partially relevant, 2=highly relevant.
    Rewards getting highly relevant docs ranked higher than partially relevant ones.
    Standard metric in TREC evaluations.
    """
    def dcg(rels: list) -> float:
        return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels[:k]))

    retrieved_rels = [qrels.get(d, 0) for d in retrieved[:k]]
    actual_dcg = dcg(retrieved_rels)
    ideal_rels = sorted(qrels.values(), reverse=True)
    ideal_dcg  = dcg(ideal_rels)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def llm_answer_score(query: str, answer: str) -> float:
    """
    GPT-4o-mini judges answer quality 0-10.
    For TREC-COVID: checks relevance, factual accuracy, and specificity.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = (
        f"Query: {query}\n\n"
        f"Answer: {answer[:600]}\n\n"
        "Rate this answer 0-10 on:\n"
        "- Relevance to the query\n"
        "- Factual accuracy for a biomedical/COVID topic\n"
        "- Whether it is specific and informative (not vague)\n"
        "Reply with a single integer 0-10."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5, temperature=0
        )
        score = float(resp.choices[0].message.content.strip())
        return min(max(score, 0), 10) / 10.0
    except Exception:
        return 0.0

# ── BenchmarkResult ────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    config_name: str
    config: dict
    per_query_results: list
    avg_precision_at_3: float
    avg_precision_at_5: float
    avg_recall_at_5: float
    avg_mrr: float
    avg_ndcg_at_10: float
    avg_llm_score: float
    avg_retrieval_latency_s: float
    avg_generation_latency_s: float
    avg_total_latency_s: float
    total_prompt_tokens: int
    total_completion_tokens: int
    num_chunks: int
    embed_time_s: float

# ── Benchmark runner ───────────────────────────────────────────────────────────

def run_benchmark(
    pipeline: RAGPipeline,
    documents: list[dict],
    config_name: str,
    queries: list[dict],
    verbose: bool = True,
) -> BenchmarkResult:

    if verbose:
        print(f"\n{'='*60}")
        print(f"CONFIG: {config_name}")
        print(f"{'='*60}")

    ingest_stats = pipeline.ingest(documents)
    if verbose:
        print(f"  Chunks: {ingest_stats['num_chunks']} | Embed: {ingest_stats['embed_time_s']}s")

    per_query = []
    total_p_tokens = total_c_tokens = 0

    for tq in queries:
        if verbose:
            print(f"\n  Q{tq['id']}: {tq['query'][:65]}...")

        result = pipeline.query(tq["query"], top_k=10)

        retrieved  = [c["doc_id"] for c in result["retrieved_chunks"]]
        relevant   = tq["relevant_docs"]
        qrels_map  = tq.get("qrels", {r: 1 for r in relevant})

        p3    = precision_at_k(retrieved, relevant, 3)
        p5    = precision_at_k(retrieved, relevant, 5)
        r5    = recall_at_k(retrieved, relevant, 5)
        rr    = reciprocal_rank(retrieved, relevant)
        ndcg  = ndcg_at_k(retrieved, qrels_map, k=10)
        llm   = llm_answer_score(tq["query"], result["answer"])

        total_p_tokens += result.get("prompt_tokens", 0)
        total_c_tokens += result.get("completion_tokens", 0)

        qr = {
            "query_id":               tq["id"],
            "query":                  tq["query"],
            "answer":                 result["answer"],
            "retrieved_doc_ids":      retrieved,
            "num_relevant_in_corpus": len(relevant),
            "precision_at_3":         round(p3, 4),
            "precision_at_5":         round(p5, 4),
            "recall_at_5":            round(r5, 4),
            "mrr":                    round(rr, 4),
            "ndcg_at_10":             round(ndcg, 4),
            "llm_score":              round(llm, 4),
            "retrieval_latency_s":    result["retrieval_latency_s"],
            "generation_latency_s":   result["generation_latency_s"],
            "total_latency_s":        result["total_latency_s"],
            "prompt_tokens":          result.get("prompt_tokens", 0),
            "completion_tokens":      result.get("completion_tokens", 0),
            "rewritten_query":        result.get("rewritten_query"),
        }
        per_query.append(qr)

        if verbose:
            print(f"    P@3={p3:.2f} P@5={p5:.2f} R@5={r5:.2f} "
                  f"MRR={rr:.2f} NDCG={ndcg:.2f} LLM={llm:.2f} | {result['total_latency_s']}s")

    def avg(k): return round(float(np.mean([q[k] for q in per_query])), 4)

    br = BenchmarkResult(
        config_name=config_name,
        config=pipeline.config,
        per_query_results=per_query,
        avg_precision_at_3=avg("precision_at_3"),
        avg_precision_at_5=avg("precision_at_5"),
        avg_recall_at_5=avg("recall_at_5"),
        avg_mrr=avg("mrr"),
        avg_ndcg_at_10=avg("ndcg_at_10"),
        avg_llm_score=avg("llm_score"),
        avg_retrieval_latency_s=avg("retrieval_latency_s"),
        avg_generation_latency_s=avg("generation_latency_s"),
        avg_total_latency_s=avg("total_latency_s"),
        total_prompt_tokens=total_p_tokens,
        total_completion_tokens=total_c_tokens,
        num_chunks=ingest_stats["num_chunks"],
        embed_time_s=ingest_stats["embed_time_s"],
    )

    if verbose:
        print(f"\n  SUMMARY → P@3={br.avg_precision_at_3} P@5={br.avg_precision_at_5} "
              f"R@5={br.avg_recall_at_5} MRR={br.avg_mrr} "
              f"NDCG@10={br.avg_ndcg_at_10} LLM={br.avg_llm_score}")

    return br

# ── Comparison table ───────────────────────────────────────────────────────────

def compare_results(results: list[BenchmarkResult]) -> str:
    header = (
        f"\n{'Config':<35} {'P@3':>5} {'P@5':>5} {'R@5':>5} "
        f"{'MRR':>5} {'NDCG':>6} {'LLM':>5} {'Lat(s)':>7} {'Chunks':>7}"
    )
    sep   = "-" * len(header)
    lines = [header, sep]
    for r in results:
        lines.append(
            f"{r.config_name:<35} {r.avg_precision_at_3:>5.3f} {r.avg_precision_at_5:>5.3f} "
            f"{r.avg_recall_at_5:>5.3f} {r.avg_mrr:>5.3f} {r.avg_ndcg_at_10:>6.3f} "
            f"{r.avg_llm_score:>5.3f} {r.avg_total_latency_s:>7.2f} {r.num_chunks:>7}"
        )
    return "\n".join(lines)

# ── Cost estimation ────────────────────────────────────────────────────────────

def estimate_cost_usd(result: BenchmarkResult, embed_model: str) -> dict:
    price  = {"small": 0.02, "large": 0.13}[embed_model]
    idx    = (result.num_chunks * 300 / 1_000_000) * price
    qemb   = (10 * 50 / 1_000_000) * price
    gen    = (result.total_prompt_tokens / 1_000_000 * 0.15 +
              result.total_completion_tokens / 1_000_000 * 0.60)
    return {
        "indexing_embed_cost": round(idx, 6),
        "query_embed_cost":    round(qemb, 6),
        "generation_cost":     round(gen, 6),
        "total_usd":           round(idx + qemb + gen, 6),
    }