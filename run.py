"""
RAG System - Main Runner (TREC-COVID Edition)
=============================================

SETUP (run once before this):
  1. Download dataset:
       python3 prepare_dataset.py
     This downloads TREC-COVID from HuggingFace and saves to data/trec_covid/

  2. Set API key:
       export OPENAI_API_KEY=sk-...

Usage:
  python3 run.py                        # full benchmark — all 6 configs x 10 queries
  python3 run.py --quick                # 3 queries only (smoke test)
  python3 run.py --config "Sentence"    # rerun only configs matching substring
  python3 run.py --config "Rerank"      # rerun only the rerank config
  python3 run.py --queries 15           # use more queries for deeper evaluation

Smart resume: results are merged into results/benchmark_summary.json.
Running --config "Sentence" preserves existing Fixed config results.
"""

import os
import json
import sys
import argparse
from pathlib import Path

from rag_system import RAGPipeline
from benchmark import (
    run_benchmark, compare_results, estimate_cost_usd,
    load_trec_queries, BenchmarkResult
)

DATA_DIR    = Path("data/trec_covid")
RESULTS_DIR = Path("results")

# ── Load dataset ───────────────────────────────────────────────────────────────

def load_documents() -> list[dict]:
    corpus_path = DATA_DIR / "corpus_subset.json"
    if not corpus_path.exists():
        print("ERROR: data/trec_covid/corpus_subset.json not found.")
        print("Run: python3 prepare_dataset.py")
        sys.exit(1)

    documents = json.loads(corpus_path.read_text())
    print(f"  Loaded {len(documents):,} documents from TREC-COVID subset")
    return documents

# ── Configurations ─────────────────────────────────────────────────────────────

CONFIGS = [
    {
        "name": "Fixed+Small+Vector",
        "kwargs": {"chunker": "fixed",    "embed_model": "small", "search_mode": "vector"},
    },
    {
        "name": "Fixed+Small+Hybrid",
        "kwargs": {"chunker": "fixed",    "embed_model": "small", "search_mode": "hybrid"},
    },
    {
        "name": "Fixed+Large+Vector",
        "kwargs": {"chunker": "fixed",    "embed_model": "large", "search_mode": "vector"},
    },
    {
        "name": "Sentence+Small+Vector",
        "kwargs": {"chunker": "sentence", "embed_model": "small", "search_mode": "vector"},
    },
    {
        "name": "Sentence+Small+Hybrid",
        "kwargs": {"chunker": "sentence", "embed_model": "small", "search_mode": "hybrid"},
    },
    {
        "name": "Sentence+Large+Hybrid+Rerank",
        "kwargs": {
            "chunker":           "sentence",
            "embed_model":       "large",
            "search_mode":       "hybrid",
            "use_reranker":      True,
            "use_query_rewriter":True,
        },
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def save_result(result: BenchmarkResult, no_cost: bool):
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{result.config_name.replace('+','_').replace(' ','_')}.json"
    data = {
        "config":      result.config,
        "config_name": result.config_name,
        "summary": {
            "avg_precision_at_3":   result.avg_precision_at_3,
            "avg_precision_at_5":   result.avg_precision_at_5,
            "avg_recall_at_5":      result.avg_recall_at_5,
            "avg_mrr":              result.avg_mrr,
            "avg_ndcg_at_10":       result.avg_ndcg_at_10,
            "avg_llm_score":        result.avg_llm_score,
            "avg_total_latency_s":  result.avg_total_latency_s,
            "num_chunks":           result.num_chunks,
            "embed_time_s":         result.embed_time_s,
        },
        "per_query": result.per_query_results,
    }
    if not no_cost:
        data["cost"] = estimate_cost_usd(result, result.config["embed_model"])
    out_path.write_text(json.dumps(data, indent=2))


def merge_and_save_summary(new_results: list[BenchmarkResult], no_cost: bool) -> list[dict]:
    """Merge new results into summary, preserving previously run configs."""
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    existing: dict[str, dict] = {}
    if summary_path.exists():
        try:
            existing = {
                r["config_name"]: r
                for r in json.loads(summary_path.read_text()).get("results", [])
            }
        except Exception:
            pass

    for r in new_results:
        existing[r.config_name] = {
            "config_name":      r.config_name,
            "config":           r.config,
            "precision_at_3":   r.avg_precision_at_3,
            "precision_at_5":   r.avg_precision_at_5,
            "recall_at_5":      r.avg_recall_at_5,
            "mrr":              r.avg_mrr,
            "ndcg_at_10":       r.avg_ndcg_at_10,
            "llm_score":        r.avg_llm_score,
            "avg_latency_s":    r.avg_total_latency_s,
            "num_chunks":       r.num_chunks,
            "cost": estimate_cost_usd(r, r.config["embed_model"]) if not no_cost else {},
        }

    order   = [c["name"] for c in CONFIGS]
    ordered = sorted(
        existing.values(),
        key=lambda x: order.index(x["config_name"]) if x["config_name"] in order else 99
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    summary_path.write_text(json.dumps({"configs_tested": len(ordered), "results": ordered}, indent=2))
    return ordered


def print_full_table(all_entries: list[dict]):
    stubs = [
        BenchmarkResult(
            config_name=r["config_name"],
            config=r["config"],
            per_query_results=[],
            avg_precision_at_3=r["precision_at_3"],
            avg_precision_at_5=r["precision_at_5"],
            avg_recall_at_5=r["recall_at_5"],
            avg_mrr=r["mrr"],
            avg_ndcg_at_10=r.get("ndcg_at_10", 0),
            avg_llm_score=r["llm_score"],
            avg_retrieval_latency_s=0,
            avg_generation_latency_s=0,
            avg_total_latency_s=r["avg_latency_s"],
            total_prompt_tokens=0,
            total_completion_tokens=0,
            num_chunks=r["num_chunks"],
            embed_time_s=0,
        )
        for r in all_entries
    ]
    print(compare_results(stubs))

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark — TREC-COVID")
    parser.add_argument("--quick",   action="store_true", help="3 queries only")
    parser.add_argument("--queries", type=int, default=10, help="Number of queries (default 10)")
    parser.add_argument("--config",  type=str, default=None, help="Substring match on config name")
    parser.add_argument("--no-cost", action="store_true",   help="Skip cost estimation")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    print("=" * 60)
    print("RAG SYSTEM BENCHMARK — TREC-COVID")
    print("Dataset: TREC-COVID (BEIR) with official NIST qrels")
    print("=" * 60)

    print("\nLoading dataset...")
    documents = load_documents()

    n_queries = 3 if args.quick else args.queries
    print(f"\nLoading {n_queries} queries...")
    queries = load_trec_queries(n=n_queries)

    configs_to_run = CONFIGS
    if args.config:
        configs_to_run = [c for c in CONFIGS if args.config.lower() in c["name"].lower()]
        if not configs_to_run:
            print(f"No config matching '{args.config}'. Available: {[c['name'] for c in CONFIGS]}")
            sys.exit(1)

    new_results = []
    for cfg in configs_to_run:
        pipeline = RAGPipeline(**cfg["kwargs"])
        result   = run_benchmark(pipeline, documents, cfg["name"], queries=queries)
        new_results.append(result)
        save_result(result, args.no_cost)

    all_entries = merge_and_save_summary(new_results, args.no_cost)

    print("\n" + "=" * 60)
    print(f"COMPARISON TABLE ({len(all_entries)} configs in summary)")
    print("=" * 60)
    print_full_table(all_entries)

    if not args.no_cost:
        print("\n\nCOST ANALYSIS")
        print("-" * 60)
        print(f"{'Config':<35} {'Embed($)':>9} {'Gen($)':>9} {'Total($)':>9}")
        print("-" * 60)
        for r in all_entries:
            cost = r.get("cost", {})
            if cost:
                print(f"{r['config_name']:<35} "
                      f"{cost.get('indexing_embed_cost',0):>9.5f} "
                      f"{cost.get('generation_cost',0):>9.5f} "
                      f"{cost.get('total_usd',0):>9.5f}")

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("Done.")

if __name__ == "__main__":
    main()