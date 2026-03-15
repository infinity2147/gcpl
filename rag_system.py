"""
RAG System - Core Engine
========================
Supports:
  - Two chunking strategies: fixed-size and sentence-based
  - Two embedding models: OpenAI text-embedding-3-small and text-embedding-3-large
  - Vector search (FAISS) and Hybrid search (FAISS + BM25)
  - Reranking via cross-encoder-style LLM scoring
  - Query rewriting
  - Answer generation grounded in retrieved context

Dataset: AI Arxiv papers (abstracts + full text) from Semantic Scholar Open Research Corpus
"""

import os
import json
import time
import re
import hashlib
import math
from dataclasses import dataclass, field
from typing import Literal
from collections import defaultdict

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────

EMBEDDING_MODELS = {
    "small": "text-embedding-3-small",   # 1536-dim, fast, cheap
    "large": "text-embedding-3-large",   # 3072-dim, better quality
}
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
RERANK_TOP_N = 5   # increased from 3 — returns more context, improves recall

# ── Data structures ─────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    method: str  # "vector", "bm25", "hybrid", "reranked"

# ── Chunking Strategies ────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    """Approximate token count using whitespace split (1 word ≈ 1.3 tokens for English)."""
    return len(text.split())

class FixedSizeChunker:
    """
    Strategy 1: Fixed word-count chunks with overlap.
    Uses word count as a proxy for token count (1 word ≈ 1.3 tokens for English text).
    Pro: Consistent size, predictable embedding quality, no external tokenizer needed.
    Con: May split mid-sentence, breaking semantic units.
    """
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size   # words per chunk
        self.overlap = overlap         # word overlap between consecutive chunks

    def chunk(self, text: str, doc_id: str, metadata: dict) -> list[Chunk]:
        words = text.split()
        chunks = []
        start = 0
        idx = 0
        stride = max(1, self.chunk_size - self.overlap)
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_fixed_{idx}",
                    doc_id=doc_id,
                    text=chunk_text,
                    metadata={**metadata, "chunk_strategy": "fixed", "chunk_index": idx}
                ))
            start += stride
            idx += 1
        return chunks

class SentenceChunker:
    """
    Strategy 2: Sentence-boundary chunks — groups sentences until a word budget is reached.
    Pro: Preserves semantic coherence; each chunk is a complete thought.
    Con: Variable chunk sizes; slightly more complex preprocessing than fixed-size.

    PDF-aware preprocessing:
      PDF extraction produces single newlines as word-wrap artifacts (not sentence breaks).
      e.g. "each layer has two\nsub-layers." is one sentence across two lines.
      We MUST collapse single newlines to spaces before splitting on sentence punctuation.
      Only double newlines (blank lines) are real paragraph boundaries.
    """
    def __init__(self, max_tokens: int = 300, overlap_sentences: int = 1):
        self.max_words = max_tokens
        self.overlap = overlap_sentences

    def _normalize_pdf_text(self, text: str) -> str:
        """
        Collapse single-newline word-wraps into spaces.
        Preserve double newlines (paragraph breaks).
        Normalize whitespace.
        """
        # Protect paragraph breaks by temporarily replacing them
        text = re.sub(r'\n{2,}', '\x00', text)
        # Collapse single newlines (PDF word-wraps) into spaces
        text = re.sub(r'\n', ' ', text)
        # Restore paragraph breaks as double newlines
        text = re.sub(r'\x00', '\n\n', text)
        # Normalize multiple spaces
        text = re.sub(r'  +', ' ', text)
        return text.strip()

    def _split_to_sentences(self, text: str) -> list[str]:
        """
        Split normalized text into sentences.
        Paragraph breaks (double newlines) are hard boundaries.
        Within paragraphs, split on '. '  '! '  '? '
        """
        sentences = []
        paragraphs = re.split(r'\n{2,}', text)
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Split on sentence-ending punctuation followed by whitespace
            parts = re.split(r'(?<=[.!?]) +', para)
            sentences.extend(p.strip() for p in parts if p.strip())
        return sentences

    def chunk(self, text: str, doc_id: str, metadata: dict) -> list[Chunk]:
        normalized = self._normalize_pdf_text(text)
        sentences = self._split_to_sentences(normalized)

        chunks = []
        current: list[str] = []
        current_words = 0
        idx = 0

        for sent in sentences:
            sw = _word_count(sent)
            # Skip noise: very short fragments (lone numbers, ref tags, etc.)
            if sw < 4:
                continue
            if current_words + sw > self.max_words and current:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_sent_{idx}",
                    doc_id=doc_id,
                    text=" ".join(current),
                    metadata={**metadata, "chunk_strategy": "sentence", "chunk_index": idx}
                ))
                current = current[-self.overlap:] if self.overlap else []
                current_words = sum(_word_count(s) for s in current)
                idx += 1
            current.append(sent)
            current_words += sw

        if current:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_sent_{idx}",
                doc_id=doc_id,
                text=" ".join(current),
                metadata={**metadata, "chunk_strategy": "sentence", "chunk_index": idx}
            ))
        return chunks

# ── Embedding Layer ────────────────────────────────────────────────────────────

def _sanitize(text: str) -> str:
    """
    Remove characters that can cause OpenAI 500 errors:
      - Null bytes and other ASCII control characters (except tab/newline)
      - Lone surrogates (broken Unicode from PDF extraction)
      - Ligature / private-use characters that some PDF fonts produce
    Also truncate to 2000 words max so no single chunk is absurdly large.
    """
    # Remove null bytes and control chars (keep \t \n \r)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    # Remove lone surrogates (invalid Unicode)
    text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    # Collapse runs of whitespace (but preserve single newlines for readability)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Hard cap: 2000 words (~2800 tokens) — prevents any single chunk blowing up
    words = text.split()
    if len(words) > 2000:
        text = ' '.join(words[:2000])
    return text.strip()


class EmbeddingEngine:
    def __init__(self, model_key: Literal["small", "large"] = "small"):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = EMBEDDING_MODELS[model_key]
        self.model_key = model_key
        self._cache: dict[str, np.ndarray] = {}

    def _token_safe_batches(self, texts: list[str], max_tokens: int = 200_000) -> list[list[str]]:
        """
        Split into batches under max_tokens (conservative 200K, well below the 300K hard limit).
        Uses word_count * 1.4 as a token estimate.
        Also caps each batch at 500 items (OpenAI's per-request item limit).
        """
        batches, current, current_tokens = [], [], 0
        for t in texts:
            est = int(len(t.split()) * 1.4) + 1
            if current and (current_tokens + est > max_tokens or len(current) >= 500):
                batches.append(current)
                current, current_tokens = [], 0
            current.append(t)
            current_tokens += est
        if current:
            batches.append(current)
        return batches

    def _embed_batch_with_retry(self, batch: list[str], max_retries: int = 4) -> list[np.ndarray]:
        """
        Call the embeddings API with exponential backoff on 5xx errors.
        On repeated failure, falls back to embedding one-by-one to isolate bad items,
        substituting a zero vector for any item that keeps failing.
        """
        from openai import InternalServerError, RateLimitError, APIStatusError

        delay = 2.0
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.client.embeddings.create(model=self.model, input=batch)
                return [np.array(e.embedding, dtype=np.float32) for e in resp.data]
            except (InternalServerError, RateLimitError) as e:
                last_err = e
                print(f"    [embed] {type(e).__name__} on attempt {attempt+1}/{max_retries}, "
                      f"retrying in {delay:.0f}s…")
                time.sleep(delay)
                delay *= 2
            except APIStatusError as e:
                # 400 bad request — won't be fixed by retrying the whole batch
                last_err = e
                break

        # Batch-level retries exhausted — fall back to one-by-one
        print(f"    [embed] Batch failed ({last_err}). Falling back to single-item mode…")
        dim = 1536 if "small" in self.model else 3072
        results = []
        for item in batch:
            for attempt in range(2):
                try:
                    r = self.client.embeddings.create(model=self.model, input=[item])
                    results.append(np.array(r.data[0].embedding, dtype=np.float32))
                    break
                except Exception:
                    time.sleep(1)
            else:
                print(f"    [embed] Skipping item (persistent error), using zero vector.")
                results.append(np.zeros(dim, dtype=np.float32))
        return results

    def embed(self, texts: list[str]) -> np.ndarray:
        # Sanitize all texts first — removes null bytes, broken Unicode, etc.
        clean_texts = [_sanitize(t) for t in texts]

        # Find which (sanitized) texts are not yet cached
        uncached_clean = [t for t in clean_texts if t not in self._cache]

        if uncached_clean:
            batches = self._token_safe_batches(uncached_clean)
            for i, batch in enumerate(batches):
                if len(batches) > 1:
                    print(f"    [embed] batch {i+1}/{len(batches)} ({len(batch)} items)…")
                embeddings = self._embed_batch_with_retry(batch)
                for text, emb in zip(batch, embeddings):
                    self._cache[text] = emb

        return np.vstack([self._cache[t] for t in clean_texts])

# ── Vector Index (FAISS) ───────────────────────────────────────────────────────

class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # inner product = cosine after normalization
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_emb: np.ndarray, k: int = TOP_K) -> list[RetrievalResult]:
        q = query_emb.copy().reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        return [
            RetrievalResult(chunk=self.chunks[idx], score=float(scores[0][i]), method="vector")
            for i, idx in enumerate(indices[0]) if idx >= 0
        ]

# ── BM25 Index ─────────────────────────────────────────────────────────────────

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]):
        self.chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = TOP_K) -> list[RetrievalResult]:
        scores = self.bm25.get_scores(query.lower().split())
        top_k = np.argsort(scores)[::-1][:k]
        # normalize BM25 scores to [0,1]
        max_s = max(scores[top_k]) if scores[top_k].max() > 0 else 1.0
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]/max_s), method="bm25")
            for i in top_k
        ]

# ── Hybrid Retrieval ───────────────────────────────────────────────────────────

def hybrid_search(
    vector_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    alpha: float = 0.6,    # weight for vector scores
    k: int = TOP_K
) -> list[RetrievalResult]:
    """
    Reciprocal Rank Fusion + weighted score combination.
    alpha=0.6 favours semantic similarity while BM25 handles exact-match terms.
    """
    scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, Chunk] = {}

    for rank, r in enumerate(vector_results):
        scores[r.chunk.chunk_id] += alpha * (1 / (60 + rank + 1))
        chunk_map[r.chunk.chunk_id] = r.chunk

    for rank, r in enumerate(bm25_results):
        scores[r.chunk.chunk_id] += (1 - alpha) * (1 / (60 + rank + 1))
        chunk_map[r.chunk.chunk_id] = r.chunk

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
    return [
        RetrievalResult(chunk=chunk_map[cid], score=scores[cid], method="hybrid")
        for cid in sorted_ids
    ]

# ── Reranker ───────────────────────────────────────────────────────────────────

class LLMReranker:
    """
    Asks the LLM to score each (query, passage) pair for relevance (0-10).
    More expensive but higher precision. Used as post-retrieval step.
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def rerank(self, query: str, results: list[RetrievalResult], top_n: int = RERANK_TOP_N) -> list[RetrievalResult]:
        scored = []
        for r in results:
            prompt = (
                f"Query: {query}\n\n"
                f"Passage: {r.chunk.text[:600]}\n\n"
                "On a scale of 0-10, how relevant is this passage to answering the query? "
                "Reply with just the number."
            )
            try:
                resp = self.client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5, temperature=0
                )
                score = float(resp.choices[0].message.content.strip())
            except Exception:
                score = r.score * 10
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        for score, r in scored[:top_n]:
            r.score = score / 10.0
            r.method = "reranked"
        return [r for _, r in scored[:top_n]]

# ── Query Rewriter ─────────────────────────────────────────────────────────────

class QueryRewriter:
    """
    Expands the user query into a more retrieval-friendly form using the LLM.
    Handles: abbreviations, implicit context, multi-hop intent.
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def rewrite(self, query: str) -> str:
        prompt = (
            "Rewrite the following search query to improve retrieval from a corpus of AI research papers. "
            "Expand acronyms, add related technical terms, and make the intent explicit. "
            "Return only the rewritten query, nothing else.\n\n"
            f"Original: {query}"
        )
        resp = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100, temperature=0
        )
        return resp.choices[0].message.content.strip()

# ── Answer Generator ───────────────────────────────────────────────────────────

class AnswerGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, query: str, results: list[RetrievalResult]) -> dict:
        context = "\n\n".join([
            f"[Source {i+1}: {r.chunk.metadata.get('title','Unknown')}]\n{r.chunk.text}"
            for i, r in enumerate(results)
        ])
        system = (
            "You are a precise research assistant. Answer the question using ONLY the provided context. "
            "If the context does not contain enough information, say so explicitly. "
            "Cite source numbers like [1], [2] when referencing specific claims."
        )
        user = f"Context:\n{context}\n\nQuestion: {query}"

        start = time.time()
        resp = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=500, temperature=0.1
        )
        latency = time.time() - start
        answer = resp.choices[0].message.content.strip()
        usage = resp.usage

        return {
            "answer": answer,
            "latency_s": round(latency, 3),
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "sources": [r.chunk.metadata.get("title", r.chunk.doc_id) for r in results]
        }

# ── Main RAG Pipeline ──────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates the full RAG pipeline.
    Configuration via constructor args — swap chunker and embedder to benchmark.
    """
    def __init__(
        self,
        chunker: Literal["fixed", "sentence"] = "fixed",
        embed_model: Literal["small", "large"] = "small",
        search_mode: Literal["vector", "hybrid"] = "vector",
        use_reranker: bool = False,
        use_query_rewriter: bool = False,
        chunk_size: int = 300,
        overlap: int = 50,
    ):
        self.config = {
            "chunker": chunker,
            "embed_model": embed_model,
            "search_mode": search_mode,
            "use_reranker": use_reranker,
            "use_query_rewriter": use_query_rewriter,
        }
        # SentenceChunker uses sentence-count overlap — 1 is correct; passing 50 would mean
        # 50 sentences of overlap (~1000 words), causing tiny chunk explosion.
        self.chunker = (
            FixedSizeChunker(chunk_size, overlap)
            if chunker == "fixed"
            else SentenceChunker(max_tokens=chunk_size, overlap_sentences=1)
        )
        self.embedder = EmbeddingEngine(embed_model)
        self.vector_index: VectorIndex | None = None
        self.bm25_index: BM25Index | None = None
        self.reranker = LLMReranker() if use_reranker else None
        self.query_rewriter = QueryRewriter() if use_query_rewriter else None
        self.all_chunks: list[Chunk] = []
        self._embed_dim = 1536 if embed_model == "small" else 3072

    def ingest(self, documents: list[dict]):
        """
        documents: list of {"doc_id": str, "text": str, "metadata": {...}}
        """
        print(f"  Ingesting {len(documents)} documents...")
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc["text"], doc["doc_id"], doc.get("metadata", {}))
            all_chunks.extend(chunks)

        print(f"  Generated {len(all_chunks)} chunks. Embedding...")
        texts = [c.text for c in all_chunks]
        t0 = time.time()
        embeddings = self.embedder.embed(texts)
        embed_time = time.time() - t0
        print(f"  Embedded in {embed_time:.2f}s.")

        self.vector_index = VectorIndex(self._embed_dim)
        self.vector_index.add(all_chunks, embeddings)

        self.bm25_index = BM25Index()
        self.bm25_index.add(all_chunks)

        self.all_chunks = all_chunks
        return {"num_chunks": len(all_chunks), "embed_time_s": round(embed_time, 2)}

    def query(self, question: str, top_k: int = TOP_K) -> dict:
        t0 = time.time()

        # Optional query rewriting
        rewritten_query = question
        if self.query_rewriter:
            rewritten_query = self.query_rewriter.rewrite(question)

        # Embed query
        q_emb = self.embedder.embed([rewritten_query])

        # Retrieve
        vector_results = self.vector_index.search(q_emb, k=top_k)

        if self.config["search_mode"] == "hybrid":
            bm25_results = self.bm25_index.search(rewritten_query, k=top_k)
            results = hybrid_search(vector_results, bm25_results, k=top_k)
        else:
            results = vector_results

        # Optional reranking
        if self.reranker:
            results = self.reranker.rerank(rewritten_query, results, top_n=RERANK_TOP_N)

        retrieval_time = time.time() - t0

        # Generate answer
        gen = AnswerGenerator().generate(question, results)

        return {
            "question": question,
            "rewritten_query": rewritten_query if self.query_rewriter else None,
            "retrieved_chunks": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "doc_id": r.chunk.doc_id,
                    "score": round(r.score, 4),
                    "method": r.method,
                    "title": r.chunk.metadata.get("title", ""),
                    "text_snippet": r.chunk.text[:200]
                }
                for r in results
            ],
            "answer": gen["answer"],
            "sources": gen["sources"],
            "retrieval_latency_s": round(retrieval_time, 3),
            "generation_latency_s": gen["latency_s"],
            "total_latency_s": round(retrieval_time + gen["latency_s"], 3),
            "prompt_tokens": gen["prompt_tokens"],
            "completion_tokens": gen["completion_tokens"],
        }