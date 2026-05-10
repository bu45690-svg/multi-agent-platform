"""
worker/rag/retriever.py

RAGRetriever — queries the FAISS index and returns ranked RetrievedChunk
objects ready to be written into SharedContext.

Usage (inside RetrievalAgent)
------------------------------
    retriever = RAGRetriever(index, store)
    chunks = await retriever.query("What is a DAG?", top_k=5, min_score=0.30)
    for chunk in chunks:
        context.add_chunk(chunk)

Design decisions
----------------
• Query text is embedded with the same backend as the index (set via EMBED_MODEL).
• Scores are cosine similarities (IndexFlatIP on unit vectors).
• min_score threshold filters out low-quality hits before returning.
• Reciprocal Rank Fusion (RRF) is available for multi-query retrieval — useful
  when DecompositionAgent produces several sub-tasks that each need retrieval.
• MMR (Maximal Marginal Relevance) re-ranking reduces redundant chunks.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from worker.context import RetrievedChunk
from worker.rag.indexer import ChunkStore, _embed_batch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.25        # cosine similarity threshold
DEFAULT_MMR_LAMBDA = 0.6        # MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity
RRF_K = 60                      # standard RRF constant


# ---------------------------------------------------------------------------
# RAGRetriever
# ---------------------------------------------------------------------------


class RAGRetriever:
    """
    Thin wrapper around a (faiss_index, ChunkStore) pair.

    Parameters
    ----------
    index       : a loaded faiss.IndexFlatIP instance
    store       : ChunkStore from build_or_load_index()
    embed_model : override EMBED_MODEL env var for query embedding
    """

    def __init__(
        self,
        index: Any,                # faiss.Index (typed as Any to avoid hard dep)
        store: ChunkStore,
        embed_model: str | None = None,
    ) -> None:
        self._index = index
        self._store = store
        self._embed_model = embed_model

    # ------------------------------------------------------------------
    # Single-query retrieval
    # ------------------------------------------------------------------

    async def query(
        self,
        text: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        use_mmr: bool = False,
        mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    ) -> list[RetrievedChunk]:
        """
        Embed `text`, search the FAISS index, and return ranked chunks.

        Parameters
        ----------
        text       : the query string
        top_k      : maximum number of chunks to return
        min_score  : cosine similarity cutoff (chunks below are dropped)
        use_mmr    : apply Maximal Marginal Relevance re-ranking
        mmr_lambda : MMR λ parameter (1.0 = relevance-only)
        """
        if not text.strip():
            logger.warning("RAGRetriever.query called with empty text")
            return []

        if len(self._store) == 0:
            logger.warning("RAGRetriever: index is empty, returning no chunks")
            return []

        # 1. Embed query
        q_vec = await self._embed_query(text)   # shape: (1, dim)

        # 2. FAISS search — fetch more than top_k to allow threshold filtering
        fetch_k = min(max(top_k * 3, 20), len(self._store))
        scores, indices = self._index.search(q_vec, fetch_k)
        scores = scores[0]    # flatten
        indices = indices[0]

        # 3. Filter by score threshold and validity
        candidates: list[tuple[float, int]] = [
            (float(s), int(i))
            for s, i in zip(scores, indices)
            if i >= 0 and float(s) >= min_score
        ]

        if not candidates:
            logger.info("RAGRetriever: no chunks above min_score=%.2f", min_score)
            return []

        # 4. Optional MMR re-ranking
        if use_mmr and len(candidates) > 1:
            candidates = await self._mmr_rerank(
                q_vec, candidates, top_k, mmr_lambda
            )
        else:
            candidates = candidates[:top_k]

        # 5. Build RetrievedChunk objects
        results: list[RetrievedChunk] = []
        for score, idx in candidates:
            raw = self._store.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=raw.chunk_id,
                    source_file=raw.source_file,
                    content=raw.content,
                    relevance_score=round(score, 4),
                    retrieval_method="faiss",
                )
            )

        logger.debug(
            "RAGRetriever: query=%r → %d chunks (top score=%.3f)",
            text[:60],
            len(results),
            results[0].relevance_score if results else 0.0,
        )
        return results

    # ------------------------------------------------------------------
    # Multi-query retrieval with Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    async def query_multi(
        self,
        queries: list[str],
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        use_mmr: bool = True,
    ) -> list[RetrievedChunk]:
        """
        Run multiple queries and fuse results with Reciprocal Rank Fusion.

        Useful when the DecompositionAgent produces several sub-queries for
        a single retrieval task.
        """
        if not queries:
            return []

        # Collect per-query ranked lists (idx, score)
        per_query: list[list[tuple[int, float]]] = []
        for q in queries:
            if not q.strip():
                continue
            q_vec = await self._embed_query(q)
            fetch_k = min(max(top_k * 3, 20), len(self._store))
            scores, indices = self._index.search(q_vec, fetch_k)
            ranked = [
                (int(i), float(s))
                for s, i in zip(scores[0], indices[0])
                if int(i) >= 0 and float(s) >= min_score
            ]
            per_query.append(ranked)

        if not per_query:
            return []

        # RRF scoring
        rrf_scores: dict[int, float] = {}
        for ranked in per_query:
            for rank, (idx, _) in enumerate(ranked):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        # Sort by RRF score
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        if use_mmr and len(fused) > 1:
            # Use the first query vector for MMR diversity
            q_vec_first = await self._embed_query(queries[0])
            candidate_tuples = [(score, idx) for idx, score in fused]
            candidate_tuples = await self._mmr_rerank(
                q_vec_first, candidate_tuples, top_k, DEFAULT_MMR_LAMBDA
            )
        else:
            candidate_tuples = [(score, idx) for idx, score in fused[:top_k]]

        results: list[RetrievedChunk] = []
        for score, idx in candidate_tuples:
            raw = self._store.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=raw.chunk_id,
                    source_file=raw.source_file,
                    content=raw.content,
                    relevance_score=round(min(score, 1.0), 4),
                    retrieval_method="faiss",
                )
            )

        logger.debug(
            "RAGRetriever.query_multi: %d queries → %d fused chunks",
            len(queries),
            len(results),
        )
        return results

    # ------------------------------------------------------------------
    # MMR re-ranking
    # ------------------------------------------------------------------

    async def _mmr_rerank(
        self,
        q_vec: np.ndarray,                   # (1, dim)
        candidates: list[tuple[float, int]], # [(score, faiss_idx), ...]
        top_k: int,
        lam: float,
    ) -> list[tuple[float, int]]:
        """
        Maximal Marginal Relevance:
            mmr(d) = λ · sim(d, q) − (1−λ) · max_{s∈selected} sim(d, s)

        Returns up to top_k (score, idx) tuples in MMR order.
        """
        if not candidates:
            return []

        # Reconstruct candidate vectors from FAISS
        idx_list = [idx for _, idx in candidates]
        vecs = np.vstack([
            self._index.reconstruct(i).reshape(1, -1) for i in idx_list
        ])  # shape: (N, dim)

        q_flat = q_vec[0]   # (dim,)

        # Relevance scores (cosine similarity to query)
        rel = np.dot(vecs, q_flat)  # (N,)

        selected: list[int] = []        # positions in idx_list
        selected_vecs: list[np.ndarray] = []
        remaining = list(range(len(idx_list)))
        result: list[tuple[float, int]] = []

        while remaining and len(result) < top_k:
            if not selected_vecs:
                # First pick: pure relevance
                best_pos = max(remaining, key=lambda p: rel[p])
            else:
                sel_mat = np.vstack(selected_vecs)  # (S, dim)
                # Redundancy: max sim to any already-selected vector
                redundancy = np.max(np.dot(vecs[remaining], sel_mat.T), axis=1)
                mmr_scores = lam * rel[remaining] - (1 - lam) * redundancy
                best_local = int(np.argmax(mmr_scores))
                best_pos = remaining[best_local]

            faiss_idx = idx_list[best_pos]
            score = float(rel[best_pos])
            result.append((score, faiss_idx))
            selected_vecs.append(vecs[best_pos])
            selected.append(best_pos)
            remaining.remove(best_pos)

        return result

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    async def _embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string. Returns shape (1, dim).
        Uses the same backend as the index so vectors are compatible.
        """
        import os
        original_model = os.environ.get("EMBED_MODEL")
        if self._embed_model:
            os.environ["EMBED_MODEL"] = self._embed_model

        vecs = await _embed_batch([text])   # shape (1, dim)

        if self._embed_model and original_model is not None:
            os.environ["EMBED_MODEL"] = original_model

        return vecs

    # ------------------------------------------------------------------
    # Convenience: retrieve for a task description from the DAG
    # ------------------------------------------------------------------

    async def retrieve_for_task(
        self,
        task_description: str,
        context_query: str = "",
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[RetrievedChunk]:
        """
        Convenience wrapper: builds a multi-query from the task description
        and the original user query, then runs RRF fusion.
        """
        queries = [q for q in [task_description, context_query] if q.strip()]
        if len(queries) > 1:
            return await self.query_multi(queries, top_k=top_k, min_score=min_score)
        return await self.query(queries[0], top_k=top_k, min_score=min_score)