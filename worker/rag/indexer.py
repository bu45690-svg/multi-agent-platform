"""
worker/rag/indexer.py

FAISS index builder for the multi-agent RAG pipeline.

Responsibilities
----------------
1. Load all *.md files from a configured docs directory.
2. Split them into overlapping chunks (configurable size + stride).
3. Embed each chunk via the Anthropic Embeddings API
   (or a local sentence-transformers model as fallback).
4. Build a FAISS IndexFlatIP (inner-product / cosine) index.
5. Persist the index + chunk metadata to disk so the worker can reload
   it on restart without re-embedding.

Called once at worker startup:
    from worker.rag.indexer import build_or_load_index
    index, store = await build_or_load_index()

The returned (index, store) pair is handed to RAGRetriever.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------

DOCS_DIR = Path(os.environ.get("DOCS_DIR", "/app/docs"))
INDEX_CACHE_DIR = Path(os.environ.get("INDEX_CACHE_DIR", "/app/.index_cache"))

CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "400"))     # tokens ≈ words here
CHUNK_STRIDE = int(os.environ.get("RAG_CHUNK_STRIDE", "100"))  # overlap in words
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "local")  # "local" or "voyage-*"

_INDEX_FILE = INDEX_CACHE_DIR / "faiss.index"
_STORE_FILE = INDEX_CACHE_DIR / "chunk_store.pkl"
_MANIFEST_FILE = INDEX_CACHE_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str
    source_file: str        # relative path from DOCS_DIR
    content: str
    char_start: int
    char_end: int


@dataclass
class ChunkStore:
    """
    Flat list of Chunk objects whose index position matches the FAISS row index.
    i.e. store.chunks[i] ↔ faiss_index.reconstruct(i)
    """
    chunks: list[Chunk] = field(default_factory=list)
    embed_dim: int = 0

    def __len__(self) -> int:
        return len(self.chunks)


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------


def _split_into_chunks(text: str, source_file: str) -> list[Chunk]:
    """
    Simple word-based sliding window chunker.  Word-based (not token-based)
    to avoid a tiktoken dependency at index time; close enough for FAISS search.
    """
    words = text.split()
    chunks: list[Chunk] = []
    start_word = 0

    while start_word < len(words):
        end_word = min(start_word + CHUNK_SIZE, len(words))
        content = " ".join(words[start_word:end_word])

        # Approximate character offsets (good enough for citations)
        char_start = sum(len(w) + 1 for w in words[:start_word])
        char_end = char_start + len(content)

        chunk_id = hashlib.sha256(
            f"{source_file}:{char_start}".encode()
        ).hexdigest()[:12]

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_file=source_file,
                content=content,
                char_start=char_start,
                char_end=char_end,
            )
        )

        if end_word == len(words):
            break
        start_word += CHUNK_SIZE - CHUNK_STRIDE   # slide with overlap

    return chunks


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------


async def _embed_local(texts: list[str]) -> np.ndarray:
    """
    Local embedding using sentence-transformers (no API key needed).
    Falls back to random unit vectors if the library is not installed, so the
    system still starts in environments without GPU/ML libraries.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        loop = asyncio.get_event_loop()
        model = SentenceTransformer("all-MiniLM-L6-v2")

        def _encode():
            return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        vecs = await loop.run_in_executor(None, _encode)
        return np.array(vecs, dtype=np.float32)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; using random embeddings (dev mode)"
        )
        dim = 384
        rng = np.random.default_rng(seed=42)
        vecs = rng.standard_normal((len(texts), dim)).astype(np.float32)
        # Normalise to unit vectors so inner product ≈ cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        return vecs / norms


async def _embed_voyage(texts: list[str], model: str) -> np.ndarray:
    """
    Voyage AI embeddings (https://docs.voyageai.com/).
    Set VOYAGE_API_KEY and EMBED_MODEL=voyage-2 to use.
    """
    try:
        import voyageai  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pip install voyageai to use Voyage embeddings") from exc

    client = voyageai.AsyncClient(api_key=os.environ.get("VOYAGE_API_KEY", ""))
    result = await client.embed(texts, model=model, input_type="document")
    return np.array(result.embeddings, dtype=np.float32)


async def _embed_batch(texts: list[str]) -> np.ndarray:
    """Dispatch to the configured embedding backend, in batches."""
    model = EMBED_MODEL.strip().lower()
    all_vecs: list[np.ndarray] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        if model == "local":
            vecs = await _embed_local(batch)
        elif model.startswith("voyage"):
            vecs = await _embed_voyage(batch, model)
        else:
            raise ValueError(f"Unknown EMBED_MODEL={model!r}")
        all_vecs.append(vecs)

    return np.vstack(all_vecs)


# ---------------------------------------------------------------------------
# Manifest — tracks which files have been indexed
# ---------------------------------------------------------------------------


def _file_manifest(docs_dir: Path) -> dict[str, str]:
    """Return {relative_path: sha256_of_content} for every .md file."""
    manifest: dict[str, str] = {}
    for path in sorted(docs_dir.rglob("*.md")):
        content = path.read_bytes()
        manifest[str(path.relative_to(docs_dir))] = hashlib.sha256(content).hexdigest()
    return manifest


def _cache_is_valid(docs_dir: Path) -> bool:
    """Return True if cached index matches the current docs directory."""
    if not (_INDEX_FILE.exists() and _STORE_FILE.exists() and _MANIFEST_FILE.exists()):
        return False
    try:
        cached = json.loads(_MANIFEST_FILE.read_text())
        return cached == _file_manifest(docs_dir)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def build_or_load_index(
    docs_dir: Path | None = None,
    force_rebuild: bool = False,
) -> tuple[Any, ChunkStore]:
    """
    Return a (faiss_index, ChunkStore) pair.

    • If a valid cache exists and force_rebuild=False, load from disk.
    • Otherwise, read all .md files, chunk them, embed, and build a new index.

    Parameters
    ----------
    docs_dir      : directory containing *.md source documents
    force_rebuild : ignore cache and rebuild from scratch
    """
    import faiss  # local import — not available in all environments

    docs_dir = docs_dir or DOCS_DIR
    INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force_rebuild and _cache_is_valid(docs_dir):
        logger.info("Loading FAISS index from cache (%s)", INDEX_CACHE_DIR)
        index = faiss.read_index(str(_INDEX_FILE))
        store: ChunkStore = pickle.loads(_STORE_FILE.read_bytes())
        logger.info("Loaded %d chunks from cache.", len(store))
        return index, store

    # ── Build from scratch ──────────────────────────────────────────────────

    md_files = sorted(docs_dir.rglob("*.md"))
    if not md_files:
        logger.warning("No .md files found in %s; building empty index.", docs_dir)
        dim = 384
        index = faiss.IndexFlatIP(dim)
        return index, ChunkStore(embed_dim=dim)

    logger.info("Building FAISS index from %d files in %s…", len(md_files), docs_dir)
    t0 = time.perf_counter()

    # 1. Load + chunk all documents
    all_chunks: list[Chunk] = []
    for path in md_files:
        rel = str(path.relative_to(docs_dir))
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = _split_into_chunks(text, rel)
        all_chunks.extend(chunks)
        logger.debug("  %s → %d chunks", rel, len(chunks))

    logger.info("Total chunks: %d", len(all_chunks))

    # 2. Embed all chunks
    texts = [c.content for c in all_chunks]
    embeddings = await _embed_batch(texts)          # shape: (N, dim)

    # 3. Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)                  # inner product on unit vecs ≈ cosine
    index.add(embeddings)

    store = ChunkStore(chunks=all_chunks, embed_dim=dim)

    # 4. Persist to cache
    faiss.write_index(index, str(_INDEX_FILE))
    _STORE_FILE.write_bytes(pickle.dumps(store))
    _MANIFEST_FILE.write_text(
        json.dumps(_file_manifest(docs_dir), indent=2)
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Index built: %d vectors, dim=%d, %.2fs elapsed", len(store), dim, elapsed
    )

    return index, store