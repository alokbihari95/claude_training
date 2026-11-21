"""Shared retrieval primitives used by all three engines: HNSW vector search, BM25, RRF fusion, rerank."""
from __future__ import annotations

import json
import re
import threading

import chromadb
from rank_bm25 import BM25Okapi

import config
from src.graph_store import GraphStore
from src.llm import get_embeddings, get_reranker

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
COLLECTION = "chunks"
_tok = re.compile(r"[a-z0-9]+(?:[./%\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _tok.findall(text.lower())


class RAGIndex:
    """Chunk store + Chroma (HNSW, cosine) + BM25 + graph, loaded from config.INDEX_DIR."""

    def __init__(self) -> None:
        self.chunks: dict[str, dict] = {}
        self.graph = GraphStore()
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._client = None
        self._col = None
        self.reload()

    @staticmethod
    def _stamp() -> float:
        p = config.INDEX_DIR / "chunks.json"
        return p.stat().st_mtime if p.exists() else 0.0

    def reload(self) -> None:
        path = config.INDEX_DIR / "chunks.json"
        self.stamp = self._stamp()
        self.chunks = {c["chunk_id"]: c for c in json.loads(path.read_text())} if path.exists() else {}
        self.graph = GraphStore.load()
        self._bm25_ids = list(self.chunks)
        self._bm25 = BM25Okapi([tokenize(self.chunks[i]["text"]) for i in self._bm25_ids]) if self._bm25_ids else None
        if config.CHROMA_DIR.exists():
            self._client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            try:
                self._col = self._client.get_collection(COLLECTION)
            except Exception:  # noqa: BLE001
                self._col = None

    @property
    def ready(self) -> bool:
        return bool(self.chunks) and self._col is not None

    @staticmethod
    def _where(areas: set[str] | None):
        if not areas:
            return None
        return {"practice_area": {"$in": sorted(areas)}}

    def embed_query(self, text: str, instruct: bool = True) -> list[float]:
        return get_embeddings().embed_query((BGE_QUERY_PREFIX if instruct else "") + text)

    def vector_search(self, vec: list[float], k: int, areas: set[str] | None = None) -> list[tuple[str, float]]:
        res = self._col.query(query_embeddings=[vec], n_results=min(k, self._col.count()), where=self._where(areas))
        return [(i, 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0])]

    def bm25_search(self, query: str, k: int, areas: set[str] | None = None) -> list[tuple[str, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: -x[1])
        out = []
        for cid, sc in ranked:
            if sc <= 0:
                break
            if areas and self.chunks[cid]["metadata"]["practice_area"] not in areas:
                continue
            out.append((cid, float(sc)))
            if len(out) >= k:
                break
        return out

    def context(self, chunk_id: str, score: float = 0.0, via: str = "") -> dict:
        c = self.chunks[chunk_id]
        return {"chunk_id": chunk_id, "text": c["text"], "metadata": c["metadata"], "score": score, "via": via}


def rrf_fuse(rank_lists: dict[str, list[str]], c: int = config.DEFAULTS.rrf_constant) -> list[tuple[str, float, list[str]]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (c + rank). Returns (id, score, sources) best first."""
    score: dict[str, float] = {}
    src: dict[str, list[str]] = {}
    for name, ids in rank_lists.items():
        for rank, cid in enumerate(ids, 1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (c + rank)
            src.setdefault(cid, []).append(name)
    return sorted(((i, s, src[i]) for i, s in score.items()), key=lambda x: -x[1])


def rerank(query: str, contexts: list[dict], top_n: int) -> list[dict]:
    if not contexts:
        return contexts
    scores = get_reranker().predict([(query, c["text"]) for c in contexts], show_progress_bar=False)
    for c, s in zip(contexts, scores):
        c["rerank_score"] = float(s)
    return sorted(contexts, key=lambda c: -c["rerank_score"])[:top_n]


_index: RAGIndex | None = None
_lock = threading.Lock()


def get_index() -> RAGIndex:
    global _index
    with _lock:
        if _index is None or _index.stamp != RAGIndex._stamp():   # index rebuilt on disk (CLI re-ingest): reload handles
            _index = RAGIndex()
        return _index


def reset_index() -> None:
    global _index
    with _lock:
        _index = None
