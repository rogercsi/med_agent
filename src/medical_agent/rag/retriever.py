from dataclasses import dataclass, field
from functools import lru_cache

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from medical_agent.config import Settings, get_settings
from medical_agent.rag.bm25_store import BM25Store, get_bm25_store
from medical_agent.rag.embedder import embed_query


@dataclass
class CandidateChunk:
    chunk_id: str
    text: str
    source: str
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float = 0.0


def _rrf_score(rank: int | None, k: int = 60) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (k + rank)


class HybridRetriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._qdrant = QdrantClient(
            host=self.settings.qdrant_host, port=self.settings.qdrant_port
        )
        self._bm25: BM25Store = get_bm25_store()

    def _dense_search(self, query: str, top_k: int) -> list[tuple[str, str, str, float]]:
        """Returns list of (chunk_id, text, source, score)."""
        vec = embed_query(query)
        results = self._qdrant.search(
            collection_name=self.settings.qdrant_collection_rag,
            query_vector=vec,
            limit=top_k,
            with_payload=True,
        )
        return [
            (
                str(r.id),
                r.payload.get("text", "") if r.payload else "",
                r.payload.get("source", "") if r.payload else "",
                r.score,
            )
            for r in results
        ]

    def retrieve(self, query: str) -> list[CandidateChunk]:
        bm25_top_k = self.settings.bm25_top_k
        dense_top_k = self.settings.dense_top_k

        bm25_hits = self._bm25.search(query, bm25_top_k)
        dense_hits = self._dense_search(query, dense_top_k)

        # Build a merged dict keyed by chunk_id
        merged: dict[str, CandidateChunk] = {}

        for rank, (cid, score) in enumerate(bm25_hits):
            merged[cid] = CandidateChunk(
                chunk_id=cid,
                text=self._bm25.id_to_text.get(cid, ""),
                source="",
                bm25_rank=rank,
            )

        for rank, (cid, text, source, _score) in enumerate(dense_hits):
            if cid in merged:
                merged[cid].dense_rank = rank
                if not merged[cid].source:
                    merged[cid].source = source
                if not merged[cid].text:
                    merged[cid].text = text
            else:
                merged[cid] = CandidateChunk(
                    chunk_id=cid,
                    text=text,
                    source=source,
                    dense_rank=rank,
                )

        # Compute RRF scores
        for chunk in merged.values():
            chunk.rrf_score = _rrf_score(chunk.bm25_rank) + _rrf_score(chunk.dense_rank)

        return sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
