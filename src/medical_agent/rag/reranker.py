from functools import lru_cache

from FlagEmbedding import FlagReranker

from medical_agent.config import get_settings
from medical_agent.rag.retriever import CandidateChunk


@lru_cache(maxsize=1)
def get_reranker() -> FlagReranker:
    settings = get_settings()
    return FlagReranker(settings.reranker_model, use_fp16=True)


def rerank(query: str, candidates: list[CandidateChunk], top_k: int) -> list[CandidateChunk]:
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [[query, c.text] for c in candidates]
    scores: list[float] = reranker.compute_score(pairs, normalize=True)

    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    return [c for c, _ in ranked[:top_k]]
