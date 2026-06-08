from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from openai import OpenAI

from medical_agent.config import get_settings
from medical_agent.rag.compressor import compress_context
from medical_agent.rag.reranker import rerank
from medical_agent.rag.retriever import CandidateChunk, HybridRetriever, get_retriever


@dataclass
class RetrievedContext:
    chunks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    raw_candidates: list[CandidateChunk] = field(default_factory=list)
    rewritten_query: str = ""
    mode: Literal["naive", "optimized"] = "optimized"
    token_before_compress: int = 0
    token_after_compress: int = 0


class RAGPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever: HybridRetriever = get_retriever()
        self._llm = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )

    def _rewrite_query(self, query: str, history_summary: str = "") -> str:
        context_hint = f"\n对话摘要：{history_summary}" if history_summary else ""
        prompt = (
            f"将以下医疗问题改写为更具体、自包含的搜索查询，保留关键医学术语，不超过50字。{context_hint}\n"
            f"原始问题：{query}\n改写后的查询："
        )
        resp = self._llm.chat.completions.create(
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
        )
        return (resp.choices[0].message.content or query).strip()

    def _dense_only(self, query: str, top_k: int = 3) -> list[CandidateChunk]:
        all_candidates = self.retriever.retrieve(query)
        dense_only = sorted(
            [c for c in all_candidates if c.dense_rank is not None],
            key=lambda c: c.dense_rank or 9999,
        )
        return dense_only[:top_k]

    def query(
        self,
        query: str,
        mode: Literal["naive", "optimized"] = "optimized",
        history_summary: str = "",
    ) -> RetrievedContext:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        if mode == "naive":
            candidates = self._dense_only(query, top_k=3)
            texts = [c.text for c in candidates]
            return RetrievedContext(
                chunks=texts,
                sources=[c.source for c in candidates],
                raw_candidates=candidates,
                rewritten_query=query,
                mode="naive",
            )

        # Optimized: rewrite → hybrid retrieve → rerank → compress
        rewritten = self._rewrite_query(query, history_summary)

        all_candidates = self.retriever.retrieve(rewritten)
        reranked = rerank(rewritten, all_candidates, self.settings.rerank_top_k)

        texts = [c.text for c in reranked]
        token_before = sum(len(enc.encode(t)) for t in texts)
        compressed = compress_context(texts, rewritten)
        token_after = sum(len(enc.encode(t)) for t in compressed)

        return RetrievedContext(
            chunks=compressed,
            sources=list({c.source for c in reranked}),
            raw_candidates=reranked,
            rewritten_query=rewritten,
            mode="optimized",
            token_before_compress=token_before,
            token_after_compress=token_after,
        )


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline()
