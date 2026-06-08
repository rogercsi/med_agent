"""Unit tests for RAG pipeline components."""
import pytest
from unittest.mock import MagicMock, patch


def test_bm25_store_add_and_search():
    from medical_agent.rag.bm25_store import BM25Store

    store = BM25Store()
    store.add("c1", "高血压患者服用ACEI出现干咳，应换用ARB类降压药")
    store.add("c2", "糖尿病患者首选二甲双胍作为基础降糖药物")
    store.add("c3", "感冒发热可以使用布洛芬或对乙酰氨基酚退热")

    results = store.search("高血压 ACEI 干咳", top_k=2)
    assert len(results) > 0
    assert results[0][0] == "c1"  # most relevant chunk


def test_bm25_store_save_load(tmp_path):
    from medical_agent.rag.bm25_store import BM25Store

    store = BM25Store()
    store.add("id1", "测试文本一")
    store.add("id2", "测试文本二")

    path = tmp_path / "test_bm25.pkl"
    store.save(path)

    loaded = BM25Store.load(path)
    assert loaded.corpus_ids == ["id1", "id2"]
    assert len(loaded.corpus_texts) == 2


def test_rrf_score():
    from medical_agent.rag.retriever import _rrf_score

    assert _rrf_score(0) == pytest.approx(1.0 / 60)
    assert _rrf_score(59) == pytest.approx(1.0 / 119)
    assert _rrf_score(None) == 0.0


def test_rag_pipeline_naive_mode(sample_chunks):
    """Naive mode skips rewriting and reranking."""
    from medical_agent.rag.pipeline import RAGPipeline
    from medical_agent.rag.retriever import CandidateChunk

    mock_candidates = [
        CandidateChunk(chunk_id=f"c{i}", text=t, source="test", dense_rank=i)
        for i, t in enumerate(sample_chunks)
    ]

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = MagicMock()
    pipeline.settings.rerank_top_k = 5
    pipeline.settings.compress_token_threshold = 1500
    pipeline.settings.llm_model = "gpt-4o-mini"
    pipeline.retriever = MagicMock()
    pipeline.retriever.retrieve.return_value = mock_candidates
    pipeline._llm = MagicMock()

    result = pipeline.query("高血压如何治疗", mode="naive")

    assert result.mode == "naive"
    assert len(result.chunks) == 3
    assert result.rewritten_query == "高血压如何治疗"


def test_compress_context_below_threshold(sample_chunks):
    """Context below threshold should not be compressed."""
    from medical_agent.rag.compressor import compress_context

    # Patch threshold to a very high value so compression never fires
    with patch("medical_agent.rag.compressor.get_settings") as mock_settings:
        mock_settings.return_value.compress_token_threshold = 999999
        result = compress_context(sample_chunks, "高血压治疗")

    assert result == sample_chunks
