import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_settings(monkeypatch):
    from medical_agent.config import Settings
    s = Settings(
        openai_api_key="sk-test",
        qdrant_host="localhost",
        qdrant_port=6333,
        llm_model="gpt-4o-mini",
        bm25_index_path="/tmp/test_bm25.pkl",
        checkpoints_db_path="/tmp/test_checkpoints.db",
        ragas_testset_path="/tmp/test_testset.json",
        ragas_baseline_path="/tmp/test_baseline.json",
    )
    monkeypatch.setattr("medical_agent.config.get_settings", lambda: s)
    return s


@pytest.fixture
def sample_chunks():
    return [
        "高血压诊断标准为收缩压≥140 mmHg和/或舒张压≥90 mmHg。",
        "一线降压药物包括ACEI、ARB、CCB和噻嗪类利尿剂。",
        "对ACEI导致的干咳不耐受患者，应换用ARB类药物。",
    ]
