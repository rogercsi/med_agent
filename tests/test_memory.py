"""Tests for Mem0 memory client."""
import pytest
from unittest.mock import MagicMock, patch


def test_add_memory_calls_mem0(mock_settings):
    mock_mem = MagicMock()

    with patch("medical_agent.memory.mem0_client.get_memory", return_value=mock_mem):
        from medical_agent.memory.mem0_client import add_memory

        messages = [
            {"role": "user", "content": "我对青霉素过敏"},
            {"role": "assistant", "content": "好的，我已记录您对青霉素过敏"},
        ]
        add_memory(messages, "patient_001")

        mock_mem.add.assert_called_once_with(messages, user_id="patient_001")


def test_search_memory_uses_filters(mock_settings):
    mock_mem = MagicMock()
    mock_mem.search.return_value = {"results": [{"memory": "患者对青霉素过敏", "id": "abc"}]}

    with patch("medical_agent.memory.mem0_client.get_memory", return_value=mock_mem):
        from medical_agent.memory.mem0_client import search_memory

        results = search_memory("过敏", "patient_001", limit=5)

        # Must use filters= not user_id= to avoid mem0ai API asymmetry bug
        mock_mem.search.assert_called_once_with(
            "过敏", filters={"user_id": "patient_001"}, limit=5
        )
        assert len(results) == 1
        assert results[0]["memory"] == "患者对青霉素过敏"


def test_search_memory_handles_list_response(mock_settings):
    mock_mem = MagicMock()
    mock_mem.search.return_value = [{"memory": "患者有2型糖尿病", "id": "xyz"}]

    with patch("medical_agent.memory.mem0_client.get_memory", return_value=mock_mem):
        from medical_agent.memory.mem0_client import search_memory

        results = search_memory("糖尿病", "patient_001")
        assert len(results) == 1
