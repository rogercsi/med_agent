"""Tests for LangGraph agent state transitions."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from medical_agent.agent.state import PatientMemory


def test_patient_memory_to_prompt_str():
    pm = PatientMemory(
        allergies=["对青霉素过敏"],
        chronic_conditions=["2型糖尿病"],
        current_medications=["二甲双胍 500mg"],
    )
    s = pm.to_prompt_str()
    assert "青霉素" in s
    assert "糖尿病" in s
    assert "二甲双胍" in s


def test_patient_memory_empty():
    pm = PatientMemory()
    assert pm.to_prompt_str() == "暂无记录"


def test_route_after_safety_emergency():
    from medical_agent.agent.edges import route_after_safety

    state = {"is_emergency": True}
    assert route_after_safety(state) == "generate_response"


def test_route_after_safety_normal():
    from medical_agent.agent.edges import route_after_safety

    state = {"is_emergency": False}
    assert route_after_safety(state) == "retrieve_context"


def test_route_after_save_no_summarize():
    from medical_agent.agent.edges import route_after_save

    with patch("medical_agent.agent.edges.get_settings") as mock_s:
        mock_s.return_value.summarize_every_n_turns = 6
        state = {"turn_count": 3}
        assert route_after_save(state) == "__end__"


def test_route_after_save_triggers_summarize():
    from medical_agent.agent.edges import route_after_save

    with patch("medical_agent.agent.edges.get_settings") as mock_s:
        mock_s.return_value.summarize_every_n_turns = 6
        state = {"turn_count": 6}
        assert route_after_save(state) == "summarize_conversation"


@pytest.mark.asyncio
async def test_node_safety_check_emergency():
    from medical_agent.agent.nodes import node_safety_check
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="我突然感到剧烈的压迫性胸痛，向左臂放射，大汗，感觉要死了")],
        "rewritten_query": "压迫性胸痛放射至左臂",
    }
    result = await node_safety_check(state)
    assert result["is_emergency"] is True
    assert len(result["emergency_keywords"]) >= 2


@pytest.mark.asyncio
async def test_node_safety_check_normal():
    from medical_agent.agent.nodes import node_safety_check
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="我最近头痛，想了解一下高血压的治疗方法")],
        "rewritten_query": "高血压治疗",
    }
    result = await node_safety_check(state)
    assert result["is_emergency"] is False
